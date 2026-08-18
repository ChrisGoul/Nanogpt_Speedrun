"""
A simplified, single-GPU version of the modded-nanogpt speedrun
(https://github.com/KellerJordan/modded-nanogpt), scaled down to train
on tiny shakespeare in a few minutes on a consumer GPU.

Speedrun tricks kept (see README.md for why each one helps):
  - Muon optimizer (Newton-Schulz orthogonalized momentum) for hidden weights,
    Adam for embeddings and the lm_head
  - Rotary position embeddings (RoPE) instead of learned position embeddings
  - QK-norm: RMS-normalize queries and keys before attention
  - ReLU^2 activation in the MLP
  - Zero-initialized output projections (attn proj + mlp proj + lm_head)
  - Untied embedding / lm_head
  - No biases anywhere, RMSNorm with no learnable params
  - Logit soft-capping with tanh
  - Trapezoidal (warmup-stable-decay) learning rate schedule

Dropped for simplicity: FP8, FlexAttention w/ sliding windows, value/skip
embeddings (U-net), distributed training, torch.compile-specific tuning.
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Approximately orthogonalize G, i.e. compute U V^T from its SVD U S V^T,
    using a quintic Newton-Schulz iteration. Runs in bfloat16 on the GPU;
    the iteration's coefficients are tuned so it converges fast even though
    it doesn't converge all the way to exactly 1 (that's fine in practice).
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # transpose so we iterate on the smaller side
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)  # ensure spectral norm <= 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon: momentum SGD, but the update matrix is orthogonalized before being
    applied. Intuition: gradient updates to a weight matrix are dominated by
    a few large singular values; orthogonalization rebalances them so all
    directions of the update contribute equally. Only makes sense for 2D
    hidden-layer weight matrices (not embeddings, not the head, not 1D params).
    """
    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                # nesterov momentum
                g = p.grad.add(buf, alpha=group["momentum"])
                g = zeropower_via_newtonschulz5(g)
                # scale the update so its RMS is comparable to Adam's
                scale = max(1, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-group["lr"] * scale)

# -----------------------------------------------------------------------------
# Model

@dataclass
class Config:
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 512
    n_experts: int = 0        # 0 = dense MLP; >0 = Mixture-of-Experts
    top_k: int = 2            # experts active per token
    moe_ff: int = 0           # hidden width per expert (0 -> auto: match dense active FLOPs)
    use_checkpoint: bool = False  # gradient checkpointing: recompute block activations in backward
    use_abacus: bool = False      # abacus embeddings: tag each digit with its place value (for length generalization)
    abacus_size: int = 64         # max place-value index (covers train + random offset + test lengths)
    tie_embeddings: bool = False  # share input embedding and output head (frees params at small scale)

def norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm without learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    """Rotary position embeddings: rotate each (even, odd) pair of head-dim
    channels by an angle proportional to the token's position. Relative
    positions then fall out of the q.k dot product for free."""
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = base ** (-torch.arange(0, head_dim, 2) / head_dim)
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_head, T, head_dim)
        T = x.size(2)
        cos, sin = self.cos[:T], self.sin[:T]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj.weight.data.zero_()  # zero-init: block starts as identity
        self.rotary = Rotary(self.head_dim, cfg.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = norm(q), norm(k)          # QK-norm
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.proj.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.fc(x)).square())  # ReLU^2

class Expert(nn.Module):
    """A single ReLU^2 MLP expert (same shape as the dense MLP, narrower)."""
    def __init__(self, cfg: Config, ff: int):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, ff, bias=False)
        self.proj = nn.Linear(ff, cfg.n_embd, bias=False)
        self.proj.weight.data.zero_()   # zero-init: block starts as identity

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())

class MoE(nn.Module):
    """Top-k Mixture-of-Experts MLP. A per-token router picks top_k of n_experts;
    each expert's hidden width is sized so the ACTIVE FLOPs match the dense MLP,
    so total params grow but compute-per-token stays constant. Emits a Switch-
    style load-balancing aux loss in self.aux.

    Dispatch is capacity-based and fully batched: tokens are packed into an
    (E, capacity, d) buffer and all experts run as a single bmm (no Python loop
    over experts). Experts are kept as 2D nn.Linear weights (so Muon handles
    them) and stacked into 3D only for the bmm."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_exp, self.top_k = cfg.n_experts, cfg.top_k
        ff = cfg.moe_ff or max(1, (4 * cfg.n_embd) // cfg.top_k)   # match dense active FLOPs
        self.router = nn.Linear(cfg.n_embd, self.n_exp, bias=False)
        self.experts = nn.ModuleList(Expert(cfg, ff) for _ in range(self.n_exp))
        self.cap_factor = 1.25
        self.aux = torch.tensor(0.0)

    def _route(self, xf):
        probs = F.softmax(self.router(xf), dim=-1)          # (N, E)
        topv, topi = probs.topk(self.top_k, dim=-1)         # (N, k)
        topv = topv / topv.sum(-1, keepdim=True)
        me = probs.mean(0)
        disp = F.one_hot(topi.reshape(-1), self.n_exp).float().mean(0)
        self.aux = self.n_exp * (me * disp).sum()           # load-balance aux
        return topv, topi

    def forward(self, x):
        # per-expert loop: fastest at THIS scale (small memory-bound matmuls),
        # where capacity-batched dispatch's padding+scatter overhead dominates.
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        topv, topi = self._route(xf)
        out = torch.zeros_like(xf)
        for e, expert in enumerate(self.experts):
            sel = (topi == e)
            tok = sel.any(-1)
            if tok.any():
                idx = tok.nonzero(as_tuple=True)[0]
                out[idx] += (topv * sel).sum(-1)[idx].unsqueeze(-1) * expert(xf[idx])
        return out.reshape(B, T, C)

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MoE(cfg) if cfg.n_experts > 0 else MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: Config, digit_ids=None):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight   # share input/output embeddings
        else:
            self.lm_head.weight.data.zero_()        # untied from wte, zero-init
        if cfg.use_abacus:
            # one learned embedding per place-value index (ones, tens, hundreds, ...)
            self.abacus_emb = nn.Embedding(cfg.abacus_size, cfg.n_embd)
            nn.init.normal_(self.abacus_emb.weight, std=0.02)
            # digit token ids stored as a buffer so they load with the checkpoint
            self.register_buffer("digit_ids", torch.tensor(digit_ids if digit_ids else [0] * 10, dtype=torch.long))

    def _abacus_pos(self, idx):
        """Place-value index of each token: 0 for the ones digit, 1 for tens, ...
        Computed by reverse-scanning each contiguous run of digit tokens."""
        is_digit = (idx.unsqueeze(-1) == self.digit_ids).any(-1)          # (B, T)
        rev = is_digit.flip(1).long()
        c = rev.cumsum(1)
        reset = torch.cummax(torch.where(rev.bool(), torch.zeros_like(c), c), dim=1).values
        right1 = (c - reset).flip(1)                                      # 1-based index from the right, 0 at non-digit
        pos = (right1 - 1).clamp(min=0)
        return pos, is_digit

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        if self.cfg.use_abacus:
            pos, is_digit = self._abacus_pos(idx)
            if self.training:                       # random per-sequence offset -> length generalization
                off = torch.randint(0, self.cfg.abacus_size // 2, (idx.size(0), 1), device=idx.device)
                pos = (pos + off).clamp(max=self.cfg.abacus_size - 1)
            x = x + self.abacus_emb(pos) * is_digit.unsqueeze(-1)
        x = norm(x)
        for block in self.blocks:
            if self.cfg.use_checkpoint and self.training:
                # recompute this block's activations during backward instead of
                # storing them — trades ~30% compute for a big activation-memory cut
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = norm(x)
        # sum MoE load-balance aux losses (0 for a dense model)
        self.aux_loss = sum((b.mlp.aux for b in self.blocks if isinstance(b.mlp, MoE)),
                            start=torch.zeros((), device=idx.device))
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)  # logit soft-capping
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

# -----------------------------------------------------------------------------
# Data loading

@torch.no_grad()
def bench_acc(model, encode, items, device, block):
    """Length-normalized multiple-choice accuracy (acc_norm) on the LIVE model —
    a lightweight capability signal to log during training."""
    model.eval()
    correct = 0
    for ctx, choices, gold in items:
        cids = encode(ctx)
        best, best_i = -1e30, 0
        for i, ch in enumerate(choices):
            xids = encode(" " + ch)
            if not xids:
                continue
            ids = (cids + xids)[-block:]
            idx = torch.tensor([ids], device=device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, _ = model(idx)
            lp = F.log_softmax(logits.float(), -1)[0]
            n = len(xids)
            s = sum(lp[len(ids) - n + j - 1, xids[j]].item() for j in range(n)) / max(len(ch), 1)
            if s > best:
                best, best_i = s, i
        correct += int(best_i == gold)
    model.train()
    return round(100 * correct / max(len(items), 1), 1)

def gpu_peak_tflops(device):
    """Best-guess bf16 dense (FP32-accumulate) peak TFLOP/s for the current GPU,
    used only to turn achieved FLOP/s into an MFU %. Consumer Ampere/Ada run
    FP32-accumulate at half the marketing FP16 rate — these are the training-
    relevant numbers. Unknown cards return 0 (MFU logging just skips)."""
    if device != "cuda" or not torch.cuda.is_available():
        return 0.0
    name = torch.cuda.get_device_name().lower()
    table = {"3050": 9.0, "3060": 13.0, "3090": 35.0, "4070": 29.0, "4080": 49.0,
             "4090": 83.0, "a100": 312.0, "h100": 989.0, "h200": 989.0, "l40": 90.0, "l4": 30.0}
    for key, tflops in table.items():
        if key in name:
            return tflops
    return 0.0

def grad_global_norm(model):
    """Total L2 norm of all gradients (the same quantity clip_grad_norm reports)."""
    gs = [p.grad for p in model.parameters() if p.grad is not None]
    if not gs:
        return 0.0
    return torch.norm(torch.stack([g.detach().norm() for g in gs])).item()

def _gather_batch(data, batch_size, block_size):
    """One vectorized fancy-index gather instead of a Python loop over the batch:
    build (batch, block+1) contiguous int64 windows in a single indexing op."""
    limit = len(data) - block_size - 1
    ix = torch.randint(limit, (batch_size,)).numpy()
    idx = ix[:, None] + np.arange(block_size + 1)[None, :]        # (bs, block+1)
    chunk = data[idx].astype(np.int64)                            # contiguous
    x = torch.from_numpy(np.ascontiguousarray(chunk[:, :-1]))
    y = torch.from_numpy(np.ascontiguousarray(chunk[:, 1:]))
    return x, y

def get_batch(data_dir: str, split: str, batch_size: int, block_size: int, device):
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    x, y = _gather_batch(data, batch_size, block_size)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

class Prefetcher:
    """Overlaps batch prep + host->device copy with GPU compute. Opens the .bin
    once, gathers each batch vectorized (no Python loop), stages it in pinned
    memory, and copies on a side CUDA stream one step ahead — so the GPU never
    stalls on the loader. Big win when the GPU is fast relative to the per-step
    work (small batch/seq on a datacenter GPU, where MFU is loader-bound)."""
    def __init__(self, data_dir, split, batch_size, block_size, device):
        self.data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
        self.bs, self.bl, self.device = batch_size, block_size, device
        self.cuda = str(device).startswith("cuda")
        self.stream = torch.cuda.Stream() if self.cuda else None
        self._preload()

    def _preload(self):
        x, y = _gather_batch(self.data, self.bs, self.bl)
        if self.cuda:
            x, y = x.pin_memory(), y.pin_memory()
            with torch.cuda.stream(self.stream):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
        self.nx, self.ny = x, y

    def next(self):
        if self.cuda:
            torch.cuda.current_stream().wait_stream(self.stream)   # ensure copy done
            self.nx.record_stream(torch.cuda.current_stream())
            self.ny.record_stream(torch.cuda.current_stream())
        x, y = self.nx, self.ny
        self._preload()                                            # kick off next copy
        return x, y

# -----------------------------------------------------------------------------
# Training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", help="subfolder with train.bin/val.bin (and optional tokenizer.json)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--vocab-size", type=int, default=50257)
    ap.add_argument("--seq", type=int, default=512, help="context / sequence length (block_size)")
    ap.add_argument("--prompt", default="The meaning of life is")
    ap.add_argument("--run", default=None, help="name for the metrics archive (defaults to --data)")
    ap.add_argument("--out", default=None, help="folder to save model.pt (defaults to --data); tokenizer is copied in")
    ap.add_argument("--experts", type=int, default=0, help=">0 enables Mixture-of-Experts")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--moe-ff", type=int, default=0, help="expert hidden width (0=auto match dense active FLOPs)")
    ap.add_argument("--aux-coef", type=float, default=0.01, help="MoE load-balance loss weight")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (may be unsupported on Windows)")
    ap.add_argument("--lr-scale", type=float, default=1.0, help="multiply both base learning rates")
    ap.add_argument("--checkpoint", action="store_true", help="gradient checkpointing (fit a bigger model in VRAM)")
    ap.add_argument("--abacus", action="store_true", help="abacus place-value embeddings (arithmetic length generalization)")
    ap.add_argument("--tie", action="store_true", help="tie input and output embeddings")
    ap.add_argument("--bench", action="store_true", help="run lightweight PIQA+HellaSwag during training (logged for the dashboard)")
    ap.add_argument("--bench-every", type=int, default=2000)
    ap.add_argument("--bench-n", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000, help="save resumable ckpt.pt every N steps (0=off)")
    ap.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt if present")
    ap.add_argument("--peak-tflops", type=float, default=0.0, help="GPU bf16 dense peak TFLOP/s for MFU (0=auto-detect)")
    args = ap.parse_args()

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(vocab_size=args.vocab_size, n_layer=args.layers,
                 n_head=args.heads, n_embd=args.dim,
                 n_experts=args.experts, top_k=args.top_k, moe_ff=args.moe_ff,
                 block_size=args.seq, use_checkpoint=args.checkpoint,
                 use_abacus=args.abacus, tie_embeddings=args.tie)
    data_dir = os.path.join(HERE, args.data)

    # tokenizer: custom BPE if the data folder ships one, else GPT-2
    tok_path = os.path.join(data_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(tok_path)
        if args.vocab_size == 50257:  # not overridden: infer from tokenizer
            cfg.vocab_size = _tok.get_vocab_size()
        encode, decode = (lambda s: _tok.encode(s).ids), _tok.decode
    else:
        import tiktoken
        _tok = tiktoken.get_encoding("gpt2")
        encode, decode = _tok.encode, _tok.decode

    batch_size = args.batch_size
    num_steps = args.steps
    cooldown_frac = 0.4  # last 40% of training decays LR to 0
    eval_every = args.eval_every

    digit_ids = [encode(str(d))[0] for d in range(10)] if cfg.use_abacus else None
    model = GPT(cfg, digit_ids=digit_ids).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if cfg.n_experts > 0:
        # active params = total minus the (n_experts - top_k)/n_experts of expert weights
        exp_params = sum(p.numel() for b in model.blocks for p in b.mlp.experts.parameters())
        active = n_params - exp_params * (cfg.n_experts - cfg.top_k) / cfg.n_experts
        print(f"device={device}, MoE {cfg.n_experts}x top-{cfg.top_k}: "
              f"total {n_params/1e6:.1f}M / active {active/1e6:.1f}M params")
    else:
        print(f"device={device}, params={n_params/1e6:.1f}M")

    if args.compile:
        orig = model
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 64   # tolerate a few shapes without thrashing
        try:
            model = torch.compile(model)
            # compilation is lazy (happens on first forward) — trigger it now with
            # a warmup at the REAL training shape so a broken backend (e.g. no Triton)
            # fails HERE and we fall back, instead of crashing mid-loop.
            _wu = torch.zeros((batch_size, cfg.block_size), dtype=torch.long, device=device)
            with torch.autocast(device, dtype=torch.bfloat16):
                model(_wu)
            print("torch.compile enabled")
        except Exception as e:
            model = orig
            print(f"torch.compile unavailable ({type(e).__name__}); continuing uncompiled", flush=True)
    # eval / generate / bench run on the UNCOMPILED module: their variable-length,
    # no_grad inputs (200 bench items, growing generate lengths) would otherwise
    # thrash torch.compile's cache and stall startup. Only the fixed-shape training
    # forward uses the compiled `model`.
    eval_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Split params: Muon for 2D hidden matrices, Adam for embedding + head.
    hidden = [p for p in model.blocks.parameters() if p.ndim == 2]
    # dedupe: when embeddings are tied, wte.weight IS lm_head.weight
    embed_head = list({id(p): p for p in [model.wte.weight, model.lm_head.weight]}.values())
    opt_muon = Muon(hidden, lr=0.02 * args.lr_scale, momentum=0.95)
    opt_adam = torch.optim.Adam(embed_head, lr=0.003 * args.lr_scale, betas=(0.9, 0.95))
    optimizers = [opt_muon, opt_adam]
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]

    # resumable checkpoint dir (same place the final model.pt lands)
    out_name = args.out or args.run or os.path.basename(data_dir.rstrip("/\\"))
    out_dir = os.path.join(HERE, out_name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(ck["model"])
        for opt, st in zip(optimizers, ck["optimizers"]):
            opt.load_state_dict(st)
        start_step = ck["step"] + 1
        print(f"resumed from {ckpt_path} at step {start_step}", flush=True)

    def save_ckpt(step):
        tmp = ckpt_path + ".tmp"
        torch.save({"step": step,
                    "model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                    "optimizers": [o.state_dict() for o in optimizers]}, tmp)
        os.replace(tmp, ckpt_path)   # atomic: never leave a half-written ckpt

    def lr_mult(step: int) -> float:
        # trapezoidal schedule: brief warmup, flat, then linear decay
        warmup = 50
        if step < warmup:
            return (step + 1) / warmup
        frac_done = step / num_steps
        if frac_done < 1 - cooldown_frac:
            return 1.0
        return (1 - frac_done) / cooldown_frac

    # live file (dashboard polls this) + a per-run archive so A/B runs don't
    # overwrite each other and can be compared afterwards
    run_name = args.run or args.data.strip("./\\").replace("/", "_") or "run"
    mode = "a" if start_step > 0 else "w"   # keep dashboard history when resuming
    log_files = [open(os.path.join(HERE, "metrics.jsonl"), mode, buffering=1),
                 open(os.path.join(HERE, f"metrics_{run_name}.jsonl"), mode, buffering=1)]

    def log(**kv):
        line = json.dumps(kv) + "\n"
        for f in log_files:
            f.write(line)

    log(event="start", num_steps=num_steps, params_m=round(n_params / 1e6, 1),
        device=device, batch_size=batch_size, block_size=cfg.block_size,
        prompt=args.prompt, data=args.data)

    bench_items = {}
    if args.bench:
        from eval_base import load_piqa, load_hellaswag
        bench_items = {"piqa": load_piqa(args.bench_n), "hellaswag": load_hellaswag(args.bench_n)}
        print(f"live bench loaded: piqa {len(bench_items['piqa'])}, hellaswag {len(bench_items['hellaswag'])}", flush=True)

    prompt_ids = torch.tensor([encode(args.prompt)], device=device)

    # compute-metrics setup: FLOPs/token = 6N (matmuls) + 12*L*d*T (attention),
    # matching nanoGPT's estimate_mfu; peak lets us report MFU %.
    peak_tflops = args.peak_tflops or gpu_peak_tflops(device)
    flops_per_token = 6 * n_params + 12 * cfg.n_layer * cfg.n_embd * cfg.block_size
    if peak_tflops:
        print(f"MFU tracking: peak {peak_tflops:.0f} TFLOP/s, {flops_per_token/1e9:.2f} GFLOP/token", flush=True)

    train_pf = Prefetcher(data_dir, "train", batch_size, cfg.block_size, device)

    t0 = time.time()
    for step in range(start_step, num_steps):
        # evaluation + sample generation
        if step % eval_every == 0 or step == num_steps - 1:
            eval_model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(20):
                    x, y = get_batch(data_dir, "val", batch_size, cfg.block_size, device)
                    with torch.autocast(device, dtype=torch.bfloat16):
                        _, loss = eval_model(x, y)
                    losses.append(loss.item())
                out = eval_model.generate(prompt_ids.clone(), max_new_tokens=64)
            sample = decode(out[0].tolist())
            print(f"step {step:4d} | val loss {np.mean(losses):.4f} | {time.time()-t0:.1f}s")
            log(event="val", step=step, val_loss=round(float(np.mean(losses)), 4),
                time_s=round(time.time() - t0, 1))
            log(event="sample", step=step, text=sample)
            if bench_items and (step % args.bench_every == 0 or step == num_steps - 1):
                scores = {k: bench_acc(eval_model, encode, v, device, cfg.block_size) for k, v in bench_items.items()}
                print("  bench: " + "  ".join(f"{k} {s}%" for k, s in scores.items()), flush=True)
                log(event="bench", step=step, **scores)
            model.train()

        # periodic resumable checkpoint (atomic write)
        if args.ckpt_every and step > start_step and step % args.ckpt_every == 0:
            save_ckpt(step)

        # training step (timed, excluding eval, for instantaneous throughput/MFU)
        log_now = (step % 5 == 0)
        if device == "cuda":
            torch.cuda.synchronize()
        step_t0 = time.time()
        x, y = train_pf.next()
        with torch.autocast(device, dtype=torch.bfloat16):
            _, loss = model(x, y)
            if cfg.n_experts > 0:
                loss = loss + args.aux_coef * model.aux_loss   # keep experts balanced
        loss.backward()
        gnorm = grad_global_norm(model) if log_now else None   # capture before zero_grad
        m = lr_mult(step)
        for opt, lrs in zip(optimizers, base_lrs):
            for group, base in zip(opt.param_groups, lrs):
                group["lr"] = base * m
            opt.step()
        model.zero_grad(set_to_none=True)
        if log_now:
            if device == "cuda":
                torch.cuda.synchronize()
            step_dt = time.time() - step_t0
            tok_s = batch_size * cfg.block_size / max(step_dt, 1e-9)
            achieved_tflops = flops_per_token * tok_s / 1e12
            kv = dict(event="train", step=step, train_loss=round(loss.item(), 4),
                      lr_mult=round(m, 4), time_s=round(time.time() - t0, 1),
                      tok_per_s=round(tok_s), grad_norm=round(gnorm, 3),
                      tflops=round(achieved_tflops, 2))
            if peak_tflops:
                kv["mfu"] = round(100 * achieved_tflops / peak_tflops, 1)
            if device == "cuda":
                kv["gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6)
            log(**kv)

    # final model.pt in out_dir (computed above, next to ckpt.pt)
    # unwrap torch.compile so the checkpoint has plain keys (no _orig_mod prefix)
    sd = (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict()
    torch.save(sd, os.path.join(out_dir, "model.pt"))
    if out_dir != data_dir and os.path.exists(tok_path):
        import shutil
        shutil.copy(tok_path, os.path.join(out_dir, "tokenizer.json"))
    print(f"saved {out_dir}/model.pt")

    # final, longer sample from the model
    eval_model.eval()
    out = eval_model.generate(prompt_ids.clone(), max_new_tokens=200)
    sample = decode(out[0].tolist())
    print("\n--- sample ---")
    print(sample)
    log(event="done", time_s=round(time.time() - t0, 1), sample=sample)
    for f in log_files:
        f.close()

if __name__ == "__main__":
    main()
