"""How big a model fits on this GPU, with vs without gradient checkpointing.
Real Muon+Adam optimizers (realistic memory), batch 32, peak VRAM + throughput.
'SPILL' = peak exceeded dedicated VRAM and paged to shared RAM (10x slower)."""
import time
import torch
from train import GPT, Config, Muon

def bench(dim, layers, heads, ckpt, batch=32, T=512, steps=10):
    dev = "cuda"
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg = Config(vocab_size=8192, n_layer=layers, n_head=heads, n_embd=dim,
                 block_size=T, use_checkpoint=ckpt)
    m = GPT(cfg).to(dev); m.train()
    hidden = [p for p in m.blocks.parameters() if p.ndim == 2]
    eh = [m.wte.weight, m.lm_head.weight]
    om = Muon(hidden, lr=0.02, momentum=0.95)
    oa = torch.optim.Adam(eh, lr=0.003, betas=(0.9, 0.95))
    x = torch.randint(0, 8192, (batch, T), device=dev)
    y = torch.randint(0, 8192, (batch, T), device=dev)
    def one():
        with torch.autocast(dev, dtype=torch.bfloat16):
            _, loss = m(x, y)
        loss.backward(); om.step(); oa.step(); m.zero_grad(set_to_none=True)
    for _ in range(4):
        one()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    tok_s = batch * T * steps / (time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 1e9
    params = sum(p.numel() for p in m.parameters()) / 1e6
    del m, om, oa
    torch.cuda.empty_cache()
    return params, peak, tok_s

if __name__ == "__main__":
    print(f"torch {torch.__version__}  |  batch 32, seq 512, Muon+Adam\n", flush=True)
    print(f"{'config':26} {'params':>8} {'peakGB':>7} {'tok/s':>8}  note", flush=True)
    print("-" * 62, flush=True)
    configs = [
        ("640x8   no-ckpt (baseline)", 640, 8, 10, False),
        ("640x8   ckpt", 640, 8, 10, True),
        ("1024x12 no-ckpt", 1024, 12, 16, False),
        ("1024x12 ckpt", 1024, 12, 16, True),
        ("1280x16 ckpt", 1280, 16, 20, True),
        ("1536x20 ckpt", 1536, 20, 24, True),
    ]
    for name, d, l, h, c in configs:
        try:
            p, pk, ts = bench(d, l, h, c)
            note = "SPILL (paged to RAM)" if (pk > 8.2 or ts < 4000) else "fits"
            print(f"{name:26} {p:6.0f}M {pk:6.1f} {ts:8.0f}  {note}", flush=True)
        except Exception as e:
            print(f"{name:26} FAILED {type(e).__name__}: {str(e)[:45]}", flush=True)
            torch.cuda.empty_cache()
