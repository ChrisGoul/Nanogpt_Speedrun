"""
SFT of the mix16 general model (156M, tied 16K vocab, 768x20) into an
instruction-following chatbot.

Same architecture/optimizer/dashboard as pretraining (imports from train.py):
  1. Start from pretrained mix16/model.pt instead of random.
  2. Loss MASKED to assistant tokens only (sft16/loss_mask.bin).
  3. Low LR, a few epochs over the SFT rows.
  4. Resumable (--ckpt-every / --resume), sleep-safe like pretraining.

Role turns are marked with the *text* strings "<|user|>"/"<|assistant|>"
(the mix16 tokenizer has no dedicated role tokens). Writes metrics.jsonl for
the live dashboard. Saves sft16/model.pt.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train import GPT, Config, Muon, HERE

SEQ_LEN = 512

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)   # seq 512 -> smaller batch
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--lr-scale", type=float, default=1.0)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = Tokenizer.from_file(os.path.join(HERE, "sft16", "tokenizer.json"))
    E = tok.token_to_id("<|endoftext|>")
    U_ids, A_ids = tok.encode("<|user|>\n").ids, tok.encode("<|assistant|>\n").ids
    cfg = Config(vocab_size=tok.get_vocab_size(), n_layer=args.layers,
                 n_head=args.heads, n_embd=args.dim, block_size=SEQ_LEN,
                 use_checkpoint=True, tie_embeddings=True)

    model = GPT(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(HERE, "mix16", "model.pt")))
    print(f"loaded mix16 base, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)

    tokens = np.memmap(os.path.join(HERE, "sft16", "tokens.bin"), dtype=np.uint16, mode="r").reshape(-1, SEQ_LEN)
    mask = np.memmap(os.path.join(HERE, "sft16", "loss_mask.bin"), dtype=np.uint8, mode="r").reshape(-1, SEQ_LEN)
    n_examples = tokens.shape[0]
    print(f"{n_examples:,} SFT examples", flush=True)

    def get_batch():
        ix = torch.randint(n_examples, (args.batch_size,))
        x = torch.from_numpy(tokens[ix].astype(np.int64)).to(device)
        m = torch.from_numpy(mask[ix].astype(np.float32)).to(device)
        return x, m

    hidden = [p for p in model.blocks.parameters() if p.ndim == 2]
    embed_head = list({id(p): p for p in [model.wte.weight, model.lm_head.weight]}.values())
    opt_muon = Muon(hidden, lr=0.004 * args.lr_scale, momentum=0.95)
    opt_adam = torch.optim.Adam(embed_head, lr=6e-4 * args.lr_scale, betas=(0.9, 0.95))
    optimizers = [opt_muon, opt_adam]
    base_lrs = [[g["lr"] for g in o.param_groups] for o in optimizers]

    out_dir = os.path.join(HERE, "sft16")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        for opt, st in zip(optimizers, ck["optimizers"]):
            opt.load_state_dict(st)
        start_step = ck["step"] + 1
        print(f"resumed from {ckpt_path} at step {start_step}", flush=True)

    def save_ckpt(step):
        tmp = ckpt_path + ".tmp"
        torch.save({"step": step, "model": model.state_dict(),
                    "optimizers": [o.state_dict() for o in optimizers]}, tmp)
        os.replace(tmp, ckpt_path)

    def lr_mult(step):
        warmup = 100
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (args.steps - step) / (args.steps - warmup))  # linear decay to 0

    def masked_loss(x, m):
        with torch.autocast(device, dtype=torch.bfloat16):
            logits, _ = model(x)
        logits = logits[:, :-1].reshape(-1, logits.size(-1)).float()
        targets = x[:, 1:].reshape(-1)
        w = m[:, 1:].reshape(-1)
        per_tok = F.cross_entropy(logits, targets, reduction="none")
        return (per_tok * w).sum() / w.sum().clamp(min=1)

    mode = "a" if start_step > 0 else "w"
    log_file = open(os.path.join(HERE, "metrics.jsonl"), mode, buffering=1)
    def log(**kv): log_file.write(json.dumps(kv) + "\n")
    log(event="start", num_steps=args.steps, params_m=round(sum(p.numel() for p in model.parameters())/1e6, 1),
        device=device, batch_size=args.batch_size, block_size=SEQ_LEN,
        prompt="<|user|>What is a dog?<|assistant|>", data="sft16")

    chat_prompts = ["What is a dog?", "Tell me a story about a cat.",
                    "What is 17 plus 25?", "Why is the sky blue?"]

    def chat_sample(q):
        ids = U_ids + tok.encode(q).ids + A_ids
        idx = torch.tensor([ids], device=device)
        model.eval()
        with torch.no_grad():
            for _ in range(120):
                with torch.autocast(device, dtype=torch.bfloat16):
                    logits, _ = model(idx[:, -SEQ_LEN:])
                nxt = torch.multinomial(F.softmax(logits[:, -1].float() / 0.7, dim=-1), 1)
                idx = torch.cat([idx, nxt], 1)
                if nxt.item() == E:
                    break
        model.train()
        full = tok.decode(idx[0].tolist())
        return f"USER: {q}\nBOT: " + full.split("<|assistant|>")[-1].replace("<|endoftext|>", "").strip()

    t0 = time.time()
    for step in range(start_step, args.steps):
        if step % args.eval_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                vl = np.mean([masked_loss(*get_batch()).item() for _ in range(10)])
            sample = "\n\n".join(chat_sample(q) for q in chat_prompts)
            print(f"step {step:5d} | loss {vl:.4f} | {time.time()-t0:.0f}s", flush=True)
            log(event="val", step=step, val_loss=round(float(vl), 4), time_s=round(time.time()-t0, 1))
            log(event="sample", step=step, text=sample)
        if args.ckpt_every and step > start_step and step % args.ckpt_every == 0:
            save_ckpt(step)

        x, m = get_batch()
        loss = masked_loss(x, m)
        loss.backward()
        mult = lr_mult(step)
        for opt, lrs in zip(optimizers, base_lrs):
            for g, base in zip(opt.param_groups, lrs):
                g["lr"] = base * mult
            opt.step()
        model.zero_grad(set_to_none=True)
        if step % 5 == 0:
            log(event="train", step=step, train_loss=round(loss.item(), 4),
                lr_mult=round(mult, 4), time_s=round(time.time()-t0, 1),
                tok_per_s=round(args.batch_size * SEQ_LEN * (step+1-start_step) / max(time.time()-t0, 1e-9)))

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    final = "\n\n".join(chat_sample(q) for q in chat_prompts)
    print("\n--- final chat ---\n" + final)
    log(event="done", time_s=round(time.time()-t0, 1), sample=final)
    log_file.close()

if __name__ == "__main__":
    main()
