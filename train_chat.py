"""
Train the Route A chatbot from scratch on multi-turn dialogue (chat/).

Same architecture/optimizer as train.py, with SFT-style masked loss (graded on
assistant turns only) - but trained from random init, since this is a fresh
16K-vocab model in a new domain (not a fine-tune of the blend model).

Writes metrics.jsonl for the dashboard (chat samples every eval). Saves
chatmodel/model.pt.
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--cooldown-frac", type=float, default=0.4)
    ap.add_argument("--init", default=None, help="subfolder with model.pt to initialize from (e.g. books); random init if omitted")
    ap.add_argument("--data", default="chat", help="subfolder with tokens.bin/loss_mask.bin/tokenizer.json")
    ap.add_argument("--out", default="chatmodel", help="subfolder to save model.pt")
    args = ap.parse_args()

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = Tokenizer.from_file(os.path.join(HERE, args.data, "tokenizer.json"))
    U = tok.token_to_id("<|user|>")
    A = tok.token_to_id("<|assistant|>")
    E = tok.token_to_id("<|endoftext|>")
    cfg = Config(vocab_size=tok.get_vocab_size(), n_layer=args.layers,
                 n_head=args.heads, n_embd=args.dim)

    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if args.init:
        model.load_state_dict(torch.load(os.path.join(HERE, args.init, "model.pt")))
        print(f"initialized from {args.init}/model.pt, {n_params/1e6:.1f}M params, vocab {cfg.vocab_size}", flush=True)
    else:
        print(f"training chatbot from scratch, {n_params/1e6:.1f}M params, vocab {cfg.vocab_size}", flush=True)

    tokens = np.memmap(os.path.join(HERE, args.data, "tokens.bin"), dtype=np.uint16, mode="r").reshape(-1, cfg.block_size)
    mask = np.memmap(os.path.join(HERE, args.data, "loss_mask.bin"), dtype=np.uint8, mode="r").reshape(-1, cfg.block_size)
    n_examples = tokens.shape[0]
    print(f"{n_examples:,} conversations", flush=True)

    def get_batch():
        ix = torch.randint(n_examples, (args.batch_size,))
        x = torch.from_numpy(tokens[ix].astype(np.int64)).to(device)
        m = torch.from_numpy(mask[ix].astype(np.float32)).to(device)
        return x, m

    hidden = [p for p in model.blocks.parameters() if p.ndim == 2]
    embed_head = [model.wte.weight, model.lm_head.weight]
    # lower LR when fine-tuning a pretrained model; full LR from scratch
    muon_lr, adam_lr = (0.006, 6e-4) if args.init else (0.02, 3e-3)
    opt_muon = Muon(hidden, lr=muon_lr, momentum=0.95)
    opt_adam = torch.optim.Adam(embed_head, lr=adam_lr, betas=(0.9, 0.95))
    optimizers = [opt_muon, opt_adam]
    base_lrs = [[g["lr"] for g in o.param_groups] for o in optimizers]

    def lr_mult(step):
        warmup = 100
        if step < warmup:
            return (step + 1) / warmup
        frac = step / args.steps
        if frac < 1 - args.cooldown_frac:
            return 1.0
        return (1 - frac) / args.cooldown_frac

    def masked_loss(x, m):
        with torch.autocast(device, dtype=torch.bfloat16):
            logits, _ = model(x)
        logits = logits[:, :-1].reshape(-1, logits.size(-1)).float()
        targets = x[:, 1:].reshape(-1)
        w = m[:, 1:].reshape(-1)
        per_tok = F.cross_entropy(logits, targets, reduction="none")
        return (per_tok * w).sum() / w.sum().clamp(min=1)

    log_file = open(os.path.join(HERE, "metrics.jsonl"), "w", buffering=1)
    def log(**kv): log_file.write(json.dumps(kv) + "\n")
    log(event="start", num_steps=args.steps, params_m=round(n_params/1e6, 1),
        device=device, batch_size=args.batch_size, block_size=cfg.block_size,
        prompt="<|user|>Hi! How are you?<|assistant|>", data="chat")

    openers = ["Hi! How are you?",
               "What is the capital of France?",
               "If there are 5 birds on a tree and 2 fly away, how many are left?",
               "Context: Tom has a red bike. Question: What color is Tom's bike?"]

    def chat_sample(q):
        ids = [U] + tok.encode(q).ids + [A]
        idx = torch.tensor([ids], device=device)
        start = idx.size(1)
        model.eval()
        with torch.no_grad():
            for _ in range(80):
                logits, _ = model(idx[:, -cfg.block_size:])
                nxt = torch.multinomial(F.softmax(logits[:, -1] / 0.8, dim=-1), 1)
                idx = torch.cat([idx, nxt], 1)
                if nxt.item() == E:
                    break
        model.train()
        return f"USER: {q}\nBOT: " + tok.decode(idx[0, start:].tolist()).strip()

    os.makedirs(os.path.join(HERE, args.out), exist_ok=True)
    t0 = time.time()
    for step in range(args.steps):
        if step % args.eval_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                vl = np.mean([masked_loss(*get_batch()).item() for _ in range(10)])
            sample = "\n\n".join(chat_sample(q) for q in openers)
            print(f"step {step:5d} | loss {vl:.4f} | {time.time()-t0:.0f}s", flush=True)
            log(event="val", step=step, val_loss=round(float(vl), 4), time_s=round(time.time()-t0, 1))
            log(event="sample", step=step, text=sample)

        x, m = get_batch()
        loss = masked_loss(x, m)
        loss.backward()
        mult = lr_mult(step)
        for opt, lrs in zip(optimizers, base_lrs):
            for g, base in zip(opt.param_groups, lrs):
                g["lr"] = base * mult
            opt.step()
        model.zero_grad(set_to_none=True)
        if step % 10 == 0:
            log(event="train", step=step, train_loss=round(loss.item(), 4),
                lr_mult=round(mult, 4), time_s=round(time.time()-t0, 1),
                tok_per_s=round(args.batch_size * cfg.block_size * (step+1) / max(time.time()-t0, 1e-9)))

    torch.save(model.state_dict(), os.path.join(HERE, args.out, "model.pt"))
    final = "\n\n".join(chat_sample(q) for q in openers)
    print("\n--- final chat ---\n" + final)
    log(event="done", time_s=round(time.time()-t0, 1), sample=final)
    log_file.close()

if __name__ == "__main__":
    main()
