"""
Supervised fine-tuning (SFT) of the pretrained blend model into a little
instruction-following / Q&A bot.

Same architecture, optimizer, and dashboard as pretraining (imports from
train.py), with three changes that define SFT:
  1. Start from the pretrained weights (blend/model.pt) instead of random.
  2. Loss is MASKED to the response tokens only (loss_mask.bin) - the model is
     graded on its answers, not on echoing the user's prompt.
  3. Low LR, a couple of epochs over the SFT pairs.

Writes metrics.jsonl in the speedrun folder so the existing dashboard shows it
live (chat-style samples every eval). Saves sft/model.pt.
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
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = Tokenizer.from_file(os.path.join(HERE, "blend", "tokenizer.json"))
    U = tok.token_to_id("<|user|>")
    A = tok.token_to_id("<|assistant|>")
    E = tok.token_to_id("<|endoftext|>")
    cfg = Config(vocab_size=tok.get_vocab_size(), n_layer=args.layers,
                 n_head=args.heads, n_embd=args.dim)

    model = GPT(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(HERE, "blend", "model.pt")))
    print(f"loaded pretrained model, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)

    # SFT data: fixed-length rows + response mask
    tokens = np.memmap(os.path.join(HERE, "sft", "tokens.bin"), dtype=np.uint16, mode="r").reshape(-1, cfg.block_size)
    mask = np.memmap(os.path.join(HERE, "sft", "loss_mask.bin"), dtype=np.uint8, mode="r").reshape(-1, cfg.block_size)
    n_examples = tokens.shape[0]
    print(f"{n_examples:,} SFT examples", flush=True)

    def get_batch():
        ix = torch.randint(n_examples, (args.batch_size,))
        x = torch.from_numpy(tokens[ix].astype(np.int64)).to(device)
        m = torch.from_numpy(mask[ix].astype(np.float32)).to(device)
        return x, m

    # same Muon(hidden) + Adam(embed/head) split, lower LRs for fine-tuning
    hidden = [p for p in model.blocks.parameters() if p.ndim == 2]
    embed_head = [model.wte.weight, model.lm_head.weight]
    opt_muon = Muon(hidden, lr=0.004, momentum=0.95)
    opt_adam = torch.optim.Adam(embed_head, lr=6e-4, betas=(0.9, 0.95))
    optimizers = [opt_muon, opt_adam]
    base_lrs = [[g["lr"] for g in o.param_groups] for o in optimizers]

    def lr_mult(step):
        warmup = 100
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (args.steps - step) / (args.steps - warmup))  # linear decay to 0

    def masked_loss(x, m):
        with torch.autocast(device, dtype=torch.bfloat16):
            logits, _ = model(x)
        # predict token t+1 from t; grade only where the *target* is a response token
        logits = logits[:, :-1].reshape(-1, logits.size(-1)).float()
        targets = x[:, 1:].reshape(-1)
        w = m[:, 1:].reshape(-1)
        per_tok = F.cross_entropy(logits, targets, reduction="none")
        return (per_tok * w).sum() / w.sum().clamp(min=1)

    log_file = open(os.path.join(HERE, "metrics.jsonl"), "w", buffering=1)
    def log(**kv): log_file.write(json.dumps(kv) + "\n")
    log(event="start", num_steps=args.steps, params_m=round(sum(p.numel() for p in model.parameters())/1e6, 1),
        device=device, batch_size=args.batch_size, block_size=cfg.block_size,
        prompt="<|user|>What is a dog?<|assistant|>", data="sft")

    chat_prompts = ["What is a dog?", "Tell me a story about a cat.",
                    "What is the sun?", "Why is the sky blue?"]

    def chat_sample(q):
        ids = [U] + tok.encode(q).ids + [A]
        idx = torch.tensor([ids], device=device)
        model.eval()
        with torch.no_grad():
            for _ in range(120):
                logits, _ = model(idx[:, -cfg.block_size:])
                nxt = torch.multinomial(F.softmax(logits[:, -1] / 0.7, dim=-1), 1)
                idx = torch.cat([idx, nxt], 1)
                if nxt.item() == E:
                    break
        model.train()
        full = tok.decode(idx[0].tolist())
        return f"USER: {q}\nBOT: " + full.split("<|assistant|>")[-1].replace("<|endoftext|>", "").strip()

    t0 = time.time()
    for step in range(args.steps):
        if step % args.eval_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                vl = np.mean([masked_loss(*get_batch()).item() for _ in range(10)])
            sample = "\n\n".join(chat_sample(q) for q in chat_prompts)
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
        if step % 5 == 0:
            log(event="train", step=step, train_loss=round(loss.item(), 4),
                lr_mult=round(mult, 4), time_s=round(time.time()-t0, 1),
                tok_per_s=round(args.batch_size * cfg.block_size * (step+1) / max(time.time()-t0, 1e-9)))

    torch.save(model.state_dict(), os.path.join(HERE, "sft", "model.pt"))
    final = "\n\n".join(chat_sample(q) for q in chat_prompts)
    print("\n--- final chat ---\n" + final)
    log(event="done", time_s=round(time.time()-t0, 1), sample=final)
    log_file.close()

if __name__ == "__main__":
    main()
