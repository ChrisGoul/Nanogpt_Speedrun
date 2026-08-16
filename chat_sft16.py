"""
Talk to the mix16 SFT chatbot (156M, tied 16K vocab, 768x20).

Interactive:  python chat_sft16.py
One-shot:     python chat_sft16.py --ask "What is a volcano?"
Options:      --temp 0.3  --top-k 40  --rep-penalty 1.3  --max-new 160

Role turns use the *text* markers "<|user|>"/"<|assistant|>" (the mix16
tokenizer has no dedicated role tokens); generates until <|endoftext|>.
Defaults are tuned to reduce the repetition loops this small model falls into:
low temperature, top-k, and a repetition penalty on already-emitted tokens.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train import GPT, Config

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default=None, help="single question, then exit")
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--model", default="sft16", help="subfolder holding model.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(os.path.join(HERE, "sft16", "tokenizer.json"))
    E = tok.token_to_id("<|endoftext|>")
    U_ids, A_ids = tok.encode("<|user|>\n").ids, tok.encode("<|assistant|>\n").ids

    cfg = Config(vocab_size=tok.get_vocab_size(), n_layer=20, n_head=12,
                 n_embd=768, block_size=512, tie_embeddings=True)
    model = GPT(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(HERE, args.model, "model.pt")))
    model.eval()

    @torch.no_grad()
    def reply(question: str) -> str:
        ids = U_ids + tok.encode(question).ids + A_ids
        idx = torch.tensor([ids], device=device)
        start = idx.size(1)
        for _ in range(args.max_new):
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, _ = model(idx[:, -cfg.block_size:])
            logits = logits[:, -1].float()
            # repetition penalty: down-weight tokens already generated this turn
            gen = idx[0, start:]
            if args.rep_penalty != 1.0 and gen.numel() > 0:
                uniq = torch.unique(gen)
                logits[0, uniq] /= args.rep_penalty
            logits = logits / args.temp
            if args.top_k:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], 1)
            if nxt.item() == E:
                break
        return tok.decode(idx[0, start:].tolist()).strip()

    if args.ask is not None:
        print(reply(args.ask))
        return

    print(f"mix16 bot ({args.model}) - type a message, Ctrl-C to quit\n")
    try:
        while True:
            q = input("you> ").strip()
            if q:
                print("bot>", reply(q), "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")

if __name__ == "__main__":
    main()
