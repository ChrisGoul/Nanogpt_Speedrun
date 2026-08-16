"""
Talk to the fine-tuned blend bot.

Interactive:   python chat.py
One-shot:      python chat.py --ask "What is a volcano?"
Options:       --temp 0.7  --max-new 160  --model sft   (or --model blend for base)

Wraps your input as <|user|>...<|assistant|> and generates until <|endoftext|>,
decoding only the newly generated tokens (so the reply is clean, no echo).
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
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--model", default="sft", help="subfolder holding model.pt (sft or blend)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(os.path.join(HERE, "blend", "tokenizer.json"))
    U = tok.token_to_id("<|user|>")
    A = tok.token_to_id("<|assistant|>")
    E = tok.token_to_id("<|endoftext|>")

    cfg = Config(vocab_size=tok.get_vocab_size(), n_layer=8, n_head=8, n_embd=512)
    model = GPT(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(HERE, args.model, "model.pt")))
    model.eval()

    @torch.no_grad()
    def reply(question: str) -> str:
        ids = [U] + tok.encode(question).ids + [A]
        idx = torch.tensor([ids], device=device)
        start = idx.size(1)
        for _ in range(args.max_new):
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, _ = model(idx[:, -cfg.block_size:])
            nxt = torch.multinomial(F.softmax(logits[:, -1] / args.temp, dim=-1), 1)
            idx = torch.cat([idx, nxt], 1)
            if nxt.item() == E:
                break
        new_tokens = idx[0, start:].tolist()
        return tok.decode(new_tokens).strip()

    if args.ask is not None:
        print(reply(args.ask))
        return

    print(f"blend bot ({args.model}) - type a message, Ctrl-C to quit\n")
    try:
        while True:
            q = input("you> ").strip()
            if q:
                print("bot>", reply(q), "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")

if __name__ == "__main__":
    main()
