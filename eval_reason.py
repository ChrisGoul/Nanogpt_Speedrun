"""
Evaluate the reasoning model on FRESH held-out problems (different RNG seed than
training, so these are unseen). For each type: feed only the "Question: ...\n",
let the model generate its chain-of-thought greedily, extract the "#### answer",
and check correctness. Broken down by problem type so we see WHAT it learned.

Run after training:  python eval_reason.py --model reason168 --n 200
"""
import argparse
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train import GPT, Config
import reason_gen

HERE = os.path.dirname(os.path.abspath(__file__))

class Solver:
    def __init__(self, model_dir, layers, heads, dim, use_abacus=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = Tokenizer.from_file(os.path.join(HERE, model_dir, "tokenizer.json"))
        self.E = self.tok.token_to_id("<|endoftext|>")
        self.cfg = Config(vocab_size=self.tok.get_vocab_size(), n_layer=layers, n_head=heads,
                          n_embd=dim, use_abacus=use_abacus)
        self.model = GPT(self.cfg).to(self.device)   # digit_ids buffer restored by load_state_dict
        self.model.load_state_dict(torch.load(os.path.join(HERE, model_dir, "model.pt")))
        self.model.eval()

    @torch.no_grad()
    def solve(self, question, max_new=220):
        ids = self.tok.encode(question).ids
        idx = torch.tensor([ids], device=self.device)
        start = idx.size(1)
        for _ in range(max_new):
            with torch.autocast(self.device, dtype=torch.bfloat16):
                logits, _ = self.model(idx[:, -self.cfg.block_size:])
            nxt = logits[:, -1].argmax(-1, keepdim=True)     # greedy — math is deterministic
            idx = torch.cat([idx, nxt], 1)
            if nxt.item() == self.E:
                break
        return self.tok.decode(idx[0, start:].tolist())

def _norm(s):
    return str(s).strip().rstrip(".").lower()

def extract(text):
    # grab everything after the last "####" up to end-of-line (number, word, or list)
    m = re.findall(r"####\s*([^\n]+)", text)
    return _norm(m[-1]) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reason168")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--show", type=int, default=2, help="print this many worked examples per type")
    args = ap.parse_args()

    solver = Solver(args.model, args.layers, args.heads, args.dim)
    reason_gen.R = random.Random(999999)   # held-out seed, disjoint from training (seed 0)

    types = {"addition": reason_gen.add_cot, "subtraction": reason_gen.sub_cot,
             "multiplication": reason_gen.mul_cot, "word problem": reason_gen.word_cot,
             "algebra": reason_gen.algebra_cot, "comparison": reason_gen.compare_cot,
             "sequence": reason_gen.seq_cot, "transitive": reason_gen.transitive_cot,
             "syllogism": reason_gen.syllogism_cot, "boolean": reason_gen.boolean_cot,
             "rule (modus ponens)": reason_gen.rule_cot, "sorting": reason_gen.sort_cot,
             "set counting": reason_gen.setcount_cot}

    print(f"evaluating {args.model} on {args.n} fresh problems per type\n", flush=True)
    total_c = total_n = 0
    for name, gen in types.items():
        correct = 0
        shown = 0
        for _ in range(args.n):
            text, gold = gen()
            q = text.split("\n")[0] + "\n"
            out = solver.solve(q)
            pred = extract(out)
            ok = (pred == _norm(gold))
            correct += ok
            if shown < args.show:
                print(f"  [{name}] {q.strip()}  -> pred {pred}, gold {gold} {'OK' if ok else 'X'}", flush=True)
                shown += 1
        acc = 100 * correct / args.n
        total_c += correct; total_n += args.n
        print(f"{name:16} {acc:5.1f}%", flush=True)
    print(f"\n{'OVERALL':16} {100*total_c/total_n:5.1f}%  ({total_c}/{total_n})", flush=True)

if __name__ == "__main__":
    main()
