"""
GSM8K transfer: how much of the model's reasoning survives real natural-language
math problems it never saw (GSM8K phrasing vs our templates)? Expect a big drop
— this quantifies the cross-format generalization gap.

Note: char-level + 512-token context means long GSM8K problems may not fully fit;
we skip ones whose prompt exceeds the budget and report how many were evaluable.
"""
import argparse
import re

from eval_base import parquet_table
from eval_reason import Solver, extract, _norm

def gold_answer(ans):
    m = re.search(r"####\s*([-\d,]+)", ans)
    return m.group(1).replace(",", "") if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reason168")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    s = Solver(args.model, 12, 16, 1024)
    t = parquet_table("openai/gsm8k", "test", must_contain="main")
    qs, ans = t.column("question").to_pylist(), t.column("answer").to_pylist()

    correct = evaluable = skipped = 0
    for i in range(min(args.n, len(qs))):
        prompt = f"Question: {qs[i].strip()}\n"
        if len(s.tok.encode(prompt).ids) > 460:      # won't fit char-level context
            skipped += 1
            continue
        evaluable += 1
        pred = extract(s.solve(prompt, max_new=250))
        gold = _norm(gold_answer(ans[i]))
        correct += (pred == gold)
    n = max(evaluable, 1)
    print(f"\nGSM8K test: {100*correct/n:.1f}% ({correct}/{evaluable} evaluable, "
          f"{skipped} skipped as too long for char-level 512 context)", flush=True)

if __name__ == "__main__":
    main()
