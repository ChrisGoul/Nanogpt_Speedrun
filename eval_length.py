"""
Length generalization: the model trained on numbers up to 3 digits. Does it
apply the column algorithm to LONGER numbers it never saw (4-7 digits)?
This is the classic wall — it separates "learned the algorithm" from
"interpolated the training range".
"""
import argparse
import random

from eval_reason import Solver, extract, _norm

def forced(d, op):
    lo, hi = 10 ** (d - 1), 10 ** d - 1
    a, b = random.randint(lo, hi), random.randint(lo, hi)
    if op == "-" and b > a:
        a, b = b, a
    ans = a + b if op == "+" else a * b if op == "x" else a - b
    return f"Question: What is {a} {op} {b}?\n", ans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reason168")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--abacus", action="store_true")
    args = ap.parse_args()
    s = Solver(args.model, args.layers, args.heads, args.dim, use_abacus=args.abacus)
    random.seed(7)

    print(f"length generalization (trained on <=3 digits), {args.n} problems each\n", flush=True)
    print(f"{'digits':>7} | {'addition':>9} {'subtraction':>12}", flush=True)
    print("-" * 34, flush=True)
    for d in (3, 4, 5, 6, 7):
        row = {}
        for op, name in (("+", "add"), ("-", "sub")):
            c = 0
            for _ in range(args.n):
                q, gold = forced(d, op)
                if extract(s.solve(q, max_new=300)) == _norm(gold):
                    c += 1
            row[name] = 100 * c / args.n
        tag = "  (trained)" if d == 3 else ""
        print(f"{d:>7} | {row['add']:>8.0f}% {row['sub']:>11.0f}%{tag}", flush=True)

if __name__ == "__main__":
    main()
