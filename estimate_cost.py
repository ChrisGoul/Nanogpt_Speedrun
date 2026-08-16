"""
Project a training run's wall-clock + cost. Two modes:

EMPIRICAL (accurate — use once you're on the target GPU): run the real config
for a few hundred steps, then extrapolate from its measured tok/s.
  python estimate_cost.py --run smoke500 --tokens 10e9 --price 2.5

ANALYTICAL (planning — use from your desk, no run needed): project from the
6ND FLOP math for a target GPU. Models the hardware you DON'T have yet.
  python estimate_cost.py --params 500e6 --tokens 10e9 --peak-tflops 989 --mfu 45 --price 2.5

The empirical mode measures real speed (captures --compile, batch, MFU, etc.);
the analytical mode assumes an MFU you supply. Use analytical to decide whether
to rent, empirical to confirm before the long run.
"""
import argparse
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

def report(tok_s, tokens, price, hours, source, extra=""):
    print(f"throughput:     {tok_s:,.0f} tok/s   ({source})")
    if extra:
        print(extra)
    print(f"target tokens:  {tokens:,.0f}")
    print("-" * 44)
    print(f"projected time: {hours:,.1f} hr  ({hours/24:.1f} days)")
    print(f"projected cost: ${hours * price:,.0f}  (at ${price}/hr)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, type=float, help="target total tokens (e.g. 10e9)")
    ap.add_argument("--price", type=float, default=2.5, help="GPU $/hr (default 2.5 = ~H100)")
    # empirical mode
    ap.add_argument("--run", help="run name -> reads metrics_<run>.jsonl (empirical mode)")
    ap.add_argument("--tail", type=int, default=20, help="recent train entries to median over")
    # analytical mode
    ap.add_argument("--params", type=float, help="model params for FLOP-based projection (e.g. 500e6)")
    ap.add_argument("--peak-tflops", type=float, default=989.0, help="target GPU bf16 peak (default 989 = H100)")
    ap.add_argument("--mfu", type=float, default=45.0, help="assumed MFU %% for analytical mode (default 45)")
    args = ap.parse_args()

    if not args.run and not args.params:
        raise SystemExit("give --run <name> (empirical) or --params <N> (analytical)")

    # ANALYTICAL: FLOPs = 6*N*D; time = FLOPs / (peak * mfu)
    if args.params:
        flops = 6 * args.params * args.tokens
        eff = args.peak_tflops * 1e12 * (args.mfu / 100)
        hours = flops / eff / 3600
        tok_s = eff / (6 * args.params)   # implied tokens/s
        print(f"mode:           analytical (6ND, {args.params/1e6:.0f}M params)")
        report(tok_s, args.tokens, args.price, hours,
               f"implied @ {args.peak_tflops:.0f} TFLOP/s x {args.mfu:.0f}% MFU")
        return

    # EMPIRICAL: median steady-state tok/s from a real run
    path = os.path.join(HERE, f"metrics_{args.run}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"no metrics at {path} — run a smoke test with --run {args.run} first")
    tps, mfus, mems = [], [], []
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") == "train" and d.get("tok_per_s"):
            tps.append(d["tok_per_s"])
            if d.get("mfu") is not None:
                mfus.append(d["mfu"])
            if d.get("gpu_mem_mb") is not None:
                mems.append(d["gpu_mem_mb"])
    if len(tps) < 3:
        raise SystemExit("not enough train entries to estimate (need a few hundred steps)")
    tail = tps[-args.tail:]
    tok_s = statistics.median(tail)
    hours = args.tokens / tok_s / 3600
    extra = ""
    if mfus:
        extra += f"MFU:            {statistics.median(mfus):.1f}%\n"
    if mems:
        extra += f"peak VRAM:      {max(mems)/1000:.1f} GB"
    print(f"mode:           empirical ({args.run}, {len(tps)} train points)")
    report(tok_s, args.tokens, args.price, hours,
           f"measured, median of last {len(tail)}", extra.rstrip("\n"))

if __name__ == "__main__":
    main()
