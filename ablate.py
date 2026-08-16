"""
Ablation runner: sweep one or more hyperparameters over a shared base config,
launch each variant as its own train.py run (sequentially — single GPU), then
print a comparison table of the final metrics.

Each variant logs to metrics_<tag>_<combo>.jsonl (so the dashboard can plot
them side by side) and saves its model under abl/<tag>_<combo>/.

Examples
--------
Depth sweep at a fixed param budget:
  python ablate.py --tag depth --grid layers=12,16,20 \
    --base "--data mix16 --tie --checkpoint --seq 256 --dim 768 --heads 12 \
            --batch-size 32 --steps 2000 --bench --eval-every 500"

Two-factor grid (cartesian product = 4 runs):
  python ablate.py --tag lr_x_dim --grid lr-scale=0.5,1.0 dim=512,768 \
    --base "--data mix16 --tie --checkpoint --seq 256 --layers 20 --heads 12 \
            --batch-size 32 --steps 1500"

--dry-run prints the commands without running them.
"""
import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

def parse_grid(tokens):
    """['layers=12,20', 'dim=512,768'] -> {'layers': ['12','20'], 'dim': ['512','768']}"""
    grid = {}
    for tok in tokens:
        key, _, vals = tok.partition("=")
        if not vals:
            sys.exit(f"bad --grid token '{tok}', expected key=v1,v2,...")
        grid[key.strip()] = [v.strip() for v in vals.split(",")]
    return grid

def combo_name(tag, combo):
    return tag + "_" + "_".join(f"{k}{v}" for k, v in combo).replace("-", "")

def read_final_metrics(run_name):
    """Pull the last train/val/bench entries from metrics_<run_name>.jsonl."""
    path = os.path.join(HERE, f"metrics_{run_name}.jsonl")
    out = {"val_loss": None, "grad_norm": None, "mfu": None, "tflops": None,
           "tok_per_s": None, "gpu_mem_mb": None, "piqa": None, "hellaswag": None}
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = d.get("event")
        if ev == "val":
            out["val_loss"] = d.get("val_loss")
        elif ev == "train":
            for k in ("grad_norm", "mfu", "tflops", "tok_per_s", "gpu_mem_mb"):
                if d.get(k) is not None:
                    out[k] = d[k]
        elif ev == "bench":
            out["piqa"], out["hellaswag"] = d.get("piqa"), d.get("hellaswag")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="name for this ablation family")
    ap.add_argument("--grid", nargs="+", required=True, help="key=v1,v2 tokens (train.py flag names)")
    ap.add_argument("--base", required=True, help="shared train.py flags (quoted)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grid = parse_grid(args.grid)
    base = shlex.split(args.base)
    keys = list(grid)
    combos = [list(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    print(f"[ablate] {args.tag}: {len(combos)} run(s) over {', '.join(keys)}\n", flush=True)

    results = []
    for combo in combos:
        name = combo_name(args.tag, combo)
        overrides = []
        for k, v in combo:
            overrides += [f"--{k}", v]
        cmd = [sys.executable, "-u", os.path.join(HERE, "train.py"), *base,
               *overrides, "--run", name, "--out", f"abl/{name}"]
        pretty = " ".join(f"{k}={v}" for k, v in combo)
        print(f"[ablate] === {name} ({pretty}) ===", flush=True)
        if args.dry_run:
            print("  " + " ".join(cmd), flush=True)
            continue
        log_path = os.path.join(HERE, f"abl_{name}.log")
        with open(log_path, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        m = read_final_metrics(name)
        m["_name"], m["_combo"], m["_rc"] = name, pretty, rc
        results.append(m)
        print(f"  done (rc={rc}) val_loss={m['val_loss']} mfu={m['mfu']} "
              f"grad_norm={m['grad_norm']} piqa={m['piqa']} hella={m['hellaswag']}\n", flush=True)

    if args.dry_run or not results:
        return

    # comparison table + CSV
    cols = ["_combo", "val_loss", "piqa", "hellaswag", "grad_norm", "mfu", "tok_per_s", "gpu_mem_mb", "_rc"]
    widths = {c: max(len(c), *(len(str(r.get(c))) for r in results)) for c in cols}
    print("[ablate] === summary ===")
    print("  " + "  ".join(c.ljust(widths[c]) for c in cols))
    for r in sorted(results, key=lambda r: (r["val_loss"] is None, r["val_loss"] or 0)):
        print("  " + "  ".join(str(r.get(c)).ljust(widths[c]) for c in cols))

    out_dir = os.path.join(HERE, "abl")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{args.tag}_summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\n[ablate] wrote {csv_path}")

if __name__ == "__main__":
    main()
