"""
Base-LM evaluation for the A/B experiment.

Uses the standard base-model methodology (plain continuation likelihood, NO
chat formatting), so numbers are comparable to published small-model results
(Pythia-70M/160M, GPT-2 124M):

  perplexity   on BOTH val sets (in-domain + out-of-domain) -- primary signal
  PIQA         binary physical commonsense        chance 50%
  ARC-Easy     4-way science reasoning            chance 25%
  HellaSwag    4-way commonsense inference        chance 25%
  LAMBADA      last-word prediction (accuracy)    chance ~0%

Multiple-choice is scored two ways, matching lm-eval-harness:
  acc      = argmax of summed log-prob of the continuation
  acc_norm = argmax of log-prob normalized by continuation length in chars

Run:  python eval_base.py --models ab_raw ab_filtered
"""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer

from train import GPT, Config

HERE = os.path.dirname(os.path.abspath(__file__))

def parquet_table(repo, split, must_contain=None, config=None):
    for rev in ("refs/convert/parquet", "main"):
        try:
            files = [f for f in list_repo_files(repo, repo_type="dataset", revision=rev)
                     if f.endswith(".parquet")]
        except Exception:
            files = []
        files = [f for f in files if f"/{split}/" in f or split in os.path.basename(f)]
        if must_contain:
            files = [f for f in files if must_contain in f]
        if files:
            return pa.concat_tables([pq.read_table(hf_hub_download(repo, f, repo_type="dataset", revision=rev))
                                     for f in sorted(files)])
    raise RuntimeError(f"no parquet split '{split}' for {repo}")

class BaseLM:
    def __init__(self, model_dir, tok_dir, n_layer, n_head, n_embd):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = Tokenizer.from_file(os.path.join(HERE, tok_dir, "tokenizer.json"))
        self.cfg = Config(vocab_size=self.tok.get_vocab_size(),
                          n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        self.model = GPT(self.cfg).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(HERE, model_dir, "model.pt")))
        self.model.eval()

    @torch.no_grad()
    def perplexity(self, bin_path, max_tokens=1_000_000):
        data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        n = min(len(data), max_tokens)
        B = self.cfg.block_size
        losses = []
        for i in range(0, n - B - 1, B):
            x = torch.from_numpy(data[i:i + B].astype(np.int64))[None].to(self.device)
            y = torch.from_numpy(data[i + 1:i + 1 + B].astype(np.int64))[None].to(self.device)
            with torch.autocast(self.device, dtype=torch.bfloat16):
                _, loss = self.model(x, y)
            losses.append(loss.item())
        m = float(np.mean(losses))
        return {"loss": m, "ppl": float(np.exp(m))}

    @torch.no_grad()
    def cont_logprob(self, context, continuation):
        """Summed log P(continuation | context), plain LM format."""
        cids = self.tok.encode(context).ids
        xids = self.tok.encode(continuation).ids
        if not xids:
            return -1e9, 1
        ids = (cids + xids)[-self.cfg.block_size:]
        idx = torch.tensor([ids], device=self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            logits, _ = self.model(idx)
        lp = F.log_softmax(logits.float(), -1)[0]
        n = len(xids)
        total = sum(lp[len(ids) - n + i - 1, xids[i]].item() for i in range(n))
        return total, max(len(continuation), 1)

    def mc(self, items):
        """items: list of (context, [choices], gold_idx) -> acc, acc_norm"""
        acc = accn = 0
        for ctx, choices, gold in items:
            scored = [self.cont_logprob(ctx, c) for c in choices]
            tot = [s for s, _ in scored]
            norm = [s / l for s, l in scored]
            acc += int(np.argmax(tot) == gold)
            accn += int(np.argmax(norm) == gold)
        n = max(len(items), 1)
        return {"acc": 100 * acc / n, "acc_norm": 100 * accn / n, "n": n}

    @torch.no_grad()
    def lambada(self, texts):
        correct = 0
        for t in texts:
            words = t.strip().split()
            if len(words) < 2:
                continue
            ctx, last = " ".join(words[:-1]), " " + words[-1]
            lids = self.tok.encode(last).ids
            cids = self.tok.encode(ctx).ids
            ids = (cids + lids)[-self.cfg.block_size:]
            idx = torch.tensor([ids], device=self.device)
            with torch.autocast(self.device, dtype=torch.bfloat16):
                logits, _ = self.model(idx)
            pred = logits[0].argmax(-1)
            start = len(ids) - len(lids)
            ok = all(pred[start + i - 1].item() == lids[i] for i in range(len(lids)))
            correct += int(ok)
        return {"acc": 100 * correct / max(len(texts), 1), "n": len(texts)}

def load_piqa(n):
    t = parquet_table("ybisk/piqa", "validation")
    g, s1, s2, lab = (t.column(c).to_pylist() for c in ("goal", "sol1", "sol2", "label"))
    return [(g[i], [s1[i], s2[i]], int(lab[i])) for i in range(min(n, len(g)))]

def load_arc_easy(n):
    t = parquet_table("allenai/ai2_arc", "test", must_contain="ARC-Easy")
    q, ch, key = (t.column(c).to_pylist() for c in ("question", "choices", "answerKey"))
    out = []
    for i in range(len(q)):
        labels, texts = list(ch[i]["label"]), list(ch[i]["text"])
        if key[i] not in labels:
            continue
        out.append((q[i], texts, labels.index(key[i])))
        if len(out) >= n:
            break
    return out

def load_hellaswag(n):
    t = parquet_table("Rowan/hellaswag", "validation")
    ctx, endings, lab = (t.column(c).to_pylist() for c in ("ctx", "endings", "label"))
    return [(ctx[i], list(endings[i]), int(lab[i])) for i in range(min(n, len(ctx))) if str(lab[i]).isdigit()]

def load_lambada(n):
    # repo holds de/en/es/fr/it configs — take English only
    t = parquet_table("EleutherAI/lambada_openai", "test", must_contain="en/")
    col = "text" if "text" in t.column_names else t.column_names[0]
    return t.column(col).to_pylist()[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ab_raw", "ab_filtered"])
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=10)
    ap.add_argument("--dim", type=int, default=640)
    ap.add_argument("--n", type=int, default=500, help="examples per benchmark")
    args = ap.parse_args()

    val_sets = {"raw": os.path.join(HERE, "ab_raw", "val.bin"),
                "filtered": os.path.join(HERE, "ab_filtered", "val.bin")}
    print("loading benchmarks...", flush=True)
    bench = {}
    for name, fn in (("PIQA", load_piqa), ("ARC-Easy", load_arc_easy), ("HellaSwag", load_hellaswag)):
        try:
            bench[name] = fn(args.n)
        except Exception as e:
            print(f"  {name}: unavailable ({e!r})", flush=True)
    try:
        lam = load_lambada(args.n)
    except Exception as e:
        lam = None
        print(f"  LAMBADA: unavailable ({e!r})", flush=True)

    for m in args.models:
        print(f"\n===== {m} =====", flush=True)
        lm = BaseLM(m, m, args.layers, args.heads, args.dim)
        for vname, vpath in val_sets.items():
            if os.path.exists(vpath):
                r = lm.perplexity(vpath)
                tag = "in-domain " if vname in m else "out-domain"
                print(f"  ppl[{vname:8}] {tag} loss {r['loss']:.4f}  ppl {r['ppl']:.2f}", flush=True)
        for name, items in bench.items():
            print(f"  {name:10} {lm.mc(items)}", flush=True)
        if lam:
            print(f"  {'LAMBADA':10} {lm.lambada(lam)}", flush=True)

if __name__ == "__main__":
    main()
