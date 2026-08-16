"""
Benchmark the fine-tuned chatbot on standard, commonly-used held-out tasks.
We deliberately picked benchmarks that map to the model's three claimed
abilities (and used only the TRAIN splits in fine-tuning, so these are unseen):

  - SQuAD v1.1 (validation)  reading comprehension   -> Exact-Match + token-F1
  - GSM8K (test)             math word problems       -> final-answer accuracy (CoT)
  - ARC-Easy (test)          multiple-choice science  -> likelihood-scored accuracy

SQuAD/GSM8K are generative (greedy). ARC uses the standard base-LM method:
score each choice by the model's length-normalized log-likelihood and pick the
best. Small samples by default so it finishes quickly on a laptop GPU.

Run after post-training:  python eval.py            (uses chatmodel/ + sftmix tokenizer)
"""
import argparse
import os
import re
import string
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer

from train import GPT, Config

HERE = os.path.dirname(os.path.abspath(__file__))

def parquet_table(repo, split, must_contain=None) -> pa.Table:
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

# ---- SQuAD metrics (official normalization) ----
def _norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def _f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0 or not p or not g:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)

class Model:
    def __init__(self, model_dir, tok_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = Tokenizer.from_file(os.path.join(HERE, tok_dir, "tokenizer.json"))
        self.U, self.A, self.E = (self.tok.token_to_id(x) for x in ("<|user|>", "<|assistant|>", "<|endoftext|>"))
        self.cfg = Config(vocab_size=self.tok.get_vocab_size(), n_layer=8, n_head=8, n_embd=512)
        self.model = GPT(self.cfg).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(HERE, model_dir, "model.pt")))
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt, max_new=200):
        ids = [self.U] + self.tok.encode(prompt).ids + [self.A]
        ids = ids[-(self.cfg.block_size - max_new):]
        idx = torch.tensor([ids], device=self.device)
        start = idx.size(1)
        for _ in range(max_new):
            with torch.autocast(self.device, dtype=torch.bfloat16):
                logits, _ = self.model(idx[:, -self.cfg.block_size:])
            nxt = logits[:, -1].argmax(-1, keepdim=True)   # greedy
            idx = torch.cat([idx, nxt], 1)
            if nxt.item() == self.E:
                break
        return self.tok.decode(idx[0, start:].tolist()).strip()

    @torch.no_grad()
    def choice_loglik(self, prompt, choice):
        pre = [self.U] + self.tok.encode(prompt).ids + [self.A]
        cont = self.tok.encode(" " + choice).ids
        ids = (pre + cont)[-self.cfg.block_size:]
        idx = torch.tensor([ids], device=self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            logits, _ = self.model(idx)
        lp = F.log_softmax(logits.float(), -1)[0]
        n = len(cont)
        total = sum(lp[len(ids) - n + i - 1, cont[i]].item() for i in range(n))
        return total / max(n, 1)   # length-normalized

def eval_squad(m, n):
    t = parquet_table("rajpurkar/squad", "validation")
    ctx, q, ans = t.column("context").to_pylist(), t.column("question").to_pylist(), t.column("answers").to_pylist()
    em = f1 = 0.0
    for i in range(min(n, len(q))):
        pred = m.generate(f"Context: {ctx[i].strip()}\n\nQuestion: {q[i].strip()}", max_new=32)
        golds = ans[i]["text"] or [""]
        em += max(_norm(pred) == _norm(g) for g in golds)
        f1 += max(_f1(pred, g) for g in golds)
    n = min(n, len(q))
    return {"n": n, "EM": 100 * em / n, "F1": 100 * f1 / n}

def eval_gsm8k(m, n):
    t = parquet_table("openai/gsm8k", "test", must_contain="main")
    q, a = t.column("question").to_pylist(), t.column("answer").to_pylist()
    num = lambda s: (re.findall(r"-?[\d,]+\.?\d*", s) or [""])[-1].replace(",", "")
    correct = 0
    for i in range(min(n, len(q))):
        pred = m.generate(q[i].strip(), max_new=256)
        gold = a[i].split("####")[-1].strip().replace(",", "")
        if num(pred) == gold and gold:
            correct += 1
    n = min(n, len(q))
    return {"n": n, "accuracy": 100 * correct / n}

def eval_arc(m, n):
    t = parquet_table("allenai/ai2_arc", "test", must_contain="ARC-Easy")
    q, ch, key = t.column("question").to_pylist(), t.column("choices").to_pylist(), t.column("answerKey").to_pylist()
    correct = 0
    for i in range(min(n, len(q))):
        labels, texts = ch[i]["label"], ch[i]["text"]
        scores = [m.choice_loglik(q[i].strip(), tx) for tx in texts]
        if list(labels)[int(np.argmax(scores))] == key[i]:
            correct += 1
    n = min(n, len(q))
    return {"n": n, "accuracy": 100 * correct / n}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chatmodel")
    ap.add_argument("--tokenizer", default="sftmix")
    ap.add_argument("--squad-n", type=int, default=500)
    ap.add_argument("--gsm8k-n", type=int, default=200)
    ap.add_argument("--arc-n", type=int, default=500)
    args = ap.parse_args()

    m = Model(args.model, args.tokenizer)
    print(f"evaluating {args.model}\n", flush=True)
    print("SQuAD  (comprehension):", eval_squad(m, args.squad_n), flush=True)
    print("GSM8K  (reasoning):    ", eval_gsm8k(m, args.gsm8k_n), flush=True)
    print("ARC-Easy (commonsense):", eval_arc(m, args.arc_n), flush=True)

if __name__ == "__main__":
    main()
