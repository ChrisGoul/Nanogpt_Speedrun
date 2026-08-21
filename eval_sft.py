"""
Instruct / SFT evaluation — the benchmarks people actually report for chat models,
to complement eval_base.py (which is CF/cloze, the right lens for BASE models).

  MCF   multiple-choice FORMULATION: present lettered options in the chat format
        and score P(answer letter). This is how instruct models are graded, and
        the CF->MCF gap tells you whether SFT taught the "answer with a letter"
        behaviour (small models often can't, and score WORSE than their CF).
  MMLU  4-way knowledge MC (MCF)                       chance 25%
  GSM8K generative: solve the word problem, extract the number, exact-match.

Everything is run in the model's chat format (<|user|> ... <|assistant|>), so it
measures the SFT model as a chat model, not a raw LM.

Run:  python eval_sft.py --model big300c_sft --data sft_big300c \
          --layers 22 --heads 16 --dim 1024 --n 500
"""
import argparse
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train import GPT, Config
from eval_base import load_piqa, load_hellaswag, load_arc_easy, parquet_table

HERE = os.path.dirname(os.path.abspath(__file__))
LETTERS = ["A", "B", "C", "D", "E"]

def load_mmlu(n):
    t = parquet_table("cais/mmlu", "test", must_contain="all")
    q = t.column("question").to_pylist()
    ch = t.column("choices").to_pylist()
    ans = t.column("answer").to_pylist()
    out = []
    for i in range(len(q)):
        if q[i] and ch[i] is not None and ans[i] is not None:
            out.append((q[i], list(ch[i]), int(ans[i])))
        if len(out) >= n:
            break
    return out

def load_gsm8k(n):
    t = parquet_table("openai/gsm8k", "test", must_contain="main")
    q = t.column("question").to_pylist()
    a = t.column("answer").to_pylist()
    out = []
    for qi, ai in zip(q, a):
        m = re.search(r"####\s*([-\d,]+)", ai or "")
        if qi and m:
            out.append((qi.strip(), m.group(1).replace(",", "")))
        if len(out) >= n:
            break
    return out

def last_number(text):
    nums = re.findall(r"-?\d[\d,]*", text or "")
    return nums[-1].replace(",", "") if nums else None

def load_truthfulqa(n):
    # MC1: one correct answer among several; score by likelihood (standard).
    t = parquet_table("truthfulqa/truthful_qa", "validation", must_contain="multiple_choice")
    q = t.column("question").to_pylist()
    mc1 = t.column("mc1_targets").to_pylist()
    out = []
    for i in range(len(q)):
        choices, labels = list(mc1[i]["choices"]), list(mc1[i]["labels"])
        gold = labels.index(1) if 1 in labels else 0
        out.append((q[i], choices, gold))
        if len(out) >= n:
            break
    return out

# --- IFEval: instruction-following with rule-based verifiers (no judge needed).
# We use the real google/IFEval prompts, scoring only those whose instructions
# are all covered by the verifiers below (a faithful subset of the harness).
def _words(s): return re.findall(r"\b\w+\b", s)
def _sentences(s): return [x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x]

IFEVAL_VERIFIERS = {
    "keywords:existence": lambda r, k: all(w.lower() in r.lower() for w in k["keywords"]),
    "keywords:frequency": lambda r, k: (r.lower().count(k["keyword"].lower()) >= k["frequency"])
        if k.get("relation") == "at least" else (r.lower().count(k["keyword"].lower()) <= k["frequency"]),
    "length_constraints:number_words": lambda r, k: (len(_words(r)) >= k["num_words"])
        if k.get("relation") == "at least" else (len(_words(r)) <= k["num_words"]),
    "length_constraints:number_sentences": lambda r, k: (len(_sentences(r)) >= k["num_sentences"])
        if k.get("relation") == "at least" else (len(_sentences(r)) <= k["num_sentences"]),
    "change_case:english_capital": lambda r, k: r.strip() == r.strip().upper() and any(c.isalpha() for c in r),
    "change_case:english_lowercase": lambda r, k: r.strip() == r.strip().lower() and any(c.isalpha() for c in r),
    "detectable_format:number_bullet_lists": lambda r, k: len(re.findall(r"^\s*[\*\-]\s", r, re.M)) == k["num_bullets"],
    "startend:end_checker": lambda r, k: r.strip().endswith(k["end_phrase"].strip()),
    "punctuation:no_comma": lambda r, k: "," not in r,
    "detectable_format:title": lambda r, k: bool(re.search(r"<<.+?>>", r)),
}

def load_ifeval(n):
    t = parquet_table("google/IFEval", "train")
    prompts = t.column("prompt").to_pylist()
    ids = t.column("instruction_id_list").to_pylist()
    kw = t.column("kwargs").to_pylist()
    out = []
    for p, idl, kwl in zip(prompts, ids, kw):
        idl = list(idl)
        if idl and all(i in IFEVAL_VERIFIERS for i in idl):   # only fully-coverable prompts
            out.append((p, idl, [dict(x) for x in kwl]))
        if len(out) >= n:
            break
    return out

class SFTModel:
    def __init__(self, model_dir, data_dir, n_layer, n_head, n_embd):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = Tokenizer.from_file(os.path.join(HERE, data_dir, "tokenizer.json"))
        self.E = self.tok.token_to_id("<|endoftext|>")
        self.U = self.tok.encode("<|user|>\n").ids
        self.A = self.tok.encode("<|assistant|>\n").ids
        self.cfg = Config(vocab_size=self.tok.get_vocab_size(), n_layer=n_layer,
                          n_head=n_head, n_embd=n_embd, block_size=1024, tie_embeddings=True)
        self.model = GPT(self.cfg).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(HERE, model_dir, "model.pt")))
        self.model.eval()
        # single-token id for each answer letter (score its logit at the answer position)
        self.letter_ids = [self.tok.encode(x).ids[0] for x in LETTERS]

    def _mcf_prompt(self, question, choices):
        opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
        text = f"{question}\n{opts}\nAnswer:"
        return self.U + self.tok.encode(text).ids + self.A

    @torch.no_grad()
    def mcf(self, items):
        correct = 0
        for q, choices, gold in items:
            ids = self._mcf_prompt(q, choices)[-self.cfg.block_size:]
            idx = torch.tensor([ids], device=self.device)
            with torch.autocast(self.device, dtype=torch.bfloat16):
                logits, _ = self.model(idx)
            last = logits[0, -1].float()
            cand = self.letter_ids[:len(choices)]
            pick = int(np.argmax([last[c].item() for c in cand]))
            correct += int(pick == gold)
        return round(100 * correct / max(len(items), 1), 1)

    @torch.no_grad()
    def generate(self, question, max_new=256, temp=0.2):
        ids = (self.U + self.tok.encode(question).ids + self.A)[-(self.cfg.block_size - max_new):]
        idx = torch.tensor([ids], device=self.device)
        start = idx.size(1)
        for _ in range(max_new):
            with torch.autocast(self.device, dtype=torch.bfloat16):
                logits, _ = self.model(idx[:, -self.cfg.block_size:])
            nxt = torch.multinomial(F.softmax(logits[:, -1].float() / temp, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if nxt.item() == self.E:
                break
        return self.tok.decode(idx[0, start:].tolist())

    def gsm8k(self, items):
        correct = 0
        for q, gold in items:
            pred = last_number(self.generate(q))
            correct += int(pred is not None and pred == gold)
        return round(100 * correct / max(len(items), 1), 1)

    @torch.no_grad()
    def _cf_logprob(self, context, continuation):
        cids = self.tok.encode(context).ids
        xids = self.tok.encode(" " + continuation).ids
        if not xids:
            return -1e9
        ids = (cids + xids)[-self.cfg.block_size:]
        idx = torch.tensor([ids], device=self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            logits, _ = self.model(idx)
        lp = F.log_softmax(logits.float(), -1)[0]
        n = len(xids)
        return sum(lp[len(ids) - n + i - 1, xids[i]].item() for i in range(n)) / max(len(continuation), 1)

    def truthfulqa_mc1(self, items):
        correct = 0
        for q, choices, gold in items:
            scores = [self._cf_logprob(q, c) for c in choices]
            correct += int(int(np.argmax(scores)) == gold)
        return round(100 * correct / max(len(items), 1), 1)

    def ifeval(self, items):
        # prompt-level strict accuracy: the response must satisfy EVERY instruction
        passed = 0
        for prompt, idl, kwl in items:
            resp = self.generate(prompt, max_new=200)
            try:
                ok = all(IFEVAL_VERIFIERS[i](resp, k) for i, k in zip(idl, kwl))
            except Exception:
                ok = False
            passed += int(ok)
        return round(100 * passed / max(len(items), 1), 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="big300c_sft", help="dir with model.pt")
    ap.add_argument("--data", default="sft_big300c", help="dir with tokenizer.json")
    ap.add_argument("--layers", type=int, default=22)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--n", type=int, default=500, help="examples per benchmark")
    ap.add_argument("--gsm-n", type=int, default=100, help="GSM8K is generative/slow -> fewer")
    args = ap.parse_args()

    print(f"loading {args.model} ...", flush=True)
    lm = SFTModel(args.model, args.data, args.layers, args.heads, args.dim)

    print("\n=== MCF (multiple-choice formulation, letter-answer, chat format) ===", flush=True)
    for name, fn in (("PIQA", load_piqa), ("ARC-Easy", load_arc_easy), ("HellaSwag", load_hellaswag), ("MMLU", load_mmlu)):
        try:
            items = fn(args.n)
            print(f"  {name:10} MCF acc {lm.mcf(items):5}%   (chance {100//len(items[0][1]) if items else '?'}%)", flush=True)
        except Exception as e:
            print(f"  {name:10} unavailable ({e!r})", flush=True)

    print("\n=== TruthfulQA (MC1, likelihood) ===", flush=True)
    try:
        tqa = load_truthfulqa(args.n)
        print(f"  TruthfulQA MC1 acc {lm.truthfulqa_mc1(tqa):5}%   (n={len(tqa)}, chance ~20%)", flush=True)
    except Exception as e:
        print(f"  TruthfulQA unavailable ({e!r})", flush=True)

    print("\n=== IFEval (instruction-following, rule-verified subset) ===", flush=True)
    try:
        ife = load_ifeval(args.gsm_n)
        print(f"  IFEval     prompt-strict {lm.ifeval(ife):5}%   (n={len(ife)}, generative)", flush=True)
    except Exception as e:
        print(f"  IFEval     unavailable ({e!r})", flush=True)

    print("\n=== Generative ===", flush=True)
    try:
        gsm = load_gsm8k(args.gsm_n)
        print(f"  GSM8K      exact-match {lm.gsm8k(gsm):5}%   (n={len(gsm)})", flush=True)
    except Exception as e:
        print(f"  GSM8K      unavailable ({e!r})", flush=True)

    print("\n=== chat samples ===", flush=True)
    for q in ["What is the capital of France?", "What is 17 plus 25?",
              "Write one sentence about dogs.", "Why is the sky blue?"]:
        print(f"  Q: {q}\n  A: {lm.generate(q, max_new=80).strip()[:300]}\n", flush=True)

if __name__ == "__main__":
    main()
