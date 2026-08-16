"""
Expanded SFT blend: turn the books-pretrained model into a chatbot that also
answers questions and does basic (chain-of-thought) reasoning.

Mix (all formatted <|user|> ... <|assistant|> ... <|endoftext|>, loss on
assistant turns only), using the SAME shared 16K tokenizer as pretraining:

  - dialogue     DailyDialog + EmpatheticDialogues   -> conversational ability
  - instruction  Dolly-15k                           -> answering in Q&A form
  - reasoning    GSM8K (worked step-by-step answers)  -> chain-of-thought  [x2]
  - commonsense  CommonsenseQA                        -> pick-and-justify
  - comprehension SQuAD (answer from given passage)   -> reasoning over context

Output: sftmix/{tokens.bin, loss_mask.bin, tokenizer.json}
Train:  python train_chat.py --data sftmix --init books ...
"""
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer

from prepare_chat import load_dailydialog, load_empathetic

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sftmix")
os.makedirs(OUT, exist_ok=True)
SEQ_LEN = 512

def parquet_table(repo, split="train", must_contain=None) -> pa.Table:
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

def dialogue_convs():
    return load_dailydialog() + load_empathetic()   # list of multi-turn lists

def dolly_pairs():
    t = parquet_table("databricks/databricks-dolly-15k")
    ins = t.column("instruction").to_pylist()
    ctx = t.column("context").to_pylist()
    resp = t.column("response").to_pylist()
    out = []
    for i, c, r in zip(ins, ctx, resp):
        if not (i and r):
            continue
        prompt = i.strip() + (("\n\n" + c.strip()) if c and c.strip() else "")
        out.append([prompt, r.strip()])
    print(f"Dolly: {len(out):,}", flush=True)
    return out

def gsm8k_pairs():
    t = parquet_table("openai/gsm8k", must_contain="main")
    q = t.column("question").to_pylist()
    a = t.column("answer").to_pylist()
    out = []
    for qi, ai in zip(q, a):
        if qi and ai:
            # keep the worked steps (chain-of-thought); tidy the final marker
            out.append([qi.strip(), ai.replace("####", "The answer is").strip()])
    print(f"GSM8K: {len(out):,}", flush=True)
    return out

def commonsenseqa_pairs():
    t = parquet_table("tau/commonsense_qa")
    q = t.column("question").to_pylist()
    ch = t.column("choices").to_pylist()
    key = t.column("answerKey").to_pylist()
    out = []
    for qi, ci, ki in zip(q, ch, key):
        if not (qi and ki):
            continue
        labels, texts = ci["label"], ci["text"]
        opts = "  ".join(f"{l}) {tx}" for l, tx in zip(labels, texts))
        try:
            ans = texts[list(labels).index(ki)]
        except ValueError:
            continue
        out.append([f"{qi.strip()}\nOptions: {opts}", f"The answer is {ki}) {ans}."])
    print(f"CommonsenseQA: {len(out):,}", flush=True)
    return out

def squad_pairs(limit=40000):
    t = parquet_table("rajpurkar/squad")
    ctx = t.column("context").to_pylist()
    q = t.column("question").to_pylist()
    ans = t.column("answers").to_pylist()
    out = []
    for c, qi, ai in zip(ctx, q, ans):
        texts = ai.get("text") if isinstance(ai, dict) else None
        if not (c and qi and texts):
            continue
        out.append([f"Context: {c.strip()}\n\nQuestion: {qi.strip()}", texts[0].strip()])
        if len(out) >= limit:
            break
    print(f"SQuAD: {len(out):,}", flush=True)
    return out

def main():
    tok = Tokenizer.from_file(os.path.join(HERE, "chat", "tokenizer.json"))
    U, A, E = (tok.token_to_id(x) for x in ("<|user|>", "<|assistant|>", "<|endoftext|>"))
    shutil.copy(os.path.join(HERE, "chat", "tokenizer.json"), os.path.join(OUT, "tokenizer.json"))

    convs = []
    convs += dialogue_convs()
    convs += dolly_pairs()
    convs += gsm8k_pairs() * 2          # upweight reasoning
    convs += commonsenseqa_pairs()
    convs += squad_pairs()
    print(f"total examples: {len(convs):,}", flush=True)

    rng = np.random.default_rng(1337)
    rng.shuffle(convs)

    rows_tok, rows_mask = [], []
    for turns in convs:
        seq, mask = [], []
        for i, turn in enumerate(turns):
            role = U if i % 2 == 0 else A
            ids = tok.encode(turn).ids
            seq.append(role); mask.append(0)
            seq.extend(ids)
            mask.extend([1 if role == A else 0] * len(ids))
        seq.append(E); mask.append(1)
        seq, mask = seq[:SEQ_LEN], mask[:SEQ_LEN]
        if sum(mask) < 2:
            continue
        pad = SEQ_LEN - len(seq)
        rows_tok.append(np.array(seq + [E] * pad, dtype=np.uint16))
        rows_mask.append(np.array(mask + [0] * pad, dtype=np.uint8))

    tokens = np.stack(rows_tok)
    masks = np.stack(rows_mask)
    tokens.tofile(os.path.join(OUT, "tokens.bin"))
    masks.tofile(os.path.join(OUT, "loss_mask.bin"))
    print(f"wrote {len(tokens):,} examples x {SEQ_LEN} "
          f"({masks.sum()/masks.size:.1%} assistant tokens under loss)", flush=True)

if __name__ == "__main__":
    main()
