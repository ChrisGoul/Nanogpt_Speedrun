"""
RAFT training data: teach the small reader to answer from retrieved passages,
ignore distractors, and ABSTAIN when the answer isn't present.

Built from SQuAD (question + golden passage + extractive answer). For each
example we assemble k passages = 1 golden + (k-1) distractors, shuffled, and:
  - grounded case: answer = "According to document [i], <span>."  (teaches citation)
  - abstain case (p_abstain): golden removed, answer = "The documents do not
    contain the answer."  (teaches it to say 'I don't know' instead of guessing)

Everything fits in the 512-token window (passages truncated around the answer).
Also writes raft/index_docs.json = the passage collection for the retriever.

Output: raft/{tokens.bin, loss_mask.bin, tokenizer.json, index_docs.json}
Train:  python train_chat.py --data raft --init books --out raftmodel ...
"""
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "raft")
os.makedirs(OUT, exist_ok=True)

SEQ_LEN = 512
K_DOCS = 3               # passages shown per example (1 golden + 2 distractors)
DOC_CHARS = 420          # truncate each passage to ~100 tokens
P_ABSTAIN = 0.25         # fraction with the golden passage removed
N_TRAIN = 45000
ABSTAIN = "The documents do not contain the answer."

def parquet_table(repo, split):
    for rev in ("refs/convert/parquet", "main"):
        try:
            files = [f for f in list_repo_files(repo, repo_type="dataset", revision=rev)
                     if f.endswith(".parquet") and (f"/{split}/" in f or split in os.path.basename(f))]
        except Exception:
            files = []
        if files:
            return pa.concat_tables([pq.read_table(hf_hub_download(repo, f, repo_type="dataset", revision=rev))
                                     for f in sorted(files)])
    raise RuntimeError(f"no parquet split '{split}' for {repo}")

def window(context, ans_start, budget=DOC_CHARS):
    """Truncate a passage to ~budget chars, keeping the answer span in view."""
    if ans_start is None or len(context) <= budget:
        return context[:budget].strip()
    lo = max(0, ans_start - budget // 2)
    return context[lo:lo + budget].strip()

def main():
    tok = Tokenizer.from_file(os.path.join(HERE, "chat", "tokenizer.json"))
    U, A, E = (tok.token_to_id(x) for x in ("<|user|>", "<|assistant|>", "<|endoftext|>"))
    shutil.copy(os.path.join(HERE, "chat", "tokenizer.json"), os.path.join(OUT, "tokenizer.json"))

    print("loading SQuAD...", flush=True)
    t = parquet_table("rajpurkar/squad", "train")
    ctx = t.column("context").to_pylist()
    q = t.column("question").to_pylist()
    ans = t.column("answers").to_pylist()

    uniq_ctx = sorted(set(ctx))
    print(f"{len(q):,} questions, {len(uniq_ctx):,} unique passages", flush=True)
    json.dump(uniq_ctx, open(os.path.join(OUT, "index_docs.json"), "w"))

    rng = np.random.default_rng(1337)
    order = rng.permutation(len(q))[:N_TRAIN]

    rows_tok, rows_mask = [], []
    n_abstain = 0
    for j in order:
        answers = ans[j]["text"]
        starts = ans[j]["answer_start"]
        if not answers:
            continue
        gold_full = ctx[j]
        gold = window(gold_full, starts[0] if starts else None)
        distractors = [window(uniq_ctx[k], None)
                       for k in rng.choice(len(uniq_ctx), size=K_DOCS, replace=False)
                       if uniq_ctx[k] != gold_full][:K_DOCS - 1]

        abstain = rng.random() < P_ABSTAIN
        if abstain:
            docs = distractors + [window(uniq_ctx[int(rng.integers(len(uniq_ctx)))], None)]
            rng.shuffle(docs)
            response = ABSTAIN
            n_abstain += 1
        else:
            docs = distractors + [gold]
            rng.shuffle(docs)
            gi = docs.index(gold) + 1
            response = f"According to document [{gi}], {answers[0].strip()}."

        prompt = "Documents:\n" + "\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs)) \
                 + f"\n\nQuestion: {q[j].strip()}"

        seq = [U] + tok.encode(prompt).ids + [A]
        mask = [0] * len(seq)
        rids = tok.encode(response).ids + [E]
        seq += rids
        mask += [1] * len(rids)
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
    print(f"wrote {len(tokens):,} examples ({n_abstain:,} abstain), "
          f"{masks.sum()/masks.size:.1%} answer tokens under loss", flush=True)

if __name__ == "__main__":
    main()
