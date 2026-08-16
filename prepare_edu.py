"""
Build the FineWeb-Edu arm (ab_edu) OFFLINE from cached edu GPT-2 shards
(fetch_edu.py downloads them first). Decode the GPT-2 tokens back to text and
re-encode with the SAME 8K tokenizer as ab_raw — identical pipeline to how the
raw arm was built, so the ONLY difference between arms is the data.

  ab_edu/ {train.bin (200M), val.bin (2M), tokenizer.json}
"""
import os
import shutil

os.environ["HF_HUB_OFFLINE"] = "1"   # read only the local cache; never hit network

import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ab_edu")
os.makedirs(OUT, exist_ok=True)

REPO = "karpathy/fineweb-edu-100B-gpt2-token-shards"
TARGET_TOKENS = 200_000_000
VAL_TOKENS = 2_000_000
EOT_GPT2 = 50256
gpt2 = tiktoken.get_encoding("gpt2")

def shard_docs(i):
    path = hf_hub_download(REPO, f"edu_fineweb_train_{i:06d}.bin", repo_type="dataset")
    n = int(np.fromfile(path, dtype=np.int32, count=256)[2])
    toks = np.fromfile(path, dtype=np.uint16, offset=256 * 4, count=n)
    breaks = np.flatnonzero(toks == EOT_GPT2)
    prev = 0
    for b in breaks:
        if b > prev:
            yield gpt2.decode(toks[prev:b].tolist())
        prev = b + 1

def main():
    tok = Tokenizer.from_file(os.path.join(HERE, "ab_raw", "tokenizer.json"))
    E = tok.token_to_id("<|endoftext|>")
    shutil.copy(os.path.join(HERE, "ab_raw", "tokenizer.json"), os.path.join(OUT, "tokenizer.json"))

    need = TARGET_TOKENS + VAL_TOKENS
    chunks, total, batch = [], 0, []
    def flush():
        nonlocal total
        for enc in tok.encode_batch(batch):
            chunks.append(np.array(enc.ids + [E], dtype=np.uint16))
            total += len(chunks[-1])
        batch.clear()

    for si in range(1, 6):
        if total >= need:
            break
        print(f"decoding+encoding edu shard {si} ({total/1e6:.0f}M/{need/1e6:.0f}M tokens)", flush=True)
        for doc in shard_docs(si):
            batch.append(doc)
            if len(batch) >= 2000:
                flush()
                if total >= need:
                    break
    if batch and total < need:
        flush()

    arr = np.concatenate(chunks)[:need]
    arr[:VAL_TOKENS].tofile(os.path.join(OUT, "val.bin"))
    arr[VAL_TOKENS:].tofile(os.path.join(OUT, "train.bin"))
    print(f"ab_edu: wrote {len(arr)-VAL_TOKENS:,} train / {VAL_TOKENS:,} val tokens", flush=True)

if __name__ == "__main__":
    main()
