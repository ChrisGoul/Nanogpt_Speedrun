"""
Download pre-tokenized FineWeb shards from kjj0/fineweb10B-gpt2 — the same
files the real modded-nanogpt speedrun trains on. Each shard is GPT-2 BPE
tokens with a 256-int32 header (magic, version, token count) followed by
uint16 token ids. We download one train shard (~100M tokens) + the val shard,
strip the headers, and write plain train.bin / val.bin (capped so a local run
stays light on disk).
"""
import os

import numpy as np
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "kjj0/fineweb10B-gpt2"

TRAIN_TOKENS = 20_000_000
VAL_TOKENS = 250_000
MAGIC, VERSION = 20240520, 1

def load_shard(fname: str) -> np.ndarray:
    path = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset")
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == MAGIC and header[1] == VERSION, "unexpected shard format"
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype=np.uint16, offset=256 * 4, count=num_tokens)
    print(f"{fname}: {num_tokens:,} tokens", flush=True)
    return tokens

def main():
    train = load_shard("fineweb_train_000001.bin")[:TRAIN_TOKENS]
    val = load_shard("fineweb_val_000000.bin")[:VAL_TOKENS]
    train.tofile(os.path.join(HERE, "train.bin"))
    val.tofile(os.path.join(HERE, "val.bin"))
    print(f"wrote train.bin ({len(train):,} tokens), val.bin ({len(val):,} tokens)")

if __name__ == "__main__":
    main()
