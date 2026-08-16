"""
A/B corpus builder: raw FineWeb vs vocabulary-filtered FineWeb, matched on
token count, sharing one tokenizer so the only difference is the DATA.

  ab_raw/      {train.bin, val.bin, tokenizer.json}   unfiltered FineWeb
  ab_filtered/ {train.bin, val.bin, tokenizer.json}   simplest KEEP_FRAC by
                                                      rare-word score

Both arms get exactly TARGET_TOKENS train + VAL_TOKENS val, encoded with one
shared 8K BPE. Each model can then be evaluated on BOTH val sets (in-domain vs
out-of-domain), which gives a guaranteed signal on top of the benchmarks.

Source: kjj0/fineweb10B-gpt2 shards (GPT-2 tokens) decoded back to text.
"""
import os
import re
import shutil
from collections import Counter

import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "ab_raw")
FILT = os.path.join(HERE, "ab_filtered")
for d in (RAW, FILT):
    os.makedirs(d, exist_ok=True)

VOCAB_SIZE = 8192
TARGET_TOKENS = 200_000_000
VAL_TOKENS = 2_000_000
KEEP_FRAC = 0.25          # keep the simplest 25% (yield-practical setting)
TOP_N = 2000              # score against the top-2K common words
MAX_SHARDS = 8            # shards 1-8 are pre-fetched by fetch_shards.py
EOT_GPT2 = 50256

word_re = re.compile(r"[a-z']+")
gpt2 = tiktoken.get_encoding("gpt2")

def shard_docs(i):
    """Decode one kjj0 FineWeb shard back into documents."""
    fname = f"fineweb_train_{i:06d}.bin"
    path = hf_hub_download("kjj0/fineweb10B-gpt2", fname, repo_type="dataset")
    header = np.fromfile(path, dtype=np.int32, count=256)
    n = int(header[2])
    toks = np.fromfile(path, dtype=np.uint16, offset=256 * 4, count=n)
    breaks = np.flatnonzero(toks == EOT_GPT2)
    prev = 0
    for b in breaks:
        if b > prev:
            yield gpt2.decode(toks[prev:b].tolist())
        prev = b + 1

def rare_frac(text, common):
    w = word_re.findall(text.lower())
    if not w:
        return 1.0
    return sum(x not in common for x in w) / len(w)

def main():
    # ---- phase 0: common-word set, score threshold, shared tokenizer ----
    print("phase 0: sampling shard 1 for vocab stats + tokenizer...", flush=True)
    sample = []
    for d in shard_docs(1):
        sample.append(d)
        if len(sample) >= 40000:
            break
    freq = Counter()
    for d in sample:
        freq.update(word_re.findall(d.lower()))
    common = set(w for w, _ in freq.most_common(TOP_N))
    scores = np.array([rare_frac(d, common) for d in sample])
    threshold = float(np.quantile(scores, KEEP_FRAC))
    simple_sample = [d for d, s in zip(sample, scores) if s <= threshold]
    print(f"  score threshold @keep{KEEP_FRAC:.0%} = {threshold:.4f} "
          f"({len(simple_sample):,}/{len(sample):,} docs)", flush=True)

    print("  training shared 8K tokenizer (mixed raw+filtered)...", flush=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    mix = sample[::4] + simple_sample[::2]      # both registers represented
    tok.train_from_iterator(mix, trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, special_tokens=["<|endoftext|>"]))
    tok.save(os.path.join(RAW, "tokenizer.json"))
    shutil.copy(os.path.join(RAW, "tokenizer.json"), os.path.join(FILT, "tokenizer.json"))
    E = tok.token_to_id("<|endoftext|>")

    # ---- phase 1: stream shards, fill both arms to the same token count ----
    need = TARGET_TOKENS + VAL_TOKENS
    arms = {"raw": {"chunks": [], "n": 0}, "filt": {"chunks": [], "n": 0}}
    batch = {"raw": [], "filt": []}

    def flush(name):
        if not batch[name]:
            return
        for enc in tok.encode_batch(batch[name]):
            a = np.array(enc.ids + [E], dtype=np.uint16)
            arms[name]["chunks"].append(a)
            arms[name]["n"] += len(a)
        batch[name] = []

    for si in range(1, MAX_SHARDS + 1):
        if arms["raw"]["n"] >= need and arms["filt"]["n"] >= need:
            break
        print(f"phase 1: shard {si} "
              f"(raw {arms['raw']['n']/1e6:.0f}M / filt {arms['filt']['n']/1e6:.0f}M "
              f"of {need/1e6:.0f}M)", flush=True)
        for doc in shard_docs(si):
            if arms["raw"]["n"] < need:
                batch["raw"].append(doc)
                if len(batch["raw"]) >= 2000:
                    flush("raw")
            if arms["filt"]["n"] < need and rare_frac(doc, common) <= threshold:
                batch["filt"].append(doc)
                if len(batch["filt"]) >= 2000:
                    flush("filt")
            if arms["raw"]["n"] >= need and arms["filt"]["n"] >= need:
                break
    flush("raw"); flush("filt")

    # trim BOTH arms to the same length — a token-matched budget is the whole
    # point of the A/B, so if one arm came up short the other must follow
    built = {n: np.concatenate(arms[n]["chunks"]) for n in ("raw", "filt")}
    n_tokens = min(len(built["raw"]), len(built["filt"]), need)
    print(f"matching both arms to {n_tokens:,} tokens "
          f"(raw had {len(built['raw'])/1e6:.0f}M, filt {len(built['filt'])/1e6:.0f}M)", flush=True)
    for name, out in (("raw", RAW), ("filt", FILT)):
        arr = built[name][:n_tokens]
        arr[:VAL_TOKENS].tofile(os.path.join(out, "val.bin"))
        arr[VAL_TOKENS:].tofile(os.path.join(out, "train.bin"))
        print(f"{name}: wrote {len(arr)-VAL_TOKENS:,} train / {VAL_TOKENS:,} val tokens",
              flush=True)
    print("done — both arms matched on token count, shared tokenizer", flush=True)

if __name__ == "__main__":
    main()
