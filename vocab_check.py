"""
Sanity check: compare the word-type distribution of our filtered Simple
English Wikipedia against real TinyStories text. Reports, for each corpus:
  - unique word types per 1M running words
  - how many word types cover 95% / 99% of all tokens
  - fraction of tokens outside the top-2K and top-20K types
Run after prepare_simplewiki.py (reuses the cached dataset + same filter).
"""
import re
from collections import Counter

import numpy as np
from datasets import load_dataset

word_re = re.compile(r"[a-z']+")
SAMPLE_WORDS = 5_000_000  # compare equal-sized samples

def stats(name, texts):
    freq = Counter()
    total = 0
    for t in texts:
        words = word_re.findall(t.lower())
        freq.update(words)
        total += len(words)
        if total >= SAMPLE_WORDS:
            break
    counts = np.array(sorted(freq.values(), reverse=True))
    cum = np.cumsum(counts) / counts.sum()
    cover95 = int(np.searchsorted(cum, 0.95) + 1)
    cover99 = int(np.searchsorted(cum, 0.99) + 1)
    oov2k = counts[2000:].sum() / counts.sum() if len(counts) > 2000 else 0.0
    oov20k = counts[20_000:].sum() / counts.sum() if len(counts) > 20_000 else 0.0
    print(f"{name:24s} | {counts.sum():>9,} words | {len(counts):>7,} types "
          f"| 95% cover: {cover95:>6,} | 99% cover: {cover99:>7,} "
          f"| OOV@2k: {oov2k:.2%} | OOV@20k: {oov20k:.2%}", flush=True)

def main():
    print("loading TinyStories sample...", flush=True)
    ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    stats("TinyStories", (d["text"] for d in ts))

    print("loading SimpleWiki (cached)...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    texts = [d["text"] for d in ds]
    stats("SimpleWiki (unfiltered)", texts)

    # reproduce prepare_simplewiki's filter
    freq = Counter()
    doc_words = [word_re.findall(t.lower()) for t in texts]
    for w in doc_words:
        freq.update(w)
    common = set(w for w, _ in freq.most_common(20_000))
    scores = [sum(w not in common for w in ws) / max(len(ws), 1) for ws in doc_words]
    cutoff = float(np.quantile(scores, 0.70))
    kept = [t for t, s in zip(texts, scores) if s <= cutoff]
    stats("SimpleWiki (filtered)", kept)

if __name__ == "__main__":
    main()
