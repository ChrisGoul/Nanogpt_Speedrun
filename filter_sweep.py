"""
Sweep the rare-word filter threshold on Simple English Wikipedia: at each
keep-fraction, what does the surviving corpus's vocabulary look like, and how
many training tokens survive? Also tries scoring with a stricter "common"
set (top-5K instead of top-20K) to see if that targets the tail better.
"""
import re
from collections import Counter

import numpy as np
from datasets import load_dataset

word_re = re.compile(r"[a-z']+")

def corpus_stats(doc_words_subset):
    freq = Counter()
    for ws in doc_words_subset:
        freq.update(ws)
    counts = np.array(sorted(freq.values(), reverse=True))
    total = counts.sum()
    cum = np.cumsum(counts) / total
    cover95 = int(np.searchsorted(cum, 0.95) + 1)
    oov2k = counts[2000:].sum() / total if len(counts) > 2000 else 0.0
    return total, len(counts), cover95, oov2k

def main():
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    doc_words = [word_re.findall(d["text"].lower()) for d in ds]
    freq = Counter()
    for ws in doc_words:
        freq.update(ws)

    print(f"{'common set':>10s} {'keep':>5s} | {'Mwords':>7s} | {'types':>8s} | "
          f"{'95% cover':>9s} | {'OOV@2k':>7s}", flush=True)
    for top_k in (20_000, 5_000):
        common = set(w for w, _ in freq.most_common(top_k))
        scores = np.array([sum(w not in common for w in ws) / max(len(ws), 1)
                           for ws in doc_words])
        for keep in (0.70, 0.50, 0.30, 0.15, 0.05):
            cutoff = np.quantile(scores, keep)
            subset = [ws for ws, s in zip(doc_words, scores) if s <= cutoff]
            total, types, cover95, oov2k = corpus_stats(subset)
            print(f"top-{top_k//1000:>2d}k {keep:>5.0%} | {total/1e6:>6.1f}M | "
                  f"{types:>8,} | {cover95:>9,} | {oov2k:>7.2%}", flush=True)

if __name__ == "__main__":
    main()
