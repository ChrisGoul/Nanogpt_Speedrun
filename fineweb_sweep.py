"""
Step 1: measure whether vocabulary-filtering FineWeb can reproduce the
TinyStories vocabulary profile, and at what yield.

Reference profile we're aiming at (measured earlier on real TinyStories):
    95%-cover vocab ~ 1,800 types     OOV@2K ~ 4.3%

We score each document by the fraction of its words that fall OUTSIDE the top-N
most common words, then keep the simplest X%. We report, for each setting:
  - surviving words, 95%-cover vocab, OOV@2K   (is it TinyStories-simple?)
  - % of source words retained                  (what yield / how much FineWeb
                                                 we'd need to hit ~1B tokens)
Also reported WITH a prose-quality guard, because "uses only common words" on
web data otherwise selects navigation menus and product listings.

Source: the kjj0/fineweb10B-gpt2 shard we already have (GPT-2 tokens), decoded
back to text and split on <|endoftext|>.
"""
import re
from collections import Counter

import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download

SOURCE_TOKENS = 30_000_000     # enough for stable statistics
EOT = 50256
word_re = re.compile(r"[a-z']+")
sent_re = re.compile(r"[.!?]")

# TinyStories reference (measured earlier in this project)
TS_COVER95, TS_OOV2K = 1800, 0.043

def load_docs():
    path = hf_hub_download("kjj0/fineweb10B-gpt2", "fineweb_train_000001.bin",
                           repo_type="dataset")
    header = np.fromfile(path, dtype=np.int32, count=256)
    n = min(int(header[2]), SOURCE_TOKENS)
    toks = np.fromfile(path, dtype=np.uint16, offset=256 * 4, count=n)
    enc = tiktoken.get_encoding("gpt2")
    print(f"decoding {n:,} tokens...", flush=True)
    # split on the document separator, decode each doc
    breaks = np.flatnonzero(toks == EOT)
    docs, prev = [], 0
    for b in breaks:
        if b > prev:
            docs.append(enc.decode(toks[prev:b].tolist()))
        prev = b + 1
    print(f"{len(docs):,} documents", flush=True)
    return docs

def is_prose(text):
    """Quality guard: real sentences, not menus/listings/spam."""
    if not (200 <= len(text) <= 20000):
        return False
    if len(sent_re.findall(text)) < 3:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    if alpha < 0.75:
        return False
    w = word_re.findall(text.lower())
    if len(w) < 50 or len(set(w)) / len(w) < 0.30:   # repetitive boilerplate
        return False
    return True

def profile(word_lists):
    freq = Counter()
    for w in word_lists:
        freq.update(w)
    counts = np.array(sorted(freq.values(), reverse=True))
    total = counts.sum()
    if total == 0:
        return 0, 0, 1.0
    cum = np.cumsum(counts) / total
    cover95 = int(np.searchsorted(cum, 0.95) + 1)
    oov2k = counts[2000:].sum() / total if len(counts) > 2000 else 0.0
    return int(total), cover95, float(oov2k)

def main():
    docs = load_docs()
    words = [word_re.findall(d.lower()) for d in docs]
    keep_prose = [is_prose(d) for d in docs]
    print(f"prose guard keeps {sum(keep_prose):,}/{len(docs):,} docs "
          f"({sum(keep_prose)/len(docs):.1%})\n", flush=True)

    freq = Counter()
    for w in words:
        freq.update(w)
    total_words = sum(len(w) for w in words)
    print(f"source: {total_words/1e6:.1f}M words, {len(freq):,} types")
    print(f"TARGET (TinyStories): 95%-cover ~{TS_COVER95}, OOV@2K ~{TS_OOV2K:.1%}\n")

    print(f"{'guard':>6} {'topN':>6} {'keep':>5} | {'Mwords':>7} {'%src':>6} | "
          f"{'95%cover':>9} {'OOV@2K':>7}")
    print("-" * 60)
    for guard in (False, True):
        mask = keep_prose if guard else [True] * len(docs)
        for top_n in (1000, 2000, 5000):
            common = set(w for w, _ in freq.most_common(top_n))
            idx = [i for i in range(len(docs)) if mask[i] and words[i]]
            scores = np.array([sum(w not in common for w in words[i]) / len(words[i])
                               for i in idx])
            for keep in (0.30, 0.10, 0.05, 0.02):
                cut = np.quantile(scores, keep)
                sel = [idx[j] for j in range(len(idx)) if scores[j] <= cut]
                tot, c95, oov = profile([words[i] for i in sel])
                print(f"{str(guard):>6} {top_n:>6} {keep:>5.0%} | {tot/1e6:>6.2f}M "
                      f"{tot/total_words:>5.1%} | {c95:>9,} {oov:>6.2%}")
        print()

    # eyeball what survives at an aggressive setting, with the guard on
    common = set(w for w, _ in freq.most_common(2000))
    idx = [i for i in range(len(docs)) if keep_prose[i] and words[i]]
    scores = np.array([sum(w not in common for w in words[i]) / len(words[i]) for i in idx])
    cut = np.quantile(scores, 0.05)
    sel = [idx[j] for j in range(len(idx)) if scores[j] <= cut][:3]
    print("=== sample survivors (guard on, top-2K, keep 5%) ===")
    for i in sel:
        print("---", docs[i][:400].replace("\n", " "), "\n")

if __name__ == "__main__":
    main()
