"""
Build the blended corpus: ~85% TinyStories + ~15% Simple English Wikipedia
(filtered hard: score each article against the top-5K common words, keep the
simplest 30% — measured OOV@2k ~12.7%). The blend's overall vocabulary
concentration is TinyStories-grade (~5-6% OOV@2k) while injecting real-world
names and facts.

One shared 8K BPE tokenizer is trained on the mix. The wiki slice is small
(~12M tokens) so it is repeated 3x in the corpus (light repetition is fine;
it raises the wiki share to ~11%). Documents are shuffled together, encoded
in chunks (the corpus is ~300M+ tokens - never hold python int lists of that
size), and written to blend/train.bin, val.bin, tokenizer.json.
Train with:  python train.py --data blend ...
"""
import os
import re
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "blend")
os.makedirs(OUT, exist_ok=True)

VOCAB_SIZE = 8192
COMMON_TOP_K = 5_000
WIKI_KEEP = 0.30
WIKI_REPEATS = 3
TARGET_TOKENS = 340_000_000  # ~40k steps x 8192 tokens/step, plus val
VAL_TOKENS = 250_000

word_re = re.compile(r"[a-z']+")

def filtered_wiki() -> list[str]:
    print("loading SimpleWiki...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    texts = [d["text"] for d in ds]
    doc_words = [word_re.findall(t.lower()) for t in texts]
    freq = Counter()
    for ws in doc_words:
        freq.update(ws)
    common = set(w for w, _ in freq.most_common(COMMON_TOP_K))
    scores = np.array([sum(w not in common for w in ws) / max(len(ws), 1)
                       for ws in doc_words])
    cutoff = np.quantile(scores, WIKI_KEEP)
    kept = [t for t, s in zip(texts, scores) if s <= cutoff]
    print(f"wiki: kept {len(kept):,}/{len(texts):,} articles", flush=True)
    return kept

def load_tinystories(max_stories: int) -> list[str]:
    """Download TinyStories parquet shards (resumable) and return story texts.
    More robust than datasets' loader over a flaky connection."""
    files = sorted(f for f in list_repo_files("roneneldan/TinyStories",
                                              repo_type="dataset")
                   if f.endswith(".parquet") and "train" in f)
    texts = []
    for f in files:
        path = hf_hub_download("roneneldan/TinyStories", f, repo_type="dataset")
        col = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        texts.extend(col)
        print(f"  {f}: +{len(col):,} -> {len(texts):,} stories", flush=True)
        if len(texts) >= max_stories:
            break
    return texts[:max_stories]

def main():
    wiki = filtered_wiki()

    print("loading TinyStories...", flush=True)
    ts_texts = load_tinystories(max_stories=1_300_000)
    print(f"TinyStories: {len(ts_texts):,} stories", flush=True)

    # ---- shared tokenizer, trained on a mixed sample ----
    def tok_iter():
        yield from wiki
        for i in range(0, len(ts_texts), 8):  # every 8th story is plenty
            yield ts_texts[i]

    print("training tokenizer...", flush=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    # <|user|> / <|assistant|> are reserved now so the SFT stage can use them
    # as single atomic tokens (they never appear in pretraining text)
    tok.train_from_iterator(tok_iter(), trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>"]))
    tok.save(os.path.join(OUT, "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")

    # ---- shuffle doc order (wiki x3 interleaved with stories), encode in chunks ----
    # doc_refs: (source, index) pairs for every document instance in the corpus
    doc_refs = [("w", i) for _ in range(WIKI_REPEATS) for i in range(len(wiki))]
    doc_refs += [("t", i) for i in range(len(ts_texts))]
    rng = np.random.default_rng(1337)
    rng.shuffle(doc_refs)

    chunks, total = [], 0
    batch, CHUNK = [], 2000

    def flush():
        nonlocal total, batch
        for enc in tok.encode_batch(batch):
            chunks.append(np.array(enc.ids + [eot], dtype=np.uint16))
            total += len(chunks[-1])
        batch = []

    for j, (src, i) in enumerate(doc_refs):
        batch.append(wiki[i] if src == "w" else ts_texts[i])
        if len(batch) >= CHUNK:
            flush()
            if total >= TARGET_TOKENS:
                break
            if (j // CHUNK) % 20 == 0:
                print(f"  {total/1e6:.0f}M tokens", flush=True)
    if batch and total < TARGET_TOKENS:
        flush()

    arr = np.concatenate(chunks)[:TARGET_TOKENS]
    val, train = arr[:VAL_TOKENS], arr[VAL_TOKENS:]
    train.tofile(os.path.join(OUT, "train.bin"))
    val.tofile(os.path.join(OUT, "val.bin"))
    print(f"wrote {len(train):,} train / {len(val):,} val tokens, vocab {VOCAB_SIZE}",
          flush=True)

if __name__ == "__main__":
    main()
