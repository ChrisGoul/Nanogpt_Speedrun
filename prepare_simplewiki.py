"""
Prepare Simple English Wikipedia for from-scratch training, TinyStories-style:
keep the world small enough for a tiny model to master.

Three complexity-reduction steps:
  1. Rare-word filter: score each article by the fraction of its words that
     are rare in the corpus; drop the hardest ~30% (technical stubs,
     proper-noun-dense pages).
  2. Custom 8K BPE tokenizer trained on the filtered corpus (vs GPT-2's 50K):
     shrinks the model's embedding+head from ~38M params to ~6M.
  3. Output goes to simplewiki/ (train.bin / val.bin / tokenizer.json);
     train.py --data simplewiki picks it all up.
"""
import os
import re
from collections import Counter

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "simplewiki")
os.makedirs(OUT, exist_ok=True)

VOCAB_SIZE = 8192
KEEP_FRAC = 0.70      # keep the simplest 70% of articles
RARE_TOP_K = 20_000   # words outside the top-20K count as "rare"
VAL_TOKENS = 250_000

def main():
    print("loading Simple English Wikipedia...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    texts = [d["text"] for d in ds]
    print(f"{len(texts):,} articles", flush=True)

    # ---- 1. rare-word filter ----
    word_re = re.compile(r"[a-z']+")
    freq = Counter()
    doc_words = []
    for t in texts:
        words = word_re.findall(t.lower())
        doc_words.append(words)
        freq.update(words)
    common = set(w for w, _ in freq.most_common(RARE_TOP_K))

    def rare_frac(words):
        if not words:
            return 1.0
        return sum(w not in common for w in words) / len(words)

    scores = [rare_frac(w) for w in doc_words]
    cutoff = float(np.quantile(scores, KEEP_FRAC))
    kept = [t for t, s in zip(texts, scores) if s <= cutoff]
    n_words = sum(len(w) for w, s in zip(doc_words, scores) if s <= cutoff)
    print(f"kept {len(kept):,}/{len(texts):,} articles "
          f"(rare-word cutoff {cutoff:.3f}), ~{n_words/1e6:.0f}M words", flush=True)

    # ---- 2. train an 8K byte-level BPE tokenizer on the kept text ----
    print("training tokenizer...", flush=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE,
                                  special_tokens=["<|endoftext|>"])
    tok.train_from_iterator(kept, trainer=trainer)
    tok.save(os.path.join(OUT, "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")

    # ---- 3. encode everything -> train.bin / val.bin ----
    print("encoding...", flush=True)
    ids = []
    for i, enc in enumerate(tok.encode_batch(kept)):
        ids.extend(enc.ids)
        ids.append(eot)
        if i % 20_000 == 0:
            print(f"  {i:,} docs, {len(ids):,} tokens", flush=True)
    arr = np.array(ids, dtype=np.uint16)
    val, train = arr[:VAL_TOKENS], arr[VAL_TOKENS:]
    train.tofile(os.path.join(OUT, "train.bin"))
    val.tofile(os.path.join(OUT, "val.bin"))
    print(f"wrote {len(train):,} train / {len(val):,} val tokens, "
          f"vocab {VOCAB_SIZE}", flush=True)

if __name__ == "__main__":
    main()
