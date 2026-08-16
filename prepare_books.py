"""
Books backbone for the chatbot.

Stage 1 pretraining needs a large, modern, dialogue-rich, fact-light language
corpus. We pull a bounded slice of a fiction corpus (BookCorpus, with fallbacks),
train ONE shared 16K tokenizer on books+dialogue, then write:

  chat/tokenizer.json          shared 16K BPE (books + dialogue)
  books/tokenizer.json         copy (so `train.py --data books` finds it)
  books/train.bin, val.bin     flat token stream for pretraining
  chat/tokens.bin, loss_mask.bin   dialogue re-encoded with the shared tokenizer

Then:  python train.py --data books ...        (pretrain from scratch)
       python train_chat.py --init books ...   (fine-tune on dialogue)
"""
import os
import shutil

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from prepare_chat import load_dailydialog, load_empathetic

HERE = os.path.dirname(os.path.abspath(__file__))
BOOKS = os.path.join(HERE, "books")
CHAT = os.path.join(HERE, "chat")
os.makedirs(BOOKS, exist_ok=True)
os.makedirs(CHAT, exist_ok=True)

VOCAB_SIZE = 16384
SEQ_LEN = 512
TARGET_BOOK_TOKENS = 100_000_000
VAL_TOKENS = 500_000

# (repo, text-column) candidates, tried in order. All modern-ish prose.
BOOK_SOURCES = [
    ("bookcorpus/bookcorpus", "text"),
    ("euclaise/writingprompts", "story"),
    ("sedthh/gutenberg_english", "TEXT"),
]

def iter_book_texts(max_chars: int):
    """Yield text from the first available source, stopping near max_chars."""
    for repo, col in BOOK_SOURCES:
        files = []
        for rev in ("refs/convert/parquet", "main"):
            try:
                files = [(rev, f) for f in list_repo_files(repo, repo_type="dataset", revision=rev)
                         if f.endswith(".parquet") and "train" in f.lower()]
            except Exception:
                files = []
            if files:
                break
        if not files:
            print(f"  {repo}: no parquet, skipping", flush=True)
            continue
        print(f"using books source: {repo} ({len(files)} shards)", flush=True)
        got = 0
        for rev, f in sorted(files):
            try:
                path = hf_hub_download(repo, f, repo_type="dataset", revision=rev)
                tbl = pq.read_table(path, columns=[col])
            except Exception as e:
                print(f"  shard {f} failed: {e}", flush=True)
                continue
            for t in tbl.column(col).to_pylist():
                if t:
                    yield t
                    got += len(t)
            print(f"  {got/1e6:.0f}M chars", flush=True)
            if got >= max_chars:
                return
        return
    raise RuntimeError("no book source available")

def main():
    # gather dialogue (reused later for tokenizer + re-encoding)
    convs = load_dailydialog() + load_empathetic()

    # pull books text (~4 chars/token, so aim for 4x the token target in chars)
    print("downloading books...", flush=True)
    books = list(iter_book_texts(max_chars=TARGET_BOOK_TOKENS * 4))
    print(f"books: {len(books):,} passages", flush=True)

    # ---- shared 16K tokenizer on books + dialogue ----
    print("training shared tokenizer...", flush=True)
    def tok_iter():
        for i in range(0, len(books), 3):   # sample books for speed
            yield books[i]
        for c in convs:
            for t in c:
                yield t
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(tok_iter(), trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>"]))
    tok.save(os.path.join(CHAT, "tokenizer.json"))
    shutil.copy(os.path.join(CHAT, "tokenizer.json"), os.path.join(BOOKS, "tokenizer.json"))
    U, A, E = (tok.token_to_id(x) for x in ("<|user|>", "<|assistant|>", "<|endoftext|>"))

    # ---- encode books -> flat token stream ----
    print("encoding books...", flush=True)
    chunks, total = [], 0
    batch = []
    def flush():
        nonlocal total
        for enc in tok.encode_batch(batch):
            chunks.append(np.array(enc.ids + [E], dtype=np.uint16))
            total += len(chunks[-1])
    for i, t in enumerate(books):
        batch.append(t)
        if len(batch) >= 2000:
            flush(); batch = []
            if total >= TARGET_BOOK_TOKENS:
                break
            if (i // 2000) % 10 == 0:
                print(f"  {total/1e6:.0f}M tokens", flush=True)
    if batch and total < TARGET_BOOK_TOKENS:
        flush()
    arr = np.concatenate(chunks)[:TARGET_BOOK_TOKENS]
    arr[:VAL_TOKENS].tofile(os.path.join(BOOKS, "val.bin"))
    arr[VAL_TOKENS:].tofile(os.path.join(BOOKS, "train.bin"))
    print(f"books: wrote {len(arr)-VAL_TOKENS:,} train / {VAL_TOKENS:,} val tokens", flush=True)

    # ---- re-encode dialogue with the shared tokenizer (multi-turn, masked) ----
    print("encoding dialogue...", flush=True)
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
    np.stack(rows_tok).tofile(os.path.join(CHAT, "tokens.bin"))
    np.stack(rows_mask).tofile(os.path.join(CHAT, "loss_mask.bin"))
    print(f"dialogue: wrote {len(rows_tok):,} conversations, vocab {VOCAB_SIZE}", flush=True)

if __name__ == "__main__":
    main()
