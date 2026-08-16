"""
Build the synthetic reasoning corpus for training.

Character/byte-level tokenizer (~257 tokens) so every DIGIT is its own
consistent token — the key choice for learning arithmetic (per "Teaching
Arithmetic to Small Transformers"). Tiny vocab also means almost all model
params go to the transformer, not the embedding table.

Generates ~200M tokens of mixed chain-of-thought problems (free, programmatic,
no API / no downloads). Writes reason/{train.bin, val.bin, tokenizer.json}.
"""
import os

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

import reason_gen

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reason")
os.makedirs(OUT, exist_ok=True)

TARGET_TOKENS = 200_000_000
VAL_TOKENS = 2_000_000

def build_byte_tokenizer():
    # pure byte-level: the 256 byte tokens + <|endoftext|>, no merges -> char-level
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {c: i for i, c in enumerate(alphabet)}
    vocab["<|endoftext|>"] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|endoftext|>"])
    return tok

def main():
    tok = build_byte_tokenizer()
    tok.save(os.path.join(OUT, "tokenizer.json"))
    E = tok.token_to_id("<|endoftext|>")
    print(f"byte tokenizer: vocab {tok.get_vocab_size()}", flush=True)

    need = TARGET_TOKENS + VAL_TOKENS
    chunks, total = [], 0
    batch = []
    def flush():
        nonlocal total
        for enc in tok.encode_batch(batch):
            chunks.append(np.array(enc.ids + [E], dtype=np.uint16))
            total += len(chunks[-1])
        batch.clear()

    while total < need:
        batch.append(reason_gen.sample())
        if len(batch) >= 4000:
            flush()
            if total // 20_000_000 != (total - 1) // 20_000_000:
                print(f"  {total/1e6:.0f}M / {need/1e6:.0f}M tokens", flush=True)
    if batch:
        flush()

    arr = np.concatenate(chunks)[:need]
    arr[:VAL_TOKENS].tofile(os.path.join(OUT, "val.bin"))
    arr[VAL_TOKENS:].tofile(os.path.join(OUT, "train.bin"))
    print(f"reason: wrote {len(arr)-VAL_TOKENS:,} train / {VAL_TOKENS:,} val tokens", flush=True)

if __name__ == "__main__":
    main()
