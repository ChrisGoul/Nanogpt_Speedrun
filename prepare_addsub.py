"""
Clean addition+subtraction corpus (<=3 digits) for the abacus length-generalization
test. Same char-level byte tokenizer. Only columnar ops (the case abacus is
designed for). Writes addsub/{train.bin, val.bin, tokenizer.json}.
"""
import os

import numpy as np

from prepare_reason import build_byte_tokenizer
import reason_gen

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "addsub")
os.makedirs(OUT, exist_ok=True)

TARGET_TOKENS = 60_000_000
VAL_TOKENS = 1_000_000
GENS = [reason_gen.add_cot, reason_gen.sub_cot]

def main():
    tok = build_byte_tokenizer()
    tok.save(os.path.join(OUT, "tokenizer.json"))
    E = tok.token_to_id("<|endoftext|>")
    print(f"byte tokenizer: vocab {tok.get_vocab_size()}", flush=True)

    need = TARGET_TOKENS + VAL_TOKENS
    chunks, total, batch = [], 0, []
    def flush():
        nonlocal total
        for enc in tok.encode_batch(batch):
            chunks.append(np.array(enc.ids + [E], dtype=np.uint16))
            total += len(chunks[-1])
        batch.clear()
    while total < need:
        batch.append(reason_gen.R.choice(GENS)()[0])
        if len(batch) >= 4000:
            flush()
    if batch:
        flush()

    arr = np.concatenate(chunks)[:need]
    arr[:VAL_TOKENS].tofile(os.path.join(OUT, "val.bin"))
    arr[VAL_TOKENS:].tofile(os.path.join(OUT, "train.bin"))
    print(f"addsub: wrote {len(arr)-VAL_TOKENS:,} train / {VAL_TOKENS:,} val tokens", flush=True)

if __name__ == "__main__":
    main()
