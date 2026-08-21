"""
SFT dataset for the mix16 general model (156M, tied 16K vocab).

The mix16 tokenizer only has <|endoftext|> as a special token (no dedicated
role tokens), so we mark turns with the *text* strings "<|user|>" /
"<|assistant|>", which the BPE encodes into a few ordinary tokens. The base
model learns these markers during SFT — no embedding surgery needed.

Reuses the proven instruction loaders from prepare_sftmix / prepare_chat:
  - dialogue     DailyDialog + EmpatheticDialogues   -> conversational ability
  - instruction  Dolly-15k                           -> answering in Q&A form
  - reasoning    GSM8K (worked step-by-step)  [x2]    -> chain-of-thought
  - commonsense  CommonsenseQA                        -> pick-and-justify
  - comprehension SQuAD (answer from passage)         -> reasoning over context

Each turn: <|user|>\n{prompt}\n<|assistant|>\n{response}<eot>, loss masked to
assistant tokens only. Fixed-length SEQ_LEN rows.

Output: sft16/{tokens.bin, loss_mask.bin, tokenizer.json}
Train:  python train_sft16.py
"""
import os
import shutil

import numpy as np
from tokenizers import Tokenizer

from prepare_sftmix import (dialogue_convs, dolly_pairs, gsm8k_pairs,
                            commonsenseqa_pairs, squad_pairs)

HERE = os.path.dirname(os.path.abspath(__file__))
# SFT_TOK dir supplies the tokenizer (must match the base model you'll fine-tune);
# SFT_OUT is where the tokenized instruction data lands. Defaults reproduce the
# original mix16 -> sft16 build; override for other models (e.g. big300c).
SRC = os.path.join(HERE, os.environ.get("SFT_TOK", "mix16"))
OUT = os.path.join(HERE, os.environ.get("SFT_OUT", "sft16"))
os.makedirs(OUT, exist_ok=True)
SEQ_LEN = 512

def main():
    tok = Tokenizer.from_file(os.path.join(SRC, "tokenizer.json"))
    E = tok.token_to_id("<|endoftext|>")
    assert E is not None, "mix16 tokenizer missing <|endoftext|>"
    # text role markers -> ordinary BPE ids (encoded once)
    U_ids = tok.encode("<|user|>\n").ids
    A_ids = tok.encode("<|assistant|>\n").ids
    shutil.copy(os.path.join(SRC, "tokenizer.json"), os.path.join(OUT, "tokenizer.json"))

    convs = []
    convs += dialogue_convs()          # multi-turn lists
    convs += dolly_pairs()             # [prompt, response]
    convs += gsm8k_pairs() * 2         # upweight reasoning
    convs += commonsenseqa_pairs()
    convs += squad_pairs()
    print(f"total examples: {len(convs):,}", flush=True)

    rng = np.random.default_rng(1337)
    rng.shuffle(convs)

    rows_tok, rows_mask = [], []
    for turns in convs:
        seq, mask = [], []
        for i, turn in enumerate(turns):
            is_asst = (i % 2 == 1)
            marker = A_ids if is_asst else U_ids
            ids = tok.encode(turn).ids
            seq.extend(marker);   mask.extend([0] * len(marker))       # marker never under loss
            seq.extend(ids);      mask.extend([1 if is_asst else 0] * len(ids))
        seq.append(E); mask.append(1)                                  # grade the stop token
        seq, mask = seq[:SEQ_LEN], mask[:SEQ_LEN]
        if sum(mask) < 2:                                              # response truncated away
            continue
        pad = SEQ_LEN - len(seq)
        rows_tok.append(np.array(seq + [E] * pad, dtype=np.uint16))
        rows_mask.append(np.array(mask + [0] * pad, dtype=np.uint8))

    tokens = np.stack(rows_tok)
    masks = np.stack(rows_mask)
    tokens.tofile(os.path.join(OUT, "tokens.bin"))
    masks.tofile(os.path.join(OUT, "loss_mask.bin"))
    print(f"wrote {len(tokens):,} examples x {SEQ_LEN} "
          f"({masks.sum()/masks.size:.1%} assistant tokens under loss)", flush=True)

if __name__ == "__main__":
    main()
