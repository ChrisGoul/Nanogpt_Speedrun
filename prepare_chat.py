"""
Route A chat corpus: real adult dialogue datasets, formatted as multi-turn
conversations for a fluent-but-factless chatbot.

Sources (both on HuggingFace):
  - DailyDialog        (~13k everyday conversations) - clean adult chitchat
  - EmpatheticDialogues (~25k) - social/emotional back-and-forth

Each conversation becomes:
  <|user|> turn0 <|assistant|> turn1 <|user|> turn2 <|assistant|> turn3 ...
and the loss is masked to the assistant turns only. A fresh 16K BPE tokenizer
is trained on the dialogue text (adult vocabulary is wider than TinyStories, and
with little factual load a bigger vocab is affordable).

Outputs to chat/:
  tokens.bin    uint16 (N, SEQ_LEN)
  loss_mask.bin uint8  (N, SEQ_LEN)  1 on assistant-turn tokens
  tokenizer.json
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "chat")
os.makedirs(OUT, exist_ok=True)

VOCAB_SIZE = 16384
SEQ_LEN = 512

def load_parquet_split(repo: str, split: str = "train") -> pa.Table:
    """Load a dataset split from its parquet files. Script-based datasets are no
    longer supported by `datasets`, so we read the auto-converted parquet on the
    refs/convert/parquet branch (falling back to parquet on the main branch)."""
    for rev in ("refs/convert/parquet", "main"):
        try:
            files = list_repo_files(repo, repo_type="dataset", revision=rev)
        except Exception:
            continue
        want = [f for f in files if f.endswith(".parquet") and f"/{split}/" in f] \
            or [f for f in files if f.endswith(".parquet") and split in os.path.basename(f)]
        if not want:
            continue
        tables = []
        for f in want:
            p = hf_hub_download(repo, f, repo_type="dataset", revision=rev)
            tables.append(pq.read_table(p))
        return pa.concat_tables(tables)
    raise RuntimeError(f"no parquet split '{split}' found for {repo}")

def load_dailydialog() -> list[list[str]]:
    tbl = load_parquet_split("li2017dailydialog/daily_dialog")
    convs = []
    for dialog in tbl.column("dialog").to_pylist():
        turns = [t.strip() for t in dialog if t and t.strip()]
        if len(turns) >= 2:
            convs.append(turns)
    print(f"DailyDialog: {len(convs):,} conversations", flush=True)
    return convs

def load_empathetic() -> list[list[str]]:
    tbl = load_parquet_split("facebook/empathetic_dialogues")
    conv_id = tbl.column("conv_id").to_pylist()
    idx = tbl.column("utterance_idx").to_pylist()
    utt = tbl.column("utterance").to_pylist()
    by_conv: dict[str, list[tuple[int, str]]] = {}
    for cid, ui, u in zip(conv_id, idx, utt):
        u = (u or "").replace("_comma_", ",").strip()
        if u:
            by_conv.setdefault(cid, []).append((int(ui), u))
    convs = []
    for turns in by_conv.values():
        turns.sort()
        seq = [u for _, u in turns]
        if len(seq) >= 2:
            convs.append(seq)
    print(f"EmpatheticDialogues: {len(convs):,} conversations", flush=True)
    return convs

def main():
    convs = load_dailydialog() + load_empathetic()
    print(f"total: {len(convs):,} conversations", flush=True)

    # ---- 16K BPE tokenizer on the dialogue text ----
    print("training tokenizer...", flush=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        (t for c in convs for t in c),
        trainers.BpeTrainer(vocab_size=VOCAB_SIZE,
                            special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>"]))
    tok.save(os.path.join(OUT, "tokenizer.json"))
    U = tok.token_to_id("<|user|>")
    A = tok.token_to_id("<|assistant|>")
    E = tok.token_to_id("<|endoftext|>")

    # ---- format multi-turn, mask assistant turns ----
    print("encoding...", flush=True)
    rows_tok, rows_mask = [], []
    for turns in convs:
        seq, mask = [], []
        for i, turn in enumerate(turns):
            role = U if i % 2 == 0 else A          # alternate speakers
            ids = tok.encode(turn).ids
            seq.append(role); mask.append(0)       # role marker: no loss
            seq.extend(ids)
            mask.extend([1 if role == A else 0] * len(ids))  # loss on assistant text
        seq.append(E); mask.append(1)              # learn to end the (assistant) turn
        # pad / truncate to SEQ_LEN
        seq, mask = seq[:SEQ_LEN], mask[:SEQ_LEN]
        if sum(mask) < 2:
            continue
        pad = SEQ_LEN - len(seq)
        rows_tok.append(np.array(seq + [E] * pad, dtype=np.uint16))
        rows_mask.append(np.array(mask + [0] * pad, dtype=np.uint8))

    tokens = np.stack(rows_tok)
    masks = np.stack(rows_mask)
    tokens.tofile(os.path.join(OUT, "tokens.bin"))
    masks.tofile(os.path.join(OUT, "loss_mask.bin"))
    print(f"wrote {len(tokens):,} examples x {SEQ_LEN} tokens "
          f"({masks.sum()/masks.size:.1%} assistant tokens under loss), "
          f"vocab {VOCAB_SIZE}", flush=True)

if __name__ == "__main__":
    main()
