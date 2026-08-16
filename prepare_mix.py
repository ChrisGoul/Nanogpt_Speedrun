"""
High-signal pretraining mix for the tied-32K ~170M general model (SmolLM-style,
adapted to what we can get reliably on this machine):

  - FineWeb-Edu   (cached llm.c shards)     -> high-quality educational web text
  - Cosmopedia    (download, best-effort)   -> synthetic textbooks (skipped if it stalls)
  - synthetic CoT (reason_gen, local/free)  -> math/reasoning signal

Trains a fresh 32K byte-level BPE on the mix, then encodes everything.
Writes mix/{train.bin, val.bin, tokenizer.json}. Reports tokens per source.
"""
import os
import time

import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

import reason_gen

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("MIX_OUT", "mix"))
os.makedirs(OUT, exist_ok=True)

VOCAB_SIZE = int(os.environ.get("MIX_VOCAB", "32000"))
VAL_TOKENS = 2_000_000
COT_FRACTION_DOCS = 120_000     # modest slice of synthetic reasoning
gpt2 = tiktoken.get_encoding("gpt2")
EOT_GPT2 = 50256

def fineweb_edu_docs(max_shards=None):
    # shards: MIX_EDU_SHARDS env (default 3). Offline by default (uses shards
    # already cached on the dev box); on a fresh cloud pod set MIX_OFFLINE=0 to
    # DOWNLOAD them, and raise MIX_EDU_SHARDS to scale the token count up
    # (each shard ~0.1B gpt2 tokens -> ~0.14B at 16K vocab).
    max_shards = max_shards or int(os.environ.get("MIX_EDU_SHARDS", "3"))
    offline = os.environ.get("MIX_OFFLINE", "1") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"          # cached shards only
    for i in range(1, max_shards + 1):
        try:
            path = hf_hub_download("karpathy/fineweb-edu-100B-gpt2-token-shards",
                                   f"edu_fineweb_train_{i:06d}.bin", repo_type="dataset")
        except Exception as e:
            print(f"  edu shard {i} unavailable ({e}); stopping", flush=True)
            break
        n = int(np.fromfile(path, dtype=np.int32, count=256)[2])
        toks = np.fromfile(path, dtype=np.uint16, offset=256 * 4, count=n)
        breaks = np.flatnonzero(toks == EOT_GPT2)
        prev = 0
        for b in breaks:
            if b > prev:
                yield gpt2.decode(toks[prev:b].tolist())
            prev = b + 1
    os.environ.pop("HF_HUB_OFFLINE", None)

def cosmopedia_docs(target=150_000):
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
    import pyarrow.parquet as pq
    from huggingface_hub import list_repo_files
    try:
        files = sorted(f for f in list_repo_files("HuggingFaceTB/cosmopedia-100k", repo_type="dataset")
                       if f.endswith(".parquet"))
    except Exception as e:
        print(f"  cosmopedia: listing failed ({e}); skipping", flush=True)
        return
    got = 0
    for f in files:
        for attempt in range(4):
            try:
                path = hf_hub_download("HuggingFaceTB/cosmopedia-100k", f, repo_type="dataset")
                for t in pq.read_table(path, columns=["text"]).column("text").to_pylist():
                    if t:
                        yield t
                        got += 1
                break
            except Exception:
                print(f"  cosmopedia {f}: retry {attempt+1}", flush=True)
                time.sleep(5 * (attempt + 1))
        if got >= target:
            return

def build():
    print("gathering FineWeb-Edu (cached)...", flush=True)
    edu = list(fineweb_edu_docs())
    print(f"  edu: {len(edu):,} docs", flush=True)

    print("gathering Cosmopedia (best-effort download)...", flush=True)
    cosmo = list(cosmopedia_docs())
    print(f"  cosmopedia: {len(cosmo):,} docs", flush=True)

    print("gathering synthetic reasoning CoT (local)...", flush=True)
    cot = [reason_gen.sample() for _ in range(COT_FRACTION_DOCS)]
    print(f"  cot: {len(cot):,} docs", flush=True)

    print("training 32K tokenizer on the mix...", flush=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    def sample_iter():
        for d in edu[::6]: yield d
        for d in cosmo[::2]: yield d
        for d in cot[::4]: yield d
    tok.train_from_iterator(sample_iter(), trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, special_tokens=["<|endoftext|>"]))
    tok.save(os.path.join(OUT, "tokenizer.json"))
    E = tok.token_to_id("<|endoftext|>")
    print(f"  tokenizer vocab {tok.get_vocab_size()}", flush=True)

    print("encoding...", flush=True)
    import random
    docs = edu + cosmo + cot
    random.Random(1337).shuffle(docs)
    chunks, total, batch = [], 0, []
    def flush():
        nonlocal total
        for enc in tok.encode_batch(batch):
            chunks.append(np.array(enc.ids + [E], dtype=np.uint16)); total += len(chunks[-1])
        batch.clear()
    for d in docs:
        batch.append(d)
        if len(batch) >= 2000:
            flush()
            if total % 40_000_000 < 200_000:
                print(f"    {total/1e6:.0f}M tokens", flush=True)
    if batch:
        flush()
    arr = np.concatenate(chunks)
    arr[:VAL_TOKENS].tofile(os.path.join(OUT, "val.bin"))
    arr[VAL_TOKENS:].tofile(os.path.join(OUT, "train.bin"))
    print(f"mix: wrote {len(arr)-VAL_TOKENS:,} train / {VAL_TOKENS:,} val tokens "
          f"(edu {len(edu):,} + cosmo {len(cosmo):,} + cot {len(cot):,} docs)", flush=True)

if __name__ == "__main__":
    build()
