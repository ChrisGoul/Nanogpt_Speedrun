"""
Build the SFT (instruction-tuning) dataset for the blend model, mixing the
same two worlds it was pretrained on:

  1. TinyStories-Instruct: instruction -> story pairs (teaches the model to
     follow a request and produce fluent text).
  2. Wiki Q/A: "What is <title>?" -> first sentences of the article, generated
     from the SAME filtered Simple-Wiki slice used in pretraining (teaches it
     to answer factual questions about things it actually saw).

Each example is formatted with the reserved role tokens:
    <|user|> {prompt} <|assistant|> {response} <|endoftext|>
and written as fixed-length rows so the loss can be masked to the response.

Outputs to sft/:
  tokens.bin   uint16, shape (N * SEQ_LEN,)
  loss_mask.bin uint8, 1 where the token is part of the response (+eot), else 0
Reuses blend/tokenizer.json (with <|user|>/<|assistant|>/<|endoftext|>).
"""
import os
import re
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sft")
os.makedirs(OUT, exist_ok=True)

SEQ_LEN = 512
N_INSTRUCT = 120_000     # TinyStories-Instruct examples
N_WIKI_QA = 80_000       # wiki Q/A examples
COMMON_TOP_K = 5_000     # same filter as the pretraining blend
WIKI_KEEP = 0.30

word_re = re.compile(r"[a-z']+")
sent_re = re.compile(r"(?<=[.!?])\s+")

QA_TEMPLATES = [
    "What is {t}?", "Tell me about {t}.", "What is {t}?",
    "Can you explain what {t} is?", "Who or what is {t}?",
]

def load_instruct(n: int) -> list[tuple[str, str]]:
    # TinyStories-Instruct is a plain-text file: records separated by
    # <|endoftext|>, each with "Summary:/Features:/Words:" instruction lines
    # then "Story:" and the story. We use the valid split (~20k records) - the
    # train split's download stalls repeatedly on this connection, and 20k
    # instruction examples is plenty alongside the wiki Q/A at this scale.
    path = hf_hub_download("roneneldan/TinyStories-Instruct",
                           "TinyStories-Instruct-valid.txt", repo_type="dataset")
    pairs, buf = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "<|endoftext|>":
                rec = "".join(buf)
                buf = []
                if "Story:" in rec:
                    prompt, story = rec.split("Story:", 1)
                    prompt, story = prompt.strip(), story.strip()
                    if prompt and story:
                        pairs.append((prompt, story))
                        if len(pairs) >= n:
                            break
            else:
                buf.append(line)
    return pairs

def load_wiki_qa(n: int) -> list[tuple[str, str]]:
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    titles = [d["title"] for d in ds]
    texts = [d["text"] for d in ds]
    # reproduce the pretraining filter so Q/A covers the same articles
    doc_words = [word_re.findall(t.lower()) for t in texts]
    freq = Counter()
    for ws in doc_words:
        freq.update(ws)
    common = set(w for w, _ in freq.most_common(COMMON_TOP_K))
    scores = np.array([sum(w not in common for w in ws) / max(len(ws), 1)
                       for ws in doc_words])
    cutoff = np.quantile(scores, WIKI_KEEP)

    rng = np.random.default_rng(0)
    pairs = []
    for i in range(len(texts)):
        if scores[i] > cutoff:
            continue
        title, body = titles[i], texts[i].strip()
        # answer: first 1-2 sentences, skipping ones that are too short/stubby
        sents = [s.strip() for s in sent_re.split(body) if len(s.strip()) > 20]
        if not sents:
            continue
        answer = " ".join(sents[:2])[:600]
        q = rng.choice(QA_TEMPLATES).format(t=title)
        pairs.append((q, answer))
        if len(pairs) >= n:
            break
    return pairs

def main():
    tok = Tokenizer.from_file(os.path.join(HERE, "blend", "tokenizer.json"))
    U = tok.token_to_id("<|user|>")
    A = tok.token_to_id("<|assistant|>")
    E = tok.token_to_id("<|endoftext|>")
    assert None not in (U, A, E), "role tokens missing from tokenizer"

    print("loading TinyStories-Instruct...", flush=True)
    pairs = load_instruct(N_INSTRUCT)
    print(f"instruct: {len(pairs):,} pairs", flush=True)
    print("building wiki Q/A...", flush=True)
    pairs += load_wiki_qa(N_WIKI_QA)
    print(f"total: {len(pairs):,} pairs", flush=True)

    rng = np.random.default_rng(1337)
    rng.shuffle(pairs)

    tokens = np.zeros((len(pairs), SEQ_LEN), dtype=np.uint16)
    mask = np.zeros((len(pairs), SEQ_LEN), dtype=np.uint8)
    kept = 0
    for prompt, resp in pairs:
        p_ids = tok.encode(prompt).ids
        r_ids = tok.encode(resp).ids
        seq = [U] + p_ids + [A] + r_ids + [E]
        m = [0] * (1 + len(p_ids) + 1) + [1] * (len(r_ids) + 1)
        seq, m = seq[:SEQ_LEN], m[:SEQ_LEN]
        if sum(m) < 2:            # response got truncated away; skip
            continue
        L = len(seq)
        tokens[kept, :L] = seq
        tokens[kept, L:] = E      # pad with eot
        mask[kept, :L] = m
        kept += 1
    tokens, mask = tokens[:kept], mask[:kept]

    tokens.tofile(os.path.join(OUT, "tokens.bin"))
    mask.tofile(os.path.join(OUT, "loss_mask.bin"))
    frac = mask.sum() / mask.size
    print(f"wrote {kept:,} examples x {SEQ_LEN} tokens "
          f"({frac:.1%} are response tokens under loss)", flush=True)

if __name__ == "__main__":
    main()
