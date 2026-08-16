"""
RAG inference: BM25 retriever over the passage index + the RAFT-trained reader.

A question -> retrieve top-k passages -> format exactly like training
("Documents: [1]... Question: ...") -> reader answers or abstains.

Interactive:  python rag.py
One-shot:     python rag.py --ask "Who wrote Hamlet?"
Options:      --k 3  --model raftmodel  --show-docs
"""
import argparse
import json
import os
import re

import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer

from train import GPT, Config

HERE = os.path.dirname(os.path.abspath(__file__))
DOC_CHARS = 420

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

class RAG:
    def __init__(self, model_dir="raftmodel", data_dir="raft", k=3):
        self.k = k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.docs = json.load(open(os.path.join(HERE, data_dir, "index_docs.json")))
        print(f"indexing {len(self.docs):,} passages (BM25)...", flush=True)
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs])

        self.tok = Tokenizer.from_file(os.path.join(HERE, data_dir, "tokenizer.json"))
        self.U, self.A, self.E = (self.tok.token_to_id(x) for x in ("<|user|>", "<|assistant|>", "<|endoftext|>"))
        self.cfg = Config(vocab_size=self.tok.get_vocab_size(), n_layer=8, n_head=8, n_embd=512)
        self.model = GPT(self.cfg).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(HERE, model_dir, "model.pt")))
        self.model.eval()
        print("reader ready", flush=True)

    def retrieve(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]
        return [self.docs[i][:DOC_CHARS].strip() for i in idx]

    @torch.no_grad()
    def answer(self, query, temp=0.3, max_new=64, return_docs=False):
        docs = self.retrieve(query)
        prompt = "Documents:\n" + "\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs)) \
                 + f"\n\nQuestion: {query.strip()}"
        ids = [self.U] + self.tok.encode(prompt).ids + [self.A]
        ids = ids[-(self.cfg.block_size - max_new):]
        idx = torch.tensor([ids], device=self.device)
        start = idx.size(1)
        for _ in range(max_new):
            with torch.autocast(self.device, dtype=torch.bfloat16):
                logits, _ = self.model(idx[:, -self.cfg.block_size:])
            nxt = torch.multinomial(F.softmax(logits[:, -1] / max(temp, 1e-5), dim=-1), 1)
            idx = torch.cat([idx, nxt], 1)
            if nxt.item() == self.E:
                break
        ans = self.tok.decode(idx[0, start:].tolist()).strip()
        return (ans, docs) if return_docs else ans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--model", default="raftmodel")
    ap.add_argument("--show-docs", action="store_true")
    args = ap.parse_args()
    rag = RAG(model_dir=args.model, k=args.k)

    def go(q):
        ans, docs = rag.answer(q, return_docs=True)
        if args.show_docs:
            for i, d in enumerate(docs):
                print(f"  [{i+1}] {d[:160]}...")
        print("A:", ans)

    if args.ask:
        go(args.ask)
        return
    print("RAG bot — ask a question, Ctrl-C to quit\n")
    try:
        while True:
            q = input("Q: ").strip()
            if q:
                go(q)
    except (KeyboardInterrupt, EOFError):
        print("\nbye")

if __name__ == "__main__":
    main()
