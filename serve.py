"""
Dashboard server: serves the static dashboard files AND runs the model behind
a POST /chat endpoint, so you can talk to the bot from the dashboard page.

Replaces `python -m http.server`. Run:  python serve.py [--port 8731] [--model sft]
"""
import argparse
import json
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train import GPT, Config

HERE = os.path.dirname(os.path.abspath(__file__))

class Bot:
    def __init__(self, model_dir: str, base: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base = base   # base LM: continue the prompt (no chat role markers)
        self.tok = Tokenizer.from_file(os.path.join(HERE, model_dir, "tokenizer.json"))
        self.E = self.tok.token_to_id("<|endoftext|>")
        self.U_ids = self.tok.encode("<|user|>\n").ids
        self.A_ids = self.tok.encode("<|assistant|>\n").ids
        # infer the architecture from the checkpoint so ANY model size loads
        sd = torch.load(os.path.join(HERE, model_dir, "model.pt"), map_location=self.device)
        n_embd = sd["wte.weight"].shape[1]
        n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        self.cfg = Config(vocab_size=self.tok.get_vocab_size(), n_layer=n_layer,
                          n_head=n_embd // 64, n_embd=n_embd, block_size=1024, tie_embeddings=True)
        self.model = GPT(self.cfg).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()
        self.lock = threading.Lock()
        print(f"bot ready ({model_dir}: {n_layer}L x {n_embd}d, {'base' if base else 'chat'}) on {self.device}", flush=True)

    @torch.no_grad()
    def reply(self, messages, temp: float = 0.3, max_new: int = 160,
              top_k: int = 40, rep_penalty: float = 1.3) -> str:
        """messages: list of {"role": "user"|"assistant", "content": str}, in order,
        ending with the latest user turn. Builds the multi-turn context and
        generates the next assistant turn. Low temp + top-k + repetition penalty
        keep this small model out of the degenerate loops it otherwise falls into."""
        with self.lock:
            ids = []
            if self.base:
                # base LM: just continue the latest user text (no chat markers)
                last = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
                ids = self.tok.encode(last).ids
            else:
                for m in messages:
                    marker = self.U_ids if m.get("role") == "user" else self.A_ids
                    ids.extend(marker)
                    ids.extend(self.tok.encode(m.get("content", "")).ids)
                ids.extend(self.A_ids)  # cue the assistant turn
            # keep the most recent context within the block size
            ids = ids[-(self.cfg.block_size - max_new):]
            idx = torch.tensor([ids], device=self.device)
            start = idx.size(1)
            for _ in range(max_new):
                with torch.autocast(self.device, dtype=torch.bfloat16):
                    logits, _ = self.model(idx[:, -self.cfg.block_size:])
                logits = logits[:, -1].float()
                gen = idx[0, start:]
                if rep_penalty != 1.0 and gen.numel() > 0:
                    logits[0, torch.unique(gen)] /= rep_penalty
                logits = logits / max(temp, 1e-5)
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                idx = torch.cat([idx, nxt], 1)
                if nxt.item() == self.E:
                    break
            return self.tok.decode(idx[0, start:].tolist()).strip()

class RagBot:
    """Retriever + reader. Each question is an independent lookup, so we use
    the latest user message as the query (no conversational history)."""
    def __init__(self, model_dir, data_dir="raft", k=3):
        from rag import RAG
        self.rag = RAG(model_dir=model_dir, data_dir=data_dir, k=k)
        self.lock = threading.Lock()

    def reply(self, messages, temp=0.3, max_new=64):
        query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                query = m.get("content", "")
                break
        with self.lock:
            ans, docs = self.rag.answer(query, temp=temp, max_new=max_new, return_docs=True)
        return ans, docs

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, bot=None, **kw):
        self.bot = bot
        super().__init__(*a, directory=HERE, **kw)

    def do_POST(self):
        if self.path != "/chat":
            self.send_error(404)
            return
        if self.bot is None:
            self.send_error(503, "model not loaded yet (still training)")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or "{}")
            # accept either full history (messages) or a single prompt
            messages = req.get("messages")
            if not messages:
                messages = [{"role": "user", "content": req.get("prompt", "")}]
            out = self.bot.reply(messages,
                                 temp=float(req.get("temp", 0.7)),
                                 max_new=int(req.get("max_new", 160)))
            # RAG engine returns (reply, retrieved_docs); plain bot returns str
            reply, docs = out if isinstance(out, tuple) else (out, None)
            body = json.dumps({"reply": reply, "docs": docs}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:  # noqa
            self.send_error(500, str(e))

    def log_message(self, *a):  # quiet
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--model", default="sft16")
    ap.add_argument("--rag", action="store_true", help="retrieve passages before reading (RAFT reader)")
    ap.add_argument("--base", action="store_true", help="base LM: continue the prompt (no chat markers)")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    try:
        bot = RagBot(args.model, k=args.k) if args.rag else Bot(args.model, base=args.base)
    except Exception as e:  # model not trained yet: still serve the live dashboard
        bot = None
        print(f"no model loaded ({e!r}); serving dashboard only, /chat disabled", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, bot=bot))
    print(f"serving dashboard{' + /chat' if bot else ' (static only)'} on http://localhost:{args.port}/dashboard.html", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()
