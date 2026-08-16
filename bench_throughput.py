"""Clean steady-state throughput benchmark: warmup then time with cuda.sync.
Compares dense vs MoE (batched dispatch) forward+backward, matched shapes."""
import time
import torch
from train import GPT, Config

def bench(n_experts, steps=25, warmup=8, B=32, T=512):
    torch.manual_seed(0)
    dev = "cuda"
    cfg = Config(vocab_size=8192, n_layer=8, n_head=10, n_embd=640, block_size=T,
                 n_experts=n_experts, top_k=2)
    m = GPT(cfg).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=1e-4)
    x = torch.randint(0, 8192, (B, T), device=dev)
    y = torch.randint(0, 8192, (B, T), device=dev)
    def one():
        with torch.autocast(dev, dtype=torch.bfloat16):
            _, loss = m(x, y)
            if n_experts > 0:
                loss = loss + 0.01 * m.aux_loss
        loss.backward(); opt.step(); m.zero_grad(set_to_none=True)
    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    dt = time.time() - t0
    tok_s = B * T * steps / dt
    total = sum(p.numel() for p in m.parameters()) / 1e6
    del m, opt
    torch.cuda.empty_cache()
    return tok_s, total

if __name__ == "__main__":
    for e in (0, 4, 8):
        tok_s, total = bench(e)
        tag = "dense" if e == 0 else f"MoE-{e}"
        print(f"{tag:8} ({total:5.1f}M params): {tok_s:8.0f} tok/s", flush=True)
