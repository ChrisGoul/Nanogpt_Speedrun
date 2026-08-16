"""Throughput sweep: torch.compile on/off x batch size, dense 50M.
Measures steady-state forward+backward tok/s (warmup + cuda.sync).
Baseline for comparison: batch 32, no compile ~ 19,778 tok/s (bench_throughput.py)."""
import time
import torch
from train import GPT, Config

def bench(batch, compile_on, steps=20, T=512):
    dev = "cuda"
    cfg = Config(vocab_size=8192, n_layer=8, n_head=10, n_embd=640, block_size=T)
    m = GPT(cfg).to(dev)
    if compile_on:
        m = torch.compile(m)
    opt = torch.optim.SGD(m.parameters(), lr=1e-4)
    x = torch.randint(0, 8192, (batch, T), device=dev)
    y = torch.randint(0, 8192, (batch, T), device=dev)
    def one():
        with torch.autocast(dev, dtype=torch.bfloat16):
            _, loss = m(x, y)
        loss.backward(); opt.step(); m.zero_grad(set_to_none=True)
    for _ in range(15 if compile_on else 8):   # extra warmup for compile
        one()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    tok_s = batch * T * steps / (time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 1e9
    del m, opt
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    return tok_s, peak

if __name__ == "__main__":
    print(f"torch {torch.__version__}", flush=True)
    for compile_on in (False, True):
        for batch in (32, 64, 96, 128):
            try:
                tok_s, peak = bench(batch, compile_on)
                print(f"compile={str(compile_on):5} batch={batch:3}: {tok_s:8.0f} tok/s  (peak {peak:.1f} GB)", flush=True)
            except Exception as e:
                print(f"compile={str(compile_on):5} batch={batch:3}: FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
                torch.cuda.empty_cache()
