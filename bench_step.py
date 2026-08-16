"""Break down where the ~1s per training step goes: data load, forward,
backward, Muon step, Adam step. cuda.synchronize() around each phase so GPU
async work is attributed correctly. Also isolates Muon's Newton-Schulz cost."""
import collections
import time

import torch

from train import GPT, Config, Muon, get_batch, zeropower_via_newtonschulz5, HERE
import os

dev = "cuda"
data_dir = os.path.join(HERE, "ab_raw")
cfg = Config(vocab_size=8192, n_layer=8, n_head=10, n_embd=640, block_size=512)
model = GPT(cfg).to(dev)
hidden = [p for p in model.blocks.parameters() if p.ndim == 2]
embed_head = [model.wte.weight, model.lm_head.weight]
opt_muon = Muon(hidden, lr=0.02, momentum=0.95)
opt_adam = torch.optim.Adam(embed_head, lr=0.003, betas=(0.9, 0.95))
sync = torch.cuda.synchronize

def step_once():
    x, y = get_batch(data_dir, "train", 32, 512, dev)
    with torch.autocast(dev, dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward(); opt_muon.step(); opt_adam.step()
    model.zero_grad(set_to_none=True)

for _ in range(10):      # warmup
    step_once()
sync()

T = collections.defaultdict(float); N = 40
for _ in range(N):
    sync(); t0 = time.time()
    x, y = get_batch(data_dir, "train", 32, 512, dev)
    sync(); t1 = time.time()
    with torch.autocast(dev, dtype=torch.bfloat16):
        _, loss = model(x, y)
    sync(); t2 = time.time()
    loss.backward()
    sync(); t3 = time.time()
    opt_muon.step()
    sync(); t4 = time.time()
    opt_adam.step()
    sync(); t5 = time.time()
    model.zero_grad(set_to_none=True)
    sync(); t6 = time.time()
    T["data load"] += t1 - t0
    T["forward"] += t2 - t1
    T["backward"] += t3 - t2
    T["Muon step"] += t4 - t3
    T["Adam step"] += t5 - t4
    T["zero_grad"] += t6 - t5

total = sum(T.values())
print(f"{'phase':12} {'ms/step':>9} {'%':>7}")
for k in ["data load", "forward", "backward", "Muon step", "Adam step", "zero_grad"]:
    print(f"{k:12} {1000*T[k]/N:9.1f} {100*T[k]/total:6.1f}", flush=True)
print(f"{'TOTAL':12} {1000*total/N:9.1f} {100.0:6.1f}")

# isolate Newton-Schulz (the orthogonalization inside Muon)
bufs = [torch.randn_like(p) for p in hidden]
sync(); t0 = time.time()
for _ in range(N):
    for b in bufs:
        zeropower_via_newtonschulz5(b)
sync()
ns = 1000 * (time.time() - t0) / N
print(f"\nNewton-Schulz alone: {ns:.1f} ms/step  (~{100*ns/(1000*total/N):.0f}% of the step; it's the bulk of Muon)")
print(f"experts/params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, {len(hidden)} Muon matrices")
