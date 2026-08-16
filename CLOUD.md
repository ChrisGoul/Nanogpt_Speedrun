# Moving to the Cloud — Training 500M–1B Models

You've hit the ceiling of the 3050: both **params** (8 GB caps you ~168M) and
**tokens** (days per run). To go bigger you rent a GPU by the hour. The good news
from the cost math: these runs are **tens of dollars, not thousands.**

---

## 1. Recommendation

**Primary: [RunPod](https://runpod.io) — a single H100 80GB (SXM).**

Why RunPod for this project:
- **Per-second billing** + start/stop — you pay only while the GPU runs.
- **Persistent network volumes** — store your data/tokenizer once, reattach to
  any pod, so you don't re-download the mix every time.
- **Wide selection** (H100 / A100 / 4090) and a cheap **Community Cloud** tier.
- It's **Linux**, which unlocks the two things Windows blocked: **`torch.compile`
  (Triton)** and full **FlashAttention** — often +30–50% throughput for free.
- SSH + Jupyter + exposed HTTP ports (so your `serve.py` dashboard works remotely).

**When to use the others:**
- **[Vast.ai](https://vast.ai)** — cheapest floor (H100 ~$1.33–1.87/hr on the
  marketplace). Reliability varies by host, but **your trainer is now checkpointed
  + resumable**, so a preempted spot instance costs minutes, not the run. Use Vast
  **spot** for the big 1B run to cut cost 40–60%.
- **[Lambda](https://lambdalabs.com)** — clean managed H100 (~$2.99/hr), **no
  spot** (you pay full price even if interruptible). Use when you want zero babysitting.

*Pricing is volatile — check the live number before you launch. Snapshot below is Aug 2026.*

| Provider | H100 80GB | A100 80GB | RTX 4090 | Spot? |
|----------|-----------|-----------|----------|-------|
| RunPod (Community) | ~$1.99/hr | ~$1.39/hr | ~$0.34/hr | yes |
| RunPod (Secure) | ~$2.89/hr | ~$1.49/hr | ~$0.69/hr | yes |
| Vast.ai (marketplace) | ~$1.33–1.87/hr | ~$1.0/hr | ~$0.25/hr | yes |
| Lambda | ~$2.99/hr | — | — | no |

**Don't use a 4090 for 500M–1B training** — 24 GB is tight for a 1B optimizer
state, and it's ~10× slower than an H100 here. (It's fine for the 156M model.)
**One H100 is more cost-efficient *and* faster than an A100** for these sizes, so
default to H100; drop to A100 only if H100s are sold out.

---

## 2. Cost & time (worked from the same 6ND math as before)

`FLOPs = 6 × params × tokens`; time = FLOPs ÷ (H100 peak 989 TFLOP/s × ~45% MFU
≈ 445 TFLOP/s effective). Cost at ~$2.5/hr H100.

| Model | Tokens | Total FLOPs | Wall-clock (1×H100) | Cost |
|-------|--------|-------------|---------------------|------|
| **500M** (Chinchilla-optimal, 20 tok/param) | 10B | 3.0e19 | **~19 hr** | **~$50** |
| 500M (undertrained, 10 tok/param) | 5B | 1.5e19 | ~9 hr | ~$25 |
| **1B** (Chinchilla-optimal) | 20B | 1.2e20 | **~75 hr (~3 days)** | **~$190** |
| 1B (undertrained, 10 tok/param) | 10B | 6.0e19 | ~37 hr | ~$95 |

Notes:
- **Cost is ~invariant to GPU count** — 4×H100 finishes the 1B run in ~19 hr but
  costs about the same total, because you pay per GPU-hour. So multi-GPU buys
  *speed*, not savings. For a first run, **one H100 is simplest.**
- **Memory is not the constraint** on an 80 GB H100 — a 1B model + optimizer +
  activations fits comfortably. Single-GPU works for both sizes.
- These assume the Linux speedups are on (`--compile`, FlashAttention). Without
  them, add ~30–50%.

---

## 3. What changes when you leave Windows/3050

Wins to turn on (they were unavailable locally):
- **`--compile`** — `torch.compile` works on Linux (Triton). Free throughput.
- **FlashAttention** — already used via `scaled_dot_product_attention`; on
  Ampere/Hopper + Linux it hits the fast kernels automatically.
- **Bigger batch / longer context** — 80 GB lets you drop gradient checkpointing
  (the ~30% recompute tax) and/or train at seq 1024–2048 instead of 256.
- **Full-rate bf16** — datacenter GPUs don't have the consumer FP32-accumulate
  halving, so real MFU climbs.

One real gap: **`train.py` is single-GPU.** For multi-GPU (the only reason you'd
want it: finishing 1B faster) it needs DDP (`torchrun` + `DistributedDataParallel`
+ a distributed sampler). Not required for a first run — a single H100 trains 1B
fine, just slower. Ask me and I'll add a DDP path.

---

## 4. Migration workflow (step by step)

> You create the account and add payment yourself — I can't do that for you.
> Never paste API keys or card details into code or config committed to the repo.

**One-time setup**
1. Sign up at RunPod, add credit.
2. Create a **Network Volume** (e.g. 100 GB) in a region that has H100s — this
   holds your datasets and checkpoints across pods.
3. Put the code on GitHub (private) so you can `git clone` onto any pod. Or plan
   to `rsync`/`scp` it up.

**Per-run**
4. **Deploy a Pod**: H100 80GB, a PyTorch template (CUDA 12.x, PyTorch ≥2.4),
   attach your network volume at e.g. `/workspace/data`.
5. **SSH in**, then:
   ```bash
   git clone <your-repo> && cd nanogpt_speedrun
   pip install torch numpy tokenizers tiktoken huggingface_hub datasets pyarrow
   ```
6. **Get the data onto the volume** (once): either rebuild it there —
   `MIX_VOCAB=16000 MIX_OUT=/workspace/data/mix500 python prepare_mix.py` — or
   `rsync` your local `mix16/` up. Building on the pod is usually faster (fat pipe
   to HuggingFace) and avoids uploading ~1 GB.
7. **Launch**, checkpointed to the persistent volume:
   ```bash
   python train.py --data /workspace/data/mix16 --run big500 \
     --out /workspace/data/big500 --compile --tie --bench \
     --dim 1280 --layers 24 --heads 20 --seq 1024 --batch-size 32 \
     --steps <see below> --ckpt-every 1000 --resume \
     --peak-tflops 989
   ```
8. **Monitor**: expose port 8731 on the pod and run `serve.py`, or just
   `tail -f` the log / scp `metrics_big500.jsonl` down and open it in your local
   dashboard.
9. **When done**: the model is already on the network volume. `scp` `model.pt`
   down, then **stop/terminate the pod** so the meter stops. The volume persists.

---

## 4b. First cloud session — copy-paste smoke test (~$1)

Do this **before** any long run. It rents one H100, runs the *real* target
architecture for 300 steps (so throughput reflects the actual config), then
extrapolates the full run's time + cost. Total: a few minutes, ~$1.

```bash
# --- on a fresh RunPod H100 pod (PyTorch template) ---
git clone <your-repo> && cd nanogpt_speedrun
pip install -q torch numpy tokenizers tiktoken huggingface_hub datasets pyarrow

# 1. Build a small data slice on the pod (fast pipe to HuggingFace).
#    Reuse the 16K mix recipe; a few shards is plenty for a throughput test.
MIX_VOCAB=16000 MIX_OUT=mix16 python prepare_mix.py

# 2. Smoke-run the REAL 500M config for 300 steps, compiled, with MFU tracking.
python train.py --data mix16 --run smoke500 --out abl/smoke500 \
  --compile --tie --peak-tflops 989 \
  --dim 1280 --layers 24 --heads 20 --seq 1024 --batch-size 32 \
  --steps 300 --eval-every 300 --ckpt-every 0

# 3. Extrapolate the full Chinchilla-optimal 500M run (10B tokens).
python estimate_cost.py --run smoke500 --tokens 10e9 --price 2.5
```

`estimate_cost.py` prints steady-state tok/s, MFU, peak VRAM, and the projected
hours + dollars. Sanity-check MFU is **40–55%** (if it's ~20%, something's wrong
— likely `--compile` failed or the batch is too small). Then decide: if the
number matches the CLOUD.md table, relaunch **without** `--steps 300`, at the
full step count (§5), **with `--ckpt-every 1000 --resume`**, and let it run.

For the 1B config, swap in `--dim 2048 --layers 24 --heads 16` and
`--tokens 20e9`. **Stop the pod the moment the smoke test ends** if you're not
proceeding immediately — the meter runs regardless.

---

## 4c. Never pay for an idle pod (`runpod_train.sh`)

**A deployed pod bills per-second at the full rate whether or not the GPU is
working** — an idle H100 you forgot about is ~$48/day for nothing. This is the #1
way to waste money. Three layers of protection:

1. **Auto-stop wrapper** — `runpod_train.sh` runs your training command and stops
   the pod on exit (finish, crash, *or* Ctrl-C), so walking away can't cost you:
   ```bash
   chmod +x runpod_train.sh
   ./runpod_train.sh python train.py --data /workspace/data/mix16 --run big500 \
     --out /workspace/data/big500 --compile --tie --bench --resume \
     --dim 1280 --layers 24 --heads 20 --seq 1024 --batch-size 32 \
     --steps 305000 --ckpt-every 1000 --peak-tflops 989
   ```
   It uses the pre-installed `runpodctl` (or the API via `RUNPOD_API_KEY`) to stop
   pod `$RUNPOD_POD_ID`. Options: `DEADLINE_HOURS=30` adds a hard backstop that
   stops the pod even if training hangs; `AUTOSTOP=terminate` also frees the pod's
   disk (safe — your data/checkpoints are on the network volume). If it can't
   self-stop it prints a loud warning so you know to stop it by hand.

2. **Account spending limit** — set one in the RunPod dashboard (Settings →
   Billing). A hard ceiling that caps total spend no matter what.

3. **Stopped ≠ terminated** — *stopping* ends GPU billing but still charges a few
   cents for the pod's disk; *terminate* removes even that. Since everything
   persists on the network volume, terminating a finished training pod is safe and
   cleanest. The volume itself bills separately (~$0.05–0.10/GB/month — pennies).

> Because training is checkpointed (`--resume`), auto-stop is free insurance: if
> the deadline or a crash stops the pod mid-run, you just redeploy and resume from
> the last `ckpt.pt` on the volume.

---

## 4d. First cloud run — 300M, direct comparison to the 156M (copy-paste)

Cheapest meaningful first run: a **300M** model on the **exact same `mix16` data**
as the 156M, so the only variable is parameters (a clean "does 2× params help?"
test). **~1 hour, ~$2.** Reuses existing data — no rebuild.

```bash
# ============================================================
#  Pod: 1x H100 PCIe (Community ~$1.99/hr), PyTorch template,
#       network volume mounted at /workspace/data.
# ============================================================

# --- FROM YOUR LAPTOP (once): upload the EXACT mix16 the 156M trained on ---
#   so the comparison is truly apples-to-apples (~780 MB).
#   Get the pod's ssh host/port from the RunPod dashboard.
rsync -avP -e "ssh -p <POD_PORT>" mix16/ root@<POD_IP>:/workspace/data/mix16/

# --- ON THE POD ---
cd /workspace && git clone <YOUR-REPO> nanogpt && cd nanogpt
pip install -q torch numpy tokenizers tiktoken huggingface_hub datasets pyarrow
chmod +x runpod_train.sh

# Train 300M, auto-stopping the pod on exit. SAME seq/batch/steps as the 156M
# (seq 256, batch 32, ~99k steps ≈ 0.8B tokens) so ONLY params differ.
DEADLINE_HOURS=3 ./runpod_train.sh python train.py \
  --data /workspace/data/mix16 --run big300 --out /workspace/data/big300 \
  --compile --tie --bench --peak-tflops 756 \
  --dim 1024 --layers 22 --heads 16 --seq 256 --batch-size 32 \
  --steps 99000 --eval-every 500 --bench-every 3000 --ckpt-every 2000 --resume
```

- **Result** lands on the persistent volume: `/workspace/data/big300/model.pt` and
  `metrics_big300.jsonl`. Compare its PIQA/HellaSwag to the 156M's **~59 / ~40**.
- **Retrieve the model**: `scp` it down *before* the pod auto-stops, or restart the
  stopped pod briefly (the volume persists either way).
- **This run is cheap enough to skip the smoke test** — just launch and watch the
  dashboard. Reserve the §4b smoke test for the expensive optimal/1B runs.

**To go bigger afterwards** (Chinchilla-optimal 300M ≈ 6B tokens, ~$17): rebuild a
larger mix ON the pod with the download path now supported —
`MIX_OFFLINE=0 MIX_EDU_SHARDS=45 MIX_VOCAB=16000 MIX_OUT=/workspace/data/mix300 python prepare_mix.py` —
then relaunch with `--data /workspace/data/mix300 --seq 1024 --steps <6e9/(batch*seq)>`.

---

## 5. Sizing a 500M / 1B config

Rough param budget (use `param_config.py` to tune): `params ≈ 12·d²·L + vocab·d`.

| Target | dim | layers | heads | ~params |
|--------|-----|--------|-------|---------|
| ~300M | 1024 | 22 | 16 | ~293M |
| ~500M | 1280 | 24 | 20 | ~510M |
| ~1B | 2048 | 24 | 16 | ~1.2B |

Then set `--steps = tokens / (batch × seq)`. E.g. 500M optimal = 10B tokens at
batch 32 × seq 1024 = 8192 tok/step → ~305,000 steps. (Rebuild the mix bigger
than 0.8B tokens first, or repeat epochs — but prefer fresh tokens at this scale.)

---

## 6. Cost-control checklist

- **Always `--ckpt-every` + `--resume`** → spot/preemption is safe. (Already wired.)
- **Use spot** (Vast, or RunPod interruptible) for the long 1B run.
- **Stop the pod the moment the run ends** — idle GPU time is the #1 way to waste money.
- **Keep data on the network volume** so you never re-download/re-tokenize.
- **Estimate first**: plug params×tokens into the table above before launching, so
  there are no surprises on the meter.
- **Start small**: do a 500-step smoke run to confirm throughput + MFU, extrapolate
  the full cost, *then* commit to the long run.

---

### Sources (pricing, Aug 2026)
- [RunPod pricing breakdown — Northflank](https://northflank.com/blog/runpod-gpu-pricing)
- [Cloud GPU pricing 2026 — SynpixCloud](https://www.synpixcloud.com/blog/cloud-gpu-pricing-comparison-2026)
- [H100 rental prices across 15+ providers — IntuitionLabs](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [RunPod vs Lambda vs Vast.ai — Klymentiev](https://klymentiev.com/blog/runpod-vs-lambda-vs-vast)
