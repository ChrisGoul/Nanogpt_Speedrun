# Architecture & Code Walkthrough

A map of this repo: what every file does, how the pieces connect, and the
conventions that hold it together. Written for the point where the project has
grown past "one train.py" into a small research harness.

The whole thing is **single-GPU, from-scratch pretraining + SFT** built on a
scaled-down [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
speedrun core, tuned to run on an 8 GB consumer card (RTX 3050) on Windows.

---

## 1. The 30-second mental model

```
prepare_*.py  ─►  <dataset>/{train.bin, val.bin, tokenizer.json}
                          │
                          ▼
      train.py  ──►  <run>/model.pt   + metrics.jsonl  + metrics_<run>.jsonl
        │  (GPT + Muon, pretraining)         │
        │                                    ▼
        │                            dashboard.html  ◄── served by serve.py
        ▼
   train_sft16.py  ──►  sft16/model.pt   (fine-tune on masked chat data)
        │
        ▼
   chat_sft16.py / serve.py /chat  ──►  talk to the model
```

Everything communicates through **three file conventions** (see §8):
1. Tokenized data as flat `uint16` `.bin` files + a `tokenizer.json`.
2. Model weights as `model.pt` (a plain `state_dict`).
3. Metrics as newline-delimited JSON (`metrics.jsonl`) that the dashboard polls.

---

## 2. Core files (start here)

| File | Role |
|------|------|
| **train.py** | The heart: model definition, Muon optimizer, data loader, training loop, live benchmarking, checkpoint/resume, compute-metrics logging. |
| **serve.py** | HTTP server that hosts the dashboard **and** runs the model behind `POST /chat`. Replaces `python -m http.server`. |
| **dashboard.html** | Zero-dependency live dashboard: loss/LR/grad-norm/bench charts, tiles (MFU, VRAM, throughput), experiments log, and a chat card. Polls `metrics.jsonl` every 2 s. |
| **experiments.json** | The experiment log rendered as a table in the dashboard. Each entry: `{name, setup, mask, result, takeaway}`. |
| **ablate.py** | Sweep runner: launches a grid of `train.py` runs and prints a comparison table. |
| **param_config.py** | Pure-math parameter-budget calculator (embedding vs attn vs MLP split) — no training. |

---

## 3. The model — `train.py`

All in one file (lines are approximate). Architecture follows the speedrun
recipe; every choice is a small win that compounds.

### `Config` (dataclass)
Every knob: `vocab_size, n_layer, n_head, n_embd, block_size`, plus feature
flags `n_experts/top_k/moe_ff` (MoE), `use_checkpoint` (gradient checkpointing),
`use_abacus/abacus_size` (place-value digit embeddings for arithmetic length
generalization), `tie_embeddings`.

### `GPT` — the network
- **Embeddings**: `wte` (token) → optional abacus digit-position add → RMSNorm.
- **Blocks** (`Block`): pre-norm `x = x + attn(norm(x))` then `x = x + mlp(norm(x))`.
  - **`CausalSelfAttention`**: fused QKV, **RoPE** (`Rotary`, non-persistent
    cos/sin buffers so seq length can change between train/SFT), **QK-norm**
    (RMS-norm q and k), `F.scaled_dot_product_attention` (FlashAttention when
    available). Output proj **zero-initialized** so each block starts as identity.
    *Full multi-head attention — no GQA.*
  - **`MLP`**: `proj(relu(fc(x))²)` — **ReLU²**, 4× expansion, zero-init proj.
  - **`MoE`** (only if `n_experts>0`): top-k router + per-expert ReLU² MLPs sized
    to match dense active FLOPs; emits a load-balance aux loss. A large-scale
    technique — measured *slower* at this scale (see experiments log).
- **Head**: `lm_head`. Tied to `wte` when `tie_embeddings` else zero-init untied.
- **Logit soft-capping**: `30*tanh(logits/30)` before the loss.
- **`forward(idx, targets)`** returns `(logits, loss)`; cross-entropy over all
  positions when `targets` given. Gradient checkpointing wraps each block in the
  backward pass when `use_checkpoint` and training.
- **`_abacus_pos`**: computes each digit's place value (ones=0, tens=1, …) via a
  reverse-scan; used only when `use_abacus`.

### Optimizer — `Muon` + Adam (the signature trick)
- **`Muon`** orthogonalizes the momentum update via a 5-step Newton-Schulz
  iteration (`zeropower_via_newtonschulz5`) before applying it. Rebalances the
  singular values of a weight update so all directions contribute equally.
- **Split** (set up in `main`): Muon drives the **2D hidden matrices**
  (`model.blocks` params with `ndim==2`); **Adam** drives the embedding + head
  (`wte`, `lm_head`), de-duplicated by `id()` so a tied weight isn't listed twice.

### Training loop (`main`)
1. Build `Config` from CLI args; infer `vocab_size` from a shipped `tokenizer.json`.
2. Build model (+ optional abacus `digit_ids`); optional `--resume` from `ckpt.pt`.
3. Muon+Adam optimizers; trapezoidal LR schedule (`lr_mult`: warmup → flat → linear decay over the last 40%).
4. Optional live benches (`--bench`): loads PIQA/HellaSwag subsets.
5. Loop: every `eval_every` → val loss + sample + (periodically) `bench_acc`;
   every `ckpt_every` → atomic `ckpt.pt`; each step → forward/backward, Muon+Adam
   step, and (every 5 steps) log **train_loss, lr_mult, tok/s, grad_norm, tflops,
   mfu, gpu_mem_mb**.
6. Save `model.pt` + copy tokenizer to the run dir.

### Instrumentation helpers (added for inspection)
- **`gpu_peak_tflops(device)`** — bf16 FP32-accumulate peak for common GPUs, used
  to turn achieved FLOP/s into an MFU %. (Consumer Ampere/Ada numbers are the
  half-rate FP32-accumulate values that training actually hits.)
- **`grad_global_norm(model)`** — total L2 norm of all grads (stability signal).
- **`bench_acc(...)`** — length-normalized multiple-choice accuracy (**acc_norm**,
  cloze/CF format) on the live model. This is what the dashboard bench chart plots.

---

## 4. Data pipeline — `prepare_*.py`

Each script downloads/synthesizes a corpus, (optionally) trains a byte-level BPE
tokenizer, and writes `<dataset>/{train.bin, val.bin, tokenizer.json}`. `.bin`
files are flat `uint16` token streams; `train.py`'s `get_batch` memory-maps them
and samples random `block_size` windows.

| Script | Dataset it builds |
|--------|-------------------|
| `prepare_data.py` | tiny-shakespeare (the original toy) |
| `prepare_simplewiki.py`, `prepare_blend.py` | Simple-Wiki, TinyStories+Wiki blend |
| `prepare_books.py` | BookCorpus |
| `prepare_edu.py`, `prepare_ab.py` | FineWeb-Edu; raw-vs-edu A/B sets |
| **`prepare_mix.py`** | **The current general-model mix**: FineWeb-Edu + Cosmopedia + synthetic CoT; trains a BPE tokenizer at `MIX_VOCAB` (env) into `MIX_OUT` (env). This built `mix/` (32K) and `mix16/` (16K). |
| `reason_gen.py`, `prepare_reason.py`, `prepare_addsub.py` | Synthetic arithmetic/logic chain-of-thought problems (13 generator types). |
| `prepare_raft.py` | SQuAD read-cite-or-abstain reader data. |
| `fetch_shards.py`, `fetch_edu.py` | Robust, timeout-hardened HuggingFace shard downloaders (this connection stalls a lot). |

**SFT data builders** (fixed-length rows + a loss mask, not a flat stream):
- `prepare_chat.py`, `prepare_sft.py`, `prepare_sftmix.py` — earlier `blend`-era
  chat/instruction sets (use a tokenizer with `<|user|>`/`<|assistant|>` tokens).
- **`prepare_sft16.py`** — the current SFT set for `mix16`. Reuses the proven
  loaders (DailyDialog, EmpatheticDialogues, Dolly, GSM8K×2, CommonsenseQA,
  SQuAD) but marks turns with the **text** strings `<|user|>`/`<|assistant|>`
  (the mix16 tokenizer has no dedicated role tokens). Writes
  `sft16/{tokens.bin (N×512), loss_mask.bin (N×512), tokenizer.json}` — the mask
  is 1 on assistant tokens only.

---

## 5. Post-training (SFT) — `train_sft16.py`

Fine-tunes `mix16/model.pt` into a chatbot. Same GPT/Muon core (imported from
`train.py`), three differences that define SFT:
1. **Init from pretrained weights**, not random.
2. **Masked loss**: `masked_loss` grades only assistant tokens via
   `loss_mask.bin` — the model is scored on its answers, not on echoing the prompt.
3. **Low LR, few epochs**, resumable (`--ckpt-every/--resume`), sleep-safe.

Older equivalents `train_sft.py` / `train_chat.py` target the `blend`/`chat`
tokenizers. `chat.py` and **`chat_sft16.py`** are CLI chat REPLs;
`chat_sft16.py` adds a repetition penalty + top-k + low temp to tame the loops a
156M model falls into.

---

## 6. Evaluation — `eval*.py`

| File | What it measures |
|------|------------------|
| `eval_base.py` | Base-LM benchmark loaders + scorers (PIQA, HellaSwag, ARC-E, LAMBADA), **CF/acc_norm** — likelihood-scored, the right format for base models. Also the source of the live-bench loaders `train.py` imports. |
| `eval.py` | General perplexity / sampling eval. |
| `eval_reason.py`, `eval_length.py`, `eval_gsm8k.py` | Reasoning model: in-distribution accuracy, length generalization, GSM8K. |
| `vocab_check.py` | Tokenizer/vocab sanity. |

**Base = CF (cloze/continuation, likelihood-scored). After SFT you can add MCF
(lettered multiple-choice).** See the benches-vs-loss discussion in the README/notes.

---

## 7. Serving & dashboard

- **`serve.py`** — `ThreadingHTTPServer` that serves the static files (dashboard,
  `metrics.jsonl`, `experiments.json`) **and** a `POST /chat` endpoint backed by
  a `Bot` (loads `sft16/model.pt`, text role markers, low-temp/top-k/rep-penalty
  sampling). A `RagBot` variant wires in `rag.py` (BM25 retrieve → read). Run:
  `python serve.py --port 8731 --model sft16`.
- **`dashboard.html`** — single self-contained file. Polls `metrics.jsonl`;
  renders: tiles (progress, val/train loss, throughput, **MFU, VRAM, grad norm**),
  loss chart, LR schedule, **gradient-norm chart**, live benchmarks, an A/B
  comparison card, model-output samples, the **experiments log** (now with a
  **Loss mask** column), and the chat card (single-turn toggle + New chat).

---

## 8. Conventions (the glue)

**Data**: `<dataset>/train.bin` + `val.bin` (`uint16`), optional `tokenizer.json`.
`train.py --data <dataset>` picks them up; if a `tokenizer.json` is present the
vocab size is inferred from it, else GPT-2 BPE is used.

**Runs & outputs**: `--run <name>` sets the metrics archive
(`metrics_<name>.jsonl`) and, with `--out <dir>`, where `model.pt` + `ckpt.pt`
land. The live `metrics.jsonl` is always overwritten by the newest run (that's
what the dashboard shows); `metrics_<name>.jsonl` is the per-run archive that
survives for comparison. **Always pass `--out`** on throwaway runs so they don't
clobber a dataset dir's baseline model.

**Checkpointing**: `--ckpt-every N` writes an atomic `ckpt.pt`
(model+optimizers+step); `--resume` continues from it (appends to metrics so the
dashboard history survives). Use on any multi-hour run.

**Metrics schema** (`metrics.jsonl`, one JSON object per line):
| `event` | fields |
|---------|--------|
| `start` | `num_steps, params_m, device, batch_size, block_size, prompt, data` |
| `train` | `step, train_loss, lr_mult, time_s, tok_per_s, grad_norm, tflops, mfu, gpu_mem_mb` |
| `val` | `step, val_loss, time_s` |
| `sample` | `step, text` |
| `bench` | `step, piqa, hellaswag` |
| `done` | `time_s, sample` |

---

## 9. Common workflows

```bash
# Build a dataset (16K-vocab general mix)
MIX_VOCAB=16000 MIX_OUT=mix16 python prepare_mix.py

# Pretrain, checkpointed + instrumented + live benches
python train.py --data mix16 --run mix16 --out mix16 --tie --bench --checkpoint \
  --seq 256 --dim 768 --layers 20 --heads 12 --batch-size 32 --steps 99000 \
  --ckpt-every 2000 --resume

# Ablate depth at a fixed budget (sequential single-GPU runs + summary table)
python ablate.py --tag depth --grid layers=12,16,20 \
  --base "--data mix16 --tie --checkpoint --seq 256 --dim 768 --heads 12 \
          --batch-size 32 --steps 2000 --bench --eval-every 500"

# Build SFT data + fine-tune
python prepare_sft16.py
python train_sft16.py --resume

# Serve dashboard + chat
python serve.py --port 8731 --model sft16     # → http://localhost:8731/dashboard.html
```

---

## 10. Sweep / benchmark scripts (utility)

`filter_sweep.py`, `fineweb_sweep.py` (data ablations); `bench_throughput.py`,
`bench_compile.py`, `bench_step.py`, `bench_checkpoint.py` (throughput/memory
probes that produced the "Throughput ceiling" and "Gradient checkpointing"
experiment-log rows). `rag.py` is the retriever used by `serve.py --rag`.

For the cloud-migration plan (bigger 500M–1B runs), see **CLOUD.md**.
