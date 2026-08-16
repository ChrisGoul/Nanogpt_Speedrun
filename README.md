# nanogpt speedrun, simplified

A scaled-down, single-GPU, single-file version of the
[modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt)
(the community effort, started by Keller Jordan, to train GPT-2 to a fixed
val loss as fast as possible — Karpathy's nanoGPT is the ancestor).

The real speedrun trains a 124M-param model on FineWeb across 8xH100.
This version trains a ~30M-param model on tiny shakespeare in a few minutes
on a consumer GPU, but keeps the interesting architectural/optimizer ideas.

## Run it

```
pip install torch numpy tiktoken
python prepare_data.py   # downloads + tokenizes tiny shakespeare
python train.py          # trains ~1000 steps, prints val loss, samples text
```

## The tricks, and why they help

**Muon optimizer** — the speedrun's biggest single win. Momentum SGD, but each
update matrix is *orthogonalized* (via a Newton-Schulz iteration) before being
applied. Gradients to a weight matrix tend to be dominated by a few directions
(large singular values); orthogonalizing rebalances the update so all
directions move equally. Only used for 2D hidden weight matrices; embeddings
and the lm_head still use Adam.

**RoPE (rotary position embeddings)** — instead of adding learned position
vectors to the input, rotate query/key channels by position-dependent angles.
Relative position then falls out of the q·k dot product naturally. Standard in
modern LLMs (Llama, etc).

**QK-norm** — RMS-normalize queries and keys before attention. Stops attention
logits from blowing up, which stabilizes training and lets you use higher
learning rates.

**ReLU² MLP** — `relu(x)²` instead of GELU. Empirically slightly better and
cheaper (from the "Primer" paper).

**Zero-init projections** — the attention output projection, MLP output
projection, and lm_head start at zero, so every residual block starts as the
identity and the model's initial output is uniform. Training starts smoother.

**Untied lm_head** — nanoGPT tied the input embedding and output head weights
(GPT-2 style). Untying costs parameters but trains faster.

**No biases, parameter-free RMSNorm** — biases and learnable norm scales add
params/steps but empirically don't help at this scale.

**Logit soft-capping** — `30·tanh(logits/30)` gently bounds the logits
(borrowed from Gemma 2); another stabilizer.

**Trapezoidal LR schedule** — short warmup, long flat stretch at max LR, then
linear decay to zero. Simpler than cosine, works as well or better, and you
can extend training by stretching the flat part.

## What was dropped (vs. the real speedrun)

- **Distributed training** (8xH100, DDP with overlapped gradient all-reduce)
- **FP8 matmuls** for the lm_head; custom Triton/compile tuning
- **FlexAttention** with long-short sliding-window attention patterns
- **Value embeddings + U-net skip connections** between layers
- **Learned attention-lambda / skip gates** and a few other micro-tweaks
- **torch.compile** — add `model = torch.compile(model)` for a speedup if
  you have a working Triton setup (tricky on Windows)

## Things to try

- Bump `n_layer`/`n_embd`/`block_size` in `Config` and see the loss curve move
- Swap Muon for Adam on all params (`hidden` list -> Adam) and compare — the
  Muon advantage should be visible even at this scale
- Delete QK-norm or zero-init and watch early-training stability change
- Plot the loss vs. step for different `cooldown_frac`
