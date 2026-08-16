#!/usr/bin/env bash
# Hands-off pod runner. After the pod boots this does EVERYTHING:
#   clone/update repo -> build data (once, if missing) -> train -> back up
#   results to HF (if HF_TOKEN set) -> stop the pod.
#
# Use it two ways:
#   1) Paste as the pod's "Docker/Start Command" when you deploy -> fully
#      automatic: deploy the pod and walk away; it trains and stops itself.
#   2) Run once after SSH:  bash pod_boot.sh
#
# REQUIRES a network volume mounted at /workspace so data + checkpoints + the
# final model survive the pod stopping. Everything below lives under /workspace.
#
# Config via env (all optional — defaults train the 300M A100 config):
#   REPO            git URL (default: the ChrisGoul repo)
#   RUN             run name / output dir under /workspace (default big300b)
#   GPU_PEAK        bf16 peak TFLOP/s for MFU logging (312 A100 / 756 H100)
#   MIX_EDU_SHARDS  FineWeb-Edu shards to build (more = more tokens)
#   ARGS            training hyperparams (see default below)
#   DEADLINE_HOURS  hard backstop (default 3 — set to the GPU's expected time +buffer)
#   HF_TOKEN/HF_REPO  if set, final model is backed up to a private HF repo
#
# NOTE: this script only starts training + stops the pod. It never touches
# billing. Your prepaid balance + "auto-recharge OFF" are the hard cost cap.
set -uo pipefail

WORK=/workspace
REPO="${REPO:-https://github.com/ChrisGoul/Nanogpt_Speedrun}"
RUN="${RUN:-big300b}"
DATA="$WORK/mix16"
OUT="$WORK/$RUN"
export GPU_PEAK="${GPU_PEAK:-312}"
export DEADLINE_HOURS="${DEADLINE_HOURS:-3}"
export RUN OUT_DIR="$OUT" DATA_DIR="$DATA"     # for save_results.py
ARGS="${ARGS:---dim 1024 --layers 22 --heads 16 --seq 1024 --batch-size 64 --steps 8000 --eval-every 500 --bench-every 2000 --ckpt-every 1000}"

cd "$WORK" || { echo "[boot] no /workspace — attach a network volume there"; exit 1; }

# 1. code: clone once, else fast-forward to latest
if [ -d nanogpt/.git ]; then
  echo "[boot] updating repo..."; git -C nanogpt pull --ff-only || true
else
  echo "[boot] cloning $REPO ..."; git clone "$REPO" nanogpt || { echo "[boot] clone failed"; exit 1; }
fi
cd nanogpt
chmod +x runpod_train.sh 2>/dev/null || true

# 2. deps (torch/numpy preinstalled on PyTorch templates)
pip install -q tokenizers tiktoken huggingface_hub datasets pyarrow

# 3. data: build onto the volume only if not already there (idempotent)
if [ ! -f "$DATA/train.bin" ]; then
  echo "[boot] building dataset onto the volume ($DATA)..."
  MIX_OFFLINE=0 MIX_EDU_SHARDS="${MIX_EDU_SHARDS:-3}" MIX_VOCAB=16000 MIX_OUT="$DATA" python prepare_mix.py \
    || { echo "[boot] dataset build failed"; exit 1; }
else
  echo "[boot] dataset already on volume — skipping build"
fi

# 4. train -> (PRESTOP: back up to HF) -> stop pod.  runpod_train.sh handles the
#    deadline backstop and the self-stop; save_results.py no-ops without HF_TOKEN.
export PRESTOP_CMD="python $WORK/nanogpt/save_results.py"
echo "[boot] launching training: RUN=$RUN GPU_PEAK=$GPU_PEAK DEADLINE_HOURS=$DEADLINE_HOURS"
./runpod_train.sh python train.py \
  --data "$DATA" --run "$RUN" --out "$OUT" \
  --compile --tie --bench --peak-tflops "$GPU_PEAK" --resume $ARGS
