#!/usr/bin/env bash
# Auto-stopping training wrapper for RunPod — so you NEVER pay for an idle pod.
#
# Runs whatever command you pass, then stops THIS pod on exit — whether the run
# finished, crashed, or you hit Ctrl-C. An optional hard deadline stops the pod
# even if training hangs. Your data/checkpoints must live on the network volume
# (a stopped/terminated pod keeps the volume but wipes local disk).
#
# Usage (on the pod):
#   chmod +x runpod_train.sh
#   ./runpod_train.sh python train.py --data /workspace/data/mix16 ... --resume
#
# Optional env vars:
#   DEADLINE_HOURS=30   hard backstop: stop the pod after N hours no matter what
#   AUTOSTOP=terminate  fully remove the pod (default: stop = ends GPU billing)
#   RUNPOD_API_KEY=...   only needed if `runpodctl` isn't pre-authenticated
#
# RUNPOD_POD_ID is set automatically inside every RunPod pod.

set -uo pipefail
ACTION="${AUTOSTOP:-stop}"          # stop (ends GPU billing) | terminate (also frees disk)
DEADLINE_HOURS="${DEADLINE_HOURS:-0}"

stop_pod() {
  local rc=$?
  echo "[autostop] run exited (code $rc) — stopping pod so billing ends" >&2
  if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "[autostop] !! RUNPOD_POD_ID unset — cannot self-stop. STOP THE POD MANUALLY." >&2
    return
  fi
  # Preferred: the pre-installed, pod-authenticated CLI.
  if command -v runpodctl >/dev/null 2>&1; then
    if [ "$ACTION" = "terminate" ]; then
      runpodctl remove pod "$RUNPOD_POD_ID" >&2 && return
    else
      runpodctl stop pod "$RUNPOD_POD_ID" >&2 && return
    fi
  fi
  # Fallback: GraphQL API (needs RUNPOD_API_KEY).
  if [ -n "${RUNPOD_API_KEY:-}" ]; then
    local mut="podStop"; [ "$ACTION" = "terminate" ] && mut="podTerminate"
    curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
      -H "Content-Type: application/json" \
      -d "{\"query\":\"mutation{${mut}(input:{podId:\\\"$RUNPOD_POD_ID\\\"})}\"}" >&2
    echo >&2 && return
  fi
  echo "[autostop] !! could not stop automatically (no runpodctl, no RUNPOD_API_KEY). STOP THE POD MANUALLY." >&2
}
trap stop_pod EXIT   # fires on success, failure, AND Ctrl-C

# Optional hard deadline: kill the run (which triggers the trap) after N hours.
if [ "$DEADLINE_HOURS" != "0" ]; then
  ( sleep "$(awk "BEGIN{print $DEADLINE_HOURS*3600}")" \
    && echo "[autostop] DEADLINE_HOURS=$DEADLINE_HOURS reached — stopping run" >&2 \
    && kill -TERM $$ ) &
fi

echo "[autostop] pod $RUNPOD_POD_ID — running: $*" >&2
"$@"
