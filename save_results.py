"""
Back up a finished run to the Hugging Face Hub so you can retrieve it from
anywhere — no pod, no scp. Uploads model.pt (or the latest ckpt.pt), the run's
metrics, and the tokenizer to a PRIVATE repo.

Designed to be the PRESTOP_CMD in runpod_train.sh (runs just before the pod
stops), but also works standalone. No-ops quietly if HF_TOKEN isn't set, so it's
safe to leave wired in.

Env:
  HF_TOKEN   your Hugging Face write token (required to do anything)
  HF_REPO    target repo id, e.g. "chrisgoul/nanogpt-big300b" (default: <run>)
  OUT_DIR    run output dir holding model.pt/ckpt.pt (default: /workspace/<RUN>)
  RUN        run name (default: big300b)
  DATA_DIR   dir holding tokenizer.json to bundle (default: /workspace/mix16)
"""
import glob
import os
import sys

RUN = os.environ.get("RUN", "big300b")
OUT_DIR = os.environ.get("OUT_DIR", f"/workspace/{RUN}")
DATA_DIR = os.environ.get("DATA_DIR", "/workspace/mix16")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO = os.environ.get("HF_REPO", RUN)

def main():
    if not HF_TOKEN:
        print("[save_results] HF_TOKEN not set — skipping backup (results remain on the volume)")
        return 0
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[save_results] huggingface_hub not installed — skipping")
        return 0

    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_REPO, exist_ok=True, private=True, repo_type="model")

    # what to upload: model.pt if present, else the newest ckpt.pt; + tokenizer + metrics
    uploads = []
    if os.path.exists(os.path.join(OUT_DIR, "model.pt")):
        uploads.append((os.path.join(OUT_DIR, "model.pt"), "model.pt"))
    ckpts = sorted(glob.glob(os.path.join(OUT_DIR, "ckpt.pt")))
    if ckpts and not uploads:
        uploads.append((ckpts[-1], "ckpt.pt"))       # fall back to checkpoint
    tok = os.path.join(DATA_DIR, "tokenizer.json")
    if os.path.exists(tok):
        uploads.append((tok, "tokenizer.json"))
    for m in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"metrics_{RUN}.jsonl")):
        uploads.append((m, os.path.basename(m)))

    if not uploads:
        print(f"[save_results] nothing to upload in {OUT_DIR} — skipping")
        return 0

    for path, name in uploads:
        mb = os.path.getsize(path) / 1e6
        print(f"[save_results] uploading {name} ({mb:.0f} MB) -> {HF_REPO}", flush=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=HF_REPO, repo_type="model")
    print(f"[save_results] done -> https://huggingface.co/{HF_REPO}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
