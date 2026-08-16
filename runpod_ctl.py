"""
runpod_ctl.py — deploy / list / stop RunPod pods from ONE command, so you can
fire many training experiments without touching the RunPod UI.

Each `run` deploys a pod that self-provisions via pod_boot.sh (clone -> build
data -> train -> back up to HF -> stop itself). So the whole loop is:

    python runpod_ctl.py run --name big500 --args "--dim 1280 --layers 24 ..."

...and later the model shows up at huggingface.co/<HF_USER>/big500 with the pod
already stopped.

--- Setup (once) --------------------------------------------------------------
    pip install runpod
Create a LOCAL .env next to this file (it is gitignored — never commit; your
GitHub repo is public). See .env.example:

    RUNPOD_API_KEY=...          # RunPod -> Settings -> API Keys
    NETWORK_VOLUME_ID=...       # RunPod -> Storage -> your volume -> its id
    HF_TOKEN=hf_...             # HF write token (backs up the model)
    HF_USER=chgoul3
    REPO_URL=https://github.com/ChrisGoul/Nanogpt_Speedrun
    IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

--- Commands ------------------------------------------------------------------
    python runpod_ctl.py gpus                 # list GPU type ids (to pick --gpu)
    python runpod_ctl.py run --name big300b   # deploy + train (defaults = 300M A100)
    python runpod_ctl.py list                 # running pods
    python runpod_ctl.py stop <pod_id>        # stop (ends GPU billing)
    python runpod_ctl.py terminate <pod_id>   # remove (also frees pod disk)

Add --dry-run to `run` to preview the deploy (secrets masked) without spending.

NOTE: this only deploys/starts/stops pods and reads GPU/pod info — it never
touches billing. Your prepaid balance + auto-recharge OFF is the hard cap.
RunPod's SDK evolves; verify the FIRST deploy with `list` + the dashboard.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

def load_env():
    p = os.path.join(HERE, ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

def need(k):
    v = os.environ.get(k)
    if not v:
        sys.exit(f"[runpod_ctl] missing {k} — set it in .env (see this file's header / .env.example)")
    return v

def main():
    load_env()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpus", help="list GPU type ids + memory")
    sub.add_parser("list", help="list your pods")
    r = sub.add_parser("run", help="deploy a self-training pod")
    r.add_argument("--name", required=True, help="run name -> HF repo + output dir")
    r.add_argument("--gpu", default="NVIDIA A100 80GB PCIe", help="gpu type id (see `gpus`)")
    r.add_argument("--peak", type=int, default=312, help="GPU bf16 peak TFLOP/s for MFU (A100 312 / H100 756)")
    r.add_argument("--deadline", type=int, default=3, help="hard auto-stop after N hours")
    r.add_argument("--args", default="", help="ARGS override passed to pod_boot/train.py (quoted)")
    r.add_argument("--cloud", default="COMMUNITY", help="COMMUNITY (cheap) or SECURE")
    r.add_argument("--disk", type=int, default=40, help="container disk GB")
    r.add_argument("--dry-run", action="store_true")
    st = sub.add_parser("stop"); st.add_argument("pod_id")
    tm = sub.add_parser("terminate"); tm.add_argument("pod_id")
    a = ap.parse_args()

    try:
        import runpod
    except ImportError:
        sys.exit("[runpod_ctl] `pip install runpod` first")
    runpod.api_key = need("RUNPOD_API_KEY")

    if a.cmd == "gpus":
        for g in runpod.get_gpus():
            print(f"{str(g.get('id')):36} {g.get('memoryInGb','?')}GB")
        return
    if a.cmd == "list":
        pods = runpod.get_pods() or []
        if not pods:
            print("(no pods)")
        for p in pods:
            gpu = (p.get("machine") or {}).get("gpuDisplayName", "")
            print(f"{p.get('id')}  {p.get('name')}  {p.get('desiredStatus')}  {gpu}")
        return
    if a.cmd == "stop":
        print(runpod.stop_pod(a.pod_id)); return
    if a.cmd == "terminate":
        print(runpod.terminate_pod(a.pod_id)); return

    # --- run: deploy a pod that trains itself and stops ---
    repo = os.environ.get("REPO_URL", "https://github.com/ChrisGoul/Nanogpt_Speedrun")
    start_cmd = (f'bash -c "cd /workspace && (git -C nanogpt pull --ff-only || '
                 f'git clone {repo} nanogpt) && bash nanogpt/pod_boot.sh"')
    env = {
        "RUN": a.name,
        "GPU_PEAK": str(a.peak),
        "DEADLINE_HOURS": str(a.deadline),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HF_REPO": f"{os.environ.get('HF_USER', '')}/{a.name}",
    }
    if a.args:
        env["ARGS"] = a.args
    cfg = dict(
        name=f"train-{a.name}",
        image_name=os.environ.get("IMAGE", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"),
        gpu_type_id=a.gpu,
        cloud_type=a.cloud,
        gpu_count=1,
        network_volume_id=need("NETWORK_VOLUME_ID"),
        volume_mount_path="/workspace",
        container_disk_in_gb=a.disk,
        docker_args=start_cmd,
        env=env,
    )
    if a.dry_run:
        masked = {**cfg, "env": {k: ("***" if ("TOKEN" in k or "KEY" in k) else v) for k, v in env.items()}}
        print("[runpod_ctl] DRY RUN — would create_pod with:")
        print(json.dumps(masked, indent=2))
        return
    pod = runpod.create_pod(**cfg)
    pid = pod.get("id") if isinstance(pod, dict) else pod
    print(f"[runpod_ctl] deployed pod {pid} — training '{a.name}'.")
    print(f"    model will appear at: https://huggingface.co/{env['HF_REPO']}")
    print(f"    stop early with: python runpod_ctl.py stop {pid}")

if __name__ == "__main__":
    main()
