"""
runpod_ctl.py — deploy / list / stop RunPod pods from ONE command, so you can
fire many training experiments without the RunPod UI.

Dependency-free: uses only Python's standard library (urllib) to call RunPod's
GraphQL API — nothing to pip install. Each `run` deploys a pod that
self-provisions via pod_boot.sh (clone -> build data -> train -> back up to HF
-> stop itself), so the whole loop is one command:

    python runpod_ctl.py run --name big500 --args "--dim 1280 --layers 24 ..."

--- Setup (once) --------------------------------------------------------------
Create a LOCAL .env next to this file (gitignored — never commit; repo is
public). See .env.example:
    RUNPOD_API_KEY=...
    NETWORK_VOLUME_ID=...
    HF_TOKEN=hf_...
    HF_USER=chgoul3
    REPO_URL=https://github.com/ChrisGoul/Nanogpt_Speedrun
    IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

--- Commands ------------------------------------------------------------------
    python runpod_ctl.py gpus                 # list GPU type ids (for --gpu)
    python runpod_ctl.py run --name big300b   # deploy + train (defaults = 300M A100)
    python runpod_ctl.py list                 # your pods
    python runpod_ctl.py stop <pod_id>        # stop (ends GPU billing)
    python runpod_ctl.py terminate <pod_id>   # remove (also frees pod disk)

Add --dry-run to `run` to preview (secrets masked) without spending.

Only deploys/starts/stops pods and reads GPU/pod info — never touches billing;
your prepaid balance + auto-recharge OFF is the hard cap. RunPod's API evolves,
so verify the FIRST deploy with `list` + the dashboard.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://api.runpod.io/graphql"

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

def gql(query, variables=None, tries=3):
    key = need("RUNPOD_API_KEY")
    body = {"query": query}
    if variables is not None:
        body["variables"] = variables
    data = json.dumps(body).encode()
    headers = {
        "Content-Type": "application/json",
        # a real User-Agent gets past Cloudflare (it 403s the default Python-urllib UA)
        "User-Agent": "runpod_ctl/1.0 (+https://github.com/ChrisGoul/Nanogpt_Speedrun)",
        "Authorization": f"Bearer {key}",           # modern auth (query ?api_key= also kept)
    }
    last = None
    for _ in range(tries):                          # retry: flaky connection
        try:
            req = urllib.request.Request(f"{API}?api_key={key}", data=data,
                                         headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            if out.get("errors"):
                sys.exit("[runpod_ctl] API error:\n" + json.dumps(out["errors"], indent=2))
            return out["data"]
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode(errors="replace")[:600]
            except Exception:
                pass
            last = f"HTTP {e.code} {e.reason}" + (f" — {detail}" if detail else "")
            if e.code in (401, 403):                # auth problem — retrying won't help
                break
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
    sys.exit(f"[runpod_ctl] request failed: {last}")

def main():
    load_env()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpus", help="list GPU type ids + memory")
    sub.add_parser("list", help="list your pods")
    r = sub.add_parser("run", help="deploy a self-training pod")
    r.add_argument("--name", required=True, help="run name -> HF repo + output dir")
    r.add_argument("--gpu", default="NVIDIA A100 80GB PCIe", help="gpu type id (see `gpus`)")
    r.add_argument("--peak", type=int, default=312, help="GPU bf16 peak TFLOP/s (A100 312 / H100 756)")
    r.add_argument("--deadline", type=int, default=3, help="hard auto-stop after N hours")
    r.add_argument("--args", default="", help="ARGS override for pod_boot/train.py (quoted)")
    r.add_argument("--cloud", default="COMMUNITY", help="COMMUNITY (cheap) or SECURE")
    r.add_argument("--disk", type=int, default=40, help="container disk GB (raise for big mixes: 20GB .bin per ~10B tokens)")
    r.add_argument("--shards", type=int, default=0,
                   help="MIX_EDU_SHARDS: FineWeb-Edu shards to build (~0.14B unique tokens each; 0=pod default of 3)")
    r.add_argument("--no-volume", action="store_true",
                   help="deploy WITHOUT the network volume -> RunPod can place the pod in ANY "
                        "datacenter with the GPU free (data ephemeral; final model still -> HF)")
    r.add_argument("--dry-run", action="store_true")
    st = sub.add_parser("stop"); st.add_argument("pod_id")
    tm = sub.add_parser("terminate"); tm.add_argument("pod_id")
    a = ap.parse_args()

    if a.cmd == "gpus":
        for g in gql("query{gpuTypes{id displayName memoryInGb}}")["gpuTypes"]:
            print(f"{str(g.get('id')):34} {g.get('memoryInGb','?')}GB  {g.get('displayName','')}")
        return
    if a.cmd == "list":
        pods = (gql("query{myself{pods{id name desiredStatus machine{gpuDisplayName}}}}")
                .get("myself", {}).get("pods") or [])
        if not pods:
            print("(no pods)")
        for p in pods:
            gpu = (p.get("machine") or {}).get("gpuDisplayName", "")
            print(f"{p.get('id')}  {p.get('name')}  {p.get('desiredStatus')}  {gpu}")
        return
    if a.cmd == "stop":
        print(gql('mutation($id:String!){podStop(input:{podId:$id}){id desiredStatus}}', {"id": a.pod_id})); return
    if a.cmd == "terminate":
        print(gql('mutation($id:String!){podTerminate(input:{podId:$id})}', {"id": a.pod_id})); return

    # --- run: deploy a pod that trains itself and stops ---
    repo = os.environ.get("REPO_URL", "https://github.com/ChrisGoul/Nanogpt_Speedrun")
    start_cmd = (f'bash -c "mkdir -p /workspace && cd /workspace && '
                 f'(git -C nanogpt pull --ff-only || git clone {repo} nanogpt) && '
                 f'bash nanogpt/pod_boot.sh"')
    env = {
        "RUN": a.name,
        "GPU_PEAK": str(a.peak),
        "DEADLINE_HOURS": str(a.deadline),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HF_REPO": f"{os.environ.get('HF_USER', '')}/{a.name}",
    }
    if a.args:
        env["ARGS"] = a.args
    if a.shards:
        env["MIX_EDU_SHARDS"] = str(a.shards)      # bigger unique-token dataset
    inp = {
        "cloudType": a.cloud,
        "gpuTypeId": a.gpu,
        "gpuCount": 1,
        "name": f"train-{a.name}",
        "imageName": os.environ.get("IMAGE", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"),
        "containerDiskInGb": a.disk,
        "dockerArgs": start_cmd,
        "ports": "22/tcp",
        "env": [{"key": k, "value": v} for k, v in env.items()],
    }
    vol = "" if a.no_volume else os.environ.get("NETWORK_VOLUME_ID", "").strip()
    if vol:
        inp["networkVolumeId"] = vol       # pins the pod to the volume's datacenter
        inp["volumeMountPath"] = "/workspace"
        inp["volumeInGb"] = 0
    else:
        print("[runpod_ctl] no network volume -> pod can land in ANY datacenter with the GPU; "
              "data/checkpoints are EPHEMERAL (final model still backs up to HF).")
    if a.dry_run:
        masked = {**inp, "env": [{"key": e["key"], "value": ("***" if ("TOKEN" in e["key"] or "KEY" in e["key"]) else e["value"])}
                                 for e in inp["env"]]}
        print("[runpod_ctl] DRY RUN — would podFindAndDeployOnDemand with:")
        print(json.dumps(masked, indent=2))
        return
    mut = ("mutation Deploy($input: PodFindAndDeployOnDemandInput!){"
           "podFindAndDeployOnDemand(input:$input){id imageName machineId}}")
    pod = gql(mut, {"input": inp})["podFindAndDeployOnDemand"]
    print(f"[runpod_ctl] deployed pod {pod.get('id')} — training '{a.name}'.")
    print(f"    model will appear at: https://huggingface.co/{env['HF_REPO']}")
    print(f"    stop early with: python runpod_ctl.py stop {pod.get('id')}")

if __name__ == "__main__":
    main()
