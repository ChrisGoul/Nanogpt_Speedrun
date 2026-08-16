"""
Robust FineWeb-Edu shard fetcher. HF downloads on this connection intermittently
wedge at the connection phase where HF_HUB_DOWNLOAD_TIMEOUT can't interrupt them.
So we run each download in a subprocess with a HARD timeout: if it wedges we kill
it and retry. hf_hub_download resumes from the .incomplete partial, so progress
accumulates across retries.

Grabs edu shards 1..N (each ~100M GPT-2 tokens ~ 200MB). 2 shards -> ~200M tokens.
"""
import os
import subprocess
import sys
import time

REPO = "karpathy/fineweb-edu-100B-gpt2-token-shards"
HARD_TIMEOUT = 200          # seconds per attempt before we kill + retry
N_SHARDS = 3

DL = ("import os; os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='20';"
      "from huggingface_hub import hf_hub_download;"
      f"hf_hub_download('{REPO}', __import__('sys').argv[1], repo_type='dataset')")

def fetch(shard):
    for attempt in range(1, 60):
        try:
            subprocess.run([sys.executable, "-c", DL, shard],
                           timeout=HARD_TIMEOUT, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"{shard}: OK", flush=True)
            return True
        except subprocess.TimeoutExpired:
            print(f"{shard}: attempt {attempt} wedged (killed), resuming...", flush=True)
        except subprocess.CalledProcessError:
            print(f"{shard}: attempt {attempt} errored, retry...", flush=True)
            time.sleep(min(3 * attempt, 20))
    print(f"{shard}: GAVE UP", flush=True)
    return False

def main():
    ok = 0
    for i in range(1, N_SHARDS + 1):
        ok += fetch(f"edu_fineweb_train_{i:06d}.bin")
    print(f"fetched {ok}/{N_SHARDS} edu shards", flush=True)

if __name__ == "__main__":
    main()
