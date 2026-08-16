"""
Pre-fetch FineWeb shards with timeouts + retries.

The HF download stalls intermittently on this connection, and when it stalls
inside the corpus builder it blocks the whole job with no progress. So we pull
the shards first, one at a time, with a short socket timeout (so a stall raises
instead of hanging) and unlimited retries. hf_hub_download resumes partial
files, so retries are cheap.

Run:  python fetch_shards.py --first 2 --last 9
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")   # must be set before import

from huggingface_hub import hf_hub_download

REPO = "kjj0/fineweb10B-gpt2"

def fetch(i, max_tries=20):
    fname = f"fineweb_train_{i:06d}.bin"
    for attempt in range(1, max_tries + 1):
        try:
            t0 = time.time()
            path = hf_hub_download(REPO, fname, repo_type="dataset")
            mb = os.path.getsize(path) / 1e6
            print(f"shard {i}: OK ({mb:.0f} MB, {time.time()-t0:.0f}s)", flush=True)
            return True
        except Exception as e:
            print(f"shard {i}: attempt {attempt} failed ({type(e).__name__}), retrying...", flush=True)
            time.sleep(min(5 * attempt, 30))
    print(f"shard {i}: GAVE UP", flush=True)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2)
    ap.add_argument("--last", type=int, default=9)
    args = ap.parse_args()
    ok = 0
    for i in range(args.first, args.last + 1):
        ok += fetch(i)
    print(f"fetched {ok}/{args.last - args.first + 1} shards", flush=True)

if __name__ == "__main__":
    main()
