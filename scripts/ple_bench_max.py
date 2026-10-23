"""PLE prefill at the model's maximum 262K context (prompt 262000 + 1 token)."""

import json
import random
import sys
import time

import requests

BASE = "http://127.0.0.1:8000/v1/completions"
MODEL = "Qwen3.8-Flash-Next"
label = sys.argv[1]
out = sys.argv[2]
round_no = int(sys.argv[3]) if len(sys.argv) > 3 else 1

rng = random.Random(880000 + round_no * 1000000)
tokens = [rng.randrange(1000, 30000) for _ in range(262000)]
start = time.perf_counter()
resp = requests.post(BASE, json={
    "model": MODEL, "prompt": tokens, "max_tokens": 1,
    "temperature": 0, "ignore_eos": True,
}, timeout=1800)
resp.raise_for_status()
body = resp.json()
elapsed = time.perf_counter() - start
row = dict(kind="prefill", label=label, tokens=body["usage"]["prompt_tokens"],
           sec=round(elapsed, 3), tok_s=round(body["usage"]["prompt_tokens"] / elapsed, 1))
print(json.dumps(row), flush=True)
with open(out, "w") as fh:
    json.dump([row], fh, indent=2)
print("WROTE", out, flush=True)
