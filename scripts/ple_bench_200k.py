"""PLE SSD benchmark at long context: 80K reference + 200K prompt.

200K pushes the prefetcher's working set (~200K * 16 rows * 320 B ~= 1.0 GiB)
past what small row caches can hold, so cache-size effects show up there.
"""

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
lengths = [81920, 204800]


def post(payload, timeout=1800):
    start = time.perf_counter()
    resp = requests.post(BASE, json=payload, timeout=timeout)
    resp.raise_for_status()
    return time.perf_counter() - start, resp.json()


rows = []

for length in lengths:
    rng = random.Random(770000 + length + round_no * 1000000)
    tokens = [rng.randrange(1000, 30000) for _ in range(length)]
    elapsed, body = post({
        "model": MODEL, "prompt": tokens, "max_tokens": 1,
        "temperature": 0, "ignore_eos": True,
    })
    assert body["usage"]["prompt_tokens"] == length
    row = dict(kind="prefill", label=label, tokens=length,
               sec=round(elapsed, 3), tok_s=round(length / elapsed, 1))
    rows.append(row)
    print(json.dumps(row), flush=True)

with open(out, "w") as fh:
    json.dump(rows, fh, indent=2)
print("WROTE", out, flush=True)
