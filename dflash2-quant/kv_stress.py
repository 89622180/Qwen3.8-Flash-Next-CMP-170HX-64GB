"""Concurrency stress test: 4 unique ~170K-token prompts (680K total > 611K pool).

Observes vLLM's queuing behavior when KV capacity is exceeded:
- vllm:num_requests_running / waiting (capacity vs deferred)
- preemption events in the engine log
- KV offloading activity (CPU tier)

Run in CT120: /root/vllm-latest/.venv/bin/python /root/dflash2-quant/kv_stress.py
"""

import json
import threading
import time
import urllib.request

from transformers import AutoTokenizer

MODEL_DIR = "/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP"
API = "http://127.0.0.1:8000"
N_REQS = 4
TARGET_TOKENS = 170_000
MAX_TOKENS = 32
TEST_CAP_S = 150

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
para = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. "
    "Bright vixens jump; dozy fowl quack. "
)
# calibrate repetitions once
probe = tok(para * 100, add_special_tokens=False)["input_ids"]
per_rep = len(probe) / 100
reps = int(TARGET_TOKENS / per_rep)
print(f"[stress] per_rep={per_rep:.1f} tokens, reps={reps} "
      f"-> ~{per_rep * reps:.0f} tokens x {N_REQS} reqs "
      f"(total ~{per_rep * reps * N_REQS / 1000:.0f}K vs pool 611K)", flush=True)

results = {}
results_lock = threading.Lock()


def fire(i: int):
    # unique nonce per request so prefix caching cannot share blocks
    prompt = f"[kv-stress nonce=req{i}-u{int(time.time())}]\n" + para * reps
    body = json.dumps({
        "model": "Qwen3.8-Flash-Next", "prompt": prompt,
        "max_tokens": MAX_TOKENS, "temperature": 0, "ignore_eos": True,
        "stream": True,
    }).encode()
    req = urllib.request.Request(f"{API}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=TEST_CAP_S) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                p = line[6:]
                if p == "[DONE]":
                    break
                d = json.loads(p)
                ch = d.get("choices") or []
                if ch and ch[0].get("text"):
                    if first is None:
                        first = time.time()
                    n += len(ch[0]["text"])
        with results_lock:
            results[i] = {"ttft": (first - t0) if first else None,
                          "done_s": time.time() - t0, "chars": n,
                          "status": "ok"}
    except Exception as e:
        with results_lock:
            results[i] = {"ttft": (first - t0) if first else None,
                          "done_s": time.time() - t0, "chars": n,
                          "status": f"err: {type(e).__name__}: {str(e)[:80]}"}
    print(f"[req{i}] {results[i]}", flush=True)


def metrics_snapshot():
    try:
        with urllib.request.urlopen(f"{API}/metrics", timeout=5) as r:
            text = r.read().decode()
        out = {}
        for line in text.splitlines():
            if line.startswith("vllm:num_requests_running") or \
               line.startswith("vllm:num_requests_waiting{") or \
               line.startswith("vllm:num_requests_waiting_by_reason") or \
               ("kv_offload" in line and not line.startswith("#")):
                k, _, v = line.rpartition(" ")
                out[k.split('model_name=')[0].rstrip(',"{ ')] = v
        return out
    except Exception as e:
        return {"err": str(e)[:60]}


print("[stress] baseline:", metrics_snapshot(), flush=True)
threads = [threading.Thread(target=fire, args=(i,)) for i in range(N_REQS)]
t_start = time.time()
for t in threads:
    t.start()

polls = []
while time.time() - t_start < TEST_CAP_S and \
        any(t.is_alive() for t in threads):
    time.sleep(4)
    snap = metrics_snapshot()
    snap["t"] = round(time.time() - t_start, 1)
    polls.append(snap)
    print(f"[poll t={snap['t']:>5.1f}s] {snap}", flush=True)

for t in threads:
    t.join(timeout=10)

print("\n[stress] === summary ===")
for i in sorted(results):
    r = results[i]
    ttft = f"{r['ttft']:.1f}s" if r["ttft"] else "never"
    print(f"  req{i}: status={r['status']:<12} ttft={ttft:<8} "
          f"total={r['done_s']:.1f}s chars={r['chars']}")
print(f"[stress] wall={time.time() - t_start:.0f}s")
