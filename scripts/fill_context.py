"""Fill the context to exactly max_model_len (262,144) and measure prefill.

v2: progress prints (flush), no pre-count roundtrip, auto-retry on 400
by parsing the exact token count from the error message.
Run with: python -u fill_context.py
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

from transformers import AutoTokenizer

MODEL_DIR = "/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP"
API = "http://127.0.0.1:8000/v1/chat/completions"
MAX_LEN = 262144
MAX_TOKENS = 256

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
print("tokenizer loaded", flush=True)

para = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. "
    "Bright vixens jump; dozy fowl quack. "
)
ids = tok(para * 10600, add_special_tokens=False)["input_ids"]
print(f"token pool: {len(ids)}", flush=True)

prefix = "[full-context fill v2 nonce=262144c]\n"
budget = 261600


def build(b: int) -> str:
    return prefix + tok.decode(ids[:b])


def send(text: str):
    data = json.dumps(
        {
            "model": "Qwen3.8-Flash-Next",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        API, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if ttft is None and chunk.get("choices"):
                ttft = time.time() - t0
            if chunk.get("usage"):
                usage = chunk["usage"]
    return time.time() - t0, ttft, usage


text = build(budget)
print(f"attempt 1: budget {budget} chars {len(text)}", flush=True)

result = None
for attempt in range(3):
    try:
        total, ttft, usage = send(text)
        result = (total, ttft, usage)
        break
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code}: {body[:300]}", flush=True)
        m = re.search(r"prompt contains at least (\d+) input tokens", body)
        if e.code == 400 and m:
            over = int(m.group(1)) + MAX_TOKENS - MAX_LEN
            budget = budget - over - 60
            text = build(budget)
            print(f"retry: budget {budget} (over by {over})", flush=True)
        else:
            raise

if result:
    total, ttft, usage = result
    pt = usage["prompt_tokens"] if usage else 0
    print(
        json.dumps(
            {
                "ttft_s": round(ttft, 3),
                "total_s": round(total, 3),
                "prompt_tokens": pt,
                "completion_tokens": usage["completion_tokens"] if usage else None,
                "context_fill_pct": round(pt / MAX_LEN * 100, 2) if pt else None,
                "prefill_tok_s": round(pt / ttft) if ttft else None,
            },
            indent=2,
        ),
        flush=True,
    )
