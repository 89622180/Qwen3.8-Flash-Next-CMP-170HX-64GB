"""Prefill/TTFT benchmark for CT120 vLLM (Qwen3.8-Flash-Next).

Usage: bench_prefill.py [target_prompt_tokens] [n_requests] [max_tokens]
Defaults: 55000 5 256

Builds ~target-token prompts with a unique prefix (prefix-cache miss),
sends sequential streaming requests, measures TTFT and prefill throughput.
"""
import json
import sys
import time
import urllib.request

from transformers import AutoTokenizer

MODEL_DIR = "/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP"
API = "http://127.0.0.1:8000/v1/chat/completions"
TARGET_PROMPT_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 55000
N_REQUESTS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 256

tok = AutoTokenizer.from_pretrained(MODEL_DIR)

para = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. "
    "Bright vixens jump; dozy fowl quack. "
)
para_ids = tok(para, add_special_tokens=False)["input_ids"]
reps = TARGET_PROMPT_TOKENS // len(para_ids) + 1
body = para * reps
body_tokens = len(tok(body, add_special_tokens=False)["input_ids"])
sys.stderr.write(f"shared body: {body_tokens} tokens\n")


def one_request(idx: int):
    # unique prefix at the START -> zero prefix-cache hit
    prompt = f"[benchmark request #{idx} nonce={idx * 7919 + 104729}]\n" + body
    data = json.dumps(
        {
            "model": "Qwen3.8-Flash-Next",
            "messages": [{"role": "user", "content": prompt}],
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
    total = time.time() - t0
    return {
        "idx": idx,
        "ttft": round(ttft, 3),
        "total": round(total, 3),
        "prompt_tokens": usage["prompt_tokens"] if usage else None,
        "completion_tokens": usage["completion_tokens"] if usage else None,
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens")
        if usage
        else None,
    }


# warmup (small, unique) - absorbs lazy init
one_request(-1)

results = [one_request(i) for i in range(N_REQUESTS)]
ttfts = [r["ttft"] for r in results]
ptoks = [r["prompt_tokens"] or 0 for r in results]
srt = sorted(ttfts)
median_ttft = srt[len(srt) // 2]
mean_ttft = sum(ttfts) / len(ttfts)
mean_ptok = sum(ptoks) / len(ptoks)
print(json.dumps(results, indent=2))
print(
    json.dumps(
        {
            "mean_ttft_s": round(mean_ttft, 3),
            "median_ttft_s": round(median_ttft, 3),
            "mean_prompt_tokens": round(mean_ptok),
            "prefill_tok_s_mean": round(mean_ptok / mean_ttft) if mean_ttft else None,
            "prefill_tok_s_median": round(mean_ptok / median_ttft) if median_ttft else None,
        },
        indent=2,
    )
)
