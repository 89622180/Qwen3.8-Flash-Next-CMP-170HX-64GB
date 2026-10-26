"""KV capacity probe: dump the hybrid KV cache group breakdown of the running
engine so we can compute exactly where the per-token bytes go and how to reach
the 2x262K (524,288 token) pool target.

Run in CT120 while the service is up:
  /root/vllm-latest/.venv/bin/python /root/dflash2-quant/kv_probe.py
"""

from __future__ import annotations

import json
import re
import subprocess

GiB = 1024 ** 3

# Draft model geometry (from its config.json)
DRAFT_LAYERS = 5
DRAFT_KV_HEADS = 8
DRAFT_HEAD_DIM = 128
BF16 = 2
FP8 = 1

draft_kv_bf16_per_token = DRAFT_LAYERS * DRAFT_KV_HEADS * DRAFT_HEAD_DIM * 2 * BF16
draft_kv_fp8_per_token = DRAFT_LAYERS * DRAFT_KV_HEADS * DRAFT_HEAD_DIM * 2 * FP8


def journal(pattern, n=800):
    out = subprocess.run(
        ["journalctl", "-u", "qwen38-032-test", "--no-pager", "-n", str(n)],
        capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if re.search(pattern, l)]


def main() -> None:
    print("=== startup memory lines (latest boot) ===")
    for l in journal(r"Model loading took|Available KV cache memory|"
                     r"GPU KV cache size|kv lcm block sizes|Auto-fit|"
                     r"Free memory on device|CUDA graph pool|"
                     r"no KV cache group|Auto-fit max_model_len")[-12:]:
        print(" ", l.split("bash[", 1)[-1][:220])

    print("\n=== derived numbers ===")
    m = {}
    for l in journal(r"GPU KV cache size: ([\d,]+) tokens")[-1:]:
        m["pool_tokens"] = int(re.search(r"GPU KV cache size: ([\d,]+) tokens", l).group(1).replace(",", ""))
    for l in journal(r"Available KV cache memory: ([\d.]+) GiB")[-1:]:
        m["kv_gib"] = float(re.search(r"Available KV cache memory: ([\d.]+) GiB", l).group(1))
    for l in journal(r"Model loading took ([\d.]+) GiB")[-1:]:
        m["weights_gib"] = float(re.search(r"Model loading took ([\d.]+) GiB", l).group(1))
    print(json.dumps(m, indent=1))

    if "pool_tokens" in m and "kv_gib" in m:
        per_tok = m["kv_gib"] * GiB / m["pool_tokens"]
        print(f"\nper-token KV bytes (pool average): {per_tok:,.0f}")
        print(f"draft KV bf16 (5L x 8kv x 128 x 2 x 2B): {draft_kv_bf16_per_token:,}")
        print(f"draft KV fp8  (5L x 8kv x 128 x 2 x 1B): {draft_kv_fp8_per_token:,}")
        target_share = per_tok - draft_kv_bf16_per_token
        print(f"implied target+other share: {target_share:,.0f} "
              f"({target_share / per_tok * 100:.0f}% of per-token cost)")
        if target_share > 0:
            fp8_draft_pool = m["kv_gib"] * GiB / (target_share + draft_kv_fp8_per_token)
            print(f"pool if draft KV were fp8 at same memory: {fp8_draft_pool:,.0f} "
                  f"tokens ({fp8_draft_pool / 262144:.2f}x 262K)")

    print("\n=== spec metrics (last) ===")
    for l in journal(r"Mean acceptance length")[-2:]:
        print(" ", l.split("bash[", 1)[-1][:200])

    print("\n=== current service env (drop-ins) ===")
    out = subprocess.run(["systemctl", "cat", "qwen38-032-test", "--no-pager"],
                         capture_output=True, text=True).stdout
    for l in out.splitlines():
        if l.startswith("Environment=QWEN_") or l.startswith("ExecStart"):
            print(" ", l)


if __name__ == "__main__":
    main()

