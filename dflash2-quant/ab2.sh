#!/bin/bash
set -uo pipefail
DROPIN=/etc/systemd/system/qwen38-032-test.service.d/zz-dflash2.conf
BF16=/models/incoai-Qwen3.8-Flash-Next-DFlash2
W8=/models/incoai-Qwen3.8-Flash-Next-DFlash2-W8A16G128-GPTQ

switch_and_wait() {
    local model=$1
    sed -i "s|^Environment=QWEN_SPEC_MODEL=.*|Environment=QWEN_SPEC_MODEL=$model|" "$DROPIN"
    rm -f /dev/shm/vllm_offload_*.mmap
    systemctl daemon-reload
    systemctl restart qwen38-032-test
    for i in $(seq 1 45); do
        sleep 10
        if curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
            echo "API up after ~$((i*10))s"; return 0
        fi
    done
    echo "API DID NOT COME UP"; return 1
}

echo "=== BF16 side ==="
switch_and_wait "$BF16" || exit 1
/root/vllm-latest/.venv/bin/python /root/head2head.py 2>&1
echo "=== BF16 acceptance ==="
journalctl -u qwen38-032-test --no-pager -n 120 2>&1 | grep -oE "Mean acceptance length: [0-9.]+,.*Avg Draft acceptance rate: [0-9.]+%" | tail -4
journalctl -u qwen38-032-test --no-pager -n 300 2>&1 | grep -aE "GPU KV cache size|Available KV|Model loading" | tail -3

echo "=== restoring W8 ==="
switch_and_wait "$W8" || exit 1
echo "=== A/B complete: W8 deployed ==="

