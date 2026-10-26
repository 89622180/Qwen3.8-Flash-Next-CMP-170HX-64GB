#!/bin/bash
# Switch qwen38-032-test to the quantized DFlash2 draft and maximize KV capacity.
# Run in CT120 as root.
#
#   bash /root/dflash2-quant/deploy.sh [NEW_DRAFT_DIR] [GPU_MEM_UTIL]
#
# Rollback: remove the added Environment= lines from the drop-in and restart.
set -euo pipefail

NEW_DRAFT="${1:-/models/incoai-Qwen3.8-Flash-Next-DFlash2-W8A16G128-GPTQ}"
GPU_MEM="${2:-0.99}"
TARGET_POOL=524288   # 2 x 262144
DROPIN=/etc/systemd/system/qwen38-032-test.service.d/zz-dflash2.conf

[[ -f $NEW_DRAFT/model.safetensors && -f $NEW_DRAFT/config.json ]] || {
    echo "ERROR: $NEW_DRAFT incomplete"; exit 1; }

cp "$DROPIN" "$DROPIN.bak-$(date +%Y%m%d-%H%M%S)"
set_env() {  # name value  -> upsert Environment= line
    if grep -q "^Environment=$1=" "$DROPIN"; then
        sed -i "s|^Environment=$1=.*|Environment=$1=$2|" "$DROPIN"
    else
        echo "Environment=$1=$2" >> "$DROPIN"
    fi
}
set_env QWEN_SPEC_MODEL "$NEW_DRAFT"
set_env QWEN_GPU_MEMORY "$GPU_MEM"
echo "drop-in now:"; grep "^Environment=" "$DROPIN"

systemctl daemon-reload
systemctl restart qwen38-032-test

echo "waiting for engine startup..."
for i in $(seq 1 60); do
    sleep 10
    if curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo "API up after ~$((i*10))s"; break
    fi
    [[ $i == 60 ]] && { echo "API did not come up"; journalctl -u qwen38-032-test -n 60 --no-pager; exit 1; }
done

echo "=== KV geometry ==="
journalctl -u qwen38-032-test --no-pager -n 500 | grep -E "kv lcm block sizes|GPU KV cache size|Auto-fit|Model loading took|Available KV cache" | tail -6
POOL=$(journalctl -u qwen38-032-test --no-pager -n 500 | grep -oE "GPU KV cache size: [0-9,]+" | tail -1 | grep -oE "[0-9,]+" | tr -d ',')
echo "pool tokens: ${POOL:-?} / target $TARGET_POOL"
echo "=== draft quantization evidence ==="
journalctl -u qwen38-032-test --no-pager -n 500 | grep -iE "AutoGPTQ|marlin|Using .* for AutoGPTQ|quantization=inc" | tail -6
echo "=== spec metrics ==="
journalctl -u qwen38-032-test --no-pager -n 500 | grep -oE "Mean acceptance length: [0-9.]+.*" | tail -3
echo "done. bench with /root/decodebench.py (baseline 140 tok/s)."

