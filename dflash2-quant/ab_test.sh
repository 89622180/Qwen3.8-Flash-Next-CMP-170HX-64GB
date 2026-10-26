#!/bin/bash
# A/B: temporarily point back to the BF16 draft, restart, bench, then restore W8.
set -uo pipefail
DROPIN=/etc/systemd/system/qwen38-032-test.service.d/zz-dflash2.conf
BF16=/models/incoai-Qwen3.8-Flash-Next-DFlash2
W8=/models/incoai-Qwen3.8-Flash-Next-DFlash2-W8A16G128-GPTQ

echo "=== switching to BF16 draft ==="
sed -i "s|^Environment=QWEN_SPEC_MODEL=.*|Environment=QWEN_SPEC_MODEL=$BF16|" "$DROPIN"
systemctl daemon-reload
systemctl restart qwen38-032-test
for i in $(seq 1 40); do
    sleep 10
    curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
done
echo "=== BF16 bench ==="
/root/vllm-latest/.venv/bin/python /root/head2head.py 2>&1
echo "=== BF16 acceptance ==="
journalctl -u qwen38-032-test --no-pager -n 150 2>&1 | grep -oE "Mean acceptance length: [0-9.]+,.*Avg Draft acceptance rate: [0-9.]+%" | tail -4

echo "=== restoring W8 draft ==="
sed -i "s|^Environment=QWEN_SPEC_MODEL=.*|Environment=QWEN_SPEC_MODEL=$W8|" "$DROPIN"
systemctl daemon-reload
systemctl restart qwen38-032-test
for i in $(seq 1 40); do
    sleep 10
    curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
done
echo "=== W8 bench ==="
/root/vllm-latest/.venv/bin/python /root/head2head.py 2>&1
echo "=== W8 acceptance ==="
journalctl -u qwen38-032-test --no-pager -n 150 2>&1 | grep -oE "Mean acceptance length: [0-9.]+,.*Avg Draft acceptance rate: [0-9.]+%" | tail -4
echo "=== A/B done (W8 left deployed) ==="

