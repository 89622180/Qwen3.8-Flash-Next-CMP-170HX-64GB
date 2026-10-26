#!/bin/bash
# Capture window: stop service -> capture activations -> ALWAYS restart service.
set -uo pipefail
LOG=/models/dflash2-quant/capture.run.log
cd /root/dflash2-quant

# Same driver/library environment as the qwen38-032-test service
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/root/vllm-latest/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda
export PATH=/root/vllm-latest/.venv/bin:/usr/local/cuda/bin:$PATH

echo "=== $(date -u +%H:%M:%S) stopping service ===" | tee -a $LOG
systemctl stop qwen38-032-test
sleep 5

echo "=== $(date -u +%H:%M:%S) starting capture ===" | tee -a $LOG
/root/vllm-latest/.venv/bin/python capture_offline.py 2>&1 | tee -a $LOG
RC=${PIPESTATUS[0]}

echo "=== $(date -u +%H:%M:%S) capture exit=$RC, restarting service ===" | tee -a $LOG
systemctl start qwen38-032-test

if [[ $RC -ne 0 ]]; then
    echo "CAPTURE FAILED (rc=$RC) - service restarted, see $LOG" | tee -a $LOG
    exit $RC
fi
echo "CAPTURE OK - hessians at /models/dflash2-quant/hessians.pt" | tee -a $LOG

