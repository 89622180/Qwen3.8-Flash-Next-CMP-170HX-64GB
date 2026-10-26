#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh" 2>/dev/null ||
    source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
cd "$VLLM_WORKDIR"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$VLLM_VENV/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2
export OMP_NUM_THREADS=${QWEN_OMP_THREADS:-1}
export MKL_NUM_THREADS=${QWEN_OMP_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
QWEN_MTP=${QWEN_MTP:-1}
spec=()
if [[ -n ${QWEN_SPEC:-} && ${QWEN_SPEC} == dflash2 ]]; then
    # DFlash2 draft: vLLM dispatches on the draft architecture, and the only
    # accepted method literal for the DFlash family is "dflash".
    # The draft inherits the target's attention backend and KV dtype unless
    # QWEN_SPEC_ATTN_BACKEND / QWEN_SPEC_KV_DTYPE override them. On sm80 the
    # inherited FlashInfer caps at UNIFORM_SINGLE_TOKEN_DECODE, which drops the
    # whole engine to cudagraph_mode=NONE once the draft verifies more than one
    # token per step, and its sliding-window group picks a 64-token block that
    # inflates the scheduler LCM. TRITON_ATTN reaches ALWAYS and uses
    # 16-aligned blocks, but refuses fp8 below SM89, hence bf16 for its KV.
    draft_args=""
    if [[ -n ${QWEN_SPEC_ATTN_BACKEND:-} ]]; then
        draft_args+=",\"attention_backend\":\"${QWEN_SPEC_ATTN_BACKEND}\""
    fi
    if [[ -n ${QWEN_SPEC_KV_DTYPE:-} ]]; then
        draft_args+=",\"kv_cache_dtype\":\"${QWEN_SPEC_KV_DTYPE}\""
    fi
    spec=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${QWEN_SPEC_MODEL:-/models/incoai-Qwen3.8-Flash-Next-DFlash2}\",\"num_speculative_tokens\":${QWEN_SPEC_TOKENS:-7}${draft_args}}")
elif (( QWEN_MTP > 0 )); then
    spec=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${QWEN_MTP}}")
fi
capture_sizes="${QWEN_CAPTURE_SIZES:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,64,128,256,512,1024,2048]}"
mode=(--compilation-config "{\"cudagraph_mode\":\"${QWEN_GRAPH_MODE:-FULL_AND_PIECEWISE}\",\"cudagraph_capture_sizes\":${capture_sizes}}")
if [[ ${QWEN_EAGER:-0} == 1 ]]; then
    mode=(--enforce-eager)
fi
kvoff=(--prefix-match-unit "${QWEN_PMU:-8}")
if [[ -n ${QWEN_KV_CPU_GB:-} && ${QWEN_KV_CPU_GB} != 0 ]]; then
    kvoff+=(--kv-offloading-size "${QWEN_KV_CPU_GB}" --kv-offloading-backend native)
    export VLLM_USE_SIMPLE_KV_OFFLOAD=1
fi
mamba_ckpt=()
if [[ ${QWEN_MAMBA_SHARED_PREFIX:-0} == 1 ]]; then
    mamba_ckpt=(--enable-mamba-shared-prefix-checkpoint)
fi
hashalgo=()
if [[ -n ${QWEN_HASH_ALGO:-} ]]; then
    hashalgo=(--prefix-caching-hash-algo "${QWEN_HASH_ALGO}")
fi
async=()
if [[ ${QWEN_ASYNC_SCHED:-0} == 1 ]]; then
    async=(--async-scheduling)
fi
cumem=()
if [[ ${QWEN_CUMEM:-0} == 1 ]]; then
    cumem=(--enable-cumem-allocator)
fi
kvdtype=()
if [[ -n ${QWEN_KV_DTYPE:-} ]]; then
    kvdtype=(--kv-cache-dtype "${QWEN_KV_DTYPE}")
fi
cors=()
if [[ -n ${QWEN_ALLOWED_ORIGINS:-} ]]; then
    cors=(--allowed-origins "${QWEN_ALLOWED_ORIGINS}")
fi
kvther=()
if [[ ${QWEN_KV_TIER_CPU_GB:-0} != 0 ]]; then
    kvther+=(--kv-offloading-size "${QWEN_KV_TIER_CPU_GB}")
    tier_sec="[]"
    if [[ ${QWEN_KV_TIER_NVME:-1} == 1 ]]; then
        tier_sec="[{\"type\":\"fs\",\"root_dir\":\"${QWEN_KV_TIER_DIR:-/kvcache/tier}\",\"n_read_threads\":${QWEN_KV_TIER_RT:-16},\"n_write_threads\":${QWEN_KV_TIER_WT:-16}}]"
    fi
    kvther+=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$(( ${QWEN_KV_TIER_CPU_GB} * 1024 * 1024 * 1024 )),\"secondary_tiers\":${tier_sec}}}")
fi
kvdisk=()
if [[ ${QWEN_KV_TIER_CPU_GB:-0} == 0 && ${QWEN_KV_DISK_CAP_GB:-0} != 0 ]]; then
    kvdisk+=(--kv-offloading-size "${QWEN_KV_DISK_CAP_GB}")
    kvdisk+=(--kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"kv_offload_backend\":\"disk\",\"disk_path\":\"${QWEN_KV_DISK_PATH:-/kvcache/vllm_kv.bin}\",\"disk_capacity_bytes\":$(( ${QWEN_KV_DISK_CAP_GB} * 1024 * 1024 * 1024 )),\"disk_buffer_slots\":${QWEN_KV_DISK_SLOTS:-4},\"use_page_cache\":true}}")
    export VLLM_USE_SIMPLE_KV_OFFLOAD=1
fi

blksz=()
if [[ -n ${QWEN_BLOCK_SIZE:-} ]]; then
    blksz=(--block-size "${QWEN_BLOCK_SIZE}")
fi

command=("$VLLM_VENV/bin/vllm" serve "$QWEN_MODEL_DIR" \
    --served-model-name Qwen3.8-Flash-Next \
    --host "${QWEN_HOST:-0.0.0.0}" --port "${QWEN_PORT:-8000}" \
    --dtype bfloat16 \
    --max-model-len "${QWEN_CONTEXT:-auto}" \
    --max-num-seqs "${QWEN_SEQS:-16}" \
    --max-num-batched-tokens "${QWEN_BATCH_TOKENS:-2048}" \
    --gpu-memory-utilization "${QWEN_GPU_MEMORY:-0.96}" \
    --safetensors-load-strategy lazy \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --additional-config "{\"ple_ssd_offload\":true,\"ple_ssd_workers\":${QWEN_SSD_WORKERS:-16},\"ple_ssd_cache_mb\":${QWEN_SSD_CACHE_MB:-512},\"ple_ssd_native_library\":\"${VLLM_WORKDIR}/optimization/ple_ssd_io.so\",\"ple_ssd_io_depth\":${QWEN_SSD_DEPTH:-256},\"ple_ssd_prefetch_tokens\":${QWEN_SSD_PREFETCH:-16384}}" \
    "${mode[@]}" "${spec[@]}" "${cors[@]}" "${kvoff[@]}" "${mamba_ckpt[@]}" "${hashalgo[@]}" "${async[@]}" "${cumem[@]}" "${kvther[@]}" "${kvdisk[@]}" "${kvdtype[@]}" "${blksz[@]}")
if [[ ${QWEN_PRINT_COMMAND:-0} == 1 ]]; then
    printf "%q " "${command[@]}"
    printf "\n"
    exit 0
fi
exec "${command[@]}"
