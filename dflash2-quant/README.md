# DFlash2 W8A16G128 量化 + CT120 部署 + KV 容量优化 — 完成报告

日期: 2026-10-26 | 目标: 量化 DFlash2 draft、部署、FP8 缓存、KV 容量 ≥ 2×262K

## 最终结果

| 指标 | 开始时 | 最终 | 目标 |
|---|---|---|---|
| GPU KV 池 | 411,612 tok | **661,601 tok (2.52×262K)** | 524,288 ✓ 超额 26% |
| 262K 并发 | 1.57x | **2.52x** | 2x ✓ |
| draft 权重 | 1.15 GB BF16 | **708 MB W8A16G128** | 量化 ✓ |
| 主缓存 dtype | fp8_e5m2 | fp8_e5m2（不变） | FP8 ✓ |
| decode 吞吐 | 140.3 tok/s | 137.4 tok/s | 无回归（噪声内） |
| spec 接受率 | 2.53-2.65 | 2.49-2.57 | 无回归（噪声内） |
| kv lcm | 3216 | 3216 | 无膨胀 ✓ |

## 三个优化杠杆（全部已部署在生产服务）

1. **draft W8A16G128 GPTQ 量化**（权重 -0.49GiB → KV +0.5GiB）
   - 算法: GPTQ，校准 = 真实 decode 激活（39 prompts × 3 轮，每模块 19.3 万行 Hessian）
   - 格式: auto_round:auto_gptq（与主模型同 INC/Marlin 路径），权重误差 mean 0.0080
   - A/B 实测: 接受率/吞吐与 BF16 完全一致
   - 产物: /models/incoai-Qwen3.8-Flash-Next-DFlash2-W8A16G128-GPTQ
2. **gpu_memory_utilization 0.975 → 0.99**（KV 10.99 → 12.47GiB）
3. **draft SWA 块 3216 → 1072**（核心突破，KV 池 +42%）
   - 根因: draft SWA 组页(65.9MB)主导池块大小，但它每请求只需窗口界定的 ~19 块
   - 1072 = 3216÷3，Triton 16 倍数 ✓，整除主块保持 LCM 3216 ✓
   - 池块大小降为 full-attn 组的 41,988,096B → 块数 203 → 318

## 生产配置（drop-in zz-dflash2.conf）

QWEN_SPEC=dflash2, QWEN_SPEC_TOKENS=4, QWEN_SEQS=4, QWEN_GPU_MEMORY=0.99,
QWEN_SPEC_ATTN_BACKEND=TRITON_ATTN, QWEN_SPEC_KV_DTYPE=bfloat16,
QWEN_SPEC_MODEL=/models/incoai-Qwen3.8-Flash-Next-DFlash2-W8A16G128-GPTQ,
VLLM_SW_BLOCK_SIZE=1072

## 代码补丁（均有 .bak-* 备份）

- vllm/model_executor/layers/attention/attention.py: VLLM_SW_BLOCK_SIZE 环境变量
  覆盖 SWA 块大小（校验内核支持 + 整除主块，否则告警忽略）
- vllm/v1/core/kv_cache_utils.py: [poolprobe] 启动诊断日志（每组 spec/页/每请求用量）

## 回滚

- 量化: drop-in 里 QWEN_SPEC_MODEL 改回 /models/incoai-Qwen3.8-Flash-Next-DFlash2
- SWA 块: 删 VLLM_SW_BLOCK_SIZE 行
- 显存: QWEN_GPU_MEMORY 改回 0.975
- 代码: cp 各 .bak-swblock-* / .bak-poolprobe-* 回原文件

## 重要教训

- **CT104/3080 会挂死宿主机**: 两次实测，pct start 104 后几分钟内核级挂死
  （journal 死于 "nvidia 0000:01:00.0: Using 39-bit DMA addresses"）。
  CT104 保持关机；量化在 CT120 CPU 上跑（GPTQ 80 秒）。
- FlashInfer+fp8 draft 在 SM80 有 page_size≤64 硬 assert → 容量塌到 15K，不可修。
- Triton+bf16 draft 是 SM80 唯一可行路径；draft KV fp8 需 SM89+。
- vLLM 的 /dev/shm KV offload mmap 泄漏: 快速重启后需手动清
  /dev/shm/vllm_offload_*.mmap（服务 ExecStopPost 只在正常停止时清理）。
