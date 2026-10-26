"""Offline activation capture for the DFlash2 draft linears.

Runs the target + DFlash2 draft offline on the 170HX with enforce_eager and an
in-process EngineCore, hooks the draft's linears, and accumulates per-module
activation Hessians H = 2 X^T X from DECODE-step inputs only.

The context-KV precompute inputs are EXCLUDED: syv-ai measured that blending
them into the k/v calibration made greedy acceptance 7% WORSE, so we follow
the decode-only distribution that ships.

MUST run while qwen38-032-test is STOPPED (needs the whole 170HX).
After capture the service is restarted with its normal config.

Usage (CT120):
  systemctl stop qwen38-032-test
  cd /root/dflash2-quant && /root/vllm-latest/.venv/bin/python capture_offline.py
  systemctl start qwen38-032-test
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # in-process EngineCore
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import json
import time

import torch
import vllm.model_executor.models.qwen3_dflash as qd
from vllm import LLM, SamplingParams

TARGET = "/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP"
DRAFT = "/models/incoai-Qwen3.8-Flash-Next-DFlash2"
OUT = "/models/dflash2-quant/hessians.pt"
ROW_LIMIT = 128          # decode draft rows = (1+4)*n_req <= 80; prefill is 100s+

HESS_KEYS = {
    "self_attn.qkv_proj": "self_attn.qkv",
    "self_attn.o_proj": "self_attn.o",
    "mlp.gate_up_proj": "mlp.gate_up",
    "mlp.down_proj": "mlp.down",
    "attention_conv.kernel_projection": "attention_conv.kernel_projection",
    "mlp_conv.kernel_projection": "mlp_conv.kernel_projection",
    "fc": "fc",
}

STATE: dict[str, list] = {}   # hkey -> [H fp32 on gpu, rows]
CAPTURE_ON = False
SKIP = False


def make_hook(hkey: str):
    def fn(mod, args):
        global SKIP
        if not CAPTURE_ON or SKIP:
            return
        x = args[0]
        if x.dim() != 2 or x.shape[0] > ROW_LIMIT:
            return
        xf = x.detach().float()
        st = STATE.get(hkey)
        if st is None:
            STATE[hkey] = [2.0 * (xf.t() @ xf), int(xf.shape[0])]
        else:
            st[0].addmm_(xf.t(), xf, beta=1.0, alpha=2.0)
            st[1] += int(xf.shape[0])
    return fn


def install_patches() -> int:
    n = 0

    orig_init = qd.DFlashQwen3Model.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        nonlocal n
        for name, mod in self.named_modules():
            for suffix, hkey in HESS_KEYS.items():
                if name == suffix or name.endswith("." + suffix):
                    prefix = name[: len(name) - len(suffix)]
                    mod.register_forward_pre_hook(make_hook(prefix + hkey))
                    n += 1

    qd.DFlashQwen3Model.__init__ = patched_init

    orig_pre = qd.DFlashQwen3Model.precompute_and_store_context_kv

    def patched_pre(self, *a, **kw):
        global SKIP
        SKIP = True
        try:
            return orig_pre(self, *a, **kw)
        finally:
            SKIP = False

    qd.DFlashQwen3Model.precompute_and_store_context_kv = patched_pre
    return n


PROMPTS = [
    # --- coding: implement ---
    "Implement a thread-safe LRU cache in Python with O(1) get and put. Include eviction callbacks and a __len__ method, then write pytest unit tests covering concurrent access.",
    "用 C++ 实现一个无锁的 MPSC 队列，要求支持动态扩容。给出完整的头文件实现和一段多线程压测代码。",
    "Write a Rust function that parses an ISO-8601 duration string (like P1DT2H30M) into a struct, handling fractional seconds and negative durations. Include doctests.",
    "实现一个 TypeScript 装饰器 @debounce(ms)，要求保留 this 绑定和泛型返回类型，并处理 leading/trailing 边沿。附带使用示例。",
    "Write a Go function that streams a large CSV file, validates each row against a schema, and batches inserts into Postgres with retries and backoff. Handle context cancellation.",
    "用 Java 实现一个生产者-消费者管道：多个生产者、单个消费者、有界队列、优雅关闭时排空队列。给出完整可编译代码。",
    # --- coding: debug ---
    "This Python code deadlocks intermittently under load. Find the bug and explain the fix:\n```python\nimport threading\nlock_a = threading.Lock()\nlock_b = threading.Lock()\ndef worker1():\n    with lock_a:\n        with lock_b:\n            process()\ndef worker2():\n    with lock_b:\n        with lock_a:\n            process()\n```",
    "我的 React 组件在 StrictMode 下 useEffect 执行了两次，导致订阅重复。下面是代码，请指出根因并给出符合 React 18 最佳实践的修复：\n```jsx\nuseEffect(() => {\n  const sub = ws.subscribe(channel, handler);\n}, []);\n```",
    "The following SQL query got 100x slower after a schema migration. Diagnose it and propose a fix:\n```sql\nSELECT o.id, c.name FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.created_at > now() - interval '30 days' ORDER BY o.total DESC LIMIT 50;\n```",
    "这段 CUDA kernel 在 sm80 上结果偶发错误，只有大 batch 时出现。分析可能的竞态或越界问题：\n```cuda\n__global__ void scatter(float* out, const int* idx, const float* in, int n) {\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i < n) out[idx[i]] += in[i];\n}\n```",
    "Debug this async Python code: it sometimes drops tasks silently.\n```python\nasync def fan_out(coros):\n    tasks = [asyncio.create_task(c) for c in coros]\n    return [t.result() for t in tasks if t.done()]\n```",
    # --- coding: refactor / review ---
    "Review this code for correctness, security, and style issues, then refactor it:\n```python\ndef get_user(id):\n    q = f\"SELECT * FROM users WHERE id = {id}\"\n    rows = db.execute(q).fetchall()\n    if len(rows) > 0:\n        return rows[0]\n    else:\n        return None\n```",
    "Refactor this 400-line God-class into cohesive units. Here is the class outline: it handles HTTP routing, JSON validation, database access, and email sending in one class. Propose the decomposition with interface signatures.",
    "审查这个 Kubernetes Deployment 的 YAML，指出影响滚动更新安全和资源效率的问题：\n```yaml\napiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: 3\n  template:\n    spec:\n      containers:\n      - name: api\n        image: api:latest\n        ports:\n        - containerPort: 8080\n```",
    # --- algorithms / math ---
    "Explain the difference between Adam and AdamW, derive the bias-correction terms, and discuss when the decoupled weight decay in AdamW matters in practice for LLM pretraining.",
    "推导 RoPE 的旋转矩阵形式，说明为什么它具有相对位置特性，以及 NTK-aware scaling 与 YaRN 在外推长上下文时的区别。",
    "Prove that the GPTQ objective (minimizing ||X(W - Q)^T||_F^2) admits the greedy error-compensation solution used in the reference implementation, and explain the role of the Cholesky factor of H^{-1}.",
    "Explain how speculative decoding preserves the target model's output distribution under rejection sampling, and derive the acceptance probability for greedy and sampled decoding.",
    "分析 Mamba/GDN 线性注意力状态空间模型与 Transformer 的计算复杂度对比，推导并行扫描形式。",
    # --- agent / tool-use style ---
    "You are a coding agent. Read the repository layout, find the function that caused issue #4231 (a race in the connection pool), and produce a patch. Repository tree:\n```\nsrc/\n  pool.py\n  connection.py\n  tests/test_pool.py\n```\nStart by listing the files you need.",
    "作为运维 agent，请排查线上服务 5xx 突增：先查日志聚合，再查最近 1 小时的部署记录，最后给出回滚或热修建议。工具调用请用 JSON 格式。",
    "You are an agent with shell access. Diagnose why the vLLM service's KV cache usage is pinned at 100% while throughput drops. Outline your investigation commands and expected outputs step by step.",
    "Agent task: the test suite fails only on CI (passes locally). Environment differs in glibc version. Plan your debugging: reproduce, bisect, isolate, fix. Show tool calls.",
    # --- long code context ---
    "Here is a 200-line Python module implementing a small expression parser (tokenizer, Pratt parser, evaluator). Extend it with variable bindings and function calls without breaking the existing API:\n```python\n# (full module omitted for brevity; assume standard Pratt parser structure)\nclass Parser:\n    def parse(self, tokens): ...\n    def parse_expression(self, rbp=0): ...\n    def nud(self, token): ...\n    def led(self, left, token): ...\n```\nList every place you would change and why.",
    "阅读下面这段分布式事务的伪代码，指出两阶段提交的阻塞点，并改造成三阶段或 Saga 模式：\n```text\ncoordinator: PREPARE all -> if all YES: COMMIT all -> DONE\nparticipant: on PREPARE: lock resources, reply YES/NO\n             on COMMIT: apply, release, ack\n```",
    # --- explanation / writing ---
    "写一篇技术博客，解释 KV cache 的内存占用计算方法，fp8 KV 的精度损失来源，以及 PagedAttention 的碎片优化原理。面向有深度学习基础的读者。",
    "Explain how a modern MoE router works (top-k gating, load balancing losses, expert capacity), and why shared experts are kept in dense precision in many open checkpoints.",
    "Explain the difference between throughput and latency optimizations in LLM serving: continuous batching, chunked prefill, prefix caching, and speculative decoding. When do they conflict?",
    "解释 PLE（n-gram 表）在 Qwen3.8-Flash-Next 中的作用，以及为什么把它 offload 到 SSD 时需要预取窗口。",
    # --- more implement variety ---
    "Write a Python script that monitors a log file, extracts JSON lines, computes a 5-minute rolling error-rate window, and posts an alert to a webhook when the rate exceeds 1%. Make it restart-safe.",
    "实现一个简单的向量数据库：支持 HNSW 索引、余弦相似度、top-k 查询和持久化。给出核心数据结构和插入/查询路径的完整代码。",
    "Write a bash script that finds the top 10 largest directories under a path, excluding mounts, sorted by apparent size, with a progress indicator and dry-run mode.",
    "用 Rust 写一个 tokio 异步 TCP 代理，支持连接池复用、超时和优雅关闭。给出 Cargo.toml 和完整 main.rs。",
    # --- mixed reasoning ---
    "A model serves at 140 tok/s with 4 speculative tokens and 87.7% per-position acceptance. Derive the expected acceptance length and the effective throughput formula, then compute what happens if acceptance drops to 80%.",
    "给定一个 64GB GPU、47.8GiB 权重、10.99GiB KV 池，计算把 draft 从 BF16 量化到 W8 后 KV 池能增加多少 token 容量（假设每 token 的 KV 字节数不变）。",
    "Explain why lcm(3216, 64) = 12864 causes 4x KV page inflation, given 3216 = 16 * 3 * 67, and why choosing a block size of 3264 = 64 * 51 eliminates it.",
    # --- short chat ---
    "What are the tradeoffs between int8 weight-only quantization and fp8 activations+weights for small draft models in speculative decoding?",
    "解释一下为什么 draft 模型的量化不会影响最终输出的正确性，只会影响接受率和吞吐。",
    "How would you validate that a quantized speculative drafter did not regress? Design the acceptance test suite.",
]


def main() -> None:
    n = install_patches()
    print(f"[capture] {n} hooks installed", flush=True)

    llm = LLM(
        model=TARGET,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.93,
        enforce_eager=True,          # hooks must fire; no cudagraph/compile
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        kv_cache_dtype="fp8_e5m2",
        mamba_cache_mode="align",
        speculative_config={
            "method": "dflash",
            "model": DRAFT,
            "num_speculative_tokens": 4,
            "attention_backend": "TRITON_ATTN",
            "kv_cache_dtype": "bfloat16",
        },
        additional_config={
            "ple_ssd_offload": True,
            "ple_ssd_workers": 16,
            "ple_ssd_cache_mb": 4096,
            "ple_ssd_native_library":
                "/root/vllm-latest/optimization/ple_ssd_io.so",
            "ple_ssd_io_depth": 256,
            "ple_ssd_prefetch_tokens": 262144,
        },
    )
    print("[capture] engine ready", flush=True)

    global CAPTURE_ON
    CAPTURE_ON = True
    t0 = time.time()
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20,
                        max_tokens=1280, ignore_eos=True)
    # two passes: sampled then greedy, to cover both decode regimes
    outs = llm.generate(PROMPTS, sp)
    outs += llm.generate(PROMPTS, SamplingParams(temperature=1.0, top_p=0.95,
                                                 top_k=20, max_tokens=1024,
                                                 ignore_eos=True, seed=1))
    outs += llm.generate(PROMPTS[: len(PROMPTS) // 2],
                         SamplingParams(temperature=0.0, max_tokens=1024,
                                        ignore_eos=True))
    CAPTURE_ON = False
    if not STATE:
        raise RuntimeError(
            "[capture] no activations captured - hooks never fired. "
            "enforce_eager may not have disabled the speculator's graph "
            "capture; check the startup log before re-running.")
    min_rows = min(st[1] for st in STATE.values())
    if min_rows < 20000:
        raise RuntimeError(
            f"[capture] insufficient rows (min {min_rows}); "
            "increase prompt count or max_tokens and re-run.")
    print(f"[capture] rows per module: min={min_rows} "
          f"max={max(st[1] for st in STATE.values())}", flush=True)
    gen_t = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[capture] generated {ntok} tokens in {gen_t:.0f}s "
          f"({ntok / gen_t:.0f} tok/s aggregate)", flush=True)

    payload = {k: {"H": st[0].cpu(), "rows": st[1]} for k, st in STATE.items()}
    torch.save(payload, OUT)
    print(f"[capture] saved {len(payload)} Hessians to {OUT}", flush=True)
    for k in sorted(payload):
        H = payload[k]["H"]
        d = torch.diagonal(H)
        print(f"  {k:45s} in={H.shape[0]:6d} rows={payload[k]['rows']:8d} "
              f"diag[min={d.min():.3e} max={d.max():.3e}]")
    total_gb = sum(v["H"].numel() for v in payload.values()) * 4 / 2**30
    print(f"[capture] hessian storage: {total_gb:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
