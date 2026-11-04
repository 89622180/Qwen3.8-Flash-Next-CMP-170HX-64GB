# 第四轮：vLLM 0.33.0 移植 + 解码 A/B + prefill 拆解（2026-11-04）

容器：CT103（DFlash2 臂，生产）与 CT120（MTP 臂），共用一张 CMP 170HX 64G（sm_80，
功率上限 250 W，由机主自管）。检查点 `Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP`，
PLE/n-gram 95.37 GiB 走 SSD，主模型 47.3 GiB 常驻显存。

## 0. 结论速览

1. 两个容器都已跑在 **vLLM 0.33.0**（stock wheel + 本仓库臂的 0.33 重移植分支）。
2. prefill 的真实形状（§4，9 个 prompt 长度 × 2 重复的 TTFT 全表）：先前那句“每请求固定开销
   +85 ms”是**只有 3 个点时硬拟直线的产物**。用 9 个点重拟：**截距只差 +8 ms（56.3 → 64.2 ms），
   斜率 −6 %（212.6 → 200.4 µs/tok）**。差距不是固定成本，是**分段的**——单 chunk 的 prompt 变慢，
   最坏在 pinned 1,298 tok：0.326 s → 0.398 s（**+72 ms / +22 %**，prefill 3982 → 3261 tok/s）；
   而 pinned ≥ 2,575 tok（≥2 个 chunk）不再吃亏——2,575 快 17 %（0.677 → 0.560 s）、9,882 快 5 %
   （2.141 → 2.025 s）、5,022 打平（+0.6 %）。
   单流解码持平（163.8 → 162.8，长样本 159.2），KV 池 557,576 → 553,415 tok（2.13× → 2.11×）。
3. 0.33 把本臂的 **fp8_e5m2 KV 读取路径删掉了**（§3）。现在走 upstream 为 sm_80 新加的
   uint8 字节解码 e4m3 路径，这是本轮唯一的强制语义变更。
4. 0.33 的新增项**逐个启动实测过 11 臂**（全表与判定在 `docs/0.33-new-features-matrix.md`）：
   只有 `rejection_sample_method=fly` 有信号——DFlash2 臂交错 A/B-A-B 三对，解码中位数
   +8.8 % / +10.9 % / +8.3 %；第六轮在 **CT120 的 MTP 臂**复测更大（+15.1 / +16.0 / +29.9 %，
   池化 **+18.8 %**，见 `docs/0.33-features-on-ct120-mtp.md`）。两臂都不花 prefill、不缩 KV 池。
   但它是引擎自己打 warning 的 **LOSSY** 验证路径（CT103 上 10 题只 2 题逐字节相同，
   CT120 上只 3 题相同），所以**不默认开启**，只做可选 drop-in
   （`configs/ct103-0.33-dflash2-fly.conf`、`configs/ct120-0.33-mtp-fly.conf`）。
   其余落不到这条臂（`monokernel` / XPress / `sleep_mode_backend` / `watchdog_config` /
   `simple_cpu_offload_num_blocks`，§5）；`draft_lm_head_quantization` 与
   `enable_adaptive_verification` **在 0.33 里根本不是 CLI flag**（argparse 报
   `unrecognized arguments`），前者走 JSON 在 MTP 臂过了 gate 但死在 sm_80 的 FP8 kernel 要求（§5）。
   新加的 sm_80 FP8 tile 覆盖量出来是三档里最快的，保持默认（§6.5）。
5. 解码三臂（长样本、配对 seed）：base 159.2 mean / `--async-scheduling` 152.8 /
   `--moe-backend marlin` 143.7 → **保留 base**；`QWEN_SPEC_TOKENS=3` 因 KV 池掉到 405,131（1.55×）被否。
6. CT120（MTP 臂）：k=5 与 k=4 解码打平（146.5 vs 146.0），但 k=4 多留 22,138 tokens 的池
   （2.09× vs 2.01×），所以 **降到 k=4**。
7. **prefill 还想拉的话，只剩 pinned ≈0.7k–1.3k 那一格**：长 prompt 0.33 已经更快，固定开销
   也只有 +8 ms，所以目标不再是 kernel 吞吐，而是那一格单 chunk 路径上多出来的 72 ms（未归因，§4）。

## 1. 装了什么

| 项 | 值 |
|---|---|
| 上游 tag | `v0.33.0`（tag 对象 `3a4d82f5`，指向 commit `4ba787ec`） |
| wheel | `vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl`，336,361,138 B |
| wheel sha256 | `4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa`（下载后实测一致） |
| 编译产物 | wheel 内含 13 个 `.so`，与 0.32 树里的 10 个顶层 `.so` 同名（`_C_stable_libtorch.abi3.so` 92,632,616 B 等） |
| 依赖核对 | 0.33.0 共 79 条核心 requires_dist；本机**只缺 1 条**：`orjson`（装 3.13.0） |
| 版本不一致但照跑 | `torch 2.13.0+cu130`（声明 ==2.14.0）、`torchvision 0.28.0`（==0.29.0）、`flashinfer-python 0.7.0.post1`（==0.7.2.post1） |

为什么不建全新 venv：这三个包（torch 2.14 + cu13 全家桶 + flashinfer 0.7.2）约 12–18 GiB，而
CT103 的 rootfs 配额只剩 17 GiB。所以用**薄 venv + `.pth` 指向 0.32 venv 的 site-packages**：
`/root/v33/lib/python3.12/site-packages/_oldvenv.pth` 里一行旧路径。`.pth` 追加的路径排在
本 venv 自己的 site-packages **之后**，所以 0.33 的 `vllm` 包赢，torch 等从旧 venv 解析。

验证（不是推理，是打印出来的）：

```
vllm 0.33.0 /root/v33/lib/python3.12/site-packages/vllm/__init__.py
torch 2.13.0+cu130 13.0
```

复现脚本：`scripts/install-vllm-033.sh`。

## 2. 臂的移植：分支 `ple-033`

0.32 时这条臂 = `vllm-project/vllm` tag `v0.32.0` + 4 个本地 commit + 4 个**未提交的工作区补丁**
（DFlash2 allow-list、SWA block override、target aux hidden、KV pool diagnostics）。之前只用
`git diff v0.32.0..HEAD` 导出过补丁，**漏掉了那 4 个未提交文件**；本轮改用 `git diff v0.32.0`
（含工作区）才拿到完整的 23 文件面。

移植做法：在 `/root/vllm-latest` 的同一个 clone 上 `git fetch --depth=1 origin tag v0.33.0`，
`git worktree add -B ple-033 /root/wt-033 v0.33.0`，然后 cherry-pick 那 4 个 commit（真 3-way），
最后用 `patch -p1 --fuzz=3` 打上未提交的 4 处。

结果：`git diff --stat v0.33.0` = **23 files, 1070 insertions(+), 39 deletions(-)**，与 0.32 臂的
23 files / 1068(+)/35(-) **文件清单完全一致**（两份 `+++` 列表 `diff` 为空），+2 行是冲突解决处的
注释。22 个 `.py` 全部 `py_compile` 通过，工作树里 `grep '^<<<<<<< '` 为空。

> 0.32 的补丁在 0.33 上直接 `patch --dry-run`：19 个文件里 **只有 2 个失败**。移植成本很低，
> 大头 `ple_ssd.py`（514 行）和 `ple_ssd_io.c`（247 行）是新增文件，永远干净。

### 四处冲突与判断

**a) `vllm/models/qwen4_exp/nvidia/ops/qsa.py` — kernel 的 fp8 判定**

```
HEAD(0.33):  is_fp8 = k_cache.dtype != torch.bfloat16
本臂(0.32):  is_fp8 = k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
```

取 HEAD。0.33 把 uint8 原始存储也算 fp8（下面还有 `if k_cache.dtype == torch.uint8:
softmax_scale *= 256.0` 的字节解码分支），我们的白名单形式反而会把 uint8 路径排除掉。

**b) 同文件 `warmup_qsa_sparse_paged_attention` 的 dtype 选择**

0.33 上游改成 `fp8_dtype = e4m3fn if has_device_capability(89) else uint8`，我们的补丁要一个
`kv_cache_dtype` 入参来选 e5m2。合并版保留上游的能力门，只在显式请求 e5m2 时覆盖：

```python
is_fp8 = kv_cache.dtype == torch.uint8
if not is_fp8:
    cache_dtype = key_cache.dtype
elif kv_cache_dtype == "fp8_e5m2":
    cache_dtype = torch.float8_e5m2
elif current_platform.has_device_capability(89):
    cache_dtype = torch.float8_e4m3fn
else:
    cache_dtype = torch.uint8
```

**c) `vllm/models/qwen4_exp/nvidia/qsa.py` — 视图与白名单**

0.33 只在 sm_89+ 把 uint8 视图成 e4m3；本臂的 commit 是无条件视图（e5m2 或 e4m3）。合并版：
e5m2 无条件视图，e4m3 保留上游能力门；`key_cache.dtype` 白名单同时保留 `torch.uint8`（上游）
和 `torch.float8_e5m2`（本臂）。

**d) `vllm/models/qwen4_exp/common/qsa_cache.py` — ring capacity**

0.33 把 `capacity = compress_ratio * cdiv(span, compress_ratio)` 换成了
`capacity = round_up(span, self.compress_ratio)`（等价），并留着“capacity 必须整除 block_size”
的 assert。本臂的 commit 是把 ring 放宽到能整除 block_size 的最小 `compress_ratio` 倍数。
合并版保留 `round_up` + 放宽循环 + 改成 `assert capacity <= block_size`。

## 3. 真正的门：0.33 不再能读 e5m2 的 KV

第一次启动在 CUDA graph 捕获阶段死在：

```
vllm/models/qwen4_exp/nvidia/ops/qsa.py:697 assert k_cache.dtype == torch.float8_e4m3fn
AssertionError
```

不是笔误，是设计：0.33 的 `_maybe_dequant` 只有 `tl.uint8`（手写 PTX，按 e4m3 语义 4 字节一组
解成 2 个 half）和 `tl.float8e4nv` 两个分支，**没有 `tl.float8e5`**，所以 typed e5m2 指针没有
读取路径。0.32 那年加的“uint8 alloc + e5m2 view”在 0.33 上属于写路径能用、读路径没有。

处理：`QWEN_KV_DTYPE=fp8`，走 upstream 为 sm_80 准备的 uint8 字节解码 e4m3 路径。这一步同时
改了数值路径（e4m3 vs e5m2），所以 §4 的对比不是纯版本对比——标清这一点，不假装。

## 4. prefill 全长度扫描：9 格 × 2 重复，两版同会话

先前只量了 2048-request 那一格，又用 1.3k / 5k / 10k 三个点拟合直线，得出“固定开销 +85 ms、
边际 −9 %”。**那个结论是拟合产物**：三点里恰好一格是回退、一格打平、一格领先，硬拟一条直线
就会把它翻译成“截距涨、斜率降”。本轮回到 0.32 把 9 个 prompt 长度各跑 2 遍（C=1，
`outsweep032` / `outsweep`，seed 5101/5102，`cached=0` ⇒ 每格都是冷前缀），两遍都列在上表，
比较时取每臂两遍里**较快的一遍**（两臂同样处理，不偏袒；见下面的星号说明）。
`pin` = 服务端上报的实际 prompt tokens（>512 的格约为请求长度的 0.61–0.64×）。

| 请求 tok | pin tok | TTFT 0.32（两遍） | TTFT 0.33（两遍） | 快的一遍 0.32 → 0.33 | Δ | prefill tok/s 0.32 → 0.33 |
|---|---|---|---|---|---|---|
| 8 | 8 / 8 | 0.036 / 0.036 | 0.038 / 0.038 | 0.036 → 0.038 s | **+2 ms（+5.6 %）** | 222 → 211 |
| 32 | 32 / 32 | 0.039 / 0.041 | 0.043 / 0.125 | 0.039 → 0.043 s | **+4 ms（+10.3 %）** | 821 → 744 |
| 128 | 128 / 128 | 0.064 / 0.065 | 0.069 / 0.127 | 0.064 → 0.069 s | **+5 ms（+7.8 %）** | 2000 → 1855 |
| 512 | 329 / 332 | 0.132 / 0.133 | 0.115 / 0.134 | 0.132 → 0.115 s | **−17 ms（−12.9 %）** | 2492 → 2887 |
| 1024 | 649 / 654 | 0.224 / 0.225 | 0.226 / 0.227 | 0.224 → 0.226 s | +2 ms（+0.9 %） | 2897 → 2894 |
| 2048 | 1289 / 1298 | 0.326 / 0.329 | 0.398 / 0.414 | 0.326 → 0.398 s | **+72 ms（+22.1 %）** | 3982 → 3261 |
| 4096 | 2575 / 2575 | 0.677 / 0.680 | 0.560 / 0.562 | 0.677 → 0.560 s | **−117 ms（−17.3 %）** | 3804 → 4598 |
| 8192 | 5022 / 5022 | 1.090 / 1.103 | 1.097 / 1.111 | 1.090 → 1.097 s | +7 ms（+0.6 %） | 4607 → 4578 |
| 16384 | 9820 / 9882 | 2.141 / 2.141 | 2.025 / 2.058 | 2.141 → 2.025 s | **−116 ms（−5.4 %）** | 4587 → 4880 |

用这 9 个点重新拟合 TTFT = 截距 + 斜率 × pin：

| 拟合范围 | 0.32 截距 / 斜率 | 0.33 截距 / 斜率 | 残差 |
|---|---|---|---|
| 全部 9 格 | 56.3 ms / 212.6 µs·tok⁻¹（4704 tok/s） | 64.2 ms / **200.4 µs·tok⁻¹（4991 tok/s）** | max ±74 ms，rms 32 ms |
| 仅 pin ≥ 1289 | 81.5 ms / 208.8（4789） | 112.0 ms / **193.4（5170）** | max ±58 ms |
| 仅 pin 329–1298 | 80.2 ms / 193.9（5157） | 26.1 ms / **289.3（3457）** | max ≤18 ms，rms ≤13 ms |

**所以“+85 ms 固定开销”不成立**——全域拟合的截距只差 8 ms。真正成立的是分段：

- 0.32 在 329–1298 这段是干净的直线（rms 13 ms，5157 tok/s），0.33 同一段只有 3457 tok/s，
  即**单 chunk 区间掉速 ~33 %**；
- 一旦进入多 chunk（pin ≥ 2575）0.33 不再吃亏：2575 那格 TTFT 降 17.3 %（3804 → 4598 tok/s），
  9882 降 5.4 %（4587 → 4880），5022 打平（+0.6 %）；
- 两版在 32 → 9882 的**总平均边际吞吐几乎相同**（4748 vs 4734 tok/s）。

**取“两遍里较快的一遍”这件事本身要打个星号。** 0.32 的两遍在小 prompt 上彼此一致
（32 tok：0.039 / 0.041；128 tok：0.064 / 0.065），而 0.33 的 **seed 5101 那两格慢 3 倍**
（32 tok：0.125 vs 0.043；128 tok：0.127 vs 0.069；512 tok：0.134 vs 0.115）。同一版本、同一配置、
只差 seed ⇒ 这是 0.33 臂上真实存在的**首过/抖动代价**（8 tok 那格两遍都是 0.038，所以不是纯
时钟爬坡）。取 min 把它从表里抹掉了，但**生产上会真的遇到**。这条只有观察，没做单独实验，
也不打算据此下结论。

也就是说 0.33 的 prefill 不是“慢 17 %”，是**把成本挪了位置**：pinned 0.65k–1.3k 这一段（代码
补全、短问答常用的量级）实打实多等 +2 → +72 ms，长上下文反而更快。你原来问的“预处理差距太大”
只对这一格成立，其余格是赚的。

**仍未归因，也说清：**已排除的候选——MoE 后端（`--moe-backend marlin` 1.3k = 3088 vs base 3122）、
`--async-scheduling`（3134 vs 3122）、FP8 tile 覆盖（§6.5：128,4 / table / 64,4 = 3122 / 3165 / 3082，
三档都在 3.1k 附近）、V1 runner（§6.6，被引擎拒）。一个只能观察到的相关性：0.33 在单 chunk 段
掉速、多 chunk 段反超，而两版的 chunk 边界同为 `--max-num-batched-tokens 2048`，所以差异在
**chunk 内的 kernel 选择/M 分派**，不在调度次数。要拿回这 72 ms 得上 profiler（在
`qsa_sparse_paged_attention` / PLE prefetch / 调度入队三处计时），改 flag 已经证明无效。
另一个必须记住的混淆项：§3 的 e5m2 → e4m3 是**跟着版本一起被强制换掉的数值路径**，所以这张表
量的是“0.33+fp8”对“0.32+e5m2”，不是纯版本对比。

其他 0.32 vs 0.33 的配对量（CT103，DFlash2 k=4，2048-in/256-out 短 ladder）：C1 单流 decode
163.8 → 162.8，C2 聚合 197.1 → 187.3，C4 301.8 → 287.0，acc_len 2.80-3.12 → 2.98-3.11，
`preempt=0`，GPU KV 557,576 → 553,415 tok，util 等效 0.9542 → 0.9523（graph 估算多占 ~0.19 GiB）。

## 5. 0.33 新特性逐项：diff 出的字段 + 11 臂启动实测的判定

不是看 release note，是把两版 dataclass 字段直接 diff 出来（`EngineArgs`、`CompilationConfig`、
`CacheConfig`、`SpeculativeConfig`、`KernelConfig`），再逐个占一次真启动验证。
**完整矩阵在 `docs/0.33-new-features-matrix.md`**，这里只留字段面的判定：

| 新增字段 | 属于 | 本臂能否用 |
|---|---|---|
| `monokernel` | EngineArgs + KernelConfig | **能启动，但完全惰性**。`grep -rn monokernel vllm/` 只命中 `vllm/models/deepseek_v41/amd/**`，判定里 `not current_platform.is_rocm()` 直接返回——这是 gfx942 的 DeepSeek 融合内核开关。实测臂 decode 140.3 / 139.4，与 base 同档 |
| `sleep_mode_backend` | EngineArgs | 不能（显存休眠，与本臂瓶颈无关） |
| `watchdog_config` | EngineArgs | 不能（可靠性） |
| `simple_cpu_offload_num_blocks` | CacheConfig | 用不上。本臂走 `OffloadingConnector` + `TieringOffloadingSpec`（16 GiB CPU 层），不是 `SimpleCPUOffloadConnector` |
| `rejection_sample_method` / `fly_window_size` | SpeculativeConfig | **能用，唯一有信号的新项**，但 LOSSY，见下 |
| `xpress_topc` | SpeculativeConfig | 不能。`speculative.py:1620` 直接 `raise ValueError("xpress_topc is only supported by XPress")` |
| `draft_lm_head_quantization` | SpeculativeConfig | **不是 CLI flag**（整棵树 `grep 'draft-lm-head'` 零命中）：第五轮按 `--draft-lm-head-quantization fp8` 传，两容器的启动日志都是 `vllm: error: unrecognized arguments`，所以我原先写的“引擎按 method 拒”是**源码文案不是日志**（第六轮更正，见 `docs/0.33-features-on-ct120-mtp.md` §0/§3）。字段只能走 `--speculative-config` JSON（`serve.sh` 现在有 `QWEN_DRAFT_LM_HEAD`）；MTP 臂**过了 gate**，2 min 41 s 后死在建权重：`FP8 PTPC online quant requires a kernel that honors per-token activation quantization; MarlinFP8 is W8A16 weight-only. Requires SM89+ for Cutlass FP8 or ROCm MI3xx for rowwise scaled_mm.` ⇒ **这张卡吃不到它，与配置无关** |
| `disable_lookahead_block_hashes` | SpeculativeConfig | **没有提升**：关掉后 decode 138.3 / median 135.3（base 158.5 / 138.9）。median 基本打平、mean 更低，而本 harness 的分辨率不到 20 %（矩阵 §3），所以结论是**没有理由开**，保持默认 |
| `enable_adaptive_verification` | SpeculativeConfig | 同样**不是 CLI flag**（`unrecognized arguments: --enable-adaptive-verification`），源码里的 `dspark` gate 从没被执行到 |
| `CompilationConfig` | — | **字段零变化**，所以 0.32 那套 cudagraph 结论（capture sizes 数的是 token）在 0.33 原样成立 |

FLy 不是 method，是挂在现有草案上的**验证期有损拒绝采样变体**（`fly` 出现在
`rejection_sample_method` 的 Literal 里，和 `standard` / `synthetic` / `block` 并列），所以
DF2 臂能装。交错 A-B-A-B 三对给 **+8.8 % / +10.9 % / +8.3 %**（中位数，剔除进循环的 run），
prefill 与 KV 池零代价。要开就一个 drop-in：`configs/ct103-0.33-dflash2-fly.conf`
（`QWEN_REJECTION_METHOD=fly` + `QWEN_FLY_WINDOW=2`，`serve.sh` 已接）。**但它是 lossy**：
引擎自己打 warning，10 题事实探子两臂同分 8/10，可 10 答里只有 2 答逐字节相同——
“分数没降”不等于“输出没变”，所以 shipped 配置不开。第六轮在 CT120 的 MTP 臂上量到 **+18.8 %**
（见 `docs/0.33-features-on-ct120-mtp.md`），性质一样。

上表只列 diff 出来的字段；**11 臂逐个启动的实测判定（含 hybrid KV manager、cascade attention、
cooperative indexer、triton MoE、mamba stochastic rounding、monokernel）在
`docs/0.33-new-features-matrix.md`**，那一轮的唯一收获就是 FLy。

引擎侧的实际差异只有两条（日志读到）：`Using HummingIndexedExperts Humming MoE backend`，和 §3
的 fp8 读取路径重写。V2 model runner **不是**差异：0.32 的日志里也有（journal 计数 9 vs 3）。

## 6. 解码 A/B

短样本判不出差异——同一份配置、同一个 seed，两次启动分别给出 **161.6 和 262.8 tok/s**
（acc_len 3.0 vs 4.9）。这条臂的 decode 被 DFlash2 接受长度主导，而接受长度随贪心生成的 token
路径摆动。于是改成 `1024` 输出 × `6` 个配对 seed（`1011…1016`），每臂 warm burst 之后再测，C=1：

| 臂 | 变量 | decode mean | median | min–max | 1.3k prefill |
|---|---|---:|---:|---|---:|
| base33 | 无（MoE oracle = humming） | **159.2** | 140.1 | 127.4–267.9 | 3090 |
| async33 | `--async-scheduling` | 152.8 | **147.2** | 131.2–181.4 | 3134 |
| marlin33 | `--moe-backend marlin` | 143.7 | 142.6 | 111.1–175.3 | 3088 |

- 256-out 的短 ladder 里 async 曾给出 C2 193.9 vs base 187.3（+3.5 %），长样本反过来，
  所以**不给它下结论**；KV 池三臂相同（553,415），`preempt=0`。
- `--moe-backend marlin` 明确更差（mean −10 %），同时证明 humming 不是那 5 % 掉速的来源。
- `QWEN_SPEC_TOKENS=3`：C1 median 160.6（与 base 161.6 打平），但 **KV 池 553,415 → 405,131
  （2.11× → 1.55×）**，两个 262K 请求放不下，否决。

**定稿：不传任何 extra 参数，DFlash2 k=4，`QWEN_KV_DTYPE=fp8`。** 工具：
`benchmarks/decode_long_sample.sh`（6 seed 配对 + 每轮 acc_len / preempt）。

> 补一句诊断（第五轮才搞清楚，见 `docs/0.33-new-features-matrix.md` §3）：这个 harness 的
> decode 是**双峰**的——acc_len 4.93–4.97 的一簇跑 240–268 tok/s，acc_len 2.45–3.2 的一簇跑
> 126–180 tok/s，落哪一簇取决于贪心有没有锁进重复循环（提示词是同一句话循环 12 遍 +
> `ignore_eos`），**与配置无关**。所以本表里 20 % 以内的差都判不出来，包括前面几个臂的
> “+3 % / −5 %”。要拿可信数字得换非重复的自然提示词，或按 acc_len 分层只取非循环 run。

### 6.5 新加的 sm_80 FP8 tile 覆盖：量了，保持默认

0.33 在 QSA 里新增一行（0.32 树里 `grep -c _is_sm80` = **0**，确认是新东西）：

```python
if is_fp8 and _is_sm80():
    # Wider tiles measured faster for the byte-decode path on A100;
    # unmeasured on sm_86, which keeps the table's tiles.
    BLOCK_N, num_warps = 128, 4
```

注释自己承认这是在 A100 上量出来的——**这不是“别人的卡”的借口**：CMP 170HX 就是 GA100，
sm_80 与 A100 同核，所以上游这条覆盖对我们是**同架构调优**，不是外推。我把它改成
`Q38_SM80_FP8_TILES` 环境变量控制（默认 `128,4` = 上游行为，`<n>,<w>` = 换 tile，
`table` = 关掉覆盖、用 GB300 表选的值），然后三臂：

| 设置 | 1.3k prefill | 5k prefill | 10k prefill | decode mean/median |
|---|---:|---:|---:|---|
| `128,4`（上游默认） | 3122 | **4625** | **4841** | 147.4 / 147.2 |
| `table`（关掉覆盖） | 3165 | 4492 | 4765 | 155.4 / 134.4 |
| `64,4` | 3082 | 4328 | 4538 | 155.1 / 138.4 |

默认那档在两个长 prompt 格上都是最快的（+3 % / +1.6 % vs `table`）。但要说清：**同一档默认值
本身在两个臂里给出了 147.4 和 159.2 的 decode mean（差 7 %）**，而三档 spread 也就 5 %，
所以这个差异量级与臂间飘移同阶——**结论是“不动 tile”，不是“tile 很有用”**。补丁保留在
site-packages 和 `ple-033` 里（备份 `/root/n1104/qsa_ops.py.pre-sm80tiles-20261104`），
但 shipped 配置不设这个 env，行为与上游一致。

### 6.6 V1 model runner：不是变量，而且被拒

0.32 的 journal 里也有 `Using V2 Model Runner`，所以它本来就不是 0.32→0.33 的差异。试着用
`VLLM_USE_V2_MODEL_RUNNER=0` 关掉它，24 s 后起不来：

```
pydantic_core.ValidationError: Value error, Model Runner V1 does not support:
DFlash candidate-head drafts
```

即本臂（DF2 草案）在 0.33 上被锁在 V2 runner 里，V1 不是可选退路。

## 7. CT120（MTP 臂）

同一个 wheel、同一份补丁，容器内 `/root/v33` + `/root/vllm-033work`（`cp -a` 自 CT103，再
`patch` 验证：两份 `site-packages/vllm` 与 worktree `diff -rq -x __pycache__` 零差异）。
CT120 的 `perf.conf` 用 `QWEN_KV_BYTES=10200547328` 把池钉在 9.50 GiB，日志确认走了
`reserved 9.5 GiB … skipped memory profiling`。

| 臂（0.33） | MTP 深度 | C1 decode 3 轮 / mean | C2 agg mean | C4 agg mean | GPU KV 池 |
|---|---|---|---:|---:|---|
| 出厂 | k=5 | 154.7 / 138.5 / 146.2 → **146.5** | 180.6 | 277.6 | 526,649 tok / **2.01×** |
| 定稿 | k=4 | 153.9 / 136.7 / 147.3 → 146.0 | 180.8 | 278.4 | 548,787 tok / **2.09×** |

配对逐轮 2 胜 1 负 → 解码打平；k=4 多 22,138 tokens 的池，且 k≤4 时 QSA ring 停在 capacity 8
（不需要 0.33 那个放宽循环）。所以 **CT120 降到 `Environment=QWEN_MTP=4`**。

### 7.5 第六轮：把卡倒到 CT120，复测 FLy 与 draft_lm_head

细节在 `docs/0.33-features-on-ct120-mtp.md`，这里只留结论：

- **FLy 在 MTP 臂上更强**：交错三对（每臂一次真启动，6 配对 seed × 1024 输出）解码中位
  **+15.1 % / +16.0 % / +29.9 %**，池化 162.9 vs 137.2 = **+18.8 %**，逐 seed 配对 fly 赢 16/18；
  按 acc_len ≥ 4.1 把两臂的饱和 run 都剔掉再比仍是 **+18.7 %**。acc_len mean 3.023 → 3.739。
  prefill 与 KV 池零代价（每臂都是 `548,787 tok / 2.09×`，`preempt=0`）。
  这条臂没碰到第四/五轮那个循环吸引子（acc_len 全程 ≤ 4.43），所以这组数比 DF2 那组干净。
  **性质仍然是 lossy**：10 题探子 8/10 → 9/10，但 10 答只 3 答逐字节相同。
- **`draft_lm_head_quantization` 吃不到**：走 JSON 进得去（`serve.sh` 的 `QWEN_DRAFT_LM_HEAD`），
  MTP 过 method gate，但建权重时 `MarlinFP8` 是 W8A16 weight-only，不满足 FP8 PTPC 的
  per-token 激活契约，要求 SM89+（原文见 §5 那行）。`nvfp4` 是 Blackwell 路径，更不可能。
- 顺带两条 0.33 的日志事实：`--mamba-cache-mode` **在 0.33 已废弃且无效果**（本臂还在传，
  改动会动 shipped argv，本轮只记录）；MTP 每次启动都 warning `num_speculative_tokens > 1
  will run multiple times of forward on same MTP layer … may result in lower acceptance rate`，
  这与标准臂里 acc_len 掉到 1.90–2.44 的那些 run 是同一件事。

## 8. 状态、产物与回滚

- **当前拿卡的是 CT120**（第六轮倒卡）：vLLM 0.33.0，`MainPID == :8000 owner`，health 200，
  argv `{"method":"mtp","num_speculative_tokens":4}` + `--kv-cache-dtype fp8`，**标准验证、无 fly**，
  KV 548,787 tok / 2.09×，drop-in 只剩 `perf.conf` + `zz-v033.conf`。
  **CT103 连容器一起停了**（两容器 `net0` 同为 192.168.9.12，同时起会撞 IP；这是本轮之前就存在
  的状态，我没改）。CT103 的 unit 仍是 `enabled` + `onboot: 1`、CT120 `onboot: 0`，所以**主机重启后
  卡会回到 CT103**；要立刻换回去：CT120 `systemctl stop qwen38-032-test.service` 释放显存，再
  `pct start 103`。
- 本轮入仓：`scripts/install-vllm-033.sh`、`scripts/serve.sh`（`QWEN_EXTRA_ARGS` 透传 + 保留
  `QWEN_KV_BYTES` 钉池 + `spec_switches()`：**FLy / fly_window / lookahead-off / draft_lm_head
  四个开关对 dflash 和 mtp 两条分支都生效**；线上 CT103 与 CT120 的副本 md5 相同
  `31bdaa2ae90b1a1b47eb14159efdd771`，仓库副本按同一 md5 校验过；默认一个都不设 ⇒ argv 与本轮之前
  完全相同，用 `QWEN_PRINT_COMMAND=1` 对拍过）、
  `configs/ct103-0.33-dflash2.conf`、`configs/ct120-0.33-mtp.conf`、
  `configs/ct103-0.33-dflash2-fly.conf`、`configs/ct120-0.33-mtp-fly.conf`、
  `benchmarks/decode_long_sample.sh`、`benchmarks/qa_lossy_check.sh`、本文与
  `docs/14600k-pcore-only-recheck.md`、`docs/0.33-new-features-matrix.md`、
  `docs/0.33-features-on-ct120-mtp.md`。
- **完整的 0.33 臂 diff（64,097 B，23 files/1070+）本轮没有进仓库。** 它在 CT103 的
  `/root/n1104/ple-033-full.patch`，sha256 `0d0436af1a1d0085f00c4fb74364a865417b0f880dd94229a7489023ed5c85a0`；
  也可在 `/root/wt-033`（分支 `ple-033`）用 `git diff v0.33.0` 重生。我试过用 Contents API 直接
  上传，但无法保证 23 KB base64 逐字节不出错，校验不过的东西不该留在仓库里，所以删了。
  要入仓只需一条命令（今晚容器能连 github.com）：
  ```bash
  cp /root/n1104/ple-033-full.patch <repo>/patches/qwen38-ple-ssd-0.33.patch && git push
  ```
  在那之前，本文 §2 的四处冲突判定 + `patches/ple_ssd.py` + `patches/*.patch` 已能手动复现这条臂。
  tile 覆盖那一块（§6.5）是本轮另加的 6 行，**不在** 上面那份 diff 里，需要时从
  `/root/n1104/patch_tiles.py` 重放。
- 回滚 0.32：删掉 `qwen38-032-test.service.d/zz-v033.conf` → `systemctl daemon-reload` →
  `systemctl restart qwen38-032-test.service`。本轮已实际走过一次这一路径（§4 的 0.32 对照
  就是这么测的，然后恢复回 0.33），所以回滚是验过的。`/root/vllm-latest`（0.32 臂）全程未改；
  另有备份 `scripts/serve.sh.pre-merged-20261104`、`scripts/serve.sh.pre-spec-switches-20261104`、
  `scripts/serve.sh.pre-specfn-20261104`、`scripts/serve.sh.pre-dlhqhook-20261104`
  （CT103 与 CT120 各一份）、`zz-v033.conf.pre-kvfp8-20261103`、
  `/root/n1104/qsa_ops.py.pre-sm80tiles-20261104`。
- 已知未做：`flashinfer 0.7.2.post1` / `torch 0.29/2.14` 的 pin 没补齐（盘只够一份 CUDA 栈）；
  pinned 0.7k–1.3k 那一格多出来的 72 ms 未归因（§4）；FLy 的“有损”只用 10 题事实探子看过，
  没有做长文本/代码的分布级对比；`draft_lm_head_quantization` 的 `nvfp4` 档没试（Blackwell 的
  W4A16 路径，机制上这张卡不可能）。
- 本轮顺手修了一个坑：CT103 的 `/models` bind 被 01:20 的自动快照流程卸掉后没挂回，
  导致 unit 的 `[Unit] ConditionPathExists` 静默拒绝启动（详见另一篇文档 §4）。
