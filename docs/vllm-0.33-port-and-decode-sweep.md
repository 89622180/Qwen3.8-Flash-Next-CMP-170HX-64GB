# 第四轮：vLLM 0.33.0 移植 + 解码 A/B + prefill 拆解（2026-11-04）

容器：CT103（DFlash2 臂，生产）与 CT120（MTP 臂），共用一张 CMP 170HX 64G（sm_80，
功率上限 250 W，由机主自管）。检查点 `Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP`，
PLE/n-gram 95.37 GiB 走 SSD，主模型 47.3 GiB 常驻显存。

## 0. 结论速览

1. 两个容器都已跑在 **vLLM 0.33.0**（stock wheel + 本仓库臂的 0.33 重移植分支）。
2. prefill 的真实形状（§4）：不是“回退 17 %”，而是**每请求固定开销 +85 ms（80 → 165 ms）、
   每 token 边际成本反而 −9 %（208 → 190 µs/tok）**。所以短 prompt（单 chunk）的 TTFT 从
   0.349 s 变成 0.410 s，而 5k / 10k token 的 prefill 0.33 反而不输／更快（10k：4841 vs 4623 tok/s）。
   单流解码持平（163.8 → 162.8，长样本 159.2），KV 池 557,576 → 553,415 tok（2.13× → 2.11×）。
3. 0.33 把本臂的 **fp8_e5m2 KV 读取路径删掉了**（§3）。现在走 upstream 为 sm_80 新加的
   uint8 字节解码 e4m3 路径，这是本轮唯一的强制语义变更。
4. 0.33 新增可调项（`monokernel` / FLy / XPress / `sleep_mode_backend` / `watchdog_config` /
   `simple_cpu_offload_num_blocks`）**都落不到这条臂**（§5）；新加的 sm_80 FP8 tile 覆盖量出来
   是三档里最快的，保持默认（§6.5）；V1 model runner 被引擎直接拒（§6.6）。
5. 解码三臂（长样本、配对 seed）：base 159.2 mean / `--async-scheduling` 152.8 /
   `--moe-backend marlin` 143.7 → **保留 base**；`QWEN_SPEC_TOKENS=3` 因 KV 池掉到 405,131（1.55×）被否。
6. CT120（MTP 臂）：k=5 与 k=4 解码打平（146.5 vs 146.0），但 k=4 多留 22,138 tokens 的池
   （2.09× vs 2.01×），所以 **降到 k=4**。
7. **要“预处理”拉回来的话，目标应该是那 85 ms 的固定开销，不是 kernel 吞吐**：kernel 已经更快了。

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

## 4. prefill 拆解：固定开销 vs 边际成本（同会话、同脚本、配对 seed）

之前只量了 2048-token 那一格（实际有效 prompt 只有 ~1289 token，属于单 chunk）就下了
“prefill −17.5 %”的结论，太粗。本轮把三臂补齐到 1.3k / 5k / 10k 三格，并回到 0.32 重测：

| 有效 prompt | 0.32.1.dev+ple (fp8_e5m2) | 0.33.0+ple (fp8 e4m3) | TTFT 0.32 → 0.33 |
|---|---:|---:|---|
| ~1,289 tok | 3716 tok/s | 3122 tok/s | 0.349 s → 0.410 s |
| ~5,020 tok | 4580 tok/s | 4625 tok/s | 1.097 s → 1.090 s |
| ~9,880 tok | 4623 tok/s | **4841 tok/s** | 2.137 s → 2.041 s |

把 TTFT 对 prompt 长度拟合一次，形状就拆开了：

| | 固定开销（截距） | 每 token 边际成本（斜率） |
|---|---:|---:|
| 0.32 | 81 ms | 208 µs/tok |
| 0.33 | **165 ms** | **190 µs/tok（−9 %）** |

所以“预处理差距”不是吞吐回退，而是**每请求多了 ≈85 ms 的固定成本**：短 prompt 全额付这笔钱，
长 prompt 被边际收益抵掉甚至反超。CT120 同理（它的 1.3k 格 0.33 实测 3280-3351，因为它是 MTP
草案、没有 DF2 的 candidate-head 那一段）。

**本轮没找到的东西，也说清：**这 85 ms 目仍未归因。已排除的候选：MoE 后端（`--moe-backend marlin`
的 1.3k prefill = 3088，与 base 3122 同）、`--async-scheduling`（异步臂 1.3k prefill 3134 vs
base 3122，没变）、FP8 tile 覆盖（§6.5）、V1 runner（§6.6，被拒）。下一步如果要拿回那 85 ms，
得上 profiler（在 `qsa_sparse_paged_attention` / PLE prefetch / 调度入队三处计时），而不是改 flag。

其他 0.32 vs 0.33 的配对量（CT103，DFlash2 k=4，2048-in/256-out 短 ladder）：C1 单流 decode
163.8 → 162.8，C2 聚合 197.1 → 187.3，C4 301.8 → 287.0，acc_len 2.80-3.12 → 2.98-3.11，
`preempt=0`，GPU KV 557,576 → 553,415 tok，util 等效 0.9542 → 0.9523（graph 估算多占 ~0.19 GiB）。

## 5. 0.33 新特性逐项，以及为什么落不到这条臂

不是看 release note，是把两版 dataclass 字段直接 diff 出来（`EngineArgs`、`CompilationConfig`、
`CacheConfig`、`SpeculativeConfig`、`KernelConfig`）：

| 新增字段 | 属于 | 本臂能否用 |
|---|---|---|
| `monokernel` | EngineArgs + KernelConfig | **不能**。`grep -rn monokernel vllm/` 只命中 `vllm/models/deepseek_v41/amd/**`，判定里 `not current_platform.is_rocm()` 直接返回——这是 gfx942 的 DeepSeek 融合内核开关 |
| `sleep_mode_backend` | EngineArgs | 不能（显存休眠，与吞名无关） |
| `watchdog_config` | EngineArgs | 不能（可靠性） |
| `simple_cpu_offload_num_blocks` | CacheConfig | 用不上。本臂走 `OffloadingConnector` + `TieringOffloadingSpec`（16 GiB CPU 层），不是 `SimpleCPUOffloadConnector` |
| `fly_*` | SpeculativeConfig | **能装但不该用**：FLy 是独立 method，引擎自己打 `FLy verification is a lossy speculative decoding method and may degrade model outputs`，还要求 `num_speculative_tokens ≥ 2`。DFlash2 不是 FLy |
| `xpress_topc` | SpeculativeConfig | 不能。`speculative.py:1620` 直接 `raise ValueError("xpress_topc is only supported by XPress")` |
| `draft_lm_head_quantization` | SpeculativeConfig | 未测（gate 文案是 “not supported for method …”，且 sm_80 上没有 fp8/nvfp4 硬件路径，要真启动才能判） |
| `disable_lookahead_block_hashes` | SpeculativeConfig | 未测（只在 `config/vllm.py` 里出现，看不出对本臂的作用面） |
| `CompilationConfig` | — | **字段零变化**，所以 0.32 那套 cudagraph 结论（capture sizes 数的是 token）在 0.33 原样成立 |

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

### 6.5 新加的 sm_80 FP8 tile 覆盖：量了，保持默认

0.33 在 QSA 里新增一行（0.32 树里 `grep -c _is_sm80` = **0**，确认是新东西）：

```python
if is_fp8 and _is_sm80():
    # Wider tiles measured faster for the byte-decode path on A100;
    # unmeasured on sm_86, which keeps the table's tiles.
    BLOCK_N, num_warps = 128, 4
```

注释自己承认这是在 A100 上量出来的。我把它改成 `Q38_SM80_FP8_TILES` 环境变量控制（默认
`128,4` = 上游行为，`<n>,<w>` = 换 tile，`table` = 关掉覆盖、用 GB300 表选的值），然后三臂：

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

## 8. 状态、产物与回滚

- **CT103 running，vLLM 0.33.0，:8000 提供 `Qwen3.8-Flash-Next`**（`MainPID == :8000 owner`，
  KV 553,415 tok / 2.11×，0 条错误）。CT120 已配好但 **stopped**（一张卡只能一个引擎；
  CT103 `onboot: 1`、CT120 `onboot: 0`，所以主机重启后还是 CT103 拿卡）。两容器 `net0` 同为
  192.168.9.12，同时跑会撞 IP——这是本轮之前就存在的状态，我没改。
- 本轮入仓：`scripts/install-vllm-033.sh`、`scripts/serve.sh`（新增 `QWEN_EXTRA_ARGS` 透传，
  保留 `QWEN_KV_BYTES` 钉池；与线上副本 md5 一致 `659c99ca2e39dfea95a2c87f0478dc18`）、
  `configs/ct103-0.33-dflash2.conf`、`configs/ct120-0.33-mtp.conf`、`benchmarks/decode_long_sample.sh`、
  本文与 `docs/14600k-pcore-only-recheck.md`。
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
  另有备份 `scripts/serve.sh.pre-merged-20261104`、`zz-v033.conf.pre-kvfp8-20261103`、
  `/root/n1104/qsa_ops.py.pre-sm80tiles-20261104`。
- 已知未做：`flashinfer 0.7.2.post1` / `torch 0.29/2.14` 的 pin 没补齐（盘只够一份 CUDA 栈）；
  `draft_lm_head_quantization` 与 `disable_lookahead_block_hashes` 未实测；那 85 ms 固定开销未归因。
- 本轮顺手修了一个坑：CT103 的 `/models` bind 被 01:20 的自动快照流程卸掉后没挂回，
  导致 unit 的 `[Unit] ConditionPathExists` 静默拒绝启动（详见另一篇文档 §4）。
