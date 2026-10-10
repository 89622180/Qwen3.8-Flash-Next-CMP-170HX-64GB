# Qwen3.8-Flash-Next 125B on CMP 170HX 64GB — PLE-SSD vLLM 调优实录

单张 NVIDIA CMP 170HX 64GB（sm_80）+ 一块 1TB NVMe，跑 klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP（125B MoE，6B 激活 + 51B n-gram 表）。
核心思路：**95.43 GiB 的 PLE/n-gram 表放 SSD（O_DIRECT 原生 Linux AIO，不进 RAM 也不进 ARC），主模型 47.05 GiB 常驻显存**。
本文记录 2026-10-10 一天之内的调优全过程：冷 prefill 从 ~3,550 → ~3,500+（32k）、**200k 长上下文从 2,264 → ~3,525 tok/s（+56 %）、TTFT 从 88.5 s → ~56.5 s**，以及每一个被否决的方案（诚实版）。

## 对比一览：作者原版 vs 本仓库最终

作者原版的公布值取自其部署仓库的 [`docs/RESULTS.md`](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx/blob/main/docs/RESULTS.md)（2026-09-19 测）；本仓库一列全部来自固定脚本 `bench.sh` / `decol.sh` 在 0.99 臂最后一轮的实测。

| 指标 | 作者原版公布值 | **本仓库最终实测** | 变化 |
|---|---:|---:|---|
| prefill 512 | 1,616 tok/s | 未测（脚本不含该长度） | — |
| prefill 2,048 | 1,915 tok/s | 未测 | — |
| prefill 8,192 | 2,636 tok/s | **3,211**（3 轮中位；3,203/3,211/2,092） | **+22 %** |
| prefill 32,768 | 2,132 tok/s | **3,536.0** | **+66 %** |
| prefill 131,072 | 未测（文档只到 32K） | **3,628.7**（36.12 s） | 长度扩展 |
| prefill 200,000 | 未测（原文：*"a full-length request was not tested"*） | **3,506.0**（TTFT 57.04 s） | 长度扩展 |
| **prefill 随长度的衰减** | 8K→32K **掉 19 %**（2,636→2,132） | 8K→32K→200K **全程不掉速**（3,211 / 3,536 / 3,506） | **长度轴被抹平** |
| C1 聚合 / 单流 decode | 103.30 / 110.99 | 117–143 / **124–153** | 聚合 +13~38 % · 单流 +12~38 % |
| C4 聚合 / 单流 | 277.22 / 77.95 | 303–316 / 86–90 | 聚合 +9~14 % · 单流 +10~15 % |
| C8 聚合 / 单流 | 428.97 / 60.63 | 464–531 / 68–77 | 聚合 +8~24 % · 单流 +12~27 % |
| C16 聚合 / 单流 | 667.13 / 47.32 | 716–745 / 54–56 | 聚合 +7~12 % · 单流 +14~18 % |
| TTFT C1 / C16 | 0.105 / 0.262 s | 0.07–0.08 / 0.17–0.18 s | −24~35 % |
| KV 池 / 显存占用 | 未公布 | **372,215 tok**；63,596 / 65,536 MiB（util 0.99，剩 1,316 MiB） | 这是本卡的**实际上限**：util=1.00 会被引擎的启动门禁拒绝（torch 只见 63.39 GiB、启动前驱动已占 0.30 GiB），详见 2.5 |
| MTP acceptance | 未公布（MTP=1） | 2.37–2.45 @ 69–73 %（第二 draft 位置命中 62 %） | — |

**两边配置差（这才是上表差距的来源）**：

| 项 | 作者原版 | 本仓库最终 |
|---|---|---|
| PLE 读 | 补丁内单线程 `ple_ssd_io.c` | 多线程 AIO（`patches/ple_ssd_io_mt.c`，`PLE_AIO_THREADS=32`） |
| 预读窗口 | `16K prompt read-ahead` | **全窗口**（2 行补丁，覆盖整条剩余 prompt） |
| chunk / MTP | 2048 / 1 | **3072 / 2** |
| 显存 util | 0.96 | **0.99** |
| CPU 绑核 | 无 | `CPUAffinity=0 3 4 6 8 10`（一个 P-core 一线程） |
| 存储 | 企业级 **SATA** SSD，ZFS 默认 recordsize | **NVMe + `recordsize=4K`** 专用子数据集 |
| RAM / ARC | 15 GiB RAM + 31 GiB swap，ARC 默认 | 62 GiB DDR4-3200 无 swap，`zfs_arc_max=8 GiB` |
| 功耗上限 | **180 W** | 300 W（= `power.max_limit`） |

> **读表须知：这张表不是同机 A/B。** 作者在 180 W + SATA + 15 GiB RAM 的参考机上测，我们在 300 W + NVMe(recordsize=4K) + 62 GiB RAM 的 CT120 上测——上表的一部分增益属于硬件（尤其 recordsize 与 SATA→NVMe），不属于代码。要剥离硬件看**纯改动收益**，看第 2 节那张同机同盘的当天基线表：**200k 2,264 → 3,506（+55 %）、TTFT 88.5 → 57.0 s**，每一步都是单变量 A/B。功耗那一行只是两边的客观差异：**本机实测 250 W→300 W 对这个工作负载的吞吐变化为零**（第 4 节），所以别把它当成可搬的优化项；显存/功耗上限怎么设由机主决定。

## 0. 来源与复现链（首版部署来源）

首版是按 [iIIusi0n/qwen3.8-flash-next-cmp170hx](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx) 这个部署仓库搭建的，跑 [klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP)（klee100 模型卡也指向同一个仓库）：

- 引擎 = **上游 vLLM @ `a5a30471ff2bb7f0824f2da10e358af98d304472`** + [iIIusi0n/qwen3.8-flash-next-cmp170hx](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx) 的 [`patches/qwen38-ple-ssd.patch`](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx/blob/main/patches/qwen38-ple-ssd.patch)（55,767 B / 13 个文件：PLE 元数据加载、异步行读/预读、CUDA graph host breaks、混合位宽 expert 覆盖、MTP 层重映射），安装指南在 [docs/GUIDE.md](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx/blob/main/docs/GUIDE.md)。
- 版本号 `0.29.1rc1.dev402+ga5a30471f.ple1` 的含义：上游 0.29.1rc1 之后 402 个提交（commit `a5a30471f`）+ ple-ssd 补丁的本地后缀 `.ple1`。
- 作者参考数字（180 W / 企业 SATA / MTP=1 / 2K chunk）：8k prefill 2,636 tok/s、单请求 decode 111 tok/s、16 路聚合 667 tok/s。本仓库的优化把它推到（300 W / NVMe / MTP=2 / 3072 chunk）：32k ~3,500、C1 decode 124–153、C16 聚合 716–745。
- **本仓库在其上叠加的改动**：全窗口预取补丁（`patches/ple_ssd-fullwindow.patch`，2 行）、多线程 AIO 读取器（`patches/ple_ssd_io_mt.c`，替代补丁里的单线程 `ple_ssd_io.c`）、`PLE_AIO_THREADS=32`、`QWEN_MTP=2`、chunk 3072、P-core 绑核、`bench.sh` 固定基准口径。

## 1. 背景：为什么这张卡能跑 125B

- **硬件**：PVE 9.2 LXC 容器（CT120），i5-14600K（6P+8E），62 GiB DDR4-3200 双通道，**CMP 170HX 64 GiB（compute 8.0，解锁后 300 W 上限）**，1 TB NVMe（ZFS `rpool`）。
- **模型**：`klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`，142.48 GiB = PLE 表 95.43 GiB（BF16，320 B/行，320,001,536 行）+ 主模型 47.05 GiB（AutoRound 3bpw，routed experts ≈2.84 bpw）。
- **引擎**：作者定制的 **vLLM `0.29.1rc1.dev402+ga5a30471f.ple1` fork**（`ple_ssd_offload` 钩子；PyPI 最新 0.31.0 没有这个钩子，"升级 vLLM" = rebase 补丁而不是 pip）。torch 2.13.0+cu130。
- **存储布局**：PLE 表放在专用 ZFS 子数据集上，**`recordsize=4K`**（父数据集 128K 会在 320 B 行上产生 ~400× 读放大——这是第一版部署只有 157 tok/s 的根因）；宿主机 `zfs_arc_max=8 GiB` 让 ZFS 元数据全命中。
- 显存账（最终 `--gpu-memory-utilization 0.99`）：预算 64,880 MiB = 权重 47.32 + 非 KV 开销 4.49 + **KV 10.53 GiB（372,215 tok）**，主机侧读回 63,596/65,536 MiB、剩 1,316 MiB。0.96 时是 304,144 tok（chunk 3072 臂）。

## 2. 优化实录（每一步都是单变量 A/B，同一固定脚本测量）

当天基线（250 W / AIO 线程 16 / MTP=1 / chunk 2048）：**32k 3,550 · 131k 2,349 · 200k 2,264 / TTFT 88.5 s**。

| # | 改动 | 200k prefill / TTFT | 结论 |
|---|---|---|---|
| 0 | 功耗 250→300 W（卡的上限） | 无变化 | prefill 在这台机器上**不是功耗指标**（GPU sm 均值 58.8 %、盘 util 66–68 %，双半忙，卡在 250 W 时本来就没被时钟卡住；实测峰值功耗 306–328 W、SM 1440 MHz、≤58 °C） |
| 1 | **全窗口预取补丁（2 行）** | 2,264 → 3,300 / 88.5 → 61.3 s | **+46 %，本日最大单笔收益** |
| 2 | `PLE_AIO_THREADS` 16→32 | 3,300 → 3,460 | +5 %（`.so` 读上限 61k→68k rows/s） |
| 3 | MTP `num_speculative_tokens` 1→2 | decode 单流 +4–10 % | acceptance 1.80→2.37–2.45（第二 draft 位置 62 % 命中）；KV 池反而 +9 % |
| 4 | chunk 2048→**3072** | 200k +1.7 %，C4 完整保住 | 均衡点（见 2.4 的取舍表） |
| 5 | `--gpu-memory-utilization` 0.96→**0.99** | 速度不变 | KV 池 304,144 → **372,215（+22 %）**，纯容量改动（见 2.5） |

最终（300 W / 32 线程 / MTP=2 / chunk 3072 / gpu-memory 0.99）：**32k ~3,500 · 131k ~3,450–3,520 · 200k ~3,525 / TTFT ~56.5 s · C4 聚合 3,535–3,649 · C1 decode 124–153 · KV 372,215 tok**。

### 2.1 全窗口预取补丁（`patches/ple_ssd-fullwindow.patch`）

**机制**：`PLEPromptPrefetcher.submit()` 只在 `Qwen4ExpModelState.add_request`（请求准入）时被调用**一次**，预取窗口被 `ple_ssd_prefetch_tokens`（默认 65,536 token）截断。短 prompt（≤65,536）整条被预取覆盖，demanded 读全是缓存命中——所以 32k 一直很快；**200k 的 prompt 只有前 32 % 被预取，剩下 68 % 每 chunk 裸付一次 demanded 读**（32,768 行 ≈ 500 ms，而它只能躲在 PLE 层前面 ~2 层的计算 ≈ 40 ms 后面）。这就是 200k 掉到 2,264、盘 util 只有 42 % 的全部原因。

**修法**（2 行，纯 Python，不用重编译）：

```diff
-        if min(self.limit, len(tokens) - start) < 2048:
+        if len(tokens) - start < 2048:
             return
...
-        tokens = tokens[start - history : start + self.limit]
+        tokens = tokens[start - history :]
```

预取速率 66k rows/s > 200k 消耗速率 36k rows/s（1.8× 余量），512 MiB 行缓存当滑动窗口用。改完后 **200k 的盘 util 42 % → 72.6 %，prefill 与 32k 同速——长度轴拉平**。`QWEN_SSD_PREFETCH` 这个 env 从此失效（`limit` 不再被使用）。

### 2.2 AIO 线程 16→32

ZFS 在提交线程里**内联完成** O_DIRECT AIO（iodepth 无效，7,447 IOPS @QD1 ≈ 7,584 @QD128），所以吞吐 = 线程数 × 单线程速率。harness 实测 `.so`：16 线程热态 61,033 rows/s、32 线程 68,041、48/64 线程饱和（68.4k/69.8k）。引擎 16→32：DEV 51k→54.8k r/s，131k +8.4 %、200k +5.3 %、32k +7.6 %。**48 线程预期 NULL，别浪费臂**。

### 2.3 MTP 1→2

acceptance 1.80→2.37–2.45（per-position 0.832 / 0.623——**第二 draft 位置 62 % 命中是 MTP=2 能赚的原因**）。单流 decode 全级别上行：C1 119–132→124–153、C4 82–84→86–90、C8 66–68→68–75、C16 50–52→54–56；聚合持平（每步多算一个 draft 的 FLOPs，赢在单流）。**KV 池 300,220 → 327,317（+9 %），白捡的容量**。MTP=3 按算术否决：第三位置命中率 ≈0.47，token/步 +5 % 换 verify FLOPs +10 %，净负。

### 2.4 chunk 3072：单流与并发的均衡点

chunk 4096 单流更好（131k +5.7 %、200k +8.5 %），但 **C4 聚合掉 25 %**（3,620 → 2,750）。机制：预取是单飞的，C4 时只有请求 1 拿到 read-ahead，请求 2–4 走 demanded 路径——4096 的每 chunk 读（65,536 行 ≈ 1.2 s）暴露量是 2048 的两倍；2048 的细粒度反而让 4 条流互相重叠得更好（C4 聚合甚至超过单流 32k 速率）。**悬崖在 3072 和 4096 之间**：

| | chunk 2048 | **chunk 3072** | chunk 4096 |
|---|---|---|---|
| 32k 单流 | 3,575 | 3,288–3,522 | 3,466–3,598 |
| 131k 单流 | 3,515 | 3,432–3,473 | 3,614–3,717 |
| 200k 单流 / TTFT | 3,467 / 57.7 s | **3,500–3,551 / 56.3–57.1 s** | 3,598–3,760 / 53–56 s |
| **C4 聚合** | 3,610–3,628 | **3,535–3,649** | 2,689–2,830 |
| KV 池 | 327,317 | 304,144 | 289,661 |

### 2.5 `--gpu-memory-utilization` 0.96→0.99：把显存吃干

目标是「再加 3 G KV」。实际预算只宽 **1,965 MiB**（62,915 → 64,880 MiB = 64 GiB × 0.99），所以落下来是 **+2 GiB 而不是 +3 GiB**：KV 池 304,144 → **372,215 tokens（+68,071，+22 %）**，日志 `Available KV cache memory: 10.53`。主机侧读回 `63,596 / 65,536 MiB used, 1,316 MiB free`，计算应用占 63,584 MiB——**63 G 这一档到了**。

速度在噪声内不变，200k 全程无 OOM：

| 指标 | 0.96（chunk 3072） | **0.99** |
|---|---|---|
| 32k 单流（3 轮） | 3,288–3,522 | 3,566 / 3,535 / 3,507（mean **3,536**） |
| 131k 单流 | 3,432–3,473 | **3,628.7**（36.12 s） |
| 200k 单流 / TTFT | 3,500–3,551 / 56.3–57.1 s | **3,506.0 / 57.04 s** |
| DEV | 54.8k r/s | 55.1k r/s、248,897 KiB/s、util 73.3 % |
| KV 池 | 304,144 | **372,215** |

**容量算术**：单条 262,144 ctx 流现在只占池的 70 %；128k 并发 = **2.84 条**（3×131,072 = 393,216 仍差 ~21k token），200k 并发 = 1.86 条。也就是说加这 2 GiB **仍然买不到第 3 条 128k 流**——0.96 时的结论「64 GiB 上 3×128k 是算术不可能」没有被推翻，只是从「差 8.8 万」缩到「差 2.1 万」。

**代价要说清**：剩 1,316 MiB 意味着这张卡上再起第二个进程已经无位置，且 CUDA 上下文的任何额外增长（更大 capture size、更长 ctx、KV 之外的临时缓冲）都会直接 OOM。这是「一台卡一个引擎、把显存全部换成 KV」的意图内取舍，不是漏配。

**为什么不能直接 1.00（实测被拒，报错原文在这里很有价值）**：`--gpu-memory-utilization` 的校验是 `Field(default=0.92, gt=0, le=1)`，1.0 是合法值，但引擎在 `init_device` 阶段按「启动时空闲显存」设门禁，CT120 上读到的是 **63.09 GiB free / 63.39 GiB total**（torch 视角的卡是 63.39 GiB，不是 nvidia-smi 的 65,536 MiB——差的 0.30 GiB 在进程起来之前就被驱动占了）。所以：

```
ValueError: Free memory on device cuda:0 (63.09/63.39 GiB) on startup is
less than desired GPU memory utilization (1.0, 63.39 GiB).
Decrease GPU memory utilization or reduce GPU memory used by other processes.
```

**结论：这台机器上 util 的上限受「启动瞬间的空闲显存」支配。** torch 视角总量 63.39 GiB，本次干净启动时空闲 63.09 GiB ⇒ 理论上最大可用 util 是 63.09/63.39 ≈ 0.9953；但 **0.995 这一臂没有测成**（被我的 A/B 驱动脚本在 145 s 误判为启动失败而回退，而本机正常 boot 需要 ~290 s），所以「0.995 能不能跑」目前是**未验证**，不要当结论用。已验证的只有：`0.99` 可跑、`1.00` 必被上面那条 ValueError 挡死。

**同一次误判回退还留下一个更值钱的观测**：脚本把 0.99 拉回来之后，引擎虽然 ready，但账目变了——`Free memory on device (62.8/63.39 GiB)`、`Available KV cache memory: 7.41 GiB`、`GPU KV cache size: 262,248 tokens`（对比干净启动的 63.09 / 10.53 GiB / 372,215）。空闲只差 0.29 GiB，KV 却少了 3.12 GiB，**说明有一次启动的「已消费显存」比另一次高约 2.8 GiB**，最可能是上一臂被 SIGKILL 后残留的 CUDA context 还没释放（systemd 当时确实报了 `Failed to kill control group … Input/output error`）。**含义：`util` 不是显存账的唯一自变量，重启前必须确认卡上没有残留 context，否则你会拿到一个「配置一样、KV 小 30 %」的引擎还以为改动生效了。** 这一轮随后在 131k prefill 处连接被拒（引擎掉线），我没有把它的任何数字写进上面的表。

顺带一条引擎自己打印的换算，解释为什么「0.99 看起来只买了 2 GiB」：

```
CUDA graph memory profiling is enabled (default since v0.21.0).
The current --gpu-memory-utilization=0.9900 is equivalent to
--gpu-memory-utilization=0.9575 without CUDA graph memory profiling.
```

即 vLLM 现在会把 CUDA graph 的显存（实测 1.36 GiB，估算 2.06 GiB）**从 KV 预算里扣掉**，所以名义 util 和实际 KV 不是线性关系；想把这部分要回来，可用的开关是 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`，或直接给 `--kv-cache-memory-bytes`（该值一旦设定就让 `gpu_memory_utilization` 失效，见 `config/cache.py:238-242`）。**这两个组合本仓库没有测过，属于「机制存在、收益未证」，别当成已验证的优化项照抄。**

**引擎自己报的并发口径（journal 原文，可直接引用）**：`GPU KV cache size: 372,215 tokens, Maximum concurrency for 262,144 tokens per request: 1.42x`。


## 3. 深度速度基准（最终配置，全部来自固定脚本 `bench.sh` / `decol.sh` / `c4prefill.sh`）

**Prefill（冷，随机 token，无前缀复用）**：32k ~3,500 · 131k ~3,450–3,520 · 200k ~3,525（TTFT 56.3–57.1 s）。设备侧：54–58k r/s、~250 MiB/s、util 71–75 %、r_await ~0.1 ms。0.99 臂（最后一轮，`gpu099-ab.log`）：32k **3,536.0** / 131k **3,628.7** / 200k **3,506.0**（TTFT 57.04 s）/ DEV 55.1k r/s、util 73.3 %——与 0.96 区间重合。同一轮里 8192 长度第三次跑出一个 2,091.9 的离群值（另两次 3,202.8 / 3,211.0），短 prompt 对瞬时干扰敏感，别拿单次 8K 数字下结论。

**长上下文 decode（200k prompt，`bench.sh` PART3）**：128 tok 窗口 138.8 tok/s、256 tok 窗口 177.3 tok/s（窗口越短越被 MTP 的整步颗粒抬高，别把两者当同一指标）。

**Decode（128 输出 token × 3 轮，短 prompt）**——下表测于 `gpu-memory 0.96` 窗口，0.99 后未重跑并发梯度（0.99 只改 KV 预算，不动计算路径）：

| 并发 | 聚合 tok/s | 单流 tok/s | median TTFT |
|---|---|---|---|
| C1 | 117–143 | 124–153 | 0.07–0.08 s |
| C4 | 303–316 | 86–90 | 0.08–0.11 s |
| C8 | 464–531 | 68–77 | 0.11–0.16 s |
| C16 | 716–745 | 54–56 | 0.17–0.18 s |

MTP acceptance 2.37–2.45 @ 69–73 %。decode 阶段盘 util 4 %——完全不碰盘。

**C4 prefill（4 路并发 32k）**：聚合 3,535–3,649 tok/s = 单流速率（并发不加分也不掉分）；4 条流按 FIFO 位置错开完成（~9/18/27/36 s），第 4 条的 TTFT ≈ 36 s——这是 chunk 预算 FIFO 分配的算术，不是队列长度限制（`max_num_seqs=16`，4 条全在 Running）。

**每 token 磁盘成本**：15.3–16.2 ops、~70 KiB（逻辑需求 5 KiB/token；13.8× 是 320 B 行对 4K record 的对齐浪费——盘在 util 72 % 时的墙是每 op 的提交/串行延迟，不是带宽）。

## 4. 加速空间结论（诚实版）

| 尝试 | 结果 | 处置 |
|---|---|---|
| 功耗 250→300 W | 吞吐零变化（卡没被时钟卡住） | 保留（温度 ≤58 °C 无代价），但**别指望它提速** |
| `--stream-interval 10` | C8 +3–10 %，**C16 聚合 −18 %**（转发延迟/背压吃掉整轮） | 回退，保持默认 1 |
| `_read` 向量化缓存（numpy 槽位表） | 逐字节等价、Python 开销 45→30 ms/批，但 131k +2.7 %、200k ±0、C4 ±0 | 回退（84 行自定义代码不值 2 %） |
| MTP=3 | 第三位置命中率 ≈0.47 ⇒ 净负 | 按算术否决，未浪费臂 |
| AIO 线程 48/64 | harness 饱和于 68–70k rows/s | NULL |
| 行缓存 2048 MiB | 命中率 ≈ 容量占比（行分布近均匀） | NULL |
| `QWEN_SSD_PREFETCH=131072` | 32k 本来就全覆盖 | NULL（全窗口补丁后该 env 已失效） |
| ARC 4 GiB vs 8 GiB | 9 轮交错 A/B 完全相同 | 8 GiB（留余量） |
| chunk 8192 | −19 %（重叠被粗化） | 否决 |

## 5. 踩坑清单

1. **ZFS 内联完成 O_DIRECT AIO**：iodepth 完全无效，只有线程数扩吞吐。别调 `ple_ssd_io_depth`。
2. **recordsize 必须 4K**：128K 记录在 320 B 行上是 ~400× 读放大（第一版 157 tok/s 的根因）；同时 ARC 要够大（8 GiB）让 ZFS 元数据全命中。
3. **测量只用一个固定脚本**（`bench.sh`）：每次换脚本数字就不可比。A/B 驱动可以重启引擎，但测量一律走固定脚本。
4. **同盘并发租户会毒化基准**：一次 333 MiB/s 的顺序 `cat`（另一个会话在搬模型）让 C4 掉 15 %、200k 掉 17 %。测前查 `zpool iostat rpool 3 2` + `pgrep -af 'cat model'`；**loadavg 把 D 状态 I/O 任务也算进去**，5/15 分钟均值高说明刚有过 I/O 风暴。
5. **`/proc/<pid>/environ` 不能证明 env 到达了 EngineCore**（vLLM 会 re-exec 重建 envp）；用实际速率证明（39.8k ops/s > 7.1k 单线程天花板）。
6. 裸 shell 探测要带 unit 的 `LD_LIBRARY_PATH`（`/usr/local/nvidia/lib64:…/nvidia/cu13/lib:/usr/local/cuda/lib64`），否则 torch 报 "0 active drivers" / "Failed to infer device type"。
7. **等价性测试要把新旧实现隔离在不同进程**：同进程跑两个表会产生假回归（一次 1.9× 的假回退就是这么来的）。
8. `.so` 的 `rows_read` 一次只允许一个调用（generation 计数器机制）——想加并发只能加线程或加大单次调用。
9. `pct snapshot` 对带 `mp0:` 挂载的 LXC 结构性拒绝（"snapshot feature is not available"）。
10. 引擎自己的负载会把 loadavg 推到 7–28——**判断窗口干净要看外部 I/O（zpool 写入、未知进程），不是 loadavg 绝对值**。
11. **`systemctl is-active` 要在容器里问**：同一 unit 名在 PVE 宿主上查会返回 `inactive`（unit 在 CT120 的 /etc 里，宿主 systemd 不认识），曾据此误判引擎掉了。用宿主侧 `nvidia-smi --query-compute-apps` 有 63,584 MiB 占用来反证引擎在跑，再进容器核对 MainPID==`:8000` 持有者。
12. **想要 "+N GiB KV" 要按 `util × 64 GiB` 的差算，不是按百分比**：0.96→0.99 只把预算从 62,915 抬到 64,880 MiB（+1,965 MiB），所以 KV 最多加 ~2 GiB；0.96 出发的理论上限是 (1.00−0.96)×64 = 2.56 GiB，而**实测能用的最高一档是 0.99**（1.00 被启动门禁拒绝，0.995 未验证——见 2.5），所以「再加 3 GiB KV」在这张卡上没有对应设置。
13. **A/B 驱动的「启动失败」判定必须比引擎的真实 boot 时间长，且不能只看一次 `is-failed`**：本机 boot ≈290 s，我第一版驱动在 145 s 就误判失败并触发回退，把 0.995 那一臂白白废掉（结论变成未测）。正确写法：`systemctl reset-failed` 后轮询「MainPID 为空 **且** `is-failed=failed` 连续 3 次」，期限给到 800 s；SIGKILL 掉的引擎还会让 systemd 报 `Failed to kill control group … Input/output error`，紧接着重启就会读到残留 context 占的显存（见 2.5 里那次 KV 缩水 3.12 GiB 的观测）。

## 6. 文件说明与复现

```
├── README.md                        # 本文
├── patches/
│   ├── ple_ssd-fullwindow.patch     # 全窗口预取补丁（2 行，本日核心收益 +46 % @200k）
│   ├── ple_ssd.py                   # 打好补丁的完整文件（直接覆盖用）
│   └── ple_ssd_io_mt.c              # 多线程 AIO 读取器源码（PLE_AIO_THREADS 控制）
├── configs/
│   ├── perf.conf                    # systemd drop-in（最终配置）
│   └── serve.sh                     # 引擎启动脚本（全 env 参数化）
└── scripts/
    ├── bench.sh                     # 唯一基准脚本 v1（身份门禁 + 3×32k + 131k/200k + 200k decode）
    ├── c4prefill.sh                 # C4 并发 prefill（4×32k）
    ├── decol.sh                     # decode/并发梯度（C1/4/8/16）
    └── mttest.py                    # .so 字节级等价性 + 速率 harness
```

**复现步骤**：

```bash
# 0. 基础引擎（按作者指南，见第 0 节的链接）
git clone https://github.com/vllm-project/vllm && cd vllm
git checkout a5a30471ff2bb7f0824f2da10e358af98d304472
curl -L -o /tmp/qwen38-ple-ssd.patch \
  https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx/raw/main/patches/qwen38-ple-ssd.patch
git apply /tmp/qwen38-ple-ssd.patch
# 按 docs/GUIDE.md 建 venv、编译 vllm 与原版 ple_ssd_io.so

# 1. 打本仓库的全窗口补丁（或直接 cp patches/ple_ssd.py 覆盖）
cd /root/vllm/src/vllm/models/qwen4_exp/nvidia/
cp ple_ssd.py ple_ssd.py.orig
patch -p1 < ple_ssd-fullwindow.patch   # 从仓库根目录: patch -p1 -d /root/vllm < patches/ple_ssd-fullwindow.patch

# 2. 编译多线程 AIO 读取器
cc -O3 -shared -fPIC -Wall -Wextra -Werror -pthread \
   patches/ple_ssd_io_mt.c -o /root/vllm/optimization/ple_ssd_io.so

# 3. drop-in（路径按你的 unit 名改）
cp configs/perf.conf /etc/systemd/system/<你的unit>.service.d/perf.conf
systemctl daemon-reload && systemctl restart <你的unit>

# 4. 基准（唯一口径）
bash scripts/bench.sh        # P1 32k×3 / P2 131k+200k / P3 200k decode 128/256
bash scripts/c4prefill.sh    # C4 并发 prefill
bash scripts/decol.sh        # decode 梯度
```

`serve.sh` 的可调 env：`QWEN_MTP`（speculative tokens）、`QWEN_BATCH_TOKENS`（chunk）、`QWEN_CAPTURE_SIZES`（**必须包含 chunk 值**，否则 prefill 掉出 full-graph 覆盖）、`QWEN_SEQS`、`QWEN_SSD_*`。`PLE_AIO_THREADS` 是 `.so` 的 env（`rows_open` 时读取，改了要重启）。

## 7. 环境版本

- vLLM `0.29.1rc1.dev402+ga5a30471f.ple1`（作者 fork，`/root/vllm/src/vllm` 源码树安装）
- torch 2.13.0+cu130 · CUDA 13 · nvidia-open 驱动（cmpunlocker 解锁 300 W）
- ZFS（`recordsize=4K` PLE 数据集、`zfs_arc_max=8589934592` 经 `/etc/tmpfiles.d/zfs-arc.conf` 持久化）
- 模型 `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`（13 个 safetensors，PLE 表在 `ple-dedicated-*` 分片）
- PVE 9.2 / LXC CT120 / i5-14600K / 62 GiB DDR4 / CMP 170HX 64 GiB

## License

笔记与脚本 MIT。vLLM（Apache-2.0）、模型权重与作者 fork 的许可证各自适用。
