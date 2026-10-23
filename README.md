# Qwen3.8-Flash-Next 125B on CMP 170HX 64GB — PLE-SSD vLLM 调优实录

单张 NVIDIA CMP 170HX 64GB（sm_80）+ 一块 1TB NVMe，跑 klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP（125B MoE，6B 激活 + 51B n-gram 表）。
核心思路：**95.43 GiB 的 PLE/n-gram 表放 SSD（O_DIRECT 原生 Linux AIO，不进 RAM 也不进 ARC），主模型 47.05 GiB 常驻显存**。
本文记录 2026-10-10 一天之内的调优全过程：冷 prefill 从 ~3,550 → ~3,500+（32k）、**200k 长上下文从 2,264 → ~3,525 tok/s（+56 %）、TTFT 从 88.5 s → ~56.5 s**，以及每一个被否决的方案（诚实版）。
2026-10-14 第二轮（chunk 上探 8K/16K、显存天花板 0.983、同日四配置 A/B）见第 8 节——**最终配置改为 chunk 2048 / util 0.99**（同日全指标最优，KV 池 393,940）。
2026-10-22/23 第三轮（PLE 行缓存大小矩阵、预取 limit 公式 90% 放宽、ARC 降 2G、recordsize 论证、util 0.995）见第 9 节——**最终配置 cache 2048 MiB / prefetch 262144 / util 0.995（KV 池 405,526）**。

## 对比一览：作者原版 vs 本仓库最终

作者原版的公布值取自其部署仓库的 [`docs/RESULTS.md`](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx/blob/main/docs/RESULTS.md)（2026-09-19 测）；本仓库一列全部来自固定脚本 `bench.sh` / `decol.sh` 在 0.99 臂最后一轮的实测。

| 指标 | 作者原版公布值 | **首轮（10-10）实测** | 变化 |
|---|---:|---:|---|
| prefill 512 | 1,616 tok/s | 1,386（10-14 补测，3 轮 1,700/1,237/1,222，首轮离群） | −14 %（短 prompt 固定开销占比大，见 8.7 漂移） |
| prefill 2,048 | 1,915 tok/s | 2,138（10-14 补测，3 轮 2,176/2,100/2,138） | **+12 %** |
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
| KV 池 / 显存占用 | 未公布 | **372,215 tok**（首轮 0.99 臂）；第三轮 0.995 臂 **405,526**（11.47 GiB，1.55× 262k） | 0.9953 才是**实际上限**：1.00 被启动门禁拒绝、0.997 启动失败（torch 只见 63.39 GiB、启动前驱动占 0.30 GiB），详见 2.5 / 9.7 |
| MTP acceptance | 未公布（MTP=1） | 2.37–2.45 @ 69–73 %（第二 draft 位置命中 62 %） | — |

> **2026-10-14 复测更新**：同脚本同配置（3072+0.99）复测得 8k 2,782 / 32k 2,924 / 131k 3,018 / 200k 2,968 / C4 2,955——比 10-10 整体低 ~15 %，是**跨天环境漂移**（见 8.7），不是配置回退。第二轮同日四配置 A/B（8.4）后**最终配置改为 chunk 2048 + util 0.99**（同日全指标最优、KV 池 393,940）。上表 10-10 数字保留作历史记录；跨天绝对数字不可比，配置结论只认同日 A/B。
>
> **2026-10-23 第三轮更新（第 9 节）**：最终配置再改为 **util 0.995（KV 池 405,526，0.997 会被启动门禁拒绝）+ PLE 行缓存 2048 MiB + 预取 262144 + limit 90 % 公式 + ARC 2 GiB**。本轮 200k 实测 ~70 s / ~2,900 tok/s（80k ~27 s、262k ~90 s），绝对值继续落在 8.7 的跨天漂移带内；配置结论同样只来自本轮同日 A/B。

**两边配置差（这才是上表差距的来源）**：

| 项 | 作者原版 | 本仓库最终 |
|---|---|---|
| PLE 读 | 补丁内单线程 `ple_ssd_io.c` | 多线程 AIO（`patches/ple_ssd_io_mt.c`，`PLE_AIO_THREADS=32`） |
| 预读窗口 | `16K prompt read-ahead` | **全窗口**（2 行补丁；第三轮起受 90 % 行缓存上限约束，见 9.3） |
| chunk / MTP | 2048 / 1 | **2048 / 2**（10-14 同日 A/B 后回到 2048，见 8.4） |
| 显存 util | 0.96 | **0.99** |
| CPU 绑核 | 无 | `CPUAffinity=0 3 4 6 8 10`（一个 P-core 一线程） |
| 存储 | 企业级 **SATA** SSD，ZFS 默认 recordsize | **NVMe + `recordsize=4K`** 专用子数据集 |
| RAM / ARC | 15 GiB RAM + 31 GiB swap，ARC 默认 | 62 GiB DDR4-3200 无 swap，`zfs_arc_max=2 GiB`（第三轮从 8 GiB 下调，见 9.4） |
| 功耗上限 | **180 W** | 300 W（= `power.max_limit`） |

> **读表须知：这张表不是同机 A/B。** 作者在 180 W + SATA + 15 GiB RAM 的参考机上测，我们在 300 W + NVMe(recordsize=4K) + 62 GiB RAM 的 CT120 上测——上表的一部分增益属于硬件（尤其 recordsize 与 SATA→NVMe），不属于代码。要剥离硬件看**纯改动收益**，看第 2 节那张同机同盘的当天基线表：**200k 2,264 → 3,506（+55 %）、TTFT 88.5 → 57.0 s**，每一步都是单变量 A/B。功耗那一行只是两边的客观差异：**本机实测 250 W→300 W 对这个工作负载的吞吐变化为零**（第 4 节），所以别把它当成可搬的优化项；显存/功耗上限怎么设由机主决定。

## 0. 来源与复现链（首版部署来源）

首版是按 [iIIusi0n/qwen3.8-flash-next-cmp170hx](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx) 这个部署仓库搭建的，跑 [klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP)（klee100 模型卡也指向同一个仓库）：

- 引擎 = **上游 vLLM @ `a5a30471ff2bb7f0824f2da10e358af98d304472`** + [iIIusi0n/qwen3.8-flash-next-cmp170hx](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx) 的 [`patches/qwen38-ple-ssd.patch`](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx/blob/main/patches/qwen38-ple-ssd.patch)（55,767 B / 13 个文件：PLE 元数据加载、异步行读/预读、CUDA graph host breaks、混合位宽 expert 覆盖、MTP 层重映射），安装指南在 [docs/GUIDE.md](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx/blob/main/docs/GUIDE.md)。
- 版本号 `0.29.1rc1.dev402+ga5a30471f.ple1` 的含义：上游 0.29.1rc1 之后 402 个提交（commit `a5a30471f`）+ ple-ssd 补丁的本地后缀 `.ple1`。
- 作者参考数字（180 W / 企业 SATA / MTP=1 / 2K chunk）：8k prefill 2,636 tok/s、单请求 decode 111 tok/s、16 路聚合 667 tok/s。本仓库的优化把它推到（300 W / NVMe / MTP=2 / 2048 chunk）：32k ~3,500、C1 decode 124–153、C16 聚合 716–745。
- **本仓库在其上叠加的改动**：全窗口预取补丁（`patches/ple_ssd-fullwindow.patch`，2 行）、多线程 AIO 读取器（`patches/ple_ssd_io_mt.c`，替代补丁里的单线程 `ple_ssd_io.c`）、`PLE_AIO_THREADS=32`、`QWEN_MTP=2`、chunk 2048、P-core 绑核、`bench.sh` 固定基准口径。

## 1. 背景：为什么这张卡能跑 125B

- **硬件**：PVE 9.2 LXC 容器（CT120），i5-14600K（6P+8E），62 GiB DDR4-3200 双通道，**CMP 170HX 64 GiB（compute 8.0，解锁后 300 W 上限）**，1 TB NVMe（ZFS `rpool`）。
- **模型**：`klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`，142.48 GiB = PLE 表 95.43 GiB（BF16，320 B/行，320,001,536 行）+ 主模型 47.05 GiB（AutoRound 3bpw，routed experts ≈2.84 bpw）。
- **引擎**：作者定制的 **vLLM `0.29.1rc1.dev402+ga5a30471f.ple1` fork**（`ple_ssd_offload` 钩子；PyPI 最新 0.31.0 没有这个钩子，"升级 vLLM" = rebase 补丁而不是 pip）。torch 2.13.0+cu130。
- **存储布局**：PLE 表放在专用 ZFS 子数据集上，**`recordsize=4K`**（父数据集 128K 会在 320 B 行上产生 ~400× 读放大——这是第一版部署只有 157 tok/s 的根因）；宿主机 `zfs_arc_max` 原设 8 GiB（第三轮实测 2G 无差异已降至 2 GiB，见 9.4）。
- 显存账（第三轮最终 `--gpu-memory-utilization 0.995`，chunk 2048）：预算 64,880 → 63,158 MiB·util = **KV 池 405,526 tok**（11.47 GiB，1.55× 262k，见 9.7）；0.99 臂此前是 393,940 tok、0.96 时 304,144 tok（chunk 越小图池估算越小、KV 越大，见 8.1 的机制）。

## 2. 优化实录（每一步都是单变量 A/B，同一固定脚本测量）

当天基线（250 W / AIO 线程 16 / MTP=1 / chunk 2048）：**32k 3,550 · 131k 2,349 · 200k 2,264 / TTFT 88.5 s**。

| # | 改动 | 200k prefill / TTFT | 结论 |
|---|---|---|---|
| 0 | 功耗 250→300 W（卡的上限） | 无变化 | prefill 在这台机器上**不是功耗指标**（GPU sm 均值 58.8 %、盘 util 66–68 %，双半忙，卡在 250 W 时本来就没被时钟卡住；实测峰值功耗 306–328 W、SM 1440 MHz、≤58 °C） |
| 1 | **全窗口预取补丁（2 行）** | 2,264 → 3,300 / 88.5 → 61.3 s | **+46 %，本日最大单笔收益** |
| 2 | `PLE_AIO_THREADS` 16→32 | 3,300 → 3,460 | +5 %（`.so` 读上限 61k→68k rows/s） |
| 3 | MTP `num_speculative_tokens` 1→2 | decode 单流 +4–10 % | acceptance 1.80→2.37–2.45（第二 draft 位置 62 % 命中）；KV 池反而 +9 % |
| 4 | chunk 2048→**3072** | 200k +1.7 %，C4 完整保住 | 均衡点（见 2.4 的取舍表；**10-14 同日 A/B 又改回 2048**，见 8.4） |
| 5 | `--gpu-memory-utilization` 0.96→**0.99** | 速度不变 | KV 池 304,144 → **372,215（+22 %）**，纯容量改动（见 2.5） |

第一轮最终（300 W / 32 线程 / MTP=2 / chunk 2048 / gpu-memory 0.99）：**32k ~3,500 · 131k ~3,450–3,520 · 200k ~3,525 / TTFT ~56.5 s · C4 聚合 3,535–3,649 · C1 decode 124–153 · KV 393,940 tok**（10-10 历史数字；10-14 口径见 8.4；**第三轮最终配置与数字见 9.9**——cache 2048 MiB / prefetch 262144 / util 0.995，200k ~70 s）。

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

> **10-23 更新（见 9.3）**：第三轮给 `submit()` 加回了 `tokens[: history + self.limit]` 截断——`limit` 重新生效，但公式从「行缓存 50 %」放宽到 **90 %**，`QWEN_SSD_PREFETCH=262144` 重新有意义（全窗口补丁 + `patches/ple_ssd-limit90.patch` = 最终形态）。

### 2.2 AIO 线程 16→32

ZFS 在提交线程里**内联完成** O_DIRECT AIO（iodepth 无效，7,447 IOPS @QD1 ≈ 7,584 @QD128），所以吞吐 = 线程数 × 单线程速率。harness 实测 `.so`：16 线程热态 61,033 rows/s、32 线程 68,041、48/64 线程饱和（68.4k/69.8k）。引擎 16→32：DEV 51k→54.8k r/s，131k +8.4 %、200k +5.3 %、32k +7.6 %。**48 线程预期 NULL，别浪费臂**。

### 2.3 MTP 1→2

acceptance 1.80→2.37–2.45（per-position 0.832 / 0.623——**第二 draft 位置 62 % 命中是 MTP=2 能赚的原因**）。单流 decode 全级别上行：C1 119–132→124–153、C4 82–84→86–90、C8 66–68→68–75、C16 50–52→54–56；聚合持平（每步多算一个 draft 的 FLOPs，赢在单流）。**KV 池 300,220 → 327,317（+9 %），白捡的容量**。MTP=3 按算术否决：第三位置命中率 ≈0.47，token/步 +5 % 换 verify FLOPs +10 %，净负。

### 2.4 chunk 的取舍：单流与并发的均衡点

chunk 4096 单流更好（131k +5.7 %、200k +8.5 %），但 **C4 聚合掉 25 %**（3,620 → 2,750）。机制：预取是单飞的，C4 时只有请求 1 拿到 read-ahead，请求 2–4 走 demanded 路径——4096 的每 chunk 读（65,536 行 ≈ 1.2 s）暴露量是 2048 的两倍；2048 的细粒度反而让 4 条流互相重叠得更好（C4 聚合甚至超过单流 32k 速率）。**悬崖在 3072 和 4096 之间**：

| | chunk 2048 | **chunk 3072** | chunk 4096 |
|---|---|---|---|
| 32k 单流 | 3,575 | 3,288–3,522 | 3,466–3,598 |
| 131k 单流 | 3,515 | 3,432–3,473 | 3,614–3,717 |
| 200k 单流 / TTFT | 3,467 / 57.7 s | **3,500–3,551 / 56.3–57.1 s** | 3,598–3,760 / 53–56 s |
| **C4 聚合** | 3,610–3,628 | **3,535–3,649** | 2,689–2,830 |
| KV 池 | 327,317 | 304,144 | 289,661 |

> **10-14 更新**：同日四配置 A/B（8.4）显示 2048 在 8k/131k/200k/C4/KV 五轴全优、4096 只有 32k 赢——**均衡点移回 2048**（10-10 那轮 3072 与 2048 本就在噪声内，见上表）。8192 在固定口径下短中长度大幅回退，正式否决（第 4 节 10-10 已记 −19 %，10-14 同日复测 −14~−32 %）。

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

**容量算术**：单条 262,144 ctx 流现在只占池的 70 %；128k 并发 = **2.84 条**（3×131,072 = 393,216 仍差 ~21k token），200k 并发 = 1.86 条。也就是说加这 2 GiB **仍然买不到第 3 条 128k 流**——0.96 时的结论「64 GiB 上 3×128k 是算术不可能」没有被推翻，只是从「差 8.8 万」缩到「差 2.1 万」。（10-14 更新：chunk 2048 臂的 KV 池是 **393,940**，3×131,072 = 393,216 **刚好装下 3 条 128k 流**——chunk 缩小把图池估算压小、KV 池变大（8.1 机制），这是 2048 的另一个白捡收益。**10-23 第三轮（见 9.7）：0.995 臂 KV 池 405,526，3 条 128k 流装下后还余 ~12.3k token，且 262k 并发提升到 1.55×。**）

**代价要说清**：剩 1,316 MiB 意味着这张卡上再起第二个进程已经无位置，且 CUDA 上下文的任何额外增长（更大 capture size、更长 ctx、KV 之外的临时缓冲）都会直接 OOM。这是「一台卡一个引擎、把显存全部换成 KV」的意图内取舍，不是漏配。

**为什么不能直接 1.00（实测被拒，报错原文在这里很有价值）**：`--gpu-memory-utilization` 的校验是 `Field(default=0.92, gt=0, le=1)`，1.0 是合法值，但引擎在 `init_device` 阶段按「启动时空闲显存」设门禁，CT120 上读到的是 **63.09 GiB free / 63.39 GiB total**（torch 视角的卡是 63.39 GiB，不是 nvidia-smi 的 65,536 MiB——差的 0.30 GiB 在进程起来之前就被驱动占了）。所以：

```
ValueError: Free memory on device cuda:0 (63.09/63.39 GiB) on startup is
less than desired GPU memory utilization (1.0, 63.39 GiB).
Decrease GPU memory utilization or reduce GPU memory used by other processes.
```

**结论：这台机器上 util 的上限受「启动瞬间的空闲显存」支配。** torch 视角总量 63.39 GiB，干净启动时空闲 63.09 GiB ⇒ 最大可用 util 是 63.09/63.39 ≈ **0.9953**。

> **10-23 第三轮实测（见 9.7）修正此前的“未验证”结论**：**0.995 已实测可跑**（KV 池 405,526，200K/262K prefill 无回归），**0.997 被启动门禁拒绝**——上限 0.9953 坐实。前一轮的“0.995 未验证”源于 A/B 驱动脚本在 145 s 误判启动失败而回退（本机正常 boot 需 ~290 s），属测量脚本失误，不是真的不能跑。

**同一次误判回退还留下一个更值钱的观测**：脚本把 0.99 拉回来之后，引擎虽然 ready，但账目变了——`Free memory on device (62.8/63.39 GiB)`、`Available KV cache memory: 7.41 GiB`、`GPU KV cache size: 262,248 tokens`（对比干净启动的 63.09 / 10.53 GiB / 372,215）。空闲只差 0.29 GiB，KV 却少了 3.12 GiB，**说明有一次启动的「已消费显存」比另一次高约 2.8 GiB**，最可能是上一臂被 SIGKILL 后残留的 CUDA context 还没释放（systemd 当时确实报了 `Failed to kill control group … Input/output error`）。**含义：`util` 不是显存账的唯一自变量，重启前必须确认卡上没有残留 context，否则你会拿到一个「配置一样、KV 小 30 %」的引擎还以为改动生效了。** 这一轮随后在 131k prefill 处连接被拒（引擎掉线），我没有把它的任何数字写进上面的表。

顺带一条引擎自己打印的换算，解释为什么「0.99 看起来只买了 2 GiB」：

```
CUDA graph memory profiling is enabled (default since v0.21.0).
The current --gpu-memory-utilization=0.9900 is equivalent to
--gpu-memory-utilization=0.9575 without CUDA graph memory profiling.
```

即 vLLM 现在会把 CUDA graph 的显存（实测 1.36 GiB，估算 2.06 GiB）**从 KV 预算里扣掉**，所以名义 util 和实际 KV 不是线性关系；想把这部分要回来，可用的开关是 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`，或直接给 `--kv-cache-memory-bytes`（该值一旦设定就让 `gpu_memory_utilization` 失效，见 `config/cache.py:238-242`）。**这两个组合本仓库没有测过，属于「机制存在、收益未证」，别当成已验证的优化项照抄。**

**引擎自己报的并发口径（journal 原文，可直接引用）**：`GPU KV cache size: 372,215 tokens, Maximum concurrency for 262,144 tokens per request: 1.42x`。


## 3. 深度速度基准（第一/二轮最终配置的实测，全部来自固定脚本 `bench.sh` / `decol.sh` / `c4prefill.sh`；第三轮口径见第 9 节）

**Prefill（冷，随机 token，无前缀复用）**：32k ~3,500 · 131k ~3,450–3,520 · 200k ~3,525（TTFT 56.3–57.1 s）。设备侧：54–58k r/s、~250 MiB/s、util 71–75 %、r_await ~0.1 ms。0.99 臂（10-10 最后一轮，`gpu099-ab.log`）：32k **3,536.0** / 131k **3,628.7** / 200k **3,506.0**（TTFT 57.04 s）/ DEV 55.1k r/s、util 73.3 %——与 0.96 区间重合。同一轮里 8192 长度第三次跑出一个 2,091.9 的离群值（另两次 3,202.8 / 3,211.0），短 prompt 对瞬时干扰敏感，别拿单次 8K 数字下结论。

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
| ARC 2–8 GiB | 9 轮交错 A/B 完全相同 | **2 GiB**（第三轮实测 8G/4G/2G 无差异后降，见 9.4） |
| chunk 8192 | −19 %（重叠被粗化） | 否决（10-14 同日复测 −14~−32 %，见 8.4） |
| chunk 16384 | 三种组合全灭（8.2） | 否决 |
| cumem allocator | 被 fork 时序 bug 挡死，单独启用零收益（8.3） | 回退 |

## 5. 踩坑清单

1. **ZFS 内联完成 O_DIRECT AIO**：iodepth 完全无效，只有线程数扩吞吐。别调 `ple_ssd_io_depth`。
2. **recordsize 必须 4K**：128K 记录在 320 B 行上是 ~400× 读放大（第一版 157 tok/s 的根因）。ARC 此前认为要够大（8 GiB）让 ZFS 元数据全命中——**第三轮实测 8G/4G/2G 无差异，已降 2 GiB**（9.4）。
3. **测量只用一个固定脚本**（`bench.sh`）：每次换脚本数字就不可比。A/B 驱动可以重启引擎，但测量一律走固定脚本。
4. **同盘并发租户会毒化基准**：一次 333 MiB/s 的顺序 `cat`（另一个会话在搬模型）让 C4 掉 15 %、200k 掉 17 %。测前查 `zpool iostat rpool 3 2` + `pgrep -af 'cat model'`；**loadavg 把 D 状态 I/O 任务也算进去**，5/15 分钟均值高说明刚有过 I/O 风暴。
5. **`/proc/<pid>/environ` 不能证明 env 到达了 EngineCore**（vLLM 会 re-exec 重建 envp）；用实际速率证明（39.8k ops/s > 7.1k 单线程天花板）。
6. 裸 shell 探测要带 unit 的 `LD_LIBRARY_PATH`（`/usr/local/nvidia/lib64:…/nvidia/cu13/lib:/usr/local/cuda/lib64`），否则 torch 报 "0 active drivers" / "Failed to infer device type"。
7. **等价性测试要把新旧实现隔离在不同进程**：同进程跑两个表会产生假回归（一次 1.9× 的假回退就是这么来的）。
8. `.so` 的 `rows_read` 一次只允许一个调用（generation 计数器机制）——想加并发只能加线程或加大单次调用。
9. `pct snapshot` 对带 `mp0:` 挂载的 LXC 结构性拒绝（"snapshot feature is not available"）。
10. 引擎自己的负载会把 loadavg 推到 7–28——**判断窗口干净要看外部 I/O（zpool 写入、未知进程），不是 loadavg 绝对值**。
11. **`systemctl is-active` 要在容器里问**：同一 unit 名在 PVE 宿主上查会返回 `inactive`（unit 在 CT120 的 /etc 里，宿主 systemd 不认识），曾据此误判引擎掉了。用宿主侧 `nvidia-smi --query-compute-apps` 有 63,584 MiB 占用来反证引擎在跑，再进容器核对 MainPID==`:8000` 持有者。
12. **想要 "+N GiB KV" 要按 `util × 64 GiB` 的差算，不是按百分比**：0.96→0.99 只把预算从 62,915 抬到 64,880 MiB（+1,965 MiB），所以 KV 最多加 ~2 GiB；0.96 出发的理论上限是 (1.00−0.96)×64 = 2.56 GiB。**实测能用的最高一档是 0.995**（0.997 被启动门禁拒绝，见 2.5 / 9.7；第三轮修正了此前“0.995 未验证”的旧结论），所以「再加 3 GiB KV」在这张卡上仍然没有对应设置——0.99→0.995 只多 +1,320 MiB（KV 393,940→405,526）。
13. **A/B 驱动的「启动失败」判定必须比引擎的真实 boot 时间长，且不能只看一次 `is-failed`**：本机 boot ≈290 s，我第一版驱动在 145 s 就误判失败并触发回退，把 0.995 那一臂白白废掉（结论变成未测）。正确写法：`systemctl reset-failed` 后轮询「MainPID 为空 **且** `is-failed=failed` 连续 3 次」，期限给到 800 s；SIGKILL 掉的引擎还会让 systemd 报 `Failed to kill control group … Input/output error`，紧接着重启就会读到残留 context 占的显存（见 2.5 里那次 KV 缩水 3.12 GiB 的观测）。
14. **cudagraph 池按估算预留、KV 池随后贪吃剩余空间**：裁剪 `QWEN_CAPTURE_SIZES`（比如封顶 2048）不会给 KV 挪出空间——估算变小 → KV 池反而涨到 321k–328k → 捕获阶段饿死。**估算大 → KV 小 → 捕获装得下**，反直觉但实测两次复现（8.1）。想加 KV 只有降 util 或缩 chunk 一条路。
15. **16K chunk 与高 util 结构性冲突**：capture 阶段能过，运行时首个长请求的 forward 激活照样 OOM（要 376 MiB、剩 119 MiB，8.2）。chunk 越大，运行时激活峰值越大，0.983 的余量装不下 16K。
16. **重复文本基准会给出假优化**：pangram 文本让 PLE 行缓存全命中（不走盘），chunk 排序与随机 token 口径**完全相反**（8192 在 pangram 下 +18 %、在随机 token 下 −14~−32 %）。先想清楚负载走不走盘再选基准（8.5）。
17. **跨天数字不可比**：同配置同脚本隔 4 天整体差 ~15 %（8.7）。配置 A/B 必须同日完成，否则结论方向都可能错。
18. **高 util 下并发 decode 会 OOM 崩引擎**：0.983/0.99 的运行时余量 <2 GiB，decol.sh 的 C8–C16 混合批（新请求 prefill 激活 1.0–2.25 GiB）会 `torch.OutOfMemoryError` 打死 EngineCore；0.96 窗口跑完全程。生产流量（顺序为主）不触发，但**跑并发梯度基准前要么降到 0.96，要么接受 C16 崩**（10-14 实测 3072+0.99 / 4096+0.983 / 8192+0.983 三臂全崩，与 chunk 无关）。

## 6. 文件说明与复现

```
├── README.md                        # 本文
├── patches/
│   ├── ple_ssd-fullwindow.patch     # 全窗口预取补丁（2 行，第一轮核心收益 +46 % @200k）
│   ├── ple_ssd-limit90.patch        # 第三轮：submit() 加回 limit 截断 + 公式 50 %→90 %
│   ├── ple_ssd.py                   # 打好全部补丁的完整文件（直接覆盖用，最终形态）
│   └── ple_ssd_io_mt.c              # 多线程 AIO 读取器源码（PLE_AIO_THREADS 控制）
├── configs/
│   ├── perf.conf                    # systemd drop-in（最终配置：cache 2048 MiB / prefetch 262144 / util 0.995）
│   └── serve.sh                     # 引擎启动脚本（全 env 参数化，含 KV offloading 开关）
└── scripts/
    ├── bench.sh                     # 唯一基准脚本 v1（身份门禁 + 3×32k + 131k/200k + 200k decode）
    ├── c4prefill.sh                 # C4 并发 prefill（4×32k）
    ├── decol.sh                     # decode/并发梯度（C1/4/8/16）
    ├── mttest.py                    # .so 字节级等价性 + 速率 harness
    ├── bench_cachehit.py            # PLE 行缓存命中口径基准（pangram 重复文本，部署名 /root/bench_prefill.py）
    ├── fill_context.py              # 打满 262,144 上下文测试（99.8 % 填满）
    ├── ple_bench_200k.py            # 第三轮口径：80K 参照 + 200K prompt（按轮数播种，防前缀缓存跨轮命中）
    └── ple_bench_max.py             # 第三轮口径：单条 262,000 prompt（模型上下文上限）
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

# 1. 打本仓库的补丁（或直接 cp patches/ple_ssd.py 覆盖 = 全窗口 + limit 90 % 最终形态）
cd /root/vllm/src/vllm/models/qwen4_exp/nvidia/
cp ple_ssd.py ple_ssd.py.orig
patch -p1 < ple_ssd-fullwindow.patch   # 从仓库根目录: patch -p1 -d /root/vllm < patches/ple_ssd-fullwindow.patch
patch -p1 < ple_ssd-limit90.patch      # 第三轮补丁（在 fullwindow 之上）

# 2. 编译多线程 AIO 读取器
cc -O3 -shared -fPIC -Wall -Wextra -Werror -pthread \
   patches/ple_ssd_io_mt.c -o /root/vllm/optimization/ple_ssd_io.so

# 3. drop-in（路径按你的 unit 名改）
cp configs/perf.conf /etc/systemd/system/<你的unit>.service.d/perf.conf
systemctl daemon-reload && systemctl restart <你的unit>

# 4. 基准（唯一口径）
bash scripts/bench.sh        # P1 32k×3 / P2 131k+200k / P3 200k decode 128/256
bash scripts/c4prefill.sh    # C4 并发 prefill
bash scripts/decol.sh        # decode 梯度（高 util 下 C16 会 OOM，见踩坑 18）
# 第二轮补充口径（8.5 / 8.8）：
python scripts/bench_cachehit.py 55000 5 256   # PLE 行缓存命中口径（pangram）
python -u scripts/fill_context.py              # 打满 262,144 上下文
# 第三轮补充口径（第 9 节，每轮换 seed 防前缀缓存命中）：
python scripts/ple_bench_200k.py <label> /tmp/r1.json 1   # 80K 参照 + 200K prompt
python scripts/ple_bench_max.py  <label> /tmp/r1.json 1   # 单条 262,000 prompt
```

`serve.sh` 的可调 env：`QWEN_MTP`（speculative tokens）、`QWEN_BATCH_TOKENS`（chunk）、`QWEN_CAPTURE_SIZES`（**必须包含 chunk 值**，否则 prefill 掉出 full-graph 覆盖）、`QWEN_SEQS`、`QWEN_SSD_*`。`PLE_AIO_THREADS` 是 `.so` 的 env（`rows_open` 时读取，改了要重启）。

## 7. 环境版本

- vLLM `0.29.1rc1.dev402+ga5a30471f.ple1`（作者 fork，`/root/vllm/src/vllm` 源码树安装）
- torch 2.13.0+cu130 · CUDA 13 · nvidia-open 驱动（cmpunlocker 解锁 300 W）
- ZFS（`recordsize=4K` PLE 数据集、`zfs_arc_max=2147483648`（2 GiB，第三轮从 8 GiB 下调——9.4 实测无差异）经 `/etc/modprobe.d/zfs.conf` 持久化）
- 模型 `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`（13 个 safetensors，PLE 表在 `ple-dedicated-*` 分片）
- PVE 9.2 / LXC CT120 / i5-14600K / 62 GiB DDR4 / CMP 170HX 64 GiB

## 8. 第二轮调优实录（2026-10-14）：chunk 上探、显存天花板与同日 A/B

目标：把 chunked prefill 提到 8192+ 并找到不爆显存的组合。结论先行：**8192 在固定口径（随机 token）下单流短中长度大幅回退，第二轮最终反而定为 chunk 2048 + 0.99**（同日四配置 A/B 全指标最优；第三轮在此基础上只动 PLE 缓存与 util，chunk 2048 不变，见 9.9）。但这轮挖出了四个值钱的机制发现（graph 池估算、16K 运行时 OOM、cumem fork bug、跨天漂移）和一个测量方法论警告（重复文本基准反转排序）。

### 8.1 显存天花板：0.983 是 8192 chunk 的上限，且裁剪 capture 列表是反作用

8192 chunk + capture 列表含 8192 图时：0.991 / 0.988 / 0.985 全部在 MTP speculator 捕获阶段 OOM（差 240–320 MiB），**0.983 可跑**（KV 池 267,937 tok，idle 63,908 / peak 64,886 MiB）。

**反直觉机制（本轮最有价值的发现）**：cudagraph 池是按「估算」预留的，KV 池随后贪吃剩余空间。裁剪 capture sizes（封顶 2048）会让估算变小 → KV 池反而涨到 321k–328k → 捕获阶段饿死，与「裁剪给 KV 挪空间」的直觉完全相反。**估算大 → KV 小 → 捕获装得下**。想加 KV 只有降 util 或缩 chunk——chunk 2048 臂的 KV 池因此到 **393,940**（比 3072 臂多 5.8 %，3×131,072 = 393,216 刚好装下 3 条 128k 流）。

### 8.2 16K chunk：三种组合全灭

| 组合 | 死法 |
|---|---|
| 16384 + capture 16384 | `profile_cudagraph_memory` 阶段 OOM（640 MiB float32 + 6.57 GiB 碎片） |
| 16384 + capture 封顶 | MTP speculator 捕获 OOM |
| 16384 + capture≤2048 + 0.983 | 启动 READY（KV 254,244），**首个 55K 请求运行时 OOM**（要 376 MiB、剩 119 MiB） |

结论：16K chunk 的 forward 激活峰值需要小 KV 池才装得下，与高 util 目标结构性冲突，**否决**。

### 8.3 cumem allocator：被 fork 的时序 bug 挡死

`--enable-cumem-allocator` + `expandable_segments` 的组合被 `_verify_kv_transfer_compat`（`config/vllm.py`）拒绝：该检查在 `model_config` 构建之前调用（fork 的调用时序 bug），cumem 豁免条件永不生效。cumem 单独启用（不带 env）能启动但**零收益**（55K pangram 5,003 vs 5,009），回退。

### 8.4 同日四配置 A/B（bench.sh / c4prefill.sh 固定口径）

| 指标 | **2048+0.99** | 3072+0.99 | 4096+0.983 | 8192+0.983 |
|---|---:|---:|---:|---:|
| prefill 8,192 | **2,844.4** | 2,781.9 | 2,628.8 | 1,900.9 |
| prefill 32,768 | 2,947.8 | 2,924.3 | **3,067.8** | 2,512.8 |
| prefill 131,072 | **3,166.8** | 3,018.3 | 3,168.6 | 3,015.5 |
| prefill 200,000 | **3,119.2** | 2,967.7 | 3,093.2 | 3,060.8 |
| 200k TTFT | 64.12 s | 67.39 s | 64.66 s | 65.34 s |
| C4 聚合（4×32k） | **3,012**（2,978/3,047） | 2,955 | 2,455（−17 %） | 未测 |
| KV 池 | **393,940** | 372,215 | 340,352 | 267,937 |
| pangram 55K（缓存命中口径） | 4,168 | 4,235* | 4,396 | **5,009** |

*3072 的 pangram 测于 0.985 臂（util 不影响速度）。

- **2048**：8k/131k/200k/C4/KV 五轴全优，只有 32k 输 4096 4 %——**第二轮最终（chunk 保持 2048 至第三轮，见 9.9）**。
- **4096**：单流长文本 +4~5 %，但 C4 聚合 −17 %（与 2.4 的 10-10 结论同形）。
- **8192**：8k −32 %、32k −14 %，只有缓存命中口径赢（见 8.5）。
- **打满上下文**：8192+0.983 臂 261,627/262,144 tok（99.8 % 填满）TTFT 53.3 s / 4,906 tok/s；2048+0.99 臂 261,627 tok / TTFT 64.4 s / 4,066 tok/s，全程无 OOM。
- **并发 decode 崩溃**：三臂在 decol.sh C8–C16 全部 `torch.OutOfMemoryError` 崩引擎（要 1.0–2.25 GiB、剩 1.0–1.6 GiB），与 chunk 无关，是高 util 的固有属性（踩坑 18）。

### 8.5 两个口径的分裂：随机 token vs PLE 行缓存命中

`scripts/bench_cachehit.py`（pangram 重复文本 → PLE 行缓存全命中 → 纯算力口径）与 `bench.sh`（随机 token → 盘瓶颈口径）给出**相反的 chunk 排序**：pangram 下 8192 比 2048 快 20 %（5,009 vs 4,168），随机 token 下 8192 比 2048 慢 14~32 %。**生产实测（640,851 tok / 185.6 s ≈ 3,454 tok/s）与随机 token 口径吻合**——日常流量里未被前缀缓存命中的部分仍走盘，所以固定口径（随机 token）才是代表口径；pangram 口径的用途是评估「PLE 全表进内存后能到多少」（见 8.6）。

### 8.6 PLE 全表进内存需要多少 RAM

表 95.43 GiB（320,001,536 行 × 320 B）+ 引擎主机侧 ~26 GiB + OS/ARC ≈ **122 GiB → 128 GiB 机器是下限（紧），192 GiB 从容**；本机 62 GiB 结构性不可能。实现上还要改读取器（现在是 O_DIRECT 走 SSD，tmpfs 不支持 O_DIRECT，需 mmap 或去 O_DIRECT）。**收益上界**：pangram 口径 5,009 tok/s ≈ 全表命中后的算力天花板，即 prefill ~3,500 → ~5,000（+40 %）、200k TTFT ~57 s → ~40 s。

> **10-23 修正（第三轮 9.2/9.3）**：本节原文「部分缓存无效（2048 MiB 行缓存已证 NULL）——要么全进，要么别进」**被第三轮推翻**。短 prompt（≤80K）下行缓存确实无差异（行分布近均匀、命中率低），但 **200K+ 长 prompt 下行缓存大小是决定性的**：512 MiB → 2 GiB 把 200K TTFT 从 102 s 拉到 70 s（预取窗口受 `cache_limit` 约束，见 9.3），且 **2 GiB + limit 90 % 公式与 4 GiB 持平**——2 GiB 已是可落地的中间态，不必全表进内存。全进（+40 %）仍是进一步的空间，但不再是「唯一出路」。

### 8.7 跨天漂移 ~15 %：同日 A/B 是硬要求

10-14 用同一脚本复测 3072+0.99：8k 2,782 / 32k 2,924 / 131k 3,018 / 200k 2,968 / C4 2,955——比 10-10 的 3,211/3,536/3,629/3,506/3,535–3,649 **整体低 ~15 %**（盘状态/环境漂移，DEV util 76.9 % vs 73.3 % 同档，无同盘干扰租户）。**跨天的绝对数字不可比，配置结论只能来自同日 A/B**（8.4 的表就是同日的）。10-10 那轮「3072 是均衡点」与 10-14 这轮「2048 全优」的差异本身就在两天的噪声内——两轮合起来看，2048 与 3072 相当、2048 略优且 KV 池更大，所以最终定 2048。

### 8.8 本轮新增测量工具

- `scripts/bench_cachehit.py`：PLE 行缓存命中口径（pangram 重复文本 + 唯一 nonce 前缀，零前缀缓存命中）。用法 `python bench_cachehit.py 55000 5 256`。
- `scripts/fill_context.py`：把上下文打满到 262,144（预算式构建 + HTTP 400 自动重试），报告填满率与 prefill 速率。

## 9. 第三轮调优实录（2026-10-22/23）：行缓存矩阵、limit 公式与 0.995

本轮口径与前两轮不同，先说清楚再上数字：

- **长度轴换成 80K / 200K / 262K**（`scripts/ple_bench_200k.py` / `scripts/ple_bench_max.py`）：80K 是「预取能覆盖」的参照长度，200K/262K 才能把行缓存的工作集（200K × 16 行 × 320 B ≈ 1.0 GiB）推出小缓存。
- **每轮换 seed**（`random.Random(770000 + length + round×1e6)`）：随机 token 跨轮重复会命中 GPU 前缀缓存，虚高到 160k tok/s；换轮换 seed 后每条都是真冷启动。
- **每个配置臂重启进程**（`systemctl restart`），避免行缓存/ARC 残留。
- 跨天漂移（8.7）依然成立：本轮 200k ~70 s / ~2,900 tok/s，绝对值比 10-10 低 ~15 %，与 10-14 复测同带；**本轮所有结论只来自本轮同日 A/B**。

### 9.1 CPU KV offload 是死容量：20 GiB → 8 GiB

`QWEN_KV_CPU_GB=20` 跑了一整天，`external prefix cache hit` 恒为 0 %——PLE SSD 路径下 CPU KV 层从不被命中，20 GiB 是纯占内存。砍到 8 GiB：prefill/decode 全指标无变化，EngineCore `RssAnon` 13.3 GiB。**结论：这个参数只决定内存占用，不决定性能。**

### 9.2 行缓存大小矩阵：80K 无差异，200K+ 决定性

| `QWEN_SSD_CACHE_MB` | 80K TTFT | 200K TTFT | 200K tok/s |
|---:|---:|---:|---:|
| 512 | 36.6 / 37.1 s | 102.43 s | 2,003 |
| 1024 | ~28 s | 95.15 s | 2,159 |
| 2048（原公式） | 28.5–29.6 s | 81.6 / 82.9 s | 2,470 |
| 4096 | 28.1–28.8 s | **70.30 / 71.09 s** | 2,881–2,913 |
| **2048（90 % 公式）** | **27.6–27.9 s** | **70.02 / 71.30 s** | **2,875–2,925** |

- 80K 下 2G 与 4G 无差异（三轮 28.46/28.72/29.55 vs 28.14–28.83，全在轮间波动内）：PLE 行访问是 ngram hash 随机行，缓存大小只影响重复命中的少量行，**预取覆盖才是关键**。
- 200K 单请求物理读 ~19.3 GiB（4K 块 × 随机 320 B 行 ≈ 12.8× 放大），与 cache/ARC 大小基本无关——差异全部来自预取窗口（见 9.3）。
- 262K（90 % 公式 + 2G）：89.95 s / 2,913 tok/s。

### 9.3 根因不是缓存，是 limit 公式：50 % → 90 %

`PLEPromptPrefetcher.__init__`（ple_ssd.py:319）：

```python
self.limit = min(limit, self.table.cache_limit // (2 * ngram.ngram_heads))
```

`cache_limit = cache_mb × 1 MiB // (row_bytes + 128)`（行 320 B + 128 B 记账 = 448 B/行），`ngram_heads = 32`：

| cache | cache_limit（行） | 50 % 公式 → 预取窗口（token） | 90 % 公式 |
|---:|---:|---:|---:|
| 2 GiB | 4,792,627 | **149,769** | 4,313,364 |
| 4 GiB | 9,585,254 | 299,539（min 后取满 262,144） | — |

200K prompt 在 2G + 50 % 公式下只有前 ~150K 被预取，**尾部 55K 落回 demanded 读关键路径**——这就是 81.6 s vs 70.3 s 的全部差距。4G 快不是缓存大，是公式允许它预取满。

**修法**（`patches/ple_ssd-limit90.patch`，两处）：

```diff
-        self.limit = min(limit, self.table.cache_limit // (2 * ngram.ngram_heads))
+        self.limit = min(limit, self.table.cache_limit * 9 // (10 * ngram.ngram_heads))
...
         tokens = tokens[start - history :]
+        # Bound read-ahead to what the shared row cache can actually hold.
+        tokens = tokens[: history + self.limit]
```

第二行把全窗口补丁（2.1）删掉的 `limit` 截断加回来——**预取窗口重新受控**，`QWEN_SSD_PREFETCH=262144` 重新有意义。90 % 余量：262,144 token 窗口需要 4.19M 行，2G 的 90 % = 4.31M 行，刚好覆盖且给 history 留 ~12 %。**2G + 90 % 与 4G 完全持平**（上表末行），1G 放宽后也只剩 135K token，仍退化——**2G 是 262,144 窗口的下限，最终定 2G**。

### 9.4 ZFS ARC 8G / 4G / 2G 无差异

一次性随机读，ARC 命中率本就 ~8.4 %（8 GiB / 95 GiB），粒度只影响 miss 代价。同日 A/B（cache 4G @200K）：ARC 4G = 70.30/71.09 s、2G = 72.21/70.82 s——无差异。`zfs_arc_max` 已从 8 GiB 降到 **2 GiB**（`/etc/modprobe.d/zfs.conf` + runtime `/sys/module/zfs/parameters/zfs_arc_max`）。注意这是 rpool 全局参数，其他 dataset/容器的缓存一起变小。

`zfs_prefetch_disable=1`：微基准下随机读 IOPS +34 %，但实机 80K（27.42 vs 28.14/28.40 s）差异在 ±1 s 噪声内——已回退，不值得全局改。

### 9.5 recordsize 4K 论证：大块是纯负优化

「/models 是 4K 块，会不会让表变大？换 512K/1M/4M 块是不是更快？」——微基准回答（`primarycache=none` 隔离，1 GiB 真实 PLE 数据，10,000 次随机 320 B `pread`）：

| recordsize | IOPS | 每逻辑读字节 | 读放大 |
|---|---:|---:|---:|
| **4K** | **6,895** | 16,785 | ~1× |
| 512K | 2,713 | 556,452 | ~60× |
| 1M | 1,976 | 1,071,254 | ~113× |
| 4M | 790 | 3,436,592 | ~362× |

- PLE 访问是 ngram hash 随机行，**无局部性 → 大块没有顺读红利，纯读放大**（512K 一次 miss 拉整块 1,638 行）。
- 磁盘占用四者相等（1.1G/1GiB；4K 多 ~20 MiB metadata，~2 %）——「4K 让表变大」不成立。
- ARC 命中率不受块大小影响。**结论：保持 4K，别为 PLE 建大块 dataset。**

### 9.6 试过并回滚的两个改动

- **预取批 256 → 2048**：200K 101 s（+43 % 慢）；→128 与 256 持平。机制：每个预取批持全局 condition 锁同步完成，批越大 demanded 读等得越久。**保留 256**（= AIO 深度）。
- **双 AIO lane**（prefetch/demand 各自 native context 并发，去掉全局让位锁）：80K 31.5 vs 28.1 s、200K 74–78 vs 70.3 s，**慢 5–10 %**。demanded 读的优先让位语义比 I/O 并发更重要——回滚。

### 9.7 显存 0.995 天花板

- 0.99 → **0.995**：KV 池 393,940 → **405,526 token（11.47 GiB，1.55× 262k）**，200K/262K prefill 无回归。
- 0.997 启动失败：vLLM 门禁要求 free(63.09 GiB) ≥ 预算(63.2 GiB)，torch 只见 63.39 GiB、启动前驱动已占 0.30 GiB——**上限 0.9953**。
- 显存账本：固定开销 52.54 GiB（权重 48.31 + peak activation 3.29 + CUDAGraph 0.94）；每 token KV 记账 **30.4 KB**（纯 attention 24.6 KB，hybrid 页对齐把 mamba 状态摊进 attention 块抬上来 23 %）；单条 262K KV = 7.42 GiB，两条 262K = 14.84 GiB 放不下（1.55×）。

### 9.8 I/O 画像与瓶颈定位

- NVMe 随机 320 B pread 上限：1 线程 6.9k / 32 线程 100k / 128 线程 157k IOPS；**生产只跑 ~40k——瓶颈在流水线不在盘**。
- GPU util 采样：42 % 样本 ≥90 %，38 % 在 1–69 %——约一半时间 GPU 在等 I/O，这是 PLE-SSD 架构的结构性上限（全表进内存才能消掉，见 8.6）。

### 9.9 最终配置与数字（2026-10-23 定稿）

```
QWEN_SSD_CACHE_MB=2048   QWEN_SSD_PREFETCH=262144   limit 90 % 公式补丁
QWEN_KV_CPU_GB=8         QWEN_GPU_MEMORY=0.995      QWEN_MTP=2
QWEN_BATCH_TOKENS=2048   PLE_AIO_THREADS=32         ARC 2 GiB   recordsize 4K
```

| 长度 | TTFT | tok/s |
|---:|---:|---:|
| 80K | 27.6–27.9 s | ~2,900–2,960 |
| 200K | 70.02–71.30 s | ~2,875–2,925 |
| 262K | 89.95 s | 2,913 |

decode ~100–108 tok/s（单流）；EngineCore RssAnon 13.3 GiB；80K 单请求物理读 8.3–9.4 GiB、200K ~19.3 GiB。

## 10. vLLM 0.32.0 移植与 A/B 对比（2026-10-23）

### 10.1 移植内容（容器内分支 `ple-latest`，CT120 `/root/vllm-latest`）

- 基线：上游 **vLLM 0.32.0**（tag `5df63a72`），PyPI wheel + editable 源码覆盖（`VLLM_PRECOMPILED_WHEEL_LOCATION` 指向从 uv 缓存重打包的 wheel）。
- `git apply -3` 作者 `qwen38-ple-ssd.patch`：9 文件干净应用，2 文件冲突手工解决：
  - `nvidia/ngram_embedding.py`：0.32.0 已把 PLE embedding 类移到 `qwen4_exp/common/ngram_embedding.py` 且 prefetch API 变为单参 `start_prefetch(ngram_ids)` + 新增抽象 `finalize_prefetch(total_tokens)`——冲突取 ours（丢 fork 内联旧类），保留 `use_reduce_scatter` 与 `ple_ssd_offload` 钩子（both）。
  - `v1/worker/gpu/cudagraph_utils.py`：合并 0.32.0 的 `capture_pool()`（cuMem 感知）与补丁的 `BreakableCUDAGraphCapture` 分支为单一 with 元组。
- `ple_ssd.py`：生产版（limit 90% 公式 + 全窗口预取）适配新 API：`start_prefetch` 单参、新增 `finalize_prefetch(total_tokens)` 包装 `_finalize_prefetch`。
- `ple_ssd_io.c`：多线程 AIO（`PLE_AIO_THREADS=32`）原样带入。
- **新踩坑（0.32.0 独有）**：`common/ngram_embedding.py` 的 `from_quant_config` 对 INCConfig 会调 `config_parser.resolve()`，而 AutoRound 的 `extra_config` regex 键是 HF 张量名（`model.language_model...`），`apply_vllm_mapper` 不改写 regex 键 → 运行时前缀（`language_model.model.layers.N.ple...`）落回 2-bit 默认值 → `NotImplementedError`。修复：PLE embedding 对 INCConfig 一律返回 Unquantized（与 ple1 fork 行为一致，PLE 表本来就是 BF16）。commit `dc53addc`。
- systemd 并列单元 `qwen38-032-test.service`（同 perf 参数、`VLLM_WORKDIR=/root/vllm-latest`），与生产 `qwen38-3bpw-ple-ssd` 互斥切换；CT120 的 mp0 `/models` 挂载在快照恢复时曾丢失，已重新 `pct set 120 -mp0 /models,mp=/models,backup=0`。

### 10.2 A/B 实测（同机同参数，seed 轮换防前缀缓存）

| 指标 | 0.29 生产（ple1 fork） | 0.32.0 移植（ple-latest） | 差异 |
|---|---:|---:|---|
| 80K prefill | 28.2 s / 2,900 tok/s | 27.5 s / 2,976 tok/s | ≈持平 |
| 200K prefill | 72.7 s / 2,818 tok/s | 72.9 s / 2,808 tok/s | ≈持平 |
| 262K prefill | 91.8 s / 2,854 tok/s | 95.0 s / 2,757 tok/s | 0.32 慢 ~3.5% |
| 200K prefill（d200k 口径） | 2,897 tok/s | 2,898 tok/s | 持平 |
| 200K decode 128 tok（单流） | **173.6 tok/s** | **184.2 tok/s** | **0.32 快 ~6%** |

结论：prefill 两版持平（262K 略慢但在单次样本噪声内），**decode 0.32.0 快约 6%**（推测来自上游调度/采样路径的常规优化）。

**2026-10-23 已切换生产到 0.32.0**：`systemctl stop/disable qwen38-3bpw-ple-ssd` + `enable/start qwen38-032-test`（同端口 8000，互斥；0.29 服务单元保留可随时回滚）。

### 10.3 0.32 新特性逐项优化实验（2026-10-23，切换生产后）

按性价比顺序对 0.32 新特性逐项实测（每项独立开关 + 重启 + A/B，seed 轮换防前缀缓存干扰）：

| # | 实验 | 结果 | 处置 |
|---|------|------|------|
| 1 | KV CPU offload 8GB 是否复活（6×80K 挤出驱逐后重发） | **仍是死容量**：重发 28.5s ≈ 首次 28.1s，`external_prefix_cache_hits_total=0`（0.32 新增的细粒度 boundary offload 路径未生效） | **砍掉 `QWEN_KV_CPU_GB=8`**，省 8GB host RAM；`--prefix-match-unit 8` 拆出独立传；回归无损失（80K 27.9s / 200K 70.1s / decode seed 方差带内 168-193 tok/s） |
| 2 | `--enable-mamba-shared-prefix-checkpoint`（MTP sibling 交汇处注册 mamba 检查点） | 分叉场景（共享 60K 前缀+不同尾）：开/关都 0 命中；superset 场景（req2=req1 全文+追加）：开/关都命中 59,992 tokens、1.2s vs 21s——**17 倍加速是 0.32 基础能力（replay_boundaries + 8-token 细粒度哈希），flag 无增益** | 保持关闭（`QWEN_MAMBA_SHARED_PREFIX=0`） |
| 3 | `QWEN_GPU_MEMORY` 0.995→1.0 | **启动即被拒**：设备空闲 63.09/63.39 GiB < 1.0×63.39，0.995 已是实际安全上限 | 回滚 0.995 |
| 4 | `--prefix-caching-hash-algo xxhash`（uv 装 xxhash 4.0.2） | prefill 持平（27.9/70.0s），decode 在 seed 方差带内；无实测收益但调度器哈希 CPU 开销理论更低 | **保留**（`QWEN_HASH_ALGO=xxhash`） |
| 5 | flashinfer-jit-cache 0.7.0.post1+cu130（`vllm download-kernels`） | **网络阻断**：flashinfer.ai 索引可达但 uv 静默不装；wheel 在 GitHub releases，PVE 直连停滞 4.5KB、socks5 代理 000；且收益本就边际（采样已用 FlashInfer、attention 走 FLASH_ATTN、sm80 不触发 Hopper 警告） | 放弃 |
| 6 | `QWEN_ASYNC_SCHED=1` 异步调度 | prefill 慢 2-3%（80K 28.6s / 200K 72.4s），decode 无增益（165.8/182.0/192.4，方差带内）；**且引擎在启动 4 分钟后无请求状态下静默死亡一次**（无 Python 堆栈，疑似原生层崩溃，与 PLE AIO 32 线程冲突） | **回滚**（移除 env） |

**意外收获**：多轮对话（superset）模式在 0.32 上命中 59,992/60,000 tokens、prefill 21s→1.2s——多轮长对话场景的续轮响应几乎免费，这是 0.32 相对 0.29 之外的实际体验提升。

最终生产配置变化（相对切换时）：`-QWEN_KV_CPU_GB=8`（offload 移除）、`+QWEN_HASH_ALGO=xxhash`、`+QWEN_MAMBA_SHARED_PREFIX=0`（serve.sh 新增开关）；GPU KV 容量不变（408,423 tokens / 262K 并发 1.55x）。

## License

笔记与脚本 MIT。vLLM（Apache-2.0）、模型权重与作者 fork 的许可证各自适用。
