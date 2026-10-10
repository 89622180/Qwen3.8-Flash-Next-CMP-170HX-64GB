# Qwen3.8-Flash-Next 125B on CMP 170HX 64GB — PLE-SSD vLLM 调优实录

单张 NVIDIA CMP 170HX 64GB（sm_80）+ 一块 1TB NVMe，跑 klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP（125B MoE，6B 激活 + 51B n-gram 表）。
核心思路：**95.43 GiB 的 PLE/n-gram 表放 SSD（O_DIRECT 原生 Linux AIO，不进 RAM 也不进 ARC），主模型 47.05 GiB 常驻显存**。
本文记录 2026-10-10 一天之内的调优全过程：冷 prefill 从 ~3,550 → ~3,500+（32k）、**200k 长上下文从 2,264 → ~3,525 tok/s（+56 %）、TTFT 从 88.5 s → ~56.5 s**，以及每一个被否决的方案（诚实版）。

参考了 [jiangrun0213/strata-v100-notes](https://github.com/jiangrun0213/strata-v100-notes) 的记录风格。

## 1. 背景：为什么这张卡能跑 125B

- **硬件**：PVE 9.2 LXC 容器（CT120），i5-14600K（6P+8E），62 GiB DDR4-3200 双通道，**CMP 170HX 64 GiB（compute 8.0，解锁后 300 W 上限）**，1 TB NVMe（ZFS `rpool`）。
- **模型**：`klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`，142.48 GiB = PLE 表 95.43 GiB（BF16，320 B/行，320,001,536 行）+ 主模型 47.05 GiB（AutoRound 3bpw，routed experts ≈2.84 bpw）。
- **引擎**：作者定制的 **vLLM `0.29.1rc1.dev402+ga5a30471f.ple1` fork**（`ple_ssd_offload` 钩子；PyPI 最新 0.31.0 没有这个钩子，"升级 vLLM" = rebase 补丁而不是 pip）。torch 2.13.0+cu130。
- **存储布局**：PLE 表放在专用 ZFS 子数据集上，**`recordsize=4K`**（父数据集 128K 会在 320 B 行上产生 ~400× 读放大——这是第一版部署只有 157 tok/s 的根因）；宿主机 `zfs_arc_max=8 GiB` 让 ZFS 元数据全命中。
- 显存账：权重 47.32 + KV 8.31（300,220 tok @ MTP=1）+ 非 KV 开销 4.49 ≈ 61.4/64 GiB。

## 2. 优化实录（每一步都是单变量 A/B，同一固定脚本测量）

当天基线（250 W / AIO 线程 16 / MTP=1 / chunk 2048）：**32k 3,550 · 131k 2,349 · 200k 2,264 / TTFT 88.5 s**。

| # | 改动 | 200k prefill / TTFT | 结论 |
|---|---|---|---|
| 0 | 功耗 250→300 W（卡的上限） | 无变化 | prefill 在这台机器上**不是功耗指标**（GPU sm 均值 58.8 %、盘 util 66–68 %，双半忙，卡在 250 W 时本来就没被时钟卡住；实测峰值功耗 306–328 W、SM 1440 MHz、≤58 °C） |
| 1 | **全窗口预取补丁（2 行）** | 2,264 → 3,300 / 88.5 → 61.3 s | **+46 %，本日最大单笔收益** |
| 2 | `PLE_AIO_THREADS` 16→32 | 3,300 → 3,460 | +5 %（`.so` 读上限 61k→68k rows/s） |
| 3 | MTP `num_speculative_tokens` 1→2 | decode 单流 +4–10 % | acceptance 1.80→2.37–2.45（第二 draft 位置 62 % 命中）；KV 池反而 +9 % |
| 4 | chunk 2048→**3072** | 200k +1.7 %，C4 完整保住 | 均衡点（见 2.4 的取舍表） |

最终（300 W / 32 线程 / MTP=2 / chunk 3072）：**32k ~3,500 · 131k ~3,450–3,520 · 200k ~3,525 / TTFT ~56.5 s · C4 聚合 3,535–3,649 · C1 decode 124–153**。

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

## 3. 深度速度基准（最终配置，全部来自固定脚本 `bench.sh` / `decol.sh` / `c4prefill.sh`）

**Prefill（冷，随机 token，无前缀复用）**：32k ~3,500 · 131k ~3,450–3,520 · 200k ~3,525（TTFT 56.3–57.1 s）。设备侧：54–58k r/s、~250 MiB/s、util 71–75 %、r_await ~0.1 ms。

**Decode（128 输出 token × 3 轮）**：

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

**复现步骤**（PVE LXC 内，vLLM ple1 fork 已装在 `/root/vllm`）：

```bash
# 1. 打补丁（或直接 cp patches/ple_ssd.py 覆盖）
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
