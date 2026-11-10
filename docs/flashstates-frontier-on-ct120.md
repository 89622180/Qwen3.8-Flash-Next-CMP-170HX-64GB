# FlashStates frontier checkpoints on CT120 (2026-11-11)

Ported `FutureMLS-Lab/vllm@flashstate` (one commit `3cfd432c`, parent `4ba787ec`
= **v0.33.0**, i.e. the same base as our `ple-033`) onto CT120's running 0.33
tree, booted it, measured it, and restored the shipped arm. Verdict: the
frontier mechanism works here, but it cannot coexist with the shipped MTP k=4
arm, and MTP is what doubles decode.

## 1. How the port landed (no git branch, and why)

CT120's 0.33 arm is a **thin venv** (`/root/v33`) whose `vllm` lives in
site-packages, not in a git worktree, so there is no branch to create for it.
The port is a **patch artifact**, applied host-side into the dataset:

- artifact: `/root/fs-vllm-only.diff` on the PVE host (752,862 B, md5
  `83fcf6e0...`, **62** `diff --git a/vllm/` headers; the flashstate diff
  filtered to `vllm/` because the installed tree carries no `tests/` or
  `csrc/`), staged into the container too at `/root/n1104/fs-vllm-only.diff`
  (same md5, verified after copy)
- applied into `/root/v33/lib/python3.12/site-packages`, log
  `n1104/fs-apply-20261111.log`: **62 `patching file`, 0 FAILED**,
  backup `n1104/site-packages-pre-fs-20261111.tgz` (588,344 B, 45 members) +
  `n1104/fs-added-files.txt` (the 17 files the diff ADDs, all 17 confirmed
  present; rollback = untar the backup, then delete that list)
- post-apply idempotency: `patch -p1 --dry-run` of the staged diff now reports
  **45 `Reversed (or previously applied)` / 45 checked, rc=1** - i.e. the 45
  modified files are in and the 17 new ones are the `hybrid_checkpoint`
  package, the Mooncake store frontier policy, the vendored `flashinfer_gdn`
  kernels and `spec_decode/stop_conditions.py`

Offline verification, both directions, before touching the live tree:

| direction | tree | result |
| --- | --- | --- |
| flashstate-only diff | CT120 live site-packages **with our 23-file arm already applied** | rc=0, **0 hunk failures**, 62 hunks checked |
| our 23-file arm | clean `flashstate` branch tree | rc=0, **0 failures**, 23 `checking file` |
| either order vs a merged single patch | clean v0.33.0 (`/root/fsbranch`) | produces **identical** trees |

## 2. The gate, read from the code (not guessed)

`check_frontier_checkpoint_config` (`v1/core/hybrid_checkpoint/manager.py:125`)
refuses unless: prefix caching on with `mamba_cache_mode=align`, **Model Runner
V2**, `prefix_cache_retention_interval 0`, no `long_prefill_token_threshold`
below the block size, no ReplaySSM, DCP/PCP size 1, and if a KV connector is
present `kv_role=kv_both` + `kv_load_failure_policy=recompute`. Under
speculative decoding it additionally demands
`speculative_config.truncate_acceptance_at_stop` and
`not use_eagle_block_drop()` (`hybrid_checkpoint/speculative.py:41,46`).

Neither field is a CLI flag in 0.33 (`grep -c` in `engine/arg_utils.py` = **0**),
so they only reach the engine through the `--speculative-config` JSON.
`scripts/serve.sh` grew `QWEN_TRUNCATE_STOP` / `QWEN_NO_EAGLE_DROP` for that
(md5 `0d8bf6cb...`, backup `serve.sh.pre-spec-json-20261111` made before the
edit; the file differs from the shipped `31bdaa2a...`). Proven GPU-free with
`QWEN_PRINT_COMMAND=1`: env unset emits exactly
`{"method":"mtp","num_speculative_tokens":4}`; env set emits
`{"method":"mtp","num_speculative_tokens":4,"truncate_acceptance_at_stop":true,"disable_eagle_block_drop":true}`.

## 3. What actually refused, and whose fault it is

The MTP k=4 + frontier boot passed both config gates (the engine logged
`speculative_config=... num_spec_tokens=4` with both fields and
`EAGLE trailing prefix-cache block dropping is disabled`) and then died in
`layout_groups`:

```
ValueError: mamba_frontier_checkpoint under speculative decoding needs
indexer rings that only the decode path reads; group 4 declares
resume_catchup 'prefill'
```

Group 4 is **our** QSA compressed-key ring:
`models/qwen4_exp/common/qsa_cache.py:870` declares `resume_catchup="prefill"`
with its own comment "The prefill path reads the open group's keys from the
ring." So this is not a FlashStates limitation — it is a property of our arm:
under spec decode the ring also holds draft rows, and the prefill path reads it,
which is exactly the corruption the gate protects. Making it `"decode"` is a
model-level change that would need proof that prefill does not read the ring.
**That proof is negative**: `models/qwen4_exp/nvidia/ops/qsa_prepare.py:355`
("Select historical rows from the ring without issuing two masked loads") and
`:375` ("Its exact MRoPE coordinates may live in this chunk or the raw-state
ring") show prefill really does read it, so the `"prefill"` declaration is
accurate and flipping it to `"decode"` would be a lie that the gate would then
let through. Backup of the untouched file:
`/root/n1104/qsa_cache.py.pre-resume-decode-20261111` (md5 `516e45f1...`).

`prefix_cache_eviction="gds"` also **requires** `mamba_frontier_checkpoint`
(`config/cache.py:544`), so the GDS eviction policy cannot be measured on its
own; frontier and GDS ship as one knob here.

## 4. Measured (interleaved A-B-A-B, one real boot per arm)

`n1104/ab_frontier.log` + `ab_frontier.jsonl`, 4-turn chain, ~55.3k-token
natural-Chinese prompt (seeded, non-repetitive), greedy, 300 <= out <= 520.
Every arm identity checked (`:8000` owner pid == MainPID) and argv verified
from `/proc/<MainPID>/cmdline`.

| arm | cold turn0 TTFT | warm TTFT (turn1-3) | frontier hits (tokens) |
| --- | --- | --- | --- |
| ON (pair 1) | 22.191 s | 0.202 / 0.156 / 0.350 | 522 / 523 / 0 |
| OFF (pair 1) | 22.532 s | 0.454 / 0.475 / 0.443 | 0 |
| ON (pair 2) | 22.501 s | 0.446 / 0.335 / 0.245 | 0 |
| OFF (pair 2) | 22.357 s | 0.495 / 0.477 / 0.558 | 0 |

- Cold prefill is a **dead heat** (22.2-22.5 s for ~55.3k tokens): frontier does
  not help the first prefill, as expected.
- **KV pool is identical in all four boots: 701,869 tokens** (MTP=0 frees the
  draft slots against the shipped 548,787 / 2.09x). Frontier + GDS costs
  nothing in pool terms, `preempt=0`, 0 error lines.
- The counter is in **tokens**, not blocks: `metrics/stats.py` documents
  `frontier_hits` as "the tokens that Mamba frontier checkpoints served past the
  block-aligned hit", produced by `kv_cache_manager.py:269` ->
  `frontier.hit_tokens()`, which returns **0 unless admission lands exactly at
  the frontier** (`frontier.py:486`, `num_hits != lookup.num_tokens -> 0`).
  So pair 1's 522 is 522 tokens of prefill that standard prefix caching cannot
  reuse, and pair 2's zeros mean "the resume did not land on a frontier", not
  "the feature is off".
- Pair 2 ON was still the fastest warm arm while counting zero frontier tokens,
  and GDS cannot be isolated (cache.py:544), so **its gain is unattributed** -
  either GDS eviction keeping the right blocks, or arm-to-arm drift; the 30-45
  min boot gap rule says the two pairs cannot be pooled.
- Decode on the frontier arm is **derived, not measured**: that streaming driver
  omitted `include_usage`, so `completion_tokens` came back null and tokens/s can
  only be bounded. Pair-2 ON's warm turns spent 5.4-5.9 s of decode, which is
  55-77 tok/s depending on whether the model emitted the `min_tokens` floor of 300
  or the 520 ceiling - against the shipped MTP arm's **measured** 90-97 tok/s (the
  shipped-arm chain used `include_usage`, so its token counts are real).

## 5. The shipped arm already has what frontier would give

Same 4-turn chain against the live shipped arm (MTP k=4,
`QWEN_MAMBA_SHARED_PREFIX=1`):

```
turn0 wall=28.509 ttft=23.112 prompt=55334 out=420 decode=77.8 tok/s
turn1 wall= 4.996 ttft= 0.439 prompt=55794 out=420 hits=55320 uncached=474 decode=92.2
turn2 wall= 3.337 ttft= 0.441 prompt=56255 out=262 hits=55784 uncached=471 decode=90.5
turn3 wall= 3.104 ttft= 0.403 prompt=56555 out=261 hits=56240 uncached=315 decode=96.6
```

Warm TTFT 0.40-0.44 s is the same band as the frontier arm's OFF comparison, and
decode is measurably better than the frontier arm's bounded 55-77 tok/s.
**Production is not losing anything today that frontier would recover**, and
taking frontier means giving up MTP.

## 6. Shipped state after this session

CT120 is back on its shipped drop-ins (`perf.conf` + `zz-finegraphs.conf` +
`zz-v033.conf`, MTP=4, fine capture sizes, `QWEN_KV_BYTES` pin), MainPID 3733
owns `:8000`, KV **548,787 tokens**, 0 error lines, `Using V2 Model Runner`,
`.13` and CT103 `.12` both answering. The flashstate patch **stays applied** and
is inert with the flags off - verified by that clean boot and the unchanged
pool. Rollback: `systemctl stop`, untar `site-packages-pre-fs-20261111.tgz` over
site-packages, delete the `fs-added-files.txt` entries.

Opt-in: `configs/ct120-0.33-frontier.conf` (MTP off, tiers off, frontier + GDS).
