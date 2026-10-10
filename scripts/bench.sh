#!/bin/bash
# bench.sh v1 — THE canonical CT120 measurement (2026-10-10). Every number quoted
# from now on must come from THIS script, unmodified. Config A/Bs may restart the
# engine, but the measurement phase is always `bash /root/bench.sh`.
# Parts: identity gate -> 3x cold prefill 8192+32768 -> 1x cold 131072+200000
#        -> streamed 200k decode 128 + 256 -> fixed-format summary.
set -uo pipefail
U=qwen38-3bpw-ple-ssd.service
R=/root/qwen38-flash-next-cmp170hx
O=/root/bench-out
PY=/root/vllm/.venv/bin/python
mkdir -p $O
TS=$(date -u +%Y%m%dT%H%M%SZ)
SEED=$(date +%s)

MP=$(systemctl show -p MainPID --value $U)
OP=$(ss -lntp 2>/dev/null | grep ':8000' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
EP=$(pgrep -f 'VLLM::Engine' | head -1)
SO=$(grep -o '/[^ ]*ple_ssd_io[^ ]*' /proc/$EP/maps 2>/dev/null | sort -u | tr '\n' ' ')
echo "=== BENCH v1 $TS identity MainPID=$MP owner=$OP same=$([ "$MP" = "$OP" ] && echo YES || echo NO) engine=$EP so=${SO:-NONE}"
[ "$MP" = "$OP" ] && [ -n "$MP" ] || { echo "ABORT-identity"; exit 1; }
[ -n "$SO" ] || { echo "ABORT-no-ple-so"; exit 1; }
echo "  so_sha=$(sha256sum /root/vllm/optimization/ple_ssd_io.so 2>/dev/null | cut -c1-12) arc=$(cat /sys/module/zfs/parameters/zfs_arc_max 2>/dev/null) kv=$(journalctl -u $U --no-pager 2>/dev/null | grep -oE 'GPU KV cache size: [0-9,]+' | tail -1)"

STAT=$O/iostat-$TS.log
timeout 900 iostat -x 1 900 > $STAT 2>&1 &
IO=$!

echo "=== PART1 cold prefill 8192+32768 x3 (seeds $SEED..+2)"
for i in 0 1 2; do
  $PY $R/benchmarks/bench_prefill.py --label bench-v1 --output $O/prefill-$TS-$i.json --seed $((SEED+i)) --repeats 1 --lengths 8192 32768 2>&1 | tail -3
done

echo "=== PART2 cold prefill 131072+200000 x1 (seed $((SEED+100)))"
$PY $R/benchmarks/bench_prefill.py --label bench-v1-200k --output $O/prefill-200k-$TS.json --seed $((SEED+100)) --repeats 1 --lengths 131072 200000 2>&1 | tail -3

echo "=== PART3 streamed 200k decode 128 then 256"
$PY /root/d200k.py 200000 128 $((SEED+200)) 2>&1 | tail -1
$PY /root/d200k.py 200000 256 $((SEED+201)) 2>&1 | tail -1

kill $IO 2>/dev/null

echo "=== SUMMARY (fixed format — compare runs on these rows only)"
$PY - "$O" "$TS" "$STAT" <<'PYEOF'
import glob, json, os, sys
O, TS, STAT = sys.argv[1], sys.argv[2], sys.argv[3]
def rows(pat):
    out = []
    for p in sorted(glob.glob(os.path.join(O, pat))):
        try:
            out += json.load(open(p))
        except Exception:
            pass
    return out
p32 = rows("prefill-%s-*.json" % TS)
if p32:
    for L in (8192, 32768):
        v = [r["prompt_tokens_s"] for r in p32 if r["prompt_tokens"] == L]
        if v:
            print("  P1  len=%-6d passes=%s  mean=%.1f tok/s" % (L, "/".join("%.1f" % x for x in v), sum(v)/len(v)))
p200 = rows("prefill-200k-%s.json" % TS)
for L in (131072, 200000):
    v = [r["prompt_tokens_s"] for r in p200 if r["prompt_tokens"] == L]
    if v:
        print("  P2  len=%-6d mean=%.1f tok/s  (%.2f s)" % (L, sum(v)/len(v), [r["latency_s"] for r in p200 if r["prompt_tokens"]==L][0]))
try:
    lines = open(STAT).read().splitlines()
    hdr = None; dev = None; samples = []
    for ln in lines:
        if ln.strip().startswith("Device"):
            hdr = ln.split()
            continue
        if hdr and ln.strip():
            f = ln.split()
            if f and f[0].startswith("nvme") and len(f) == len(hdr):
                if dev is None:
                    dev = f[0]
                if f[0] == dev:
                    d = dict(zip(hdr, f))
                    try:
                        rs = float(d.get("r/s", 0))
                        if rs > 3:
                            samples.append((rs, float(d.get("rkB/s", 0) or 0), float(d.get("await", 0) or 0), float(d.get("%util", 0) or 0)))
                    except ValueError:
                        pass
    if samples:
        n = len(samples)
        print("  DEV %s n=%d r/s=%.1fk rkB/s=%.0f await=%.2fms util=%.1f%%" % (
            dev, n, sum(s[0] for s in samples)/n/1000, sum(s[1] for s in samples)/n,
            sum(s[2] for s in samples)/n, sum(s[3] for s in samples)/n))
except Exception as e:
    print("  DEV parse failed:", e)
PYEOF
echo "=== BENCH v1 DONE $TS"
