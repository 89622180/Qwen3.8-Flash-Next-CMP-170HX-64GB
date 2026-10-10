#!/bin/bash
# c4prefill.sh v1 — fixed C4-prefill companion to bench.sh: 4x concurrent bench_prefill.py 32768
set -uo pipefail
U=qwen38-3bpw-ple-ssd.service
R=/root/qwen38-flash-next-cmp170hx
O=/root/bench-out
PY=/root/vllm/.venv/bin/python
mkdir -p $O
MP=$(systemctl show -p MainPID --value $U)
OP=$(ss -lntp 2>/dev/null | grep ':8000' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
echo "=== C4PREFILL v1 $(date -u +%FT%TZ) identity MainPID=$MP owner=$OP same=$([ "$MP" = "$OP" ] && echo YES || echo NO)"
[ "$MP" = "$OP" ] || { echo "ABORT-identity"; exit 1; }
SEED=$(date +%s)
for round in 0 1; do
  S=$((SEED + round * 10))
  echo "=== round $round: 4x concurrent cold prefill 32768 (seeds $S..$((S+3))) $(date -u +%T)"
  T0=$(date +%s.%N)
  for i in 0 1 2 3; do
    $PY $R/benchmarks/bench_prefill.py --label c4-$round-$i --output $O/c4prefill-$round-$i.json --seed $((S+i)) --repeats 1 --lengths 32768 > /dev/null 2>&1 &
  done
  wait
  T1=$(date +%s.%N)
  WALL=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.2f", b-a}')
  AGG=$(awk -v w=$WALL 'BEGIN{printf "%.1f", 4*32768/w}')
  echo "  wall=${WALL}s  aggregate=${AGG} tok/s (131072 tokens)"
  $PY - "$O" "$round" <<'PYEOF'
import json, sys
O, rnd = sys.argv[1], sys.argv[2]
for i in range(4):
    r = json.load(open(f"{O}/c4prefill-{rnd}-{i}.json"))[0]
    print("    stream%d latency=%.2f s  solo-rate=%.1f tok/s" % (i, r["latency_s"], r["prompt_tokens_s"]))
PYEOF
done
echo "=== C4PREFILL DONE $(date -u +%FT%TZ)"
