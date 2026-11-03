#!/usr/bin/env bash
# Decode-dominated C=1 probe with enough sample to say anything at all.
#
# Why this exists: the fixed 256-output ladder swings by more than the effect being
# chased. The same config and the same prompt seed produced 161.6 and 262.8 tok/s on
# two different boots here (acc_len 3.0 vs 4.9), because on the DFlash2 arm decode
# speed is acceptance-dominated and acceptance follows the greedy token path.
# So: 1024 output tokens, 6 seeds, paired across arms, clocks warmed first.
#
# usage: TAG=myarm bash decode_long_sample.sh   (needs ../bench_conc_text.py in /root)
set -uo pipefail
U=${U:-qwen38-032-test.service}
PY=${PY:-/root/v33/bin/python}
O=${O:-/root/n1104/out6}
MODEL=${MODEL:-Qwen3.8-Flash-Next}
SEEDS=${SEEDS:-"1011 1012 1013 1014 1015 1016"}
TAG=${TAG:?set TAG}
mkdir -p "$O"

snap() {
  curl -s --max-time 10 localhost:8000/metrics |
    grep -E "^vllm:(spec_decode_num_[a-z_]+_total|num_preemptions_total|num_requests_running)" > "$1"
}

echo "=== tag=$TAG start=$(date +%T) ==="
MP=$(systemctl show -p MainPID --value "$U")
OP=$(ss -lntp 2>/dev/null | grep ':8000' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
echo "MainPID=$MP owner=$OP same=$([ "$MP" = "$OP" ] && echo YES || echo NO)"
tr '\0' '\n' < /proc/"$MP"/cmdline | grep -E 'speculative|kv-cache-dtype|moe-backend' | head -4

# warm the clocks, discarded
$PY /root/bench_conc_text.py 1 2048 256 991 > /dev/null 2>&1

for s in $SEEDS; do
  snap "$O/$TAG-$s-before"
  $PY /root/bench_conc_text.py 1 2048 1024 "$s" > "$O/$TAG-$s.json" 2>&1
  snap "$O/$TAG-$s-after"
  $PY - "$TAG" "$s" "$O" <<'PYEOF'
import json, re, sys
tag, s, O = sys.argv[1], sys.argv[2], sys.argv[3]
def load(p):
    d = {}
    try:
        for line in open(p):
            parts = line.split()
            if len(parts) < 2:
                continue
            k = re.sub(r'^vllm:', '', parts[0].split('{')[0])
            d[k] = d.get(k, 0) + float(parts[-1])
    except Exception:
        pass
    return d
b, a = load(f"{O}/{tag}-{s}-before"), load(f"{O}/{tag}-{s}-after")
d = {k: a.get(k, 0) - b.get(k, 0) for k in set(a) | set(b)}
dr = d.get("spec_decode_num_drafts_total", 0)
ac = d.get("spec_decode_num_accepted_tokens_total", 0)
try:
    j = json.loads(open(f"{O}/{tag}-{s}.json").read().strip().splitlines()[-1])
    print("%-10s seed=%s decode=%.1f agg=%.1f ttft=%.3f prefill=%.0f acc_len=%.3f preempt=%d" % (
        tag, s, j["per_stream_decode_tok_s"][0], j["agg_output_tok_s"],
        j["ttft_s"][0], j["prefill_tok_s"][0], (1 + ac / dr) if dr else 0,
        d.get("num_preemptions_total", 0)))
except Exception as e:
    print(tag, s, "ERR", e)
PYEOF
done

$PY - "$TAG" "$O" <<'PYEOF'
import json, sys, statistics as st
tag, O = sys.argv[1], sys.argv[2]
v, pr = [], []
for s in (1011, 1012, 1013, 1014, 1015, 1016):
    try:
        j = json.loads(open(f"{O}/{tag}-{s}.json").read().strip().splitlines()[-1])
        v.append(j["per_stream_decode_tok_s"][0]); pr.append(j["prefill_tok_s"][0])
    except Exception:
        pass
if v:
    print("SUMMARY %-10s n=%d decode mean=%.1f median=%.1f min=%.1f max=%.1f prefill mean=%.0f" % (
        tag, len(v), st.mean(v), st.median(v), min(v), max(v), st.mean(pr)))
PYEOF
echo "=== done $(date +%T) ==="
