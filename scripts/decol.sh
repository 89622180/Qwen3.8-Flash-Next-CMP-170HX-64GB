#!/bin/bash
set -uo pipefail
U=qwen38-3bpw-ple-ssd.service
R=/root/qwen38-flash-next-cmp170hx
O=/root/bench-out
PY=/root/vllm/.venv/bin/python
mkdir -p $O

MP=$(systemctl show -p MainPID --value $U)
OP=$(ss -lntp 2>/dev/null | grep ':8000' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
EP=$(pgrep -f 'EngineCore' | tail -1)
echo "=== identity MainPID=$MP port-owner=$OP same=$([ "$MP" = "$OP" ] && echo YES || echo NO)  engine_pid=$EP"
[ "$MP" != "$OP" ] && { echo "ABORT"; exit 1; }

echo "=== DIRECT PROOF of the reader thread count (engine task comm names) ==="
for p in $(pgrep -f 'EngineCore'); do
  echo "  pid=$p so=$(grep -c 'ple_ssd_io' /proc/$p/maps 2>/dev/null) tasks=$(ls /proc/$p/task | wc -l) env=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep PLE_AIO_THREADS | tr '\n' ' ')"
  for t in /proc/$p/task/*; do cat "$t/comm" 2>/dev/null; done | sort | uniq -c | sort -rn | head -14 | sed 's/^/    /'
done
echo "  sha of live so: $(sha256sum /root/vllm/optimization/ple_ssd_io.so | cut -c1-24)"

echo "=== decode / concurrency sweep  $(date -u +%FT%TZ)"
timeout 1500 iostat -x 1 1500 > $O/iostat-decode-mt.log 2>&1 &
IO=$!
$PY $R/benchmarks/bench_server.py --model Qwen3.8-Flash-Next --label ct120-mt \
   --output $O/server-mt.json --concurrency 1 4 8 16 --tokens 128 --rounds 3 2>&1 | tail -22 | sed 's/^/  /'
kill $IO 2>/dev/null

echo "--- decode nvme accounting ---"
awk '/^Device/{for(i=1;i<=NF;i++)c[$i]=i;next}
     $1=="nvme0n1" && ("r/s" in c) && $c["r/s"]>3 {
       n++; rs+=$c["r/s"]; rk+=$c["rkB/s"]; aq+=$c["aqu-sz"]; aw+=$c["r_await"]; ut+=$c["%util"];
       if($c["r/s"]>pmx){pmx=$c["r/s"];prk=$c["rkB/s"]}
     }
     END{if(n==0){print "  no samples";exit}
       printf "  samples=%d mean r/s=%.0f mean rkB/s=%.0f (%.0f MiB/s) mean aqu-sz=%.2f mean r_await=%.2f ms mean util=%.1f%%\n",n,rs/n,rk/n,rk/n/1024,aq/n,aw/n,ut/n
       printf "  peak r/s=%.0f rkB/s=%.0f => bytes/op=%.0f\n",pmx,prk,(pmx>0?prk*1024/pmx:0)}' $O/iostat-decode-mt.log

echo "=== spec decode + host contention ==="
journalctl -u $U --no-pager -n 600 2>/dev/null | grep -E 'Mean acceptance length|Speculat|Prefix cache hit' | tail -4 | sed 's/^/  /'
echo "  loadavg: $(cat /proc/loadavg)   nproc=$(nproc)"
echo "  other CTs on this drive:"; pct status 110 2>/dev/null | grep -E '^status' | sed 's/^/    CT110 /'
free -g | head -2 | sed 's/^/  /'
echo "=== DECODE SEQUENCE FINISHED $(date -u +%FT%TZ) ==="
