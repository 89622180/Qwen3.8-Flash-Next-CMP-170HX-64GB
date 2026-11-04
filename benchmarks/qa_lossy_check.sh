#!/usr/bin/env bash
# Byte-diff + substring score for the two 1024-out harness caveats: "is this arm's speed
# win paid for in output quality?" (FLy is LOSSY; MoE backend / KV dtype swaps also move
# the greedy token path.)
#
# usage: TAG=base bash qa_lossy_check.sh   (run once per arm, then diff /root/n1104/qa)
# The score is deliberately coarse (expected substring inside 48 tokens); the digest is
# what tells you how much the text actually moved.
PY=${PY:-/root/v33/bin/python}
TAG=${TAG:?set TAG}
mkdir -p /root/n1104/qa
"$PY" - "$TAG" <<'PYEOF'
import hashlib, sys, requests
tag = sys.argv[1]
URL = "http://127.0.0.1:8000/v1/completions"
CASES = [
    ("中国的首都是哪座城市？只回答城市名。", "北京"),
    ("水的化学式是什么？", "H2O"),
    ("平年一年有多少天？只回答数字。", "365"),
    ("3 的 4 次方等于多少？只回答数字。", "81"),
    ("《红楼梦》前八十回的作者是谁？", "曹雪芹"),
    ("地球到太阳的平均距离约为多少亿公里？", "1.5"),
    ("光在真空中的速度约为多少 km/s？", "30"),
    ("127 是质数吗？只回答 是 或 否。", "是"),
    ("长江发源于哪座山脉？", "唐古拉"),
    ("声音在 15℃ 空气中约为多少 m/s？", "340"),
]
ok = 0
for i, (q, exp) in enumerate(CASES):
    r = requests.post(URL, json={"model": "Qwen3.8-Flash-Next", "prompt": q,
                                 "max_tokens": 48, "temperature": 0, "stream": False},
                      timeout=300)
    txt = r.json()["choices"][0]["text"]
    hit = exp in txt
    ok += hit
    open("/root/n1104/qa/%s-%d.txt" % (tag, i), "w").write(txt)
    print("  %2d %s %s | %s" % (i, "OK  " if hit else "MISS",
                                hashlib.md5(txt.encode()).hexdigest()[:10],
                                txt.replace("\n", " ")[:58]))
print("QA tag=%s score=%d/%d" % (tag, ok, len(CASES)))
print("compare arms with:  for i in 0 1 2 3 4 5 6 7 8 9; do cmp -s <(cat /root/n1104/qa/base-$i.txt) <(cat /root/n1104/qa/fly-$i.txt) || echo differ $i; done")
PYEOF
