#!/usr/bin/env bash
# Install vLLM 0.33.0 + the PLE-SSD/DFlash2 arm for this checkpoint, without
# touching the existing 0.32 install.
#
# What this actually does, and why it is shaped this way:
#   * 0.33.0's own metadata asks for torch==2.14.0 / torchvision==0.29.0 /
#     flashinfer-python==0.7.2.post1. The 0.32 arm ships torch 2.13.0+cu130 and
#     flashinfer 0.7.0.post1, and of 0.33.0's 79 core dependencies exactly ONE is
#     genuinely absent (orjson). So the new engine is installed as a thin venv
#     that reuses the old venv's dependency set through a .pth file, instead of
#     re-downloading ~15 GiB of CUDA wheels into a 50 GiB container quota.
#   * vllm's compiled kernels come from the stock wheel (10 top-level .so plus
#     vllm_flash_attn/ and third_party/deep_gemm/), NOT from a source build.
#   * The arm itself (PLE-SSD offload + INC/AutoRound PLE handling + QSA fp8 +
#     DFlash2 allow-list + KV-pool diagnostics) is a pure-Python patch applied
#     on top of the installed package; the only native artifact is the
#     pre-existing optimization/ple_ssd_io.so, which is C and version-independent.
#
# Rollback = delete /root/v33 and /root/vllm-033work, and the zz-v033.conf
# drop-in. The 0.32 tree (/root/vllm-latest) is never modified by this script.
set -euo pipefail

VLLM_WHEEL_URL=${VLLM_WHEEL_URL:-https://files.pythonhosted.org/packages/8b/a8/92e278093df40fde4c4bfd6572704bb8f398d6a2280d8c82cb5efaf12047/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl}
VLLM_WHEEL_SHA=${VLLM_WHEEL_SHA:-4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa}
VENV=${VENV:-/root/v33}
WORKDIR=${WORKDIR:-/root/vllm-033work}
OLD_VENV_SITE=${OLD_VENV_SITE:-/root/vllm-latest/.venv/lib/python3.12/site-packages}
PLE_PATCH=${PLE_PATCH:-/root/n1104/ple-033-full.patch}
WHEELS=${WHEELS:-/root/wheels33}

mkdir -p "$WHEELS" "$(dirname "$PLE_PATCH")"
NAME=$(basename "$VLLM_WHEEL_URL")

# The full arm diff ships in this repo base64+gzip, because it is 64 KiB of diff.
# Decode it if the plain patch is not already there.
if [ ! -s "$PLE_PATCH" ] && [ -f "$PWD/patches/qwen38-ple-ssd-0.33.patch.gz.b64" ]; then
    base64 -d "$PWD/patches/qwen38-ple-ssd-0.33.patch.gz.b64" | gunzip > "$PLE_PATCH"
fi
if [ -s "$PLE_PATCH" ]; then
    echo "0d0436af1a1d0085f00c4fb74364a865417b0f880dd94229a7489023ed5c85a0  $PLE_PATCH" | sha256sum -c -
fi

if [ ! -s "$WHEELS/$NAME" ]; then
    curl -fL --http2 -o "$WHEELS/$NAME.part" "$VLLM_WHEEL_URL"
    echo "$VLLM_WHEEL_SHA  $WHEELS/$NAME.part" | sha256sum -c -
    mv "$WHEELS/$NAME.part" "$WHEELS/$NAME"
fi
echo "$VLLM_WHEEL_SHA  $WHEELS/$NAME" | sha256sum -c -

/usr/bin/python3.12 -m venv "$VENV"
# Reuse the 0.32 venv's dependency set. A .pth appends the path AFTER this
# venv's own site-packages, so the 0.33 vllm package wins and torch etc.
# resolve from the old venv.
echo "$OLD_VENV_SITE" > "$VENV/lib/python3.12/site-packages/_oldvenv.pth"

"$VENV/bin/pip" install --no-deps --no-index --find-links "$WHEELS" "$WHEELS/$NAME"
# 0.33.0's only core dependency that this box did not already have.
"$VENV/bin/pip" install --no-deps orjson

# The arm. Paths in the patch are repo-relative (a/vllm/...), and site-packages
# carries the same vllm/ layout, so -p1 works. tests/ hunks are refused because a
# wheel does not ship tests/, so patch exits 1 for those and nothing else.
cd "$VENV/lib/python3.12/site-packages"
patch -p1 --fuzz=3 < "$PLE_PATCH" || true
find vllm -name '*.rej' -print -delete

# Byte-parity check against the git worktree the patch was produced in, if present.
if [ -d /root/wt-033/vllm ]; then
    diff -rq -x '__pycache__' vllm /root/wt-033/vllm && echo "site-packages == worktree"
fi

mkdir -p "$WORKDIR/optimization"
cp -p /root/vllm-latest/optimization/ple_ssd_io.so "$WORKDIR/optimization/"

"$VENV/bin/python" - <<'EOF'
import vllm
print("vllm", vllm.__version__, vllm.__file__)
from vllm.models.qwen4_exp.nvidia import ple_ssd
print("ple_ssd", ple_ssd.__file__)
EOF

cat <<'EOF'
Now point the unit at it with a drop-in (zz-v033.conf):

  [Service]
  Environment=VLLM_VENV=/root/v33
  Environment=VLLM_WORKDIR=/root/vllm-033work
  Environment=QWEN_KV_DTYPE=fp8

QWEN_KV_DTYPE must be fp8, not fp8_e5m2: 0.33's QSA kernel reads an fp8 cache
only as e4m3 (uint8 byte-decode below sm_89), and asserts float8_e4m3fn for a
typed cache.

Leave LD_LIBRARY_PATH alone: it still has to name the OLD venv's cu13 lib dir,
because torch is the old venv's torch.
EOF
