#!/usr/bin/env python3
"""Standalone test of the PLE reader: byte-exactness (vs an independent buffered
os.pread, and vs the same .so built single-threaded) plus rows/s scaling."""
import array, ctypes, hashlib, os, random, sys

SO = sys.argv[1] if len(sys.argv) > 1 else "/root/vllm/optimization/ple_ssd_io.so"
ROWS = int(os.environ.get("T_ROWS", "49152"))     # rows demanded by one 2048-token chunk
ROW_BYTES = 320

lib = ctypes.CDLL(SO, use_errno=True)
lib.rows_open.argtypes = [ctypes.c_uint]
lib.rows_open.restype = ctypes.c_void_p
lib.rows_close.argtypes = [ctypes.c_void_p]
lib.rows_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
lib.rows_read.restype = ctypes.c_int

M = "/models4k/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
fds, bfds, spans = [], [], []
for p in sorted(os.listdir(M)):
    if not p.endswith(".safetensors"):
        continue
    full = os.path.join(M, p)
    size = os.stat(full).st_size
    if size < (2 << 30):
        continue
    with open(full, "rb") as f:
        hdr = int.from_bytes(f.read(8), "little")
    start = 8 + hdr
    fd = os.open(full, os.O_RDONLY | os.O_DIRECT)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    fds.append(fd)
    bfds.append(os.open(full, os.O_RDONLY))
    spans.append((start, size - start))
print("reader=%s  PLE_AIO_THREADS=%s  shards=%d  span=%.1f GiB  rows=%d" % (
    SO, os.environ.get("PLE_AIO_THREADS", "unset->1"), len(fds),
    sum(b - a for a, b in spans) / 2**30, ROWS))

random.seed(7331)
pick = []
for _ in range(ROWS):
    k = random.randrange(len(fds))
    lo, hi = spans[k]
    pick.append((k, lo + random.randrange((hi - ROW_BYTES) // ROW_BYTES) * ROW_BYTES))

fdarr = array.array("i", [fds[k] for k, _ in pick])
offarr = array.array("Q", [o for _, o in pick])
buf = ctypes.create_string_buffer(ROWS * ROW_BYTES)

r = lib.rows_open(256)
if not r:
    raise SystemExit("rows_open failed: %s" % os.strerror(ctypes.get_errno()))
t0 = time_ = 0.0
import time
t0 = time.time()
rc = lib.rows_read(r, fdarr.buffer_info()[0], offarr.buffer_info()[0],
                   ROWS, ROW_BYTES, ctypes.byref(buf))
el = time.time() - t0
lib.rows_close(r)
if rc:
    raise SystemExit("rows_read returned %d (%s)" % (rc, os.strerror(-rc)))
data = buf.raw
print("  rows/s=%8.0f   %.3f s for %d rows   sha256=%s" % (
    ROWS / el, el, ROWS, hashlib.sha256(data).hexdigest()))

bad = 0
for i, (k, off) in enumerate(pick[:400]):
    if os.pread(bfds[k], ROW_BYTES, off) != data[i * ROW_BYTES:(i + 1) * ROW_BYTES]:
        bad += 1
uniq = len({data[i * ROW_BYTES:(i + 1) * ROW_BYTES] for i in range(ROWS)})
print("  vs buffered os.pread (400 sampled rows): mismatches=%d   unique rows=%d/%d" % (bad, uniq, ROWS))
for fd in fds + bfds:
    os.close(fd)
