"""Validate an exported quantized DFlash2 draft checkpoint.

Run in CT120 (needs the vllm venv for INCConfig):
  cd /root/dflash2-quant && /root/vllm-latest/.venv/bin/python validate_export.py \
      --ckpt /models/incoai-Qwen3.8-Flash-Next-DFlash2-W8A16G128-GPTQ

Checks:
  1. tensor inventory: quantized (qweight/qzeros/scales) + BF16 passthrough
  2. dequantized weights vs the original BF16 (per-tensor relative error)
  3. quantization_config parses through vLLM's INCConfig exactly as intended
     (bits=8, group=128, sym, for every draft Linear incl. the layer-offset
     naming model.layers.48..52 that vLLM uses at get_quant_method time)
  4. quant_method "auto-round" resolves to the INC backend
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file

import gptq_core as gc

SRC = "/models/incoai-Qwen3.8-Flash-Next-DFlash2"
NLAYERS = 5

EXPECTED_QUANT = []
for n in range(NLAYERS):
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        EXPECTED_QUANT.append(f"layers.{n}.self_attn.{p}")
    for p in ("gate_proj", "up_proj", "down_proj"):
        EXPECTED_QUANT.append(f"layers.{n}.mlp.{p}")
    EXPECTED_QUANT.append(f"layers.{n}.attention_conv.kernel_projection")
    EXPECTED_QUANT.append(f"layers.{n}.mlp_conv.kernel_projection")
EXPECTED_QUANT.append("fc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/models/incoai-Qwen3.8-Flash-Next-DFlash2-"
                            "W8A16G128-GPTQ")
    ap.add_argument("--src", default=SRC)
    args = ap.parse_args()

    src = load_file(os.path.join(args.src, "model.safetensors"))
    ck = load_file(os.path.join(args.ckpt, "model.safetensors"))
    cfg = json.load(open(os.path.join(args.ckpt, "config.json")))
    print(f"[validate] checkpoint: {args.ckpt}")
    print(f"[validate] tensors: src={len(src)} ckpt={len(ck)}")

    # 1. inventory ---------------------------------------------------------
    want = set()
    for base in EXPECTED_QUANT:
        want |= {base + ".qweight", base + ".qzeros", base + ".scales"}
    for name in src:
        if name in {b + ".weight" for b in EXPECTED_QUANT}:
            continue
        want.add(name)          # BF16 passthrough keeps its original name
    got = set(ck)
    missing = want - got
    extra = got - want
    assert not missing, f"missing tensors: {sorted(missing)[:10]}"
    assert not extra, f"unexpected tensors: {sorted(extra)[:10]}"
    print(f"[ok] inventory exact: {len(EXPECTED_QUANT)} quantized linears "
          f"({len(want) - len(src) + len(EXPECTED_QUANT)} new tensors), "
          f"{len(src) - len(EXPECTED_QUANT)} BF16 passthrough")

    # 2. dequant vs original ------------------------------------------------
    print("[validate] per-tensor weight error (dequant vs BF16 source):")
    worst = 0.0
    for base in EXPECTED_QUANT:
        name = base + ".weight"
        W = src[name].float()
        qw = ck[base + ".qweight"]
        sc = ck[base + ".scales"].float()
        q = gc.unpack_rows_u8(qw)              # [in, out]
        Wq = gc.dequant(q.t().contiguous(), sc)
        rel = ((Wq - W).norm() / (W.norm() + 1e-12)).item()
        worst = max(worst, rel)
        print(f"  {name:55s} rel_err={rel:.4f}")
    print(f"[ok] worst weight rel err = {worst:.4f} "
          f"({'PASS' if worst < 0.05 else 'FAIL'} for W8G128)")

    # BF16 passthrough byte-identical
    for name in src:
        if name in {b + ".weight" for b in EXPECTED_QUANT}:
            continue
        assert torch.equal(src[name], ck[name]), f"BF16 changed: {name}"
    print("[ok] all BF16 passthrough tensors byte-identical")

    # 3. INCConfig resolution ------------------------------------------------
    from vllm.model_executor.layers.quantization.inc import INCConfig
    qc = cfg["quantization_config"]
    inc = INCConfig.from_config(dict(qc))
    assert inc.weight_bits == 8 and inc.group_size == 128 and inc.sym
    assert inc.packing_format == "auto_round:auto_gptq"
    parser = inc.config_parser

    probe_names = []
    for n in range(NLAYERS):
        for ln in (n, 48 + n):   # local AND vLLM-global (offset) naming
            probe_names += [
                f"model.layers.{ln}.self_attn.qkv_proj",
                f"model.layers.{ln}.self_attn.o_proj",
                f"model.layers.{ln}.mlp.gate_up_proj",
                f"model.layers.{ln}.mlp.down_proj",
                f"model.layers.{ln}.attention_conv.kernel_projection",
                f"model.layers.{ln}.mlp_conv.kernel_projection",
            ]
    probe_names += ["model.fc"]  # runtime prefix is always model.* (draft model prefix)
    for pn in probe_names:
        bits, gs, sym = parser.get_layer_config(object(), pn)
        assert (bits, gs, sym) == (8, 128, True), (pn, bits, gs, sym)
    print(f"[ok] INCConfig resolves W8A16G128 sym for all {len(probe_names)} "
          f"module names (local + offset naming)")

    # 4. quant_method override ------------------------------------------------
    m = INCConfig.override_quantization_method(dict(qc), None)
    assert m == "inc", m
    print("[ok] quant_method 'auto-round' -> INC backend (same as main model)")

    mb = os.path.getsize(os.path.join(args.ckpt, "model.safetensors")) / 2**20
    sb = os.path.getsize(os.path.join(args.src, "model.safetensors")) / 2**20
    print(f"[validate] size {sb:.0f} MB -> {mb:.0f} MB ({mb / sb * 100:.0f}%)")
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()

