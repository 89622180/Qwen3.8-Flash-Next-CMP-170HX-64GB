"""Quantize the DFlash2 draft to W8A16G128 (auto_round:auto_gptq format) with GPTQ.

Usage:
  python run_gptq.py --hessians /models/dflash2-quant/hessians.pt \
      --out /models/incoai-Qwen3.8-Flash-Next-DFlash2-W8A16G128-GPTQ
  python run_gptq.py --synthetic ...   # dry-run without captured activations

Runs on CPU or CUDA (CT104's 3080). Never touches the original draft dir.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from safetensors.torch import load_file, save_file

import gptq_core as gc

SRC = "/models/incoai-Qwen3.8-Flash-Next-DFlash2"
NLAYERS = 5

# checkpoint tensor name -> hessian key (layer-local names; the loader maps
# q/k/v -> qkv_proj and gate/up -> gate_up_proj via its stacked weights mapper)
QUANT_MAP = {}
for n in range(NLAYERS):
    QUANT_MAP[f"layers.{n}.self_attn.q_proj.weight"] = f"layers.{n}.self_attn.qkv"
    QUANT_MAP[f"layers.{n}.self_attn.k_proj.weight"] = f"layers.{n}.self_attn.qkv"
    QUANT_MAP[f"layers.{n}.self_attn.v_proj.weight"] = f"layers.{n}.self_attn.qkv"
    QUANT_MAP[f"layers.{n}.self_attn.o_proj.weight"] = f"layers.{n}.self_attn.o"
    QUANT_MAP[f"layers.{n}.mlp.gate_proj.weight"] = f"layers.{n}.mlp.gate_up"
    QUANT_MAP[f"layers.{n}.mlp.up_proj.weight"] = f"layers.{n}.mlp.gate_up"
    QUANT_MAP[f"layers.{n}.mlp.down_proj.weight"] = f"layers.{n}.mlp.down"
    QUANT_MAP[f"layers.{n}.attention_conv.kernel_projection.weight"] = (
        f"layers.{n}.attention_conv.kernel_projection")
    QUANT_MAP[f"layers.{n}.mlp_conv.kernel_projection.weight"] = (
        f"layers.{n}.mlp_conv.kernel_projection")
QUANT_MAP["fc.weight"] = "fc"


def synthetic_hessian(in_f: int, rows: int = 4096, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(rows, in_f, generator=g)
    return 2.0 * (X.t() @ X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--hessians", default="/models/dflash2-quant/hessians.pt")
    ap.add_argument("--out",
                    default="/models/incoai-Qwen3.8-Flash-Next-DFlash2-W8A16G128-GPTQ")
    ap.add_argument("--synthetic", action="store_true",
                    help="dry-run: GPTQ with synthetic isotropic Hessians")
    ap.add_argument("--no-kernel-proj", action="store_true",
                    help="keep conv kernel_projection in BF16")
    ap.add_argument("--percdamp", type=float, default=0.01)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gptq] device={device}  src={args.src}  out={args.out}")
    if device == "cuda":
        print(f"[gptq] gpu={torch.cuda.get_device_name(0)}")

    hess = None
    if not args.synthetic:
        hess = torch.load(args.hessians, map_location="cpu")
        print(f"[gptq] loaded {len(hess)} Hessians from {args.hessians}")
    else:
        print("[gptq] SYNTHETIC dry-run (format validation only)")

    src_tensors = load_file(os.path.join(args.src, "model.safetensors"))
    print(f"[gptq] source tensors: {len(src_tensors)}")

    out_tensors = {}
    report = {}
    t0 = time.time()
    for name, W in src_tensors.items():
        if name in QUANT_MAP and not (args.no_kernel_proj and
                                      "kernel_projection" in name):
            hkey = QUANT_MAP[name]
            in_f = W.shape[1]
            if hess is not None:
                entry = hess.get(hkey)
                assert entry is not None, f"missing hessian for {hkey}"
                H = entry["H"].float()
                rows = entry["rows"]
                assert H.shape == (in_f, in_f), (hkey, H.shape, in_f)
            else:
                H = synthetic_hessian(in_f)
                rows = H.shape[0]
            Wd = W.to(device)
            Hd = H.to(device)
            codes, scales = gc.gptq(Wd, Hd, percdamp=args.percdamp)
            Wq = gc.dequant(codes, scales)
            rel = ((Wq - Wd).norm() / (Wd.norm() + 1e-12)).item()
            report[name] = {"hessian": hkey, "rows": rows, "rel_err": round(rel, 5)}
            tensors = gc.export_tensors(name.rsplit(".", 1)[0],
                                        codes.cpu(), scales.cpu())
            out_tensors.update(tensors)
            print(f"[gptq] {name:55s} in={in_f:6d} rows={rows:7d} "
                  f"rel_err={rel:.4f}")
            del Wd, Hd, codes, scales, Wq
            if device == "cuda":
                torch.cuda.empty_cache()
        else:
            out_tensors[name] = W  # BF16 passthrough, byte-identical

    os.makedirs(args.out, exist_ok=True)
    save_file(out_tensors, os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})

    # config.json = original + quantization_config
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    cfg["quantization_config"] = dict(gc.QUANTIZATION_CONFIG)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    # separate file, same content (mirrors the main model's layout and
    # INCConfig.get_config_filenames())
    with open(os.path.join(args.out, "quantization_config.json"), "w") as f:
        json.dump(dict(gc.QUANTIZATION_CONFIG), f, indent=2)

    with open(os.path.join(args.out, "quant_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    src_mb = os.path.getsize(os.path.join(args.src, "model.safetensors")) / 2**20
    out_mb = os.path.getsize(os.path.join(args.out, "model.safetensors")) / 2**20
    errs = [v["rel_err"] for v in report.values()]
    print(f"[gptq] done in {time.time() - t0:.0f}s")
    print(f"[gptq] tensors: {len(out_tensors)} (quantized matrices: {len(report)})")
    print(f"[gptq] size: {src_mb:.0f} MB -> {out_mb:.0f} MB "
          f"({out_mb / src_mb * 100:.0f}%)")
    print(f"[gptq] weight rel err: mean={sum(errs)/len(errs):.4f} "
          f"max={max(errs):.4f}")
    print(f"[gptq] output: {args.out}")


if __name__ == "__main__":
    main()

