"""GPTQ W8A16 group-128 symmetric quantization + auto_round:auto_gptq packing.

Format oracle (verified against the running main model
/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP, tensor
model.language_model.layers.0.linear_attn.in_proj_qkv, in=2560/out=10240):

  qweight: [in//4, out]  int32   4x int8 packed along the IN dim,
                              byte k (LSB first) = in-row 4*i+k
                              (== vllm quant_utils.pack_quantized_values_into_int32
                               with packed_dim=0)
  qzeros : [in//128, out//4] int32, constant 0x7F7F7F7F (inert for sym:
                              kernel uses the uint8b128 bias)
  scales : [in//128, out]  float16  W[i,j] = scales[i//128, j] * (q[i,j] - 128)

Dequant semantics: uint8b128 => value = stored - 128; scales may be signed
(we emit positive scales, which is a valid subset).
"""

from __future__ import annotations

import torch

MAXQ = 127                 # symmetric code range; stored code = code + 128
QZ_FILL = 0x7F7F7F7F       # matches the main model's qzeros byte pattern
GROUP = 128
BITS = 8


# ----------------------------------------------------------------------------
# packing / unpacking (byte-order identical to vLLM's own packer)
# ----------------------------------------------------------------------------

def pack_rows_u8(q_t: torch.Tensor) -> torch.Tensor:
    """q_t [in, out] uint8 biased codes -> [in//4, out] int32, LSB = first row."""
    in_f, out_f = q_t.shape
    assert in_f % 4 == 0, in_f
    assert out_f % 4 == 0, out_f
    res = torch.zeros((in_f // 4, out_f), dtype=torch.int32)
    for i in range(4):
        res |= q_t[i::4].to(torch.int32) << (8 * i)
    return res


def unpack_rows_u8(qw: torch.Tensor) -> torch.Tensor:
    """qw [in//4, out] int32 -> [in, out] uint8 biased codes."""
    in_p, out_f = qw.shape
    res = torch.empty((in_p * 4, out_f), dtype=torch.uint8)
    for i in range(4):
        res[i::4] = ((qw >> (8 * i)) & 0xFF).to(torch.uint8)
    return res


def make_qzeros(n_groups: int, out_f: int) -> torch.Tensor:
    assert out_f % 4 == 0, out_f
    return torch.full((n_groups, out_f // 4), QZ_FILL, dtype=torch.int32)


# ----------------------------------------------------------------------------
# quantizers
# ----------------------------------------------------------------------------

def rtn(W: torch.Tensor, group: int = GROUP):
    """Round-to-nearest baseline. W [out, in] -> codes [out,in] uint8, scales."""
    out_f, in_f = W.shape
    codes = torch.zeros(out_f, in_f, dtype=torch.uint8)
    scales = torch.zeros(in_f // group, out_f, dtype=torch.float32)
    for g in range(in_f // group):
        gs, ge = g * group, (g + 1) * group
        Wg = W[:, gs:ge]
        s = Wg.abs().amax(dim=1) / MAXQ
        s = torch.where(s == 0, torch.ones_like(s), s)
        c = torch.clamp(torch.round(Wg / s.unsqueeze(1)), -MAXQ, MAXQ)
        codes[:, gs:ge] = (c + 128).to(torch.uint8)
        scales[g] = s
    return codes, scales


def gptq(W: torch.Tensor, H: torch.Tensor, group: int = GROUP,
         percdamp: float = 0.01, blocksize: int = GROUP):
    """Canonical GPTQ (IST-DASLab) with group-wise symmetric scales.

    W: [out, in] float32; H: [in, in] float32 == 2 X^T X (activation Hessian).
    Returns codes [out, in] uint8 (biased +128), scales [in//group, out] f32.
    """
    assert blocksize == group, "block size must equal group size here"
    W = W.float().clone()
    H = H.float().clone()
    out_f, in_f = W.shape
    assert in_f % group == 0, (in_f, group)
    dev = W.device

    dead = torch.diagonal(H) <= 0
    if dead.any():
        W[:, dead] = 0
        H[dead.nonzero().flatten(), dead.nonzero().flatten()] = 1.0
    damp = percdamp * torch.diagonal(H).mean()
    eye = torch.eye(in_f, device=dev, dtype=H.dtype)
    H = H + damp * eye

    Hinv = None
    d = damp
    for _ in range(8):
        try:
            U = torch.linalg.cholesky(H, upper=True)
            Hinv = torch.linalg.cholesky(
                torch.cholesky_inverse(U, upper=True), upper=True)
            break
        except Exception:
            d = d * 10.0 if d > 0 else damp
            H = H + d * eye
    assert Hinv is not None, "Cholesky failed after damping retries"

    codes = torch.zeros(out_f, in_f, dtype=torch.uint8)
    scales = torch.zeros(in_f // group, out_f, dtype=torch.float32)

    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        g = i1 // group
        s = W1.abs().amax(dim=1) / MAXQ
        s = torch.where(s == 0, torch.ones_like(s), s)
        scales[g] = s
        for j in range(count):
            col = i1 + j
            w = W1[:, j]
            dd = Hinv[col, col]
            c = torch.clamp(torch.round(w / s), -MAXQ, MAXQ)
            q = c * s
            Q1[:, j] = q
            codes[:, col] = (c + 128).to(torch.uint8)
            err = (w - q) / dd
            E1[:, j] = err
            if j + 1 < count:
                W1[:, j + 1:] -= err.unsqueeze(1) * Hinv[col, col + 1:i2].unsqueeze(0)
        if i2 < in_f:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
        W[:, i1:i2] = Q1
    return codes, scales


def dequant(codes: torch.Tensor, scales: torch.Tensor, group: int = GROUP):
    """codes [out, in] uint8 biased, scales [in//group, out] -> W [out, in] f32."""
    out_f, in_f = codes.shape
    W = torch.empty(out_f, in_f, dtype=torch.float32, device=codes.device)
    for g in range(in_f // group):
        gs, ge = g * group, (g + 1) * group
        W[:, gs:ge] = (codes[:, gs:ge].float() - 128.0) * scales[g].unsqueeze(1)
    return W


# ----------------------------------------------------------------------------
# export helpers
# ----------------------------------------------------------------------------

def export_tensors(prefix: str, codes: torch.Tensor, scales: torch.Tensor,
                   group: int = GROUP) -> dict:
    """Build the checkpoint tensors for one quantized Linear."""
    out_f, in_f = codes.shape
    q_t = codes.t().contiguous()                       # [in, out]
    qw = pack_rows_u8(q_t)                             # [in//4, out]
    qz = make_qzeros(in_f // group, out_f)             # [in//group, out//4]
    sc = scales.contiguous().to(torch.float16)         # [in//group, out]
    return {
        prefix + ".qweight": qw,
        prefix + ".qzeros": qz,
        prefix + ".scales": sc,
    }


QUANTIZATION_CONFIG = {
    "bits": BITS,
    "group_size": GROUP,
    "sym": True,
    "data_type": "int",
    "packing_format": "auto_round:auto_gptq",
    "quant_method": "auto-round",
    "block_name_to_quantize": "model",
    "backend": "auto",
    "desc_act": False,
    "iters": 0,
    "autoround_version": "0.16.0-local-gptq",
}

