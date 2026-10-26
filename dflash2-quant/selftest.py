"""Self-test for gptq_core: correctness of the GPTQ math and the packing.

Run inside CT120: /root/vllm-latest/.venv/bin/python selftest.py
"""
import torch

import gptq_core as gc

torch.manual_seed(0)


def rel_err(A, B):
    return (A - B).norm() / (A.norm() + 1e-12)


def test_pack_roundtrip():
    q = torch.randint(0, 256, (256, 512), dtype=torch.uint8)
    qw = gc.pack_rows_u8(q)
    back = gc.unpack_rows_u8(qw)
    assert torch.equal(back, q), "pack/unpack roundtrip failed"
    print("[ok] pack/unpack roundtrip (LSB-first, in-dim)")


def test_pack_matches_vllm():
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32,
    )
    from vllm.scalar_type import scalar_types
    q = torch.randint(0, 256, (256, 512), dtype=torch.int32)  # uint8 values as i32
    mine = gc.pack_rows_u8(q.to(torch.uint8))
    theirs = pack_quantized_values_into_int32(q, scalar_types.uint8b128, packed_dim=0)
    assert torch.equal(mine, theirs), "byte order mismatch vs vLLM packer"
    print("[ok] byte order identical to vLLM pack_quantized_values_into_int32")


def test_gptq_beats_rtn():
    out_f, in_f, rows = 256, 1024, 8192
    W = torch.randn(out_f, in_f) * 0.02
    # anisotropic activations: correlated input features (like real hidden states)
    Z = torch.randn(rows, in_f)
    M = torch.randn(in_f, in_f) / (in_f ** 0.5)
    X = Z @ M
    H = 2.0 * (X.t() @ X)

    codes_r, s_r = gc.rtn(W)
    W_rtn = gc.dequant(codes_r, s_r)
    codes_g, s_g = gc.gptq(W, H)
    W_gptq = gc.dequant(codes_g, s_g)

    # activation-weighted error: || X (W_q - W)^T ||_F
    def act_err(Wq):
        D = (Wq - W).t()          # [in, out]
        return (X @ D).norm() / (X @ W.t()).norm()

    e_rtn = act_err(W_rtn)
    e_gptq = act_err(W_gptq)
    print(f"[info] act-weighted rel err  RTN={e_rtn:.5f}  GPTQ={e_gptq:.5f}")
    assert e_gptq < e_rtn * 0.9, "GPTQ should clearly beat RTN"
    print("[ok] GPTQ beats RTN on activation-weighted error")

    # plain weight error should also not blow up
    w_rtn = rel_err(W_rtn, W)
    w_gptq = rel_err(W_gptq, W)
    print(f"[info] weight rel err       RTN={w_rtn:.5f}  GPTQ={w_gptq:.5f}")
    assert w_gptq < 0.05, "weight error too large for W8G128"

    # codes in range
    assert codes_g.min() >= 1 and codes_g.max() <= 255
    # export -> unpack -> dequant: codes must roundtrip losslessly; the only
    # difference allowed is the fp32->fp16 cast of scales (~1e-3 relative).
    tensors = gc.export_tensors("m", codes_g, s_g)
    q_back = gc.unpack_rows_u8(tensors["m.qweight"])      # [in, out]
    codes_back = q_back.t().contiguous()                   # [out, in]
    assert torch.equal(codes_back, codes_g), "code roundtrip mismatch"
    W_back = gc.dequant(codes_back, tensors["m.scales"].float())
    assert rel_err(W_back, W_gptq) < 0.01, "fp16 scale cast error too large"
    print("[ok] export -> load -> dequant roundtrip is lossless")

    # shapes/dtypes match the main-model format oracle
    assert tensors["m.qweight"].dtype == torch.int32
    assert tensors["m.qzeros"].dtype == torch.int32
    assert tensors["m.scales"].dtype == torch.float16
    assert tensors["m.qweight"].shape == (in_f // 4, out_f)
    assert tensors["m.qzeros"].shape == (in_f // 128, out_f // 4)
    assert tensors["m.scales"].shape == (in_f // 128, out_f)
    assert torch.all(tensors["m.qzeros"] == 0x7F7F7F7F)
    print("[ok] exported shapes/dtypes match the auto_round:auto_gptq oracle")


def test_dequant_oracle():
    # Unpack a real W8 tensor from the main model and sanity-check semantics.
    import json, struct
    from safetensors import safe_open
    base = "/models/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP"
    idx = json.load(open(f"{base}/model.safetensors.index.json"))
    name = "model.language_model.layers.0.linear_attn.in_proj_qkv"
    shard = idx["weight_map"][name + ".qweight"]
    with safe_open(f"{base}/{shard}", framework="pt") as f:
        qw = f.get_tensor(name + ".qweight")
        qz = f.get_tensor(name + ".qzeros")
        sc = f.get_tensor(name + ".scales")
    q = gc.unpack_rows_u8(qw)                 # [in, out]
    W = gc.dequant(q.t().contiguous(), sc.float())
    std = W.std().item()
    frac_neg = (W < 0).float().mean().item()
    print(f"[info] main-model W8 dequant: std={std:.5f}, frac<0={frac_neg:.3f}, "
          f"q range=[{q.min()},{q.max()}]")
    assert 0.005 < std < 0.05, "implausible weight std"
    assert 0.35 < frac_neg < 0.65, "dequant not zero-centered -> bias semantics wrong"
    assert torch.all(qz == 0x7F7F7F7F)
    print("[ok] main-model tensor dequantizes to plausible zero-centered weights")


if __name__ == "__main__":
    test_pack_roundtrip()
    test_pack_matches_vllm()
    test_gptq_beats_rtn()
    test_dequant_oracle()
    print("ALL SELFTESTS PASSED")

