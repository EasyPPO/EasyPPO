#!/usr/bin/env python3
"""Small B200 kernel check: no model download, dataset, or training job."""

import json

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    assert torch.cuda.is_available(), "A CUDA GPU is required"
    capability = torch.cuda.get_device_capability()
    assert capability == (10, 0), f"Expected B200 SM100, got {capability}"
    torch.manual_seed(0)
    device = torch.device("cuda")

    # Exercise BF16 GEMM, then FlashAttention's native forward/backward kernels.
    a = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
    product = a @ a.T
    assert torch.isfinite(product).all()
    q, k, v = [
        torch.randn(2, 128, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            q.float().transpose(1, 2),
            k.float().transpose(1, 2),
            v.float().transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)
    torch.testing.assert_close(output.float(), expected, atol=0.03, rtol=0.03)
    output.float().square().mean().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    torch.cuda.synchronize()
    print(json.dumps({
        "status": "ok",
        "device": torch.cuda.get_device_name(),
        "compute_capability": capability,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "checks": ["bf16_matmul", "flash_attention_vs_sdpa", "flash_attention_backward"],
        "models_loaded": False,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
