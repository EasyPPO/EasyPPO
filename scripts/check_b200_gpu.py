#!/usr/bin/env python3
"""Small B200 kernel check: no model download, dataset, or training job."""

import json

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel


def check_fla():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    assert qwen.chunk_gated_delta_rule is chunk_gated_delta_rule
    inputs = [torch.randn(1, 256, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    inputs += [
        -torch.rand(1, 256, 4, device="cuda", dtype=torch.float32),
        torch.rand(1, 256, 4, device="cuda", dtype=torch.bfloat16),
    ]
    grad_output = torch.randn_like(inputs[2])
    results = []
    for kernel in (chunk_gated_delta_rule, qwen.torch_chunk_gated_delta_rule):
        tensors = [value.clone().requires_grad_() for value in inputs]
        output, _ = kernel(
            *tensors[:3],
            g=tensors[3],
            beta=tensors[4],
            use_qk_l2norm_in_kernel=True,
            output_final_state=False,
        )
        gradients = torch.autograd.grad(output, tensors, grad_outputs=grad_output)
        assert torch.isfinite(output).all()
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        results.append((output.detach(), gradients))

    def relative_error(actual, expected):
        return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()

    output_error = relative_error(results[0][0], results[1][0])
    gradient_errors = [relative_error(a, b) for a, b in zip(results[0][1], results[1][1], strict=True)]
    assert output_error < 0.03, output_error
    assert max(gradient_errors) < 0.05, gradient_errors
    return {"output_relative_error": output_error, "gradient_relative_errors_q_k_v_g_beta": gradient_errors}


def check_qwen_model():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=["linear_attention", "full_attention"],
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        use_cache=False,
    )
    model = Qwen3_5ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).train()
    assert model.config._attn_implementation == "flash_attention_2"
    assert model.model.layers[0].linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
    tokens = torch.randint(0, config.vocab_size, (1, 128), device="cuda")
    # Exercise the same packed position-ID layout used by remove-padding training.
    position_ids = torch.arange(64, device="cuda").repeat(2).unsqueeze(0)
    output = model(input_ids=tokens, position_ids=position_ids, labels=tokens)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    return {"loss": output.loss.item(), "parameters_with_gradients": len(gradients)}


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
    q, k, v = [torch.randn(2, 128, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
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
    fla_result = check_fla()
    model_result = check_qwen_model()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "ok",
                "device": torch.cuda.get_device_name(),
                "compute_capability": capability,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "checks": [
                    "bf16_matmul",
                    "flash_attention_vs_sdpa",
                    "flash_attention_backward",
                    "fla_vs_torch_forward_backward",
                    "qwen3_5_fla_fa2_forward_backward",
                ],
                "fla": fla_result,
                "qwen3_5": model_result,
                "pretrained_models_loaded": False,
                "training_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
