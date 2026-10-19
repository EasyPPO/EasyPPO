"""Load a Miles-trained critic using Hugging Face model classes.

scripts/miles/critic_to_hf.py writes the backbone under the base model's parameter names and the value head
separately. The critic is the base model with those parameters replaced; what the critic never trained (the
vision tower, the language-model head) stays as in the base model and is not used.
"""

import json
from pathlib import Path

import torch


def load_critic(base_dir: str, critic_dir: str, device: str = "cpu", dtype=torch.float32):
    """Returns (model, value_head): `value_head(hidden)` maps last-layer hidden states [..., H] to values [...]."""
    import safetensors.torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_dir)
    state = safetensors.torch.load_file(str(Path(critic_dir) / "critic.safetensors"))
    meta = json.loads((Path(critic_dir) / "meta.json").read_text())
    own = model.state_dict()
    # The checkpoint file names parameters as the vision-language model does; the text-only model that
    # AutoModelForCausalLM builds has no `language_model` level.
    state = {(k if k in own else k.replace("model.language_model.", "model.", 1)): v for k, v in state.items()}
    missing = sorted(k for k in state if k not in own)
    assert not missing, f"{len(missing)} critic parameters are not in the model, e.g. {missing[:5]}"
    assert len(state) == meta["converted_parameters"]
    model.load_state_dict(state, strict=False)
    model = model.to(device=device, dtype=dtype).eval()
    saved = torch.load(Path(critic_dir) / "value_head.pt")
    head = torch.nn.Linear(saved["weight"].shape[1], 1, bias=saved.get("bias") is not None)
    head.weight.data.copy_(saved["weight"])
    if saved.get("bias") is not None:
        head.bias.data.copy_(saved["bias"].reshape(1))
    head = head.to(device=device, dtype=dtype).eval()
    return model, lambda hidden: head(hidden.to(dtype)).squeeze(-1)
