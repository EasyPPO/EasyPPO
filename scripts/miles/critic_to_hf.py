"""Convert a critic checkpoint written by Miles (Megatron torch_dist) to Hugging Face parameter names.

Miles' own tools/convert_torch_dist_to_hf.py needs a `common.pt` that the trainer's checkpoints do not have, and
would write the one-row value head as `lm_head.weight`. This script takes the model arguments from the base
model's Hugging Face config instead, and writes

  OUT/critic.safetensors   the backbone, under the names the base model uses
  OUT/value_head.pt        {"weight": [1, hidden], "bias": [1] or None}: the critic's output layer
  OUT/meta.json            where it came from, and the check of the names against the base model

The backbone is loaded on top of the base model (agentic_critic.critic_checkpoint.load_critic), so the files of
the base model that the critic does not train (tokenizer, vision tower) are not copied.

Runs inside the Miles image (needs Miles importable), CPU only:
  python scripts/miles/critic_to_hf.py --input-dir CKPT/checkpoints_critic/iter_XXXXXXX --base-hf-dir MODEL --out OUT
"""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import safetensors
import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp

ap = argparse.ArgumentParser()
ap.add_argument("--input-dir", required=True)
ap.add_argument("--base-hf-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--miles", default=str(Path(__file__).resolve().parents[2]))
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("convert_torch_dist_to_hf", Path(args.miles) / "tools/convert_torch_dist_to_hf.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf  # noqa: E402
from miles.utils.hf_utils.config import load_hf_config  # noqa: E402

hf_config = load_hf_config(args.base_hf_dir)
text = getattr(hf_config, "text_config", hf_config)
model_args = SimpleNamespace(
    hidden_size=text.hidden_size, num_attention_heads=text.num_attention_heads, num_query_groups=text.num_key_value_heads,
    kv_channels=getattr(text, "head_dim", None), num_layers=text.num_hidden_layers, vocab_size=None, sglang_enable_ep_moe=False,
)  # fmt: skip
model_name = type(hf_config).__name__.lower()

state = {}
dist_cp.state_dict_loader._load_state_dict(
    state, storage_reader=tool.WrappedStorageReader(args.input_dir), planner=tool.EmptyStateDictLoadPlanner(), no_dist=True
)
backbone, head, bias = {}, None, None
for name, param in tool.get_named_params(model_args, state):
    if name == "module.module.output_layer.weight":
        assert param.shape == (1, text.hidden_size), f"not a value head: {tuple(param.shape)}"
        head = param.float().clone()
        continue
    if name == "module.module.output_layer.bias":
        bias = param.float().clone()
        continue
    for hf_name, hf_param in convert_to_hf(model_args, model_name, name, param):
        backbone[hf_name] = hf_param.contiguous()
assert head is not None, "the checkpoint has no output layer"

index = {}  # parameter name -> shape, read from the headers of the base model's files
for f in sorted(Path(args.base_hf_dir).glob("*.safetensors")):
    with safetensors.safe_open(str(f), framework="pt") as reader:
        index.update({k: tuple(reader.get_slice(k).get_shape()) for k in reader.keys()})
wrong_shape = sorted(k for k, v in backbone.items() if k in index and tuple(v.shape) != index[k])
assert not wrong_shape, f"{len(wrong_shape)} parameters have another shape than in the base model, e.g. {wrong_shape[:5]}"
unknown = sorted(k for k in backbone if k not in index)
untouched = sorted(k for k in index if k not in backbone)
assert not unknown, f"{len(unknown)} converted names are not parameters of the base model, e.g. {unknown[:5]}"

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
safetensors.torch.save_file(backbone, str(out / "critic.safetensors"))
torch.save({"weight": head, "bias": bias}, out / "value_head.pt")
(out / "meta.json").write_text(json.dumps({
    "input_dir": args.input_dir, "base_hf_dir": args.base_hf_dir, "converted_parameters": len(backbone),
    "base_parameters_left_as_they_are": len(untouched), "examples_left": untouched[:8],
    "value_head_norm": float(head.norm()), "value_head_bias": None if bias is None else float(bias),
}, indent=1))  # fmt: skip
print(f"{len(backbone)} parameters converted; {len(untouched)} of the base model left as they are (e.g. {untouched[:4]})")
print(f"value head norm {float(head.norm()):.4f}; written to {out}")
