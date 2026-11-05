"""Native checkpoint loading preserves a fresh value head and resumes trained heads."""

import ast
import logging
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class _ValueHead(torch.nn.Linear):
    def __init__(self):
        super().__init__(4, 1)


class _ValueModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.output_layer = _ValueHead()
        self.post_process = True

    def sharded_state_dict(self):
        # LanguageModule requires output_layer.weight on post-process stages;
        # hiding only the module would leave this native contract inconsistent.
        state = {name: SimpleNamespace(data=value) for name, value in self.state_dict().items()}
        if self.post_process:
            state["output_layer.weight"].allow_shape_mismatch = True
        return state


def _checkpoint_loader(args, native_loader):
    # Execute the production entry point, replacing only unavailable Megatron
    # imports. The model, head, state dict and weight mutations are real torch.
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/checkpoint.py"
    names = {"_hide_critic_value_head", "load_checkpoint"}
    nodes = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in names]
    scope = {
        "get_args": lambda: args,
        "Path": Path,
        "logger": logging.getLogger(__name__),
        "_is_dir_nonempty": lambda path: True,
        "_is_megatron_checkpoint": lambda path: True,
        "is_dsv4_model": lambda args: False,
        "_load_checkpoint_megatron": native_loader,
        "is_lora_enabled": lambda args: False,
        "contextmanager": contextmanager,
        "nullcontext": nullcontext,
        "unwrap_model": lambda models: models,
        "LinearForLastLayer": _ValueHead,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), scope)
    return scope["load_checkpoint"]


@pytest.mark.parametrize("fail", [False, True])
def test_pretrained_native_load_excludes_and_restores_fresh_scalar_head(tmp_path, fail):
    model = _ValueModel()
    head = model.output_layer
    original = {name: value.clone() for name, value in head.state_dict().items()}
    parameters = dict(model.named_parameters())
    masters = {name: parameter.detach().clone().float() for name, parameter in parameters.items()}
    args = SimpleNamespace(load=str(tmp_path), _initialize_critic_value_head=True)

    def native_loader(**kwargs):
        [chunk] = kwargs["ddp_model"]
        assert not any(name.startswith("output_layer.") for name in chunk.state_dict())
        assert not any(name.startswith("output_layer.") for name in chunk.sharded_state_dict())
        if fail:
            raise OSError("checkpoint read failed")
        with torch.no_grad():
            chunk.backbone.weight.fill_(0.125)
            # Native optimizer param groups still own every Parameter, including
            # the temporarily hidden head, when refreshing their master copies.
            for name, parameter in parameters.items():
                masters[name].copy_(parameter)
        return 0, 0

    load = _checkpoint_loader(args, native_loader)
    if fail:
        with pytest.raises(OSError, match="checkpoint read failed"):
            load([model], None, None, None, False)
    else:
        assert load([model], None, None, None, False) == (0, 0, False)
        torch.testing.assert_close(model.backbone.weight, torch.full_like(model.backbone.weight, 0.125))
        torch.testing.assert_close(masters["backbone.weight"], model.backbone.weight)
    assert model.output_layer is head
    assert model.post_process is True
    for name, value in original.items():
        torch.testing.assert_close(model.output_layer.state_dict()[name], value, atol=0, rtol=0)
        torch.testing.assert_close(masters[f"output_layer.{name}"], value, atol=0, rtol=0)


@pytest.mark.parametrize("skip_optimizer", [False, True])
def test_native_critic_resume_restores_value_head_and_keeps_checkpoint_step(tmp_path, skip_optimizer):
    model = _ValueModel()
    args = SimpleNamespace(
        load=str(tmp_path), _initialize_critic_value_head=False, no_load_optim=skip_optimizer, ckpt_step=39
    )

    def native_loader(**kwargs):
        [chunk] = kwargs["ddp_model"]
        assert "output_layer.weight" in chunk.state_dict()
        assert "output_layer.weight" in chunk.sharded_state_dict() and chunk.post_process
        assert args.ckpt_step == 39 and args.no_load_optim == skip_optimizer
        with torch.no_grad():
            chunk.output_layer.weight.fill_(1.5)
        return 39, 0

    assert _checkpoint_loader(args, native_loader)([model], None, None, None, False) == (39, 0, False)
    torch.testing.assert_close(model.output_layer.weight, torch.full_like(model.output_layer.weight, 1.5))


def test_actor_checkpoint_load_keeps_its_vocabulary_head(tmp_path):
    model = torch.nn.Module()
    model.output_layer = torch.nn.Linear(4, 128)
    args = SimpleNamespace(load=str(tmp_path), _initialize_critic_value_head=False)

    def native_loader(**kwargs):
        assert kwargs["ddp_model"][0].output_layer.weight.shape == (128, 4)
        return 9, 0

    assert _checkpoint_loader(args, native_loader)([model], None, None, None, False) == (9, 0, False)
