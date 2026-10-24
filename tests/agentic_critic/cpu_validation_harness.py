"""CPU substitutes for CUDA allocation, Megatron scheduling and object transport.

Replay loading, DP partitioning, microbatch packing, response shifting, aggregation
and native dumps all use production code. This does not test CUDA kernels or NCCL.
"""

import ast
import copy
import dataclasses
import random
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.data.rollout import get_batch, get_data_iterator, get_rollout_data
from miles.backends.training_utils.loss.hub.logit_processors import get_values
from miles.backends.training_utils.metrics import train_dump
from miles.backends.training_utils.metrics.log_utils import aggregate_forward_results
from miles.utils import object_store

ROOT = Path(__file__).resolve().parents[2]


def isolated_definition(path, name, scope, *, owner=None):
    """Keep the actual function body; avoid importing GPU-only module dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    if owner:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    if name == "evaluate_critic":
        node.decorator_list = []  # logging wrapper only
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), scope)
    return scope[name]


class MemoryStore:
    def __init__(self):
        self.rows = {}
        self.next_id = 0
        self.releases = 0

    def put(self, *, value, value_spec):
        ref = object_store._MooncakeStoreObjectRef(payload=self.next_id)
        self.next_id += 1
        self.rows[ref.payload] = copy.deepcopy(value)
        return ref

    def get(self, ref):
        def release(value):
            self.releases += 1

        return object_store.ObjectStoreGetResult(copy.deepcopy(self.rows[ref.payload]), release)

    def remove(self, ref):
        del self.rows[ref.payload]


class TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, input_ids, **kwargs):
        # Depend on the preceding token, so an off-by-one response slice fails.
        return self.dropout(input_ids.float()).unsqueeze(-1) * self.weight


class NoDumper:
    def __init__(self, *args, **kwargs):
        pass

    def wrap_forward_step(self, fn):
        return fn

    def finalize(self, model):
        pass


def make_worker(monkeypatch, args, *, dp_size=8):
    state = SimpleNamespace(
        effective_dp=SimpleNamespace(rank=0, size=dp_size),
        cp=SimpleNamespace(rank=0, size=1),
        tp=SimpleNamespace(rank=0, size=1),
        is_pp_last_stage=True,
        vpp_size=1,
    )
    monkeypatch.setattr(parallel, "_parallel_state", state)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: state.effective_dp.rank)
    store = MemoryStore()
    monkeypatch.setattr(object_store, "get_instance", lambda: store)

    def schedule(*, forward_step_func, data_iterator, model, num_microbatches, **kwargs):
        assert kwargs["forward_only"] and kwargs["collect_non_loss_data"]
        results = []
        for _ in range(num_microbatches):
            output, callback = forward_step_func(data_iterator[0], model[0])
            assert not output.requires_grad
            results.append(callback(output))
        return results

    forward = isolated_definition(
        "miles/backends/megatron_utils/model.py",
        "forward_only",
        dict(
            torch=torch,
            partial=partial,
            get_batch=get_batch,
            get_packed_seq_params=lambda batch, args: None,
            DumperMegatronUtil=NoDumper,
            DumperPhase=SimpleNamespace(FWD_ONLY=0),
            get_model_config=lambda model: SimpleNamespace(timers=None),
            get_forward_backward_func=lambda: schedule,
            mpu=SimpleNamespace(is_pipeline_last_stage=lambda: True),
            aggregate_forward_results=aggregate_forward_results,
        ),
    )
    # Use real RandomState capture/restore, replacing only Megatron's CUDA RNG API.
    random_state = isolated_definition(
        "miles/backends/megatron_utils/initialize.py",
        "RandomState",
        dict(
            dataclasses=dataclasses,
            random=random,
            np=np,
            _get_all_rng_states=lambda: (torch.get_rng_state(),),
            _set_all_rng_states=lambda value: torch.set_rng_state(value),
        ),
    )
    evaluate = isolated_definition(
        "miles/backends/megatron_utils/actor.py",
        "evaluate_critic",
        dict(
            copy=copy,
            RandomState=random_state,
            get_rollout_data=get_rollout_data,
            get_data_iterator=get_data_iterator,
            forward_only=forward,
            get_values=get_values,
            train_dump=train_dump,
        ),
        owner="MegatronTrainRayActor",
    )
    worker = SimpleNamespace(
        args=args, role="critic", model=[TinyCritic()], _heartbeat=SimpleNamespace(bump=lambda: None)
    )
    return worker, partial(evaluate, worker), state, store


def launch_args(mode):
    return SimpleNamespace(
        critic_context_mode=mode,
        global_batch_size=512,
        n_samples_per_prompt=16,
        rollout_batch_size=32,
        reward_key=None,
        advantage_estimator="ppo",
        rewards_normalization=False,
        critic_return_std_floor=0.075,
        rollout_top_logprobs_num=0,
        use_dynamic_global_batch_size=False,
        balance_data=True,
        filter_truncated_in_actor_loss=True,
        delay_split_train_data_by_dp=False,
        enable_witness=False,
        qkv_format="thd",
        data_pad_size_multiplier=8,
        allgather_cp=False,
        true_on_policy_mode=False,
        use_dynamic_batch_size=True,
        use_rollout_entropy=False,
        offload_train=False,
        custom_megatron_before_log_prob_hook_path=None,
        seq_length=4096,
        micro_batch_size=1,
        save_debug_train_data=None,
    )
