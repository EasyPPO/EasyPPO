import torch
from megatron.core.optimizer import USING_PYTORCH_OPTIMIZER
from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer


def cpu_adam_checkpoint_optimizers(optimizer):
    """Full CPU Adam offload keeps FP32 masters in Megatron's hybrid optimizer."""
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    for member in getattr(optimizer, "chained_optimizers", (optimizer,)):
        inner = getattr(member, "optimizer", None)
        if (
            isinstance(inner, HybridDeviceOptimizer)
            and member.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
            and inner.param_update_in_fp32
            and inner.gpu_optimizer is None
            and inner.cpu_optimizer_cls is torch.optim.AdamW
        ):
            yield inner


def prepare_cpu_adam_checkpoint(inner) -> None:
    if inner.state:
        return
    for adam in inner.cpu_optimizers:
        for group in adam.param_groups:
            for param in group["params"]:
                # AdamW's allocator requires a gradient to select the parameter.
                # It only inspects it; reuse a detached view without allocating a gradient.
                old_grad = param.grad
                try:
                    if old_grad is None:
                        param.grad = param.detach()
                    adam._init_group(dict(group, params=[param]), [], [], [], [], [], [])
                finally:
                    param.grad = old_grad
    inner._sync_sub_optimizers_state_to_hdo()


@torch.no_grad()
def restore_cpu_adam_masters(inner) -> None:
    # dp_reshardable replaces HDO state tensors after its load hook. Native
    # fp32 parameters may also have a distinct CPU copy when offloaded.
    for param, master in inner.param_to_inner_param.items():
        master.copy_(inner.state[param]["master_param"])
        inner.state[param]["master_param"] = master
    inner._sync_hdo_state_to_sub_optimizers()


def reset_optimizer_state(
    optimizer: MegatronOptimizer, *, stream_optimizer_state_to_disk: bool, chunked_optimizer_state_offload: bool
) -> None:
    assert isinstance(optimizer, ChainedOptimizer)
    assert not optimizer.config.offload_optimizer_states
    assert (
        not chunked_optimizer_state_offload
    ), "--chunked-optimizer-state-offload keeps the canonical moments outside the optimizer state this clears"
    assert not optimizer.config.fp16, (
        "an fp16 optimizer keeps a loss scaler with a scale and a growth tracker beside the state this clears, so "
        "the reset would leave the trainer with the scaler of the rollouts it just discarded"
    )
    assert not stream_optimizer_state_to_disk, (
        "--stream-optimizer-state-to-disk keeps the moments in its own store rather than in the master optimizer "
        "state this walks, so the reset would clear nothing and the self-check would pass on an empty walk"
    )

    for chained in optimizer.chained_optimizers:
        if chained.is_stub_optimizer:
            continue

        inner = chained.optimizer
        inner.state.clear()
        for param_group in inner.param_groups:
            param_group.pop("step", None)

        if chained.init_state_fn is not None and not USING_PYTORCH_OPTIMIZER:
            chained.init_state_fn(inner, chained.config)

        chained.zero_grad(set_to_none=True)

    _check_optimizer_state_empty(optimizer)


def _check_optimizer_state_empty(optimizer: MegatronOptimizer) -> None:
    checked_optimizers = 0
    checked_params = 0
    checked_state_keys: set[str] = set()

    for chained in optimizer.chained_optimizers:
        if chained.is_stub_optimizer:
            continue

        checked_optimizers += 1
        inner = chained.optimizer
        for param_group in inner.param_groups:
            assert "step" not in param_group
            for param in param_group["params"]:
                checked_params += 1
                assert param.grad is None

        for state in inner.state.values():
            checked_state_keys |= set(state)
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    assert torch.count_nonzero(value) == 0
                else:
                    assert value == 0

    assert checked_optimizers > 0
    assert checked_params > 0
    # if you use a non-adam optimizer, add the asserts for its state keys here
    assert checked_state_keys == (set() if USING_PYTORCH_OPTIMIZER else {"exp_avg", "exp_avg_sq"})
