"""Independent HF memory critic for Miles' Megatron actor PPO path.

The actor and critic occupy the same trainer GPUs sequentially. Critic parameters
and gradients are BF16, with chunked FP32 gradient reduction. FP32 CPU master
weights and Adam states are partitioned across workers. Full-detach retains detached attention KV and hybrid
recurrent/conv states between segments; actor-summary clears all three.

Only PP=CP=1, complete episode shards, gamma=lambda=1,
CPU offload and synchronous native checkpoints are supported. GPU capacity and
throughput must be measured on the target cluster.
"""

import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from agentic_critic.cpu_sharded_adam import CPUShardedAdam
from agentic_critic.critic_input import Segment
from agentic_critic.critic_modes import ValueHead
from agentic_critic.hf_memory_backbone import load_memory_backbone
from agentic_critic.memory_critic import MemoryCritic, MemoryEpisode, flatten_episode_values, train_memory_minibatches

from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.ray.train_actor import TrainRayActor
from miles.utils import object_store
from miles.utils.data import process_rollout_data
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)


class HFMemoryCriticTrainRayActor(TrainRayActor):
    def init(
        self,
        args,
        role,
        *,
        with_ref=False,
        with_opd_teacher=False,
        recv_ckpt_src_rank=None,
        indep_dp_info,
        indep_dp_store_addr,
    ):
        if role != "critic" or with_ref or with_opd_teacher or recv_ckpt_src_rank is not None:
            raise ValueError("HF memory worker only supports an independent critic without fault recovery")
        self._init_common(args, role)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.train_parallel_config = dict(
            dp_size=dist.get_world_size() // args.tensor_model_parallel_size,
            cp_size=1,
            vpp_size=None,
            microbatch_group_size_per_vp_stage=1,
        )
        self._asleep = True
        if args.debug_rollout_only:
            return 0
        torch.manual_seed(args.seed)
        # Text-only construction avoids a vision tower; the LM output projection
        # is discarded because values use only the backbone's hidden states.
        backbone = load_memory_backbone(args.hf_checkpoint)
        if getattr(backbone.config, "model_type", None) not in ("qwen3_5", "qwen3_5_text"):
            raise ValueError("HF memory critic currently supports the Qwen3.5 text backbone only")
        head = ValueHead(backbone.config.hidden_size)
        torch.nn.init.normal_(head.linear.weight, std=args.init_method_std)
        head = head.to(dtype=torch.bfloat16)
        self.model = MemoryCritic(backbone, head).to(device=self.device).eval()
        self.optimizer = CPUShardedAdam(
            self.model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        self.optimizer_steps = 0
        start = self._restore()
        self._asleep = False
        if dist.get_rank() == 0:
            tracking.init_tracking(args, primary=False)
        self.sleep()
        return start

    def _restore(self):
        directory = Path(self.args.load)
        pointer = directory / "memory_critic_latest.json"
        if not pointer.is_file():
            if directory.resolve() != Path(self.args.hf_checkpoint).resolve():
                raise ValueError(f"No native HF memory critic checkpoint in {directory}")
            return 0
        record = json.loads(pointer.read_text())
        checkpoint_dir = directory / record["directory"]
        state = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=False)
        weights_only = self.args.finetune or self.args.no_load_optim
        if not weights_only and (
            state["mode"] != self.args.critic_context_mode
            or state["world_size"] != dist.get_world_size()
            or state["actor_tp_size"] != self.args.tensor_model_parallel_size
        ):
            raise ValueError("HF memory critic resume requires the same context mode and trainer topology")
        self.model.load_state_dict(state["model"])
        if weights_only:
            # Loading only the BF16 replica must also replace the FP32 master;
            # otherwise the first update would restore the pre-load weights.
            self.optimizer.reset_masters_from_model()
            return 0
        self.optimizer_steps = state["optimizer_steps"]
        if not self.args.no_load_optim:
            local = torch.load(
                checkpoint_dir / f"optimizer_rank_{dist.get_rank():05d}.pt", map_location="cpu", weights_only=False
            )
            self.optimizer.load_state_dict(local["optimizer"])
            if not self.args.no_load_rng:
                torch.set_rng_state(local["rng"])
                torch.cuda.set_rng_state(local["cuda_rng"])
        return state["rollout_id"] + 1

    def _move_to(self, device):
        self.model.to(device)
        clear_memory()
        dist.barrier(group=get_gloo_group())

    def sleep(self):
        if self.args.debug_rollout_only:
            return
        self.optimizer.zero_grad()
        self._move_to("cpu")
        self._asleep = True

    def wake_up(self):
        if self.args.debug_rollout_only:
            return
        if self._asleep:
            self._move_to(self.device)
            self._asleep = False

    def _get_parallel_config(self):
        return self.train_parallel_config

    def train(self, rollout_id, rollout_data_ref, witness_info=None, attempt=0, external_data=None):
        if witness_info is not None or attempt or external_data is not None:
            raise ValueError("HF memory critic does not accept witnesses, retries or external values")
        self._heartbeat.bump()
        if self.args.debug_rollout_only:
            return TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
        self.wake_up()
        data, store_result = process_rollout_data(
            self.args,
            rollout_data_ref,
            dist.get_rank() // self.args.tensor_model_parallel_size,
            dist.get_world_size() // self.args.tensor_model_parallel_size,
            witness_info=None,
        )
        with store_result:
            episodes = [
                MemoryEpisode([Segment(**part) for part in parts], float(reward))
                for parts, reward in zip(data["critic_segments"], data["rewards"], strict=True)
            ]
            if self.args.critic_return_std_floor is not None and "critic_loss_weights" not in data:
                raise ValueError("Memory critic requires rollout-computed inverse-STD weights when enabled")
            values = self._train_episodes(
                rollout_id,
                episodes,
                minibatch_indices=data["critic_minibatch_indices"],
                loss_weights=data.get("critic_loss_weights", [1.0] * len(episodes)),
            )
            flattened = [flatten_episode_values(ep, own) for ep, own in zip(episodes, values, strict=True)]
            shipped = [flattened[i][start : start + length] for i, start, length in data["value_slices"]]
            reference = object_store.get_instance().put(
                value={"values": shipped}, value_spec={"values": object_store.ValueSpec(codec="typed_ragged")}
            )
        self._heartbeat.bump()
        return TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=reference)

    def _train_episodes(self, rollout_id, episodes, *, minibatch_indices, loss_weights):
        expected_steps = self.args.critic_num_steps_per_rollout or self.args.num_steps_per_rollout or 1
        if len(minibatch_indices) != expected_steps:
            raise ValueError("Memory critic minibatch schedule does not match its configured optimizer steps")
        # Tensor-parallel actor siblings read identical episode shards. Count
        # and gradient sums both include those replicas, so duplication cancels
        # exactly in the global token mean. Each worker returns its own values.
        result = train_memory_minibatches(
            self.model,
            self.optimizer,
            episodes,
            minibatch_indices=minibatch_indices,
            loss_weights=loss_weights,
            mode=self.args.critic_context_mode,
            device=self.device,
            value_clip=self.args.value_clip,
            loss_coef=self.args.value_loss_coef,
            learning_rate=self.args.lr,
            warmup_steps=self.args.lr_warmup_iters,
            optimizer_steps=self.optimizer_steps,
            clip_grad=self.args.clip_grad,
            on_episode=self._heartbeat.bump,
        )
        self.optimizer_steps = result.optimizer_steps
        self._log_metrics(
            rollout_id,
            episodes,
            result.values,
            result.loss_sum,
            result.token_count,
            result.grad_norms[-1],
            result.learning_rates[-1],
        )
        return result.values

    def _log_metrics(self, rollout_id, episodes, values, loss_sum, count, grad_norm, lr):
        stats = torch.zeros(5, dtype=torch.float64, device=self.device)
        stats[0] = loss_sum
        for ep, own in zip(episodes, values, strict=True):
            for seg, value in zip(ep.segments, own, strict=True):
                selected = value[torch.tensor(seg.loss_mask, dtype=torch.bool)].double()
                residual = ep.reward - selected
                stats[1] += len(selected) * ep.reward
                stats[2] += len(selected) * ep.reward**2
                stats[3] += residual.sum().to(self.device)
                stats[4] += residual.square().sum().to(self.device)
        dist.all_reduce(stats)
        if dist.get_rank() == 0:
            var_returns = stats[2] / count - (stats[1] / count).square()
            var_residuals = stats[4] / count - (stats[3] / count).square()
            metrics = {
                "critic/step": rollout_id,
                "critic/value_loss": float(stats[0] / count),
                "critic/grad_norm": grad_norm,
                "critic/lr": lr,
                "critic/optimizer_step": self.optimizer_steps,
                "critic/optimizer_steps_per_rollout": (
                    self.args.critic_num_steps_per_rollout or self.args.num_steps_per_rollout or 1
                ),
            }
            if self.args.log_critic_explained_variance:
                metrics["critic/explained_variance"] = float(1 - var_residuals / (var_returns + 1e-5))
            logger.info("HF memory critic: %s", metrics)
            tracking.log(self.args, metrics, step_key="critic/step")

    def save_model(self, rollout_id, force_sync=False):
        if not self.args.save or self.args.debug_rollout_only:
            return
        self.wake_up()
        directory = Path(self.args.save)
        checkpoint_dir = directory / f"memory_critic_{rollout_id:07d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not self.args.no_save_optim:
            torch.save(
                dict(
                    optimizer=self.optimizer.state_dict(),
                    rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),
                ),
                checkpoint_dir / f"optimizer_rank_{dist.get_rank():05d}.pt",
            )
        if dist.get_rank() == 0:
            torch.save(
                dict(
                    model={key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                    rollout_id=rollout_id,
                    optimizer_steps=self.optimizer_steps,
                    mode=self.args.critic_context_mode,
                    world_size=dist.get_world_size(),
                    actor_tp_size=self.args.tensor_model_parallel_size,
                ),
                checkpoint_dir / "model.pt",
            )
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            temporary = directory / "memory_critic_latest.json.tmp"
            temporary.write_text(json.dumps(dict(directory=checkpoint_dir.name)))
            os.replace(temporary, directory / "memory_critic_latest.json")
        dist.barrier(group=get_gloo_group())

    def update_weights(self, info):
        raise ValueError("Critic weights are never published to actor rollout engines")
