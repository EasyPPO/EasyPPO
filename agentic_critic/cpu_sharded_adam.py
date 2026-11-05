"""FP32 CPU master parameters/Adam states sharded over replicated BF16 models.

Each rank keeps a complete forward model, but owns only a deterministic subset
of its FP32 master parameters. Gradients are reduced in FP32 bounded chunks,
copied to their owning CPU optimizer, globally clipped, and updated weights are
broadcast back. This deliberately favors simple, cache-compatible semantics
over communication efficiency; there is no overlap with model computation.
"""

import math

import torch
import torch.distributed as dist


class CPUShardedAdam:
    def __init__(self, parameters, *, lr, betas, eps, weight_decay, chunk_elements=16_777_216):
        self.parameters = list(parameters)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.chunk_elements = chunk_elements
        sizes = [0] * self.world_size
        self.owners = [0] * len(self.parameters)
        for index in sorted(range(len(self.parameters)), key=lambda i: self.parameters[i].numel(), reverse=True):
            owner = min(range(self.world_size), key=sizes.__getitem__)
            self.owners[index] = owner
            sizes[owner] += self.parameters[index].numel()
        self.masters = {
            i: torch.nn.Parameter(p.detach().to(device="cpu", dtype=torch.float32).clone())
            for i, p in enumerate(self.parameters)
            if self.owners[i] == self.rank
        }
        if not self.masters:
            raise ValueError("CPU-sharded Adam requires at least one parameter tensor per rank")
        self.optimizer = torch.optim.AdamW(
            self.masters.values(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, *, clip_grad: float) -> float:
        square_norm = 0.0
        for index, parameter in enumerate(self.parameters):
            master = self.masters.get(index)
            if master is not None:
                master.grad = torch.empty_like(master)
            gradient = None if parameter.grad is None else parameter.grad.reshape(-1)
            for start in range(0, parameter.numel(), self.chunk_elements):
                end = min(start + self.chunk_elements, parameter.numel())
                chunk = (
                    torch.zeros(end - start, dtype=torch.float32, device=parameter.device)
                    if gradient is None
                    else gradient[start:end].float().clone()
                )
                if self.world_size > 1:
                    dist.all_reduce(chunk)
                square_norm += float(chunk.double().square().sum())
                if master is not None:
                    master.grad.reshape(-1)[start:end].copy_(chunk.cpu())
            parameter.grad = None
        norm = math.sqrt(square_norm)
        if not math.isfinite(norm):
            raise ValueError(f"Nonfinite memory critic gradient norm: {norm}")
        factor = min(1.0, clip_grad / (norm + 1e-6))
        for master in self.masters.values():
            master.grad.mul_(factor)
        self.optimizer.step()
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters):
                if index in self.masters:
                    source = self.masters[index].reshape(-1)
                    destination = parameter.reshape(-1)
                    for start in range(0, parameter.numel(), self.chunk_elements):
                        end = min(start + self.chunk_elements, parameter.numel())
                        destination[start:end].copy_(source[start:end])
                if self.world_size > 1:
                    dist.broadcast(parameter, src=self.owners[index])
        self.optimizer.zero_grad(set_to_none=True)
        return norm

    def state_dict(self):
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "owners": self.owners,
            "masters": {i: master.detach() for i, master in self.masters.items()},
            "optimizer": self.optimizer.state_dict(),
        }

    def reset_masters_from_model(self):
        """Refresh FP32 masters after loading model weights without optimizer state."""
        with torch.no_grad():
            for index, master in self.masters.items():
                master.copy_(self.parameters[index].detach().float().cpu())
        self.optimizer.state.clear()

    def load_state_dict(self, state):
        if (state["world_size"], state["rank"], state["owners"]) != (self.world_size, self.rank, self.owners):
            raise ValueError("CPU-sharded Adam resume requires the original DP topology and parameter ownership")
        with torch.no_grad():
            for index, master in self.masters.items():
                master.copy_(state["masters"][index])
        self.optimizer.load_state_dict(state["optimizer"])
