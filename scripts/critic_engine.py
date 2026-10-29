"""Train a Hugging Face critic on stored Miles rollouts.

Supports shared, full, full_detach, state, kv and attention_matching context modes.
Each rollout receives a value pass followed by noise-normalized, clipped value
regression on episode mini-batches, with gradient clipping and AdamW updates.
Values are saved before updates in Miles' critic debug-dump format.

This is a standalone critic-only backend; compressed KV is not integrated into
Miles' Megatron actor/critic training loop. Parameters use float32 with optional
bfloat16 autocast; mini-batches follow episode order in the rollout file.

Usage: python scripts/critic_engine.py --model MODEL --rollouts ROLLOUTS --mode kv --out OUTPUT
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentic_critic.critic_input import Segment  # noqa: E402
from agentic_critic.critic_modes import MODES  # noqa: E402
from agentic_critic.memory_config import resolve_memory_config  # noqa: E402
from agentic_critic.critic_train import AdamWHalfMoments, clipped_value_loss, problem_weights, read_episode  # noqa: E402


def load_step(path: Path) -> list[dict]:
    """Episodes of one rollout file, in file order; each with its actor samples sorted by segment."""
    by_episode: dict[int, list[dict]] = {}
    for s in torch.load(path, weights_only=False)["samples"]:
        by_episode.setdefault(s["rollout_id"], []).append(s)
    episodes = []
    for rid, samples in by_episode.items():
        samples.sort(key=lambda s: s["metadata"]["segment"])
        assert [s["metadata"]["segment"] for s in samples] == list(range(len(samples)))
        episodes.append({
            "rollout_id": rid, "problem": samples[0]["metadata"]["problem"], "reward": float(samples[0]["reward"]),
            "indices": [s["index"] for s in samples],
            "segments": [Segment(list(s["tokens"]), s["response_length"], list(s["loss_mask"]), s["metadata"].get("summary_span")) for s in samples],
        })  # fmt: skip
    return episodes


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(1 - (target - pred).var(correction=0) / (target.var(correction=0) + 1e-5))


def self_check(model, head, episodes: list[dict], mode: str, kv: dict | None, device, autocast, n: int) -> dict:
    """Checks on the real model, before any update, that the setting reads what it is meant to read.

    On the first `n` episodes with at least two segments:
      first_segment   the values on the first segment equal those of the actor-context forward (nothing precedes it);
      later_segments  for settings with a memory, the values after a compaction differ from the actor-context ones
                      (the memory is really read);
      train_forward   the forward used for training (checkpointed layers, read-only memory) gives the values of
                      the forward used for the value pass;
      kv_keep_all     in the kv setting, keeping every key reproduces full_detach.
    Returns the largest differences; raises if a check fails.
    """
    def values(ep, m, train=False, kv_cfg=None):
        got = {}

        def keep(pairs):
            for i, v in pairs:
                got[i] = v.detach().float()

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx, autocast():
            read_episode(model, head, ep["segments"], m, device, keep, train=train, kv=kv_cfg)
        return [got[i] for i in range(len(ep["segments"]))]

    def diff(a, b):
        return max(float((x - y).abs().max()) for x, y in zip(a, b, strict=True))

    def scale(a):
        return max(float(x.abs().max()) for x in a)

    report = {"episodes": 0, "first_segment": 0.0, "later_segments": None, "train_forward": 0.0, "kv_keep_all": None}
    for ep in [e for e in episodes if len(e["segments"]) >= 2][:n]:
        own, actor = values(ep, mode, kv_cfg=kv), values(ep, "shared")
        tol = 2e-2 * max(scale(own), 1.0)  # bfloat16 autocast: two passes of the same computation agree to about 1e-2 relative
        report["episodes"] += 1
        report["first_segment"] = max(report["first_segment"], diff(own[:1], actor[:1]))
        assert diff(own[:1], actor[:1]) <= tol, f"first segment differs from the actor-context forward by {diff(own[:1], actor[:1])}"
        if mode != "shared":
            later = diff(own[1:], actor[1:])
            report["later_segments"] = later if report["later_segments"] is None else min(report["later_segments"], later)
            assert later > tol, "the values after a compaction equal the actor-context ones: the memory is not being read"
        trained = values(ep, mode, train=True, kv_cfg=kv)
        report["train_forward"] = max(report["train_forward"], diff(own, trained))
        assert diff(own, trained) <= tol, f"training forward differs from the value pass by {diff(own, trained)}"
        if mode in ("kv", "attention_matching"):
            keep_all, detach = values(ep, mode, kv_cfg={**kv, "ratio": 1.0}), values(ep, "full_detach")
            report["kv_keep_all"] = max(report["kv_keep_all"] or 0.0, diff(keep_all, detach))
            assert diff(keep_all, detach) <= tol, f"kv with every key kept differs from full_detach by {diff(keep_all, detach)}"
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None, help="default: every rollout file")
    ap.add_argument("--kv-ratio", type=float, default=None, help="fraction retained (default: 0.1); summary probes compact only the pre-summary block")
    ap.add_argument("--kv-method", default=None, choices=["evict", "am", "am_cv", "am_ridge"],
                    help="default: evict for kv; am_ridge for attention_matching")
    ap.add_argument("--kv-queries", default=None, choices=["summary", "hindsight", "causal"])
    ap.add_argument("--kv-query-count", type=int, default=None,
                    help="kv/hindsight only: first N next-segment tokens (default: 400); 0 uses legacy uniform sampling; summary probes always use all tokens")
    ap.add_argument("--kv-ridge", type=float, default=None, help="positive relative ridge for am_ridge/am_cv (default: 1e-4)")
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--value-clip", type=float, default=0.2)
    ap.add_argument("--value-loss-coef", type=float, default=0.5)
    ap.add_argument("--return-std-floor", type=float, default=0.075)
    ap.add_argument("--mini-batches", type=int, default=4)
    ap.add_argument("--head-init-std", type=float, default=0.02, help="as Miles and EasyPPO initialise the value head")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-autocast", action="store_true")
    ap.add_argument("--full-precision-moments", action="store_true")
    ap.add_argument("--wait-for-end", action="store_true", help="the rollout directory is still being filled: wait for the next step until a file END appears")
    ap.add_argument("--self-check", type=int, default=2, help="episodes to run the correctness checks on before training (0: skip)")
    ap.add_argument("--save-model", action="store_true", help="store the trained backbone (bfloat16, about 18 GB for 9B) next to the value head")
    args = ap.parse_args()
    # Preserve the legacy CLI's eviction default; the named AM mode fixes its algorithm and query source.
    supplied = dict(ratio=args.kv_ratio, method=args.kv_method or ("evict" if args.mode == "kv" else None),
                    queries=args.kv_queries, query_count=args.kv_query_count, ridge=args.kv_ridge)
    try:
        kv = resolve_memory_config(args.mode, {k: v for k, v in supplied.items() if v is not None})
    except ValueError as exc:
        ap.error(str(exc))

    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device=args.device, dtype=torch.float32)
    if kv is not None:
        from agentic_critic.kv_compact import validate_model

        validate_model(model)
    model.train()
    linear = torch.nn.Linear(model.config.hidden_size, 1).to(args.device)
    torch.nn.init.normal_(linear.weight, std=args.head_init_std)
    torch.nn.init.zeros_(linear.bias)

    def head(hidden: torch.Tensor) -> torch.Tensor:
        return linear(hidden.float()).squeeze(-1)

    # lm_head is not part of the critic (a no-op for the optimiser: it never receives a gradient)
    params = [p for n, p in model.named_parameters() if not n.startswith("lm_head")] + list(linear.parameters())
    opt = AdamWHalfMoments(params, lr=args.lr, weight_decay=args.weight_decay, moment_dtype=torch.float32 if args.full_precision_moments else torch.bfloat16)
    autocast = lambda: torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16, enabled=not args.no_autocast)  # noqa: E731

    def rollout_files():
        """Rollout files in step order. With --wait-for-end the directory may still be growing: a step that is not
        there yet is waited for, until a file named END says that no more will come."""
        k = 0
        while args.steps is None or k < args.steps:
            path = Path(args.rollouts) / f"rollout_{k}.pt"
            while args.wait_for_end and not path.exists() and not (Path(args.rollouts) / "END").exists():
                time.sleep(30)
            if not path.exists():
                return
            yield path
            k += 1

    first = Path(args.rollouts) / "rollout_0.pt"
    out = Path(args.out)
    (out / "dump").mkdir(parents=True, exist_ok=True)
    log = {"args": vars(args), "memory_config": kv, "steps": []}
    if args.self_check:
        log["self_check"] = self_check(model, head, load_step(first), args.mode, kv, args.device, autocast, args.self_check)
        print("self-check passed:", json.dumps(log["self_check"]), flush=True)
    start = time.time()
    for path in rollout_files():
        step = int(path.stem.split("_")[1])
        episodes = load_step(path)

        # 1. value pass: the critic's predictions on this rollout before it trains on it
        old, critic_tokens = {}, 0
        with torch.no_grad(), autocast():
            for ep in episodes:
                def keep(pairs, ep=ep, old=old):
                    for i, v in pairs:
                        old[(ep["rollout_id"], i)] = v.float()
                read_episode(model, head, ep["segments"], args.mode, args.device, keep, train=False, kv=kv)
                critic_tokens += sum(len(s.tokens) for s in ep["segments"])
        masks = {(ep["rollout_id"], i): torch.tensor(s.loss_mask, dtype=torch.bool, device=args.device) for ep in episodes for i, s in enumerate(ep["segments"])}
        torch.save({
            "rollout_id": step, "rank": 0,
            "values": [old[(ep["rollout_id"], i)].cpu() for ep in episodes for i in range(len(ep["segments"]))],
            "loss_masks": [masks[(ep["rollout_id"], i)].long().cpu() for ep in episodes for i in range(len(ep["segments"]))],
            "sample_indices": [idx for ep in episodes for idx in ep["indices"]],
        }, out / "dump" / f"{step}_0.critic.pt")  # fmt: skip
        v_all = torch.cat([old[k][masks[k]] for k in old])
        g_all = torch.cat([torch.full((int(masks[(ep["rollout_id"], i)].sum()),), ep["reward"], device=args.device) for ep in episodes for i in range(len(ep["segments"]))])
        ev = explained_variance(v_all, g_all)
        # 2. EasyPPO's critic update on the same rollout
        rewards: dict[str, list[float]] = {}
        for ep in episodes:
            rewards.setdefault(ep["problem"], []).append(ep["reward"])
        weights = problem_weights(rewards, args.return_std_floor)
        size = -(-len(episodes) // args.mini_batches)
        updates = []
        for b in range(0, len(episodes), size):
            batch = episodes[b : b + size]
            n_tokens = sum(int(masks[(ep["rollout_id"], i)].sum()) for ep in batch for i in range(len(ep["segments"])))
            stats = {"loss": 0.0, "clipped": 0.0}
            opt.zero_grad(set_to_none=True)
            for ep in batch:
                def backward(pairs, ep=ep, masks=masks, old=old, weights=weights, stats=stats, n_tokens=n_tokens):
                    total = 0.0
                    for i, v in pairs:
                        key = (ep["rollout_id"], i)
                        loss, binds = clipped_value_loss(v[masks[key]], old[key][masks[key]], ep["reward"], args.value_clip)
                        total = total + weights[ep["problem"]] * loss.sum()
                        stats["clipped"] += float(binds.sum())
                    total = args.value_loss_coef * total / max(n_tokens, 1)
                    stats["loss"] += float(total)
                    if total.requires_grad:
                        total.backward()
                with autocast():
                    read_episode(model, head, ep["segments"], args.mode, args.device, backward, train=True, kv=kv)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.clip_grad))
            opt.step()
            updates.append({"value_loss": stats["loss"], "value_clipfrac": stats["clipped"] / max(n_tokens, 1), "grad_norm": grad_norm})
        row = {"step": step, "explained_variance": ev, "mse": float(((v_all - g_all) ** 2).mean()), "bias": float((v_all - g_all).mean()),
               "episodes": len(episodes), "critic_tokens": critic_tokens, "updates": updates,
               "seconds": round(time.time() - start), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None}  # fmt: skip
        log["steps"].append(row)
        (out / "log.json").write_text(json.dumps(log, indent=1))
        print(f"critic {step}: explained_variance={ev:.6f} loss={[round(u['value_loss'], 4) for u in updates]} "
              f"clipfrac={[round(u['value_clipfrac'], 3) for u in updates]} grad_norm={[round(u['grad_norm'], 2) for u in updates]} "
              f"{row['seconds']}s peak {row['peak_gb']} GB", flush=True)  # fmt: skip
    torch.save({"weight": linear.weight.detach().cpu(), "bias": linear.bias.detach().cpu()}, out / "value_head.pt")
    if args.save_model:
        import safetensors.torch

        safetensors.torch.save_file({n: p.detach().to(torch.bfloat16).cpu() for n, p in model.named_parameters() if not n.startswith("lm_head")},
                                    str(out / "critic_bf16.safetensors"))  # fmt: skip
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
