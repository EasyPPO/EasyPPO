# EasyPPO on Miles

This branch is a standalone Miles implementation. The root README's verl launch commands apply to the `main` branch. Use
the commands below for this branch.

## Included functionality

- EasyPPO's actor-only truncation filter, noise-normalized value regression,
  independent critic mini-batches, value-loss coefficient and explained variance.
- Separate actor and critic inputs: `shared` or `full_history`. Full-history
  values are mapped from one episode sample back to the actor's segment samples.
- Agentic Frontier-CS episodes, judge feedback, actor context compaction, and
  token-level reward/loss-mask construction.
- A standalone Hugging Face critic trainer supporting `shared`, `full`,
  `full_detach`, `state`, and `kv`, with memory-aware checkpointing and attention.
- Qwen3.5 hybrid-cache compaction using eviction or Attention Matching, including
  causal-mask and ridge variants, and causal or hindsight query selection.
- Prompt-data preparation, stored-episode conversion, and Miles critic
  checkpoint export/load utilities.

Research reports, result data, probes, ablation runners, plotting scripts,
machine-specific launchers, and GPU reservation/queue tools are not included.

## Installation

Use a Miles-compatible GPU environment with Megatron and SGLang, following
[Miles installation](../../docs/getting-started/installation.md). The imported
code targets Transformers 5.12.1. From this checkout:

```bash
git submodule update --init third_party/compaction
python -m pip install -e '.[agentic]'
```

The KV algorithm dependency is pinned as a Git submodule. An existing checkout
can instead be selected with `KV_COMPACTION_REPO=/path/to/compaction`.

## Terminal training

Use the [terminal training guide](TERMINAL.md) for full TMAX training with the
mini-swe-agent harness, thinking enabled, and official TB2.1 validation.
The launcher reuses the EasyPPO settings below.

## Frontier-CS training

Run the Frontier-CS HTTP judge separately, with the problem IDs used by your
dataset. The optional C++ judge Dockerfile is
[`docker/Dockerfile.fcs-judge-cpp`](../../docker/Dockerfile.fcs-judge-cpp);
its build context is the FrontierSmith `Frontier-CS/algorithmic` directory.

Prepare one JSONL row per problem:

```bash
python scripts/make_miles_fcs_data.py \
  --problems-dir /path/to/algorithmic/problems --out /path/to/problems.jsonl
```

The built-in wrapper constructs chat turns and computes rewards itself. Do not
apply a second chat template or reward function:

```bash
python examples/easyppo/run_qwen3_5_9b_frontier_cs.py \
  --prompt-data /path/to/problems.jsonl \
  --agentic-generate-function-path agentic_critic.miles_fcs.generate \
  --critic-context-mode shared \
  --model-dir /path/to/models --data-dir /path/to/data --output-dir /path/to/output \
  --extra-args '--fcs-judge-url http://127.0.0.1:8081 --fcs-compact-threshold 12000'
```

The launcher prepares the initial model/checkpoint and submits training through
Miles. `--num-critic-only-steps` controls warmup. To train only the critic, set it
equal to `--num-rollout`. Single-turn training is also available by passing
`--custom-rm-path your_module.reward` instead of the agentic wrapper.

`full_history` keeps each critic episode and its actor segments on the same
data-parallel rank. It requires dynamic batching, context parallel size 1,
virtual pipeline size 1, and complete episode batches divisible by the
parallel/critic-step configuration. Custom train-data conversion and delayed DP
splitting are unsupported. The token budget must fit the full critic episode
and its actor segments; full-history inputs can exceed actor context lengths.

## Standalone critic training with memory

Convert serialized `Episode` event logs to Miles rollout files if needed:

```bash
python scripts/miles/episodes_to_rollouts.py \
  --episodes /path/to/episodes.jsonl --out /path/to/rollouts --steps 30
```

Then train the Hugging Face critic:

```bash
python scripts/critic_engine.py \
  --model /path/to/hf-model --rollouts /path/to/rollouts --out /path/to/critic \
  --mode kv --kv-method evict --kv-ratio 0.25 --kv-queries causal --save-model
```

Other context modes are `shared`, `full`, `full_detach`, and `state`. The trainer
stores value dumps, optimizer-step metrics, and a value head; `--save-model` also
saves the backbone. `--self-check` checks equivalent forward paths before
training. Memory is recomputed with current weights and detached between segments.

KV/state memory training is available in this standalone trainer, not in the
Miles Megatron PPO loop. `hindsight` query selection uses later rollout tokens;
it is retained from the source implementation and has not been validated as an
unbiased baseline for actor updates. The command above uses causal queries.

## Critic checkpoint conversion

```bash
python scripts/miles/critic_to_hf.py \
  --input-dir /path/to/checkpoints_critic/iter_0000100 \
  --base-hf-dir /path/to/hf-model --out /path/to/converted-critic
```

Use `agentic_critic.critic_checkpoint.load_critic` to load the converted backbone
and value head. This export format is separate from the standalone trainer's
`--save-model` output.

## CPU checks

In an environment with PyTorch, Transformers 5.12.1, pytest, pytest-asyncio,
NumPy, PyYAML, and the Miles import dependencies installed:

```bash
OMP_NUM_THREADS=1 python -m pytest tests/agentic_critic \
  --confcutdir=tests/agentic_critic -o addopts='' -q
```

The imported Miles regression tests cover critic scheduling, sample/value
mapping, episode weighting, value dumps and numerical parity with EasyPPO.
For the parity test, point `EASYPPO_REPO` at a separate checkout of the verl-based
EasyPPO branch with its reference dependencies installed.

## Sources

- Miles: `radixark/miles@3ad61b8df7a9ff07c8fd3fc6f3f6ee25a50ce36d`.
- Functional additions and all 15 Miles patches:
  `joyemang33/agentic-critic@8a8b2b4902164beb3f2eb4ae1d71518731f8bd33`.
- KV algorithms: `adamzweiger/compaction@2020f6d37940fe013e7e7f1ebf409e5d107c5670`.
- Root README and logo:
  `dyxg/EasyPPO_fork@0a490a88ad7b84a8f7974fe168f0ba63600b1e31`.

The source algorithms and their existing limitations are preserved. Migration
changes remove research-only code, make paths relative/configurable, package
`agentic_critic`, and retain functional regression tests.
