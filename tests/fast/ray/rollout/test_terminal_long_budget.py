"""A 120k generated-token episode survives compaction and training conversion."""

from copy import deepcopy

import pytest
from agentic_critic.terminal_env import Completion, TerminalAgent, TerminalConfig
from tests.agentic_critic.test_terminal_env import Tokenizer, completion
from tests.fast.ray.rollout.test_memory_critic_minibatches import _executor

from miles.utils.data import process_rollout_data
from miles.utils.types import Sample


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
async def test_cumulative_budget_and_training_pack_include_different_tokens(monkeypatch, mode):
    config = TerminalConfig(
        max_steps=300,
        max_context_tokens=32768,
        compact_threshold=24576,
        max_new_tokens=8192,
        summary_tokens=2048,
        max_total_tokens=120000,
    )

    async def decode(prompt, limit):
        is_summary = "Summarize progress for a fresh context" in Tokenizer().decode(prompt[-1000:])
        suffix = "progress" if is_summary else "\n```bash\ninspect\n```"
        # Responses may finish below their per-call ceiling. Leave room for
        # observations and a complete summary inside the 32k actor window.
        generated = min(limit, 1024 if is_summary else 3072)
        padding = generated - len(completion(suffix).tokens)
        if padding < 0:
            return Completion([ord("x")] * limit, [-0.5] * limit, "x" * limit, "length")
        return completion("x" * padding + suffix)

    async def execute(_command, _timeout):
        return {"output": "observation" * 200, "returncode": 0}

    trace = await TerminalAgent(tokenizer=Tokenizer(), decode=decode, execute=execute, config=config).run_async("task")
    assert trace.generated_tokens == 120000 and trace.truncated and trace.n_compactions > 1
    assert {call["kind"] for call in trace.calls} == {"action", "summary"}
    samples = []
    for segment, fields in enumerate(trace.sample_fields()):
        sample = Sample(
            index=0,
            rollout_id=0,
            group_index=0,
            tokens=fields["tokens"],
            response_length=fields["response_length"],
            loss_mask=fields["loss_mask"],
            reward=1.0,
            status=Sample.Status.TRUNCATED,
            metadata={"segment": segment},
        )
        for key, value in fields.items():
            if key.startswith("critic_"):
                setattr(sample, key, value)
        samples.append(sample)
    assert max(len(sample.tokens) for sample in samples) <= 32768
    assert 131072 < sum(len(sample.tokens) for sample in samples) <= 262144
    executor = _executor(
        monkeypatch, mode=mode, dp_size=1, balance=False, samples=samples, num_groups=1, fanout=1, critic_steps=1
    )
    executor.args.num_critic_only_steps = 30
    executor.args.max_tokens_per_gpu = 131072
    source = deepcopy(samples)
    with pytest.raises(ValueError, match="exceeds --max-tokens-per-gpu"):
        await executor.get(0)
    # get() consumes/mutates sample conversion fields; restore the source batch.
    executor._get_rollout_data.return_value = (source, {}, {})
    executor.args.max_tokens_per_gpu = 262144
    pack = await executor.get(0)
    actor, actor_result = process_rollout_data(executor.args, pack.data_ref, 0, 1, witness_info=None)
    critic, critic_result = process_rollout_data(executor.args, pack.for_critic().data_ref, 0, 1, witness_info=None)
    with actor_result, critic_result:
        assert sum(sum(mask) for mask in actor["loss_masks"]) == 120000
        assert sum(sum(mask) for mask in critic["loss_masks"]) == 120000
        responses = [
            tokens[-length:] for tokens, length in zip(critic["tokens"], critic["response_lengths"], strict=True)
        ]
        shipped = [responses[row][start : start + length] for row, start, length in critic["value_slices"]]
        assert shipped == [
            tokens[-length:] for tokens, length in zip(actor["tokens"], actor["response_lengths"], strict=True)
        ]
