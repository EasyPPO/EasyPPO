from argparse import Namespace

import numpy as np
import pytest

from miles.rollout.generate_utils.rollout_topk_logprobs import extract_rollout_topk_logprobs
from miles.utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_args
from tests.fast.fixtures.score_centering_fixtures import _flat_output_top_logprobs


def _args(**overrides: object) -> Namespace:
    values = dict(
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
        rollout_top_p=1.0,
        rollout_top_k=-1,
        use_opd=False,
        opd_log_prob_top_k=0,
        opd_top_k_strategy="only-student",
    )
    return Namespace(**(values | overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"rollout_top_logprobs_num": 128},
        {"rollout_top_logprobs_num": 128, "rollout_sampling_logprobs_mode": "support", "rollout_top_k": 64},
        {
            "rollout_top_logprobs_num": 64,
            "rollout_sampling_logprobs_mode": "support",
            "rollout_top_p": 0.9,
            "rollout_top_k": 64,
        },
        {"rollout_sampling_logprobs_mode": "support", "rollout_top_k": 64},
        {"rollout_sampling_logprobs_mode": "support", "rollout_top_p": 0.9, "rollout_top_k": 64},
        {
            "rollout_top_logprobs_num": 8,
            "use_opd": True,
            "opd_log_prob_top_k": 4,
            "opd_top_k_strategy": "only-teacher",
        },
    ],
)
def test_consistent_settings_pass(overrides: dict) -> None:
    validate_rollout_topk_logprobs_args(_args(**overrides))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"rollout_top_logprobs_num": -1}, "non-negative"),
        (
            {"rollout_top_logprobs_num": 32, "rollout_sampling_logprobs_mode": "support", "rollout_top_k": 64},
            ">= --rollout-top-k",
        ),
        ({"rollout_top_logprobs_num": 8, "use_opd": True, "opd_log_prob_top_k": 4}, "OPD student"),
    ],
)
def test_inconsistent_settings_are_rejected(overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_rollout_topk_logprobs_args(_args(**overrides))


@pytest.mark.parametrize("k", [2, 3])
def test_flat_candidates_match_json_with_padding_and_truncation(k):
    rows = [
        [(-0.5, 2, None), (-np.inf, 3, None), (-1.5, 4, None)],
        [(-0.75, 5, None), (-1.25, 6, None), (-np.inf, 7, None)],
    ]
    sampled = {"output_token_logprobs": [rows[0][0], rows[1][0]]}

    actual = extract_rollout_topk_logprobs(sampled | _flat_output_top_logprobs(rows), k)
    expected = extract_rollout_topk_logprobs(sampled | {"output_top_logprobs": rows}, k)

    for flat, nested in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(flat, nested)
        assert flat.shape == (2, k)
    ids, logps = actual
    assert ids.dtype == np.int32 and logps.dtype == np.float32
    assert ids[0, 1] == -1 and np.isneginf(logps[0, 1])


@pytest.mark.parametrize(
    "case,match",
    [
        ("token_count", "every generated token"),
        ("width", "fewer than"),
        ("nan", "invalid log probability"),
        ("positive_inf", "invalid log probability"),
        ("null_prefix", "every generated token"),
    ],
)
def test_invalid_flat_candidates_are_rejected(case, match):
    rows = [[(-0.5, 2, None), (-1.0, 3, None)]]
    sampled = [rows[0][0]]
    if case == "token_count":
        sampled *= 2
    if case in ("nan", "positive_inf"):
        rows[0][1] = (np.nan if case == "nan" else np.inf, 3, None)
    meta = {"output_token_logprobs": sampled} | _flat_output_top_logprobs(
        rows, null_prefix=int(case == "null_prefix")
    )

    with pytest.raises(ValueError, match=match):
        extract_rollout_topk_logprobs(meta, 3 if case == "width" else 2)
