"""Eviction skips the bias and value fits but keeps exactly the keys the full algorithm selects."""

import os

import pytest
import torch

from agentic_critic import kv_compact

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(kv_compact.UPSTREAM, "compaction")), reason="upstream compaction repo not available")


@pytest.mark.parametrize("seed", range(3))
def test_selection_is_the_one_of_the_full_algorithm(seed):
    torch.manual_seed(seed)
    keys, values, queries = torch.randn(60, 16), torch.randn(60, 16), torch.randn(200, 16) * 3
    full, select = kv_compact._algorithm(), kv_compact._algorithm(select_only=True)
    _, _, _, want = full.compute_compacted_cache(keys, values, queries, 15, None)
    _, beta, got = select._select_keys_highest_attention(keys, queries, 15, None)
    assert list(got) == list(want) and float(beta.abs().max()) == 0.0
