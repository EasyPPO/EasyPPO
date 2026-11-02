"""Load the pinned upstream AM algorithm without replacing its methods."""

import os
import sys
from pathlib import Path

UPSTREAM_COMMIT = "2020f6d37940fe013e7e7f1ebf409e5d107c5670"


def algorithm():
    # Optional dependency: ordinary actor/shared/full forwards do not need AM.
    root = Path(os.environ.get("KV_COMPACTION_REPO", Path(__file__).resolve().parents[1] / "third_party/compaction"))
    if not (root / "compaction/algorithms/highest_attention_keys.py").is_file():
        raise FileNotFoundError("Initialize third_party/compaction or set KV_COMPACTION_REPO to the pinned upstream checkout")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from compaction.algorithms.highest_attention_keys import HighestAttentionKeysCompaction

    # examples/qa_demo.py: official HighestAttnKeys demo settings. C2 uses the
    # upstream unregularized least-squares solver, not our former V0 ridge.
    return HighestAttentionKeysCompaction(
        score_method="rms", nnls_iters=2, nnls_lower_bound=0.05,
        nnls_upper_bound=20.0, c2_method="lsq", c2_ridge_lambda=0, c2_solver="lstsq",
    )
