# AIME datasets

The canonical parquet files are committed directly to the EasyPPO release,
so a normal training launch only verifies them and does not download data.

Prompt text is preserved verbatim for reproducibility, including its original
language. The AIME validation prompts are English.

| Output | Role | Canonical rows | Pinned upstream revision |
| --- | --- | ---: | --- |
| `dapo-math-17k.parquet` | training | 17,917 | `BytedTsinghua-SIA/DAPO-Math-17k@65877096c24ffa7abc4e4fa5edb95cf3413a5674` |
| `aime-2024.parquet` | validation only | 30 | `BytedTsinghua-SIA/AIME-2024@aa49075e24ad594b79fdf0bdcefa735c2181be67` |

| Built-in file | Size | SHA-256 |
| --- | ---: | --- |
| `dapo-math-17k.parquet` | 2,311,643 bytes | `134671de0cd455477e3bbca80f125ea32094084ef1ae62e2fa3e1b066414ba4c` |
| `aime-2024.parquet` | 19,113 bytes | `91f8ef5168ae2a7db9bc6860211b731de3b9358a993088c29a6d07cda90b0dd9` |

The pinned upstream files contain 100 copies of each DAPO row and 32 copies
of each AIME row. The preparation pipeline verifies the source SHA-256 values
and keeps one row per `extra_info.index`. verl then controls the actual number
of responses with `rollout.n=16` for training and `val_kwargs.n=32` for AIME.

All canonical rows are validated to contain a one-message user prompt,
`data_source=math_dapo`, `reward_model.ground_truth`, and the
`rule-lighteval/MATH_v2` reward style.

`scripts/prepare_aime_data.sh` remains available to reproduce or repair
the built-in files from the pinned upstream revisions.
