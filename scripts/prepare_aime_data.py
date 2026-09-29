#!/usr/bin/env python3
"""Canonicalize and validate the pinned DAPO/AIME parquet files.

The upstream files intentionally repeat DAPO-Math-17k 100 times and AIME-2024
32 times. veRL already controls sampling through train epochs and rollout.n, so
this recipe recovers one row per upstream extra_info.index before training.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

REQUIRED_COLUMNS = {"data_source", "prompt", "reward_model", "extra_info"}
EXPECTED_DATA_SOURCE = "math_dapo"
EXPECTED_REWARD_STYLE = "rule-lighteval/MATH_v2"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_path: Path
    output_path: Path
    unique_rows: int
    upstream_repetitions: int


def _index_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_columns(path: Path, parquet: pq.ParquetFile) -> None:
    columns = set(parquet.schema_arrow.names)
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")


def validate_canonical(path: Path, expected_rows: int) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing parquet: {path}")

    parquet = pq.ParquetFile(path)
    _validate_columns(path, parquet)
    if parquet.metadata.num_rows != expected_rows:
        raise RuntimeError(f"{path} has {parquet.metadata.num_rows} rows; expected {expected_rows} canonical rows")

    seen_indices: set[str] = set()
    row_number = 0
    columns = ["data_source", "prompt", "reward_model", "extra_info"]
    for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
        values = batch.to_pydict()
        for data_source, prompt, reward_model, extra_info in zip(
            values["data_source"],
            values["prompt"],
            values["reward_model"],
            values["extra_info"],
            strict=True,
        ):
            row_number += 1
            if data_source != EXPECTED_DATA_SOURCE:
                raise RuntimeError(
                    f"{path} row {row_number}: data_source={data_source!r}, expected {EXPECTED_DATA_SOURCE!r}"
                )
            if not isinstance(prompt, list) or len(prompt) != 1:
                raise RuntimeError(f"{path} row {row_number}: prompt must contain exactly one message")
            message = prompt[0]
            if message.get("role") != "user" or not str(message.get("content", "")).strip():
                raise RuntimeError(f"{path} row {row_number}: invalid user prompt")
            if "Answer:" not in message["content"]:
                raise RuntimeError(f"{path} row {row_number}: prompt does not request the Answer: format")
            if not isinstance(reward_model, dict) or not str(reward_model.get("ground_truth", "")).strip():
                raise RuntimeError(f"{path} row {row_number}: missing reward_model.ground_truth")
            if reward_model.get("style") != EXPECTED_REWARD_STYLE:
                raise RuntimeError(
                    f"{path} row {row_number}: reward style={reward_model.get('style')!r}, "
                    f"expected {EXPECTED_REWARD_STYLE!r}"
                )
            if not isinstance(extra_info, dict) or extra_info.get("index") is None:
                raise RuntimeError(f"{path} row {row_number}: missing extra_info.index")
            key = _index_key(extra_info["index"])
            if key in seen_indices:
                raise RuntimeError(f"{path} row {row_number}: duplicate extra_info.index={key}")
            seen_indices.add(key)

    if len(seen_indices) != expected_rows:
        raise RuntimeError(f"{path} has {len(seen_indices)} unique indices; expected {expected_rows}")
    print(f"[data] verified {path}: rows={expected_rows}, unique_indices={len(seen_indices)}")


def canonicalize(spec: DatasetSpec, force: bool) -> None:
    if spec.output_path.exists() and not force:
        try:
            validate_canonical(spec.output_path, spec.unique_rows)
            print(f"[data] reusing canonical {spec.name}: {spec.output_path}")
            return
        except RuntimeError as exc:
            print(f"[data] rebuilding invalid {spec.name} output: {exc}")

    if not spec.input_path.is_file() or spec.input_path.stat().st_size == 0:
        raise RuntimeError(f"missing upstream parquet: {spec.input_path}")

    parquet = pq.ParquetFile(spec.input_path)
    _validate_columns(spec.input_path, parquet)
    expected_raw_rows = spec.unique_rows * spec.upstream_repetitions
    if parquet.metadata.num_rows != expected_raw_rows:
        raise RuntimeError(
            f"{spec.input_path} has {parquet.metadata.num_rows} rows; expected {expected_raw_rows} "
            f"({spec.unique_rows} x {spec.upstream_repetitions})"
        )

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = spec.output_path.with_name(f".{spec.output_path.name}.part.{os.getpid()}")
    temporary.unlink(missing_ok=True)

    counts: Counter[str] = Counter()
    seen_indices: set[str] = set()
    writer: pq.ParquetWriter | None = None
    try:
        for batch in parquet.iter_batches(batch_size=65_536):
            extra_info_position = batch.schema.get_field_index("extra_info")
            extra_infos = batch.column(extra_info_position).to_pylist()
            selected: list[int] = []
            for position, extra_info in enumerate(extra_infos):
                if not isinstance(extra_info, dict) or extra_info.get("index") is None:
                    raise RuntimeError(f"{spec.input_path}: missing extra_info.index")
                key = _index_key(extra_info["index"])
                counts[key] += 1
                if key not in seen_indices:
                    seen_indices.add(key)
                    selected.append(position)

            if selected:
                canonical_batch = batch.take(pa.array(selected, type=pa.int64()))
                if writer is None:
                    writer = pq.ParquetWriter(temporary, canonical_batch.schema, compression="zstd")
                writer.write_table(pa.Table.from_batches([canonical_batch]))
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        if writer is None:
            raise RuntimeError(f"{spec.input_path} contained no rows")
        writer.close()

    multiplicities = set(counts.values())
    if len(seen_indices) != spec.unique_rows or multiplicities != {spec.upstream_repetitions}:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{spec.input_path}: unique_indices={len(seen_indices)}, multiplicities={sorted(multiplicities)}; "
            f"expected {spec.unique_rows} unique indices repeated {spec.upstream_repetitions} times"
        )

    validate_canonical(temporary, spec.unique_rows)
    os.replace(temporary, spec.output_path)
    print(
        f"[data] canonicalized {spec.name}: {expected_raw_rows} upstream rows -> "
        f"{spec.unique_rows} rows at {spec.output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=Path)
    parser.add_argument("--validation-input", type=Path)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        validate_canonical(args.train_output, 17_917)
        validate_canonical(args.validation_output, 30)
        return

    if args.train_input is None or args.validation_input is None:
        raise SystemExit("--train-input and --validation-input are required unless --verify-only is used")

    canonicalize(
        DatasetSpec("DAPO-Math-17k", args.train_input, args.train_output, 17_917, 100),
        force=args.force,
    )
    canonicalize(
        DatasetSpec("AIME-2024", args.validation_input, args.validation_output, 30, 32),
        force=args.force,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
