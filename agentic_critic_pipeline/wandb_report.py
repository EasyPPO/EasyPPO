"""Dump every value of a finished critic comparison to Weights & Biases.

What goes where, for each analysis suite written by ``visualize.sh``:

* every scalar of ``analysis.json``, ``figures.json`` and ``diagnostics.json`` -> run summary,
  under ``<suite>/<module>/<json path>``;
* every list of rows in those files (curves, profiles, per-index jumps, ...) -> a ``wandb.Table``
  under ``tables/<suite>/<module>/<json path>``;
* every PNG -> ``images/<suite>/<module>/<file>``;
* the per-batch training curves -> ``curve/...`` plotted against ``curve/batch``;
* a curated ``headline/...`` set of the values the experiments are read by;
* the raw JSON / Markdown files -> one ``critic-analysis`` artifact.

The dump is a side effect and must never fail a run: problems (missing files, invalid JSON,
missing fields, tracking errors) are collected as ``issues`` and the process exits 0. A
local ``wandb_dump/manifest.json`` and ``wandb_dump/summary.json`` are always written, and
after upload the stored run summary is read back and compared key by key with what was sent.
"""

import argparse
import json
import math
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

MODES = ("actor_summary", "full_history")
BUCKETS = ("all", "0", "1", "2", "3", "4+")
MODULES = {"analysis": "analysis.json", "figures": "figures.json", "diagnostics": "diagnostics.json"}
METRICS = ("ev", "within_batch_ev", "mse", "mae", "bias")
EVENT_FIELDS = (
    "signed_jump",
    "abs_jump",
    "error_improvement",
    "signed_error_before",
    "signed_error_after",
    "abs_error_before",
    "abs_error_after",
    "sq_error_before",
    "sq_error_after",
    "sq_error_improvement",
    "moved_closer",
)
RETENTION_FIELDS = (
    "reward_correlation_before",
    "reward_correlation_after",
    "reward_correlation_change",
    "retained_correlation",
    "slope",
)
MISSING = object()


def suites(root):
    """``(suite id, {module: directory})`` for every analysis ``visualize.sh`` writes."""
    root = Path(root)
    out = [("main", {m: root / m for m in MODULES})]
    out += [(f"input_swap_{t}", {m: root / f"input_swap_{t}" / m for m in MODULES}) for t in MODES]
    validation = {"analysis": "analyze", "figures": "figures", "diagnostics": "diagnostics"}
    out.append(("validation/main", {m: root / "validation" / d for m, d in validation.items()}))
    out += [
        (f"validation/input_swap_{t}", {m: root / "validation" / f"input_swap_{t}" / d for m, d in validation.items()})
        for t in MODES
    ]
    return out


def headline_label(suite, phase):
    """Readable headline prefix: which data the numbers come from, and which comparison."""
    split = "validation" if suite.startswith("validation/") else ("train_online" if phase == "pre_update" else "test")
    name = suite.split("/", 1)[-1]
    comparison = "main" if name == "main" else name.replace("input_swap_", "swap_trained_")
    return f"headline/{split}/{comparison}"


def lookup(obj, path):
    for part in path:
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return MISSING
    return obj


class Dump:
    """Everything one run will receive, plus the issues found while assembling it."""

    def __init__(self):
        self.summary, self.tables, self.images, self.curves = {}, {}, {}, {}
        self.files, self.issues, self.sources = [], [], {}
        self.headline_expected, self.headline_missing = [], []

    def issue(self, kind, where, detail=""):
        self.issues.append(dict(kind=kind, where=str(where), detail=str(detail)[:2000]))

    @staticmethod
    def scalar(value):
        """JSON-safe scalar, or MISSING for values that are not scalars."""
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        try:  # numpy scalars from in-process callers
            item = value.item()
        except Exception:
            return MISSING
        return Dump.scalar(item) if not hasattr(item, "item") else MISSING

    def put(self, key, value, where=""):
        clean = self.scalar(value)
        if clean is MISSING:
            clean = json.dumps(value, default=str)
        elif clean is None and isinstance(value, float):
            self.issue("nonfinite", key, f"{value!r} logged as null ({where})")
        if key in self.summary and self.summary[key] != clean:
            self.issue("conflict", key, f"{self.summary[key]!r} vs {clean!r}")
        self.summary[key] = clean

    def flatten(self, obj, prefix, where):
        """Scalars to summary keys, lists of rows to tables, other lists verbatim as JSON."""
        counts = dict(scalars=0, tables=0, lists=0)

        def walk(node, key):
            if isinstance(node, dict):
                if not node:
                    self.put(key, "{}", where)
                    counts["lists"] += 1
                for name, child in node.items():
                    walk(child, f"{key}/{name}")
            elif isinstance(node, list):
                if node and all(isinstance(x, dict) for x in node):
                    self.table(key.replace(prefix, f"tables/{prefix}", 1), node, where)
                    counts["tables"] += 1
                else:
                    self.put(key, json.dumps(node, default=str), where)
                    counts["lists"] += 1
            else:
                self.put(key, node, where)
                counts["scalars"] += 1

        walk(obj, prefix)
        return counts

    def table(self, key, rows, where=""):
        columns, data = [], []
        for row in rows:
            flat = {}

            def walk(node, name, flat=flat):
                if isinstance(node, dict):
                    for child_name, child in node.items():
                        walk(child, f"{name}/{child_name}" if name else str(child_name))
                elif isinstance(node, list):
                    flat[name] = json.dumps(node, default=str)
                else:
                    value = self.scalar(node)
                    flat[name] = json.dumps(node, default=str) if value is MISSING else value

            walk(row, "")
            for name in flat:
                if name not in columns:
                    columns.append(name)
            data.append(flat)
        self.tables[key] = (columns, [[row.get(c) for c in columns] for row in data])

    def headline(self, key, source, path, where):
        self.headline_expected.append(key)
        value = lookup(source, path)
        if value is MISSING:
            self.headline_missing.append(key)
            self.issue("missing_field", where, "/".join(map(str, path)) + f" -> {key}")
            return
        self.put(key, value, where)


def interval_specs(key, path):
    return [(f"{key}/{part}", path + (part,)) for part in ("value", "low", "high")]


def analysis_headlines(dump, suite, analysis, where):
    for phase, data in (analysis.get("phases") or {}).items():
        label = headline_label(suite, phase)
        specs = [
            (f"{label}/episodes", ("by_compactions", "all", "episodes")),
            (f"{label}/problems", ("by_compactions", "all", "problems")),
            (f"{label}/segments", ("by_compactions", "all", "segments")),
            (f"{label}/constant_baseline/mse", ("by_compactions", "all", "constant_baseline", "mse")),
            (f"{label}/oracle_constant_mse", ("by_compactions", "all", "oracle_constant_mse")),
        ]
        for bucket in BUCKETS:
            base = ("by_compactions", bucket)
            for mode in MODES:
                specs.append((f"{label}/tokens/{bucket}/{mode}", base + (mode, "tokens")))
                specs += [(f"{label}/{m}/{bucket}/{mode}", base + (mode, m)) for m in METRICS]
            for m in METRICS:
                specs += interval_specs(f"{label}/{m}/{bucket}/full_minus_actor", base + ("full_minus_actor", m))
        for kind in ("at_compaction", "at_midpoint"):
            specs += [(f"{label}/{kind}/events", (kind, "events")), (f"{label}/{kind}/problems", (kind, "problems"))]
            for field in EVENT_FIELDS:
                specs += [(f"{label}/{kind}/{field}/{mode}", (kind, field, mode)) for mode in MODES]
                specs += interval_specs(f"{label}/{kind}/{field}/full_minus_actor", (kind, field, "full_minus_actor"))
        for index in BUCKETS[2:]:
            base = ("by_compaction_index", index)
            specs.append((f"{label}/by_compaction_index/{index}/events", base + ("events",)))
            for field in ("abs_jump", "error_improvement", "sq_error_improvement", "moved_closer"):
                specs += [
                    (f"{label}/by_compaction_index/{index}/{field}/{mode}", base + (field, mode)) for mode in MODES
                ]
                specs.append(
                    (
                        f"{label}/by_compaction_index/{index}/{field}/full_minus_actor",
                        base + (field, "full_minus_actor", "value"),
                    )
                )
        for kind in ("compaction", "midpoint"):
            base = ("retention", kind)
            specs.append((f"{label}/retention/{kind}/events", base + ("events",)))
            for mode in MODES:
                specs += [
                    (f"{label}/retention/{kind}/{field}/{mode}", base + (mode, field)) for field in RETENTION_FIELDS
                ]
            specs += interval_specs(
                f"{label}/retention/{kind}/change/full_minus_actor", base + ("change_full_minus_actor",)
            )
        specs += [
            (f"{label}/variance_share/{mode}", ("variance_share", mode, "between_segment_fraction")) for mode in MODES
        ]
        for key, path in specs:
            dump.headline(key, data, path, f"{where}:phases/{phase}")
    provenance = analysis.get("provenance") or {}
    dump.put(f"headline/provenance/{suite}/comparison_kind", provenance.get("comparison_kind"), where)
    for mode in MODES:
        dump.put(f"headline/provenance/{suite}/{mode}/trained_mode", lookup(provenance, (mode, "trained_mode")), where)


def diagnostics_headlines(dump, suite, diagnostics, where):
    swap = "input_swap" in suite
    for phase, data in (diagnostics.get("phases") or {}).items():
        label = headline_label(suite, phase)
        specs = []
        residual = data.get("residual_history") or {}
        if residual.get("status") == "insufficient_problems":
            dump.put(f"{label}/residual_history/status", "insufficient_problems", where)
        else:
            for name in ("r2", "squared_error_explained"):
                base = ("residual_history", "metrics", name)
                specs += [(f"{label}/residual_history/{name}/{mode}", base + (mode,)) for mode in MODES]
                specs.append(
                    (f"{label}/residual_history/{name}/full_minus_actor", base + ("full_minus_actor", "value"))
                )
        for group, names in (
            ("cumulative_removed_thirds", ("lowest_third", "middle_third", "highest_third")),
            ("actor_positions", tuple((lookup(data, ("slices", "actor_positions")) or {}).keys())),
        ):
            for name in names:
                base = ("slices", group, name, "full_minus_actor", "ev")
                specs += [
                    (f"{label}/slices/{group}/{name}/ev/{mode}", ("slices", group, name, mode, "ev")) for mode in MODES
                ]
                specs += interval_specs(f"{label}/slices/{group}/{name}/ev/full_minus_actor", base)
        if phase == "heldout":
            for bucket in BUCKETS:
                for metric in ("norm", "gsnr", "same_problem_different_episode_cosine"):
                    base = ("gradients", bucket, "metrics", metric)
                    if lookup(data, ("gradients", bucket, "segments")) == 0:
                        continue  # no segment had this many compactions: nothing to report
                    specs += [(f"{label}/gradients/{bucket}/{metric}/{mode}", base + (mode,)) for mode in MODES]
                    specs.append(
                        (f"{label}/gradients/{bucket}/{metric}/full_minus_actor", base + ("full_minus_actor", "value"))
                    )
        if swap:
            for field in ("passed", "segments", "max_segment_mean_difference", "mean_token_absolute_difference"):
                specs.append((f"{label}/zero_compaction_check/{field}", ("zero_compaction_check", field)))
        for key, path in specs:
            dump.headline(key, data, path, f"{where}:phases/{phase}")
    for phase, data in (diagnostics.get("score_only_baseline") or {}).items():
        label = headline_label(suite, phase)
        for variant, result in (data.get("variants") or {}).items():
            dump.put(f"{label}/score_only/{variant}/status", result.get("status"), where)
            for bucket in BUCKETS:
                value = lookup(result, ("by_compactions", bucket, "ev"))
                if value is not MISSING:
                    dump.put(f"{label}/score_only/{variant}/ev/{bucket}", value, where)


def curves(dump, suite, analysis):
    for phase, data in (analysis.get("phases") or {}).items():
        for row in data.get("curve") or []:
            batch = row.get("batch")
            if not isinstance(batch, int):
                continue
            step = dump.curves.setdefault(batch, {})
            for mode in MODES:
                for metric in METRICS + ("tokens",):
                    value = Dump.scalar(lookup(row, (mode, metric)))
                    if value is not MISSING:
                        step[f"curve/{suite}/{phase}/{mode}/{metric}"] = value
            for metric in METRICS:
                value = Dump.scalar(lookup(row, ("full_minus_actor", metric, "value")))
                if value is not MISSING:
                    step[f"curve/{suite}/{phase}/full_minus_actor/{metric}"] = value


def collect_suites(dump, root):
    for suite, directories in suites(root):
        for module, directory in directories.items():
            where = directory / MODULES[module]
            if not where.is_file():
                dump.sources[str(where)] = dict(status="missing")
                dump.issue("missing_file", where)
                continue
            try:
                data = json.loads(where.read_text())
            except Exception as error:
                dump.sources[str(where)] = dict(status="invalid_json")
                dump.issue("invalid_json", where, error)
                continue
            dump.files.append(where)
            try:
                counts = dump.flatten(data, f"{suite}/{module}", str(where))
                dump.sources[str(where)] = dict(status="ok", **counts)
                if module == "analysis":
                    analysis_headlines(dump, suite, data, str(where))
                    curves(dump, suite, data)
                elif module == "diagnostics":
                    diagnostics_headlines(dump, suite, data, str(where))
            except Exception:
                dump.sources[str(where)] = dict(status="error")
                dump.issue("error", where, traceback.format_exc())
            for extra in sorted(directory.glob("*.md")):
                dump.files.append(extra)
        for module, directory in directories.items():
            for image in sorted(directory.rglob("*.png")) if directory.is_dir() else []:
                relative = image.relative_to(directory).with_suffix("").as_posix()
                dump.images[f"images/{suite}/{module}/{relative}"] = image


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))])


def collect_episodes(dump, split, path):
    """Dataset facts every comparison depends on, streamed one episode at a time."""
    path = Path(path)
    prefix = f"data/{split}"
    if not path.is_file():
        dump.issue("missing_file", path, f"episodes for {split}")
        return
    rewards, segments, full_tokens, actor_tokens, problems, compactions = [], [], [], [], set(), {}
    try:
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    episode = json.loads(line)
                    parts = episode["segments"]
                    rewards.append(float(episode["reward"]))
                    problems.add(str(episode["problem_id"]))
                    segments.append(len(parts))
                    full_tokens.append(sum(len(s["input_ids"]) for s in parts))
                    actor_tokens.append(sum(sum(s["loss_mask"]) for s in parts))
                    bucket = str(len(parts) - 1) if len(parts) <= 4 else "4+"
                    compactions[bucket] = compactions.get(bucket, 0) + 1
                except Exception as error:
                    dump.issue("invalid_episode", f"{path}:{number}", error)
    except Exception as error:
        dump.issue("error", path, error)
        return
    n = len(rewards)
    dump.put(f"{prefix}/episodes", n)
    dump.put(f"{prefix}/problems", len(problems))
    if n:
        mean = sum(rewards) / n
        dump.put(f"{prefix}/reward/mean", mean)
        dump.put(f"{prefix}/reward/std", math.sqrt(sum((r - mean) ** 2 for r in rewards) / n))
        dump.put(f"{prefix}/reward/min", min(rewards))
        dump.put(f"{prefix}/reward/max", max(rewards))
        dump.put(f"{prefix}/reward/zero_fraction", sum(r == 0 for r in rewards) / n)
        dump.put(f"{prefix}/segments/mean", sum(segments) / n)
        dump.put(f"{prefix}/segments/max", max(segments))
        for name, values in (("full_tokens", full_tokens), ("actor_tokens", actor_tokens)):
            dump.put(f"{prefix}/{name}/mean", sum(values) / n)
            for q in (0.5, 0.9, 0.99):
                dump.put(f"{prefix}/{name}/p{int(q * 100)}", quantile(values, q))
            dump.put(f"{prefix}/{name}/max", max(values))
        for bucket in ("0", "1", "2", "3", "4+"):
            dump.put(f"{prefix}/episodes_by_compactions/{bucket}", compactions.get(bucket, 0))
    cohort = path.parent / "cohort.json"
    if cohort.is_file():
        try:
            dump.flatten(json.loads(cohort.read_text()), f"{prefix}/cohort", str(cohort))
            dump.files.append(cohort)
        except Exception as error:
            dump.issue("invalid_json", cohort, error)


def collect_training(dump, root):
    for mode in MODES:
        path = Path(root) / mode / "train" / "miles_run.json"
        if not path.is_file():
            dump.issue("missing_file", path)
            continue
        try:
            dump.flatten(json.loads(path.read_text()), f"training/{mode}", str(path))
            dump.files.append(path)
        except Exception as error:
            dump.issue("invalid_json", path, error)


def check_coverage(dump):
    """Every scalar leaf of every loaded source must be in the summary with its exact value."""
    checked, wrong = 0, 0
    for source, info in dump.sources.items():
        if info.get("status") != "ok":
            continue
        path = Path(source)
        suite = next((s for s, dirs in suites(dump.root) for m, d in dirs.items() if d / MODULES[m] == path), None)
        module = next(m for m, name in MODULES.items() if name == path.name)
        data = json.loads(path.read_text())

        def walk(node, key, suite=suite, module=module):
            nonlocal checked, wrong
            if isinstance(node, dict):
                for name, child in node.items():
                    walk(child, f"{key}/{name}")
            elif isinstance(node, list) and node and all(isinstance(x, dict) for x in node):
                table = key.replace(f"{suite}/{module}", f"tables/{suite}/{module}", 1)
                checked += 1
                if table not in dump.tables or len(dump.tables[table][1]) != len(node):
                    wrong += 1
                    dump.issue("coverage", table, "table missing or row count differs")
            elif not isinstance(node, list):
                checked += 1
                expected = Dump.scalar(node)
                if dump.summary.get(key, MISSING) != expected:
                    wrong += 1
                    dump.issue("coverage", key, f"expected {expected!r}, got {dump.summary.get(key, 'MISSING')!r}")

        walk(data, f"{suite}/{module}")
    return dict(checked=checked, wrong=wrong)


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return ""


def compare_summary(sent, stored):
    """Keys whose stored value differs from what was sent (tables/images are checked by presence)."""
    wrong = []
    for key, value in sent.items():
        got = stored.get(key, MISSING)
        if got is MISSING:
            wrong.append(dict(key=key, sent=value, stored="MISSING"))
        elif isinstance(value, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
            if not math.isclose(value, got, rel_tol=1e-12, abs_tol=0.0):
                wrong.append(dict(key=key, sent=value, stored=got))
        elif got != value:
            wrong.append(dict(key=key, sent=value, stored=got))
    return wrong


def upload(dump, args, status):
    try:
        import wandb
    except Exception as error:
        dump.issue("tracking", "import wandb", error)
        status["upload"] = "failed"
        return
    modes = [args.mode] + (["offline"] if args.mode == "online" else [])
    run = None
    for mode in modes:
        try:
            run = wandb.init(
                project=args.project,
                entity=args.entity or None,
                group=args.group,
                name=args.run_name,
                job_type="critic-analysis",
                mode=mode,
                dir=str(args.wandb_dir),
                config=dict(output_root=str(args.output_root), git_commit=git_commit(), group=args.group),
                settings=wandb.Settings(init_timeout=float(args.init_timeout)),
            )
            status["mode"] = mode
            break
        except Exception as error:
            dump.issue("tracking", f"wandb.init mode={mode}", error)
    if run is None:
        status["upload"] = "failed"
        return
    status.update(run_id=run.id, run_url=getattr(run, "url", None), run_dir=getattr(run, "dir", None))
    try:
        run.define_metric("curve/batch")
        run.define_metric("curve/*", step_metric="curve/batch")
    except Exception as error:
        dump.issue("tracking", "define_metric", error)
    for batch in sorted(dump.curves):
        try:
            run.log({"curve/batch": batch, **dump.curves[batch]})
        except Exception as error:
            dump.issue("tracking", f"curve batch {batch}", error)
    media = {}
    for key, (columns, rows) in dump.tables.items():
        try:
            media[key] = wandb.Table(columns=columns, data=rows, allow_mixed_types=True)
        except Exception as error:
            dump.issue("tracking", f"table {key}", error)
    for key, path in dump.images.items():
        try:
            media[key] = wandb.Image(str(path))
        except Exception as error:
            dump.issue("tracking", f"image {key}", error)
    keys = list(media)
    for start in range(0, len(keys), 50):
        chunk = {k: media[k] for k in keys[start : start + 50]}
        try:
            run.log(chunk)
        except Exception:
            for key, value in chunk.items():
                try:
                    run.log({key: value})
                except Exception as error:
                    dump.issue("tracking", f"log {key}", error)
    items = list(dump.summary.items())
    for start in range(0, len(items), 1000):
        chunk = dict(items[start : start + 1000])
        try:
            run.summary.update(chunk)
        except Exception:
            for key, value in chunk.items():
                try:
                    run.summary[key] = value
                except Exception as error:
                    dump.issue("tracking", f"summary {key}", error)
    try:
        artifact = wandb.Artifact(re.sub(r"[^A-Za-z0-9._-]", "-", f"critic-analysis-{args.group}"), "critic-analysis")
        for path in dict.fromkeys(dump.files):
            try:
                artifact.add_file(str(path), name=os.path.relpath(path, args.output_root))
            except Exception as error:
                dump.issue("tracking", f"artifact file {path}", error)
        for name in ("manifest.json", "summary.json"):
            local = args.local_dir / name
            if local.is_file():
                artifact.add_file(str(local), name=f"wandb_dump/{name}")
        run.log_artifact(artifact)
    except Exception as error:
        dump.issue("tracking", "artifact", error)
    run_dir = getattr(run, "dir", None)
    try:
        run.finish()
        status["upload"] = "ok"
    except Exception as error:
        dump.issue("tracking", "finish", error)
        status["upload"] = "failed"
        return
    verify(dump, args, status, run, run_dir)


def read_transaction_log(path):
    """Final summary of an offline run, rebuilt from its ``.wandb`` transaction log.

    The log is a leveldb-style record stream (7-byte header, 32 KiB blocks, records of
    crc32/length/type + payload) of ``wandb_internal_pb2.Record`` messages. As on sync,
    the summary is the last value of every history key, overridden by explicit summary
    updates.
    """
    from wandb.proto import wandb_internal_pb2

    data = Path(path).read_bytes()
    if data[:4] != b":W&B":
        raise ValueError(f"Not a wandb transaction log: {path}")
    block, position, pending, summary = 32768, 7, b"", {}

    def assign(item, value):
        # Media (tables, images) arrive as nested keys: [summary key, field].
        path = [item.key] if item.key else list(item.nested_key)
        node = summary
        for part in path[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[path[-1]] = value

    while position + 7 <= len(data):
        if block - position % block < 7:  # block trailer padding
            position += block - position % block
            continue
        length = int.from_bytes(data[position + 4 : position + 6], "little")
        kind = data[position + 6]
        payload = data[position + 7 : position + 7 + length]
        position += 7 + length
        if kind == 0 and length == 0:
            continue
        pending = payload if kind in (1, 2) else pending + payload
        if kind not in (1, 4):
            continue
        record = wandb_internal_pb2.Record()
        record.ParseFromString(pending)
        pending = b""
        which = record.WhichOneof("record_type")
        if which == "history":
            for item in record.history.item:
                assign(item, json.loads(item.value_json))
        elif which == "summary":
            for item in record.summary.update:
                assign(item, json.loads(item.value_json))
            for item in record.summary.remove:
                summary.pop(item.key or "/".join(item.nested_key), None)
    return summary


def verify(dump, args, status, run, run_dir):
    """Read the stored summary back and compare every sent key with it."""
    stored = None
    try:
        if status.get("mode") == "offline":
            logs = sorted(Path(run_dir).parent.glob("*.wandb"))
            if not logs:
                raise FileNotFoundError(f"No .wandb transaction log next to {run_dir}")
            stored = read_transaction_log(logs[-1])
            status["transaction_log"] = str(logs[-1])
            status["verified_against"] = "offline .wandb transaction log"
        else:
            import wandb

            api_run = wandb.Api(timeout=60).run(f"{run.entity}/{run.project}/{run.id}")
            stored = dict(api_run.summary._json_dict)
            status["verified_against"] = "wandb API"
    except Exception as error:
        dump.issue("verify", "read stored summary", error)
        status["verified"] = None
        return
    wrong = compare_summary(dump.summary, stored)
    missing_media = [k for k in list(dump.tables) + list(dump.images) if k not in stored]
    status["verified"] = not wrong and not missing_media
    status["verify_mismatches"] = len(wrong)
    status["verify_missing_media"] = len(missing_media)
    for item in wrong[:200]:
        dump.issue("verify", item["key"], f"sent {item['sent']!r}, stored {item['stored']!r}")
    for key in missing_media[:200]:
        dump.issue("verify", key, "table/image missing from stored summary")
    if status.get("mode") != "offline":
        try:
            api_run.summary["dump/verified"] = status["verified"]
            api_run.summary["dump/verify_mismatches"] = len(wrong) + len(missing_media)
            api_run.summary.update()
        except Exception as error:
            dump.issue("verify", "write verification result", error)


def write_local(dump, args, status):
    try:
        args.local_dir.mkdir(parents=True, exist_ok=True)
        (args.local_dir / "summary.json").write_text(json.dumps(dump.summary, indent=1, sort_keys=True) + "\n")
        manifest = dict(
            status=status,
            counts=dict(
                summary_keys=len(dump.summary),
                tables=len(dump.tables),
                table_rows=sum(len(rows) for _, rows in dump.tables.values()),
                images=len(dump.images),
                curve_steps=len(dump.curves),
                files=len(dict.fromkeys(dump.files)),
            ),
            headline=dict(
                expected=len(dump.headline_expected),
                missing=dump.headline_missing,
            ),
            sources=dump.sources,
            tables={key: dict(columns=len(cols), rows=len(rows)) for key, (cols, rows) in dump.tables.items()},
            issues=dump.issues,
        )
        (args.local_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    except Exception:
        traceback.print_exc()


def run(args):
    dump = Dump()
    dump.root = Path(args.output_root)
    status = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), group=args.group, run_name=args.run_name)
    try:
        collect_suites(dump, args.output_root)
        collect_training(dump, args.output_root)
        for item in args.episodes:
            split, _, path = item.partition("=")
            if not path:
                dump.issue("argument", item, "expected split=path")
                continue
            collect_episodes(dump, split, path)
        status["coverage"] = check_coverage(dump)
    except Exception:
        dump.issue("error", "collect", traceback.format_exc())
    dump.put("dump/issues_before_upload", len(dump.issues))
    dump.put("dump/headline_missing", len(dump.headline_missing))
    dump.put("dump/git_commit", git_commit())
    dump.put("dump/summary_keys", len(dump.summary) + 1)
    status["upload"] = "skipped"
    write_local(dump, args, status)
    if args.mode != "disabled":
        try:
            upload(dump, args, status)
        except Exception:
            dump.issue("error", "upload", traceback.format_exc())
            status["upload"] = "failed"
    status["issues"] = len(dump.issues)
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_local(dump, args, status)
    print(
        f"wandb dump: {len(dump.summary)} values, {len(dump.tables)} tables, {len(dump.images)} images; "
        f"upload={status.get('upload')} verified={status.get('verified')} issues={len(dump.issues)}; "
        f"manifest {args.local_dir / 'manifest.json'}"
    )
    return dump, status


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--episodes", nargs="*", default=[], help="split=path entries for dataset statistics")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "agentic-critic"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--group", default=os.environ.get("WANDB_GROUP", ""))
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--mode",
        choices=("online", "offline", "disabled"),
        default=(
            os.environ.get("WANDB_MODE") if os.environ.get("WANDB_MODE") in ("online", "offline", "disabled") else None
        ),
    )
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--init-timeout", type=float, default=120)
    parser.add_argument("--strict", action="store_true", help="Exit 1 when any issue was recorded (tests only)")
    args = parser.parse_args(argv)
    args.output_root = args.output_root.resolve()
    args.group = args.group or args.output_root.name
    args.run_name = args.run_name or f"{args.group}-analysis"
    if args.mode is None:
        args.mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    args.local_dir = args.output_root / "wandb_dump"
    args.wandb_dir = (args.wandb_dir or args.local_dir).resolve()
    args.wandb_dir.mkdir(parents=True, exist_ok=True)
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return 0
    try:
        dump, status = run(args)
    except Exception:
        traceback.print_exc()
        return 1 if args.strict else 0
    return 1 if args.strict and dump.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
