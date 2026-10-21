"""Exploratory experiment-10 controls; never replace primary paired metrics."""

from collections import defaultdict

import numpy as np

from .analyze import boundary_events


def build_robustness(pairs, *, window, bootstrap, seed, jump_row, retention_row):
    """Keep episode identities paired and report empty/undefined slices honestly."""

    def events_for(width):
        boundary, midpoint = boundary_events(pairs, window=width)
        # Stored edge arrays must actually contain the requested window.
        return {
            kind: [e for e in events if all(min(len(s["before"]), len(s["after"])) >= width for s in e["sides"])]
            for kind, events in (("compaction", boundary), ("midpoint", midpoint))
        }

    def index_rows(events, **labels):
        return [
            retention_row(
                [e for e in events if (str(e["index"]) if e["index"] < 3 else "3+") == index],
                bootstrap,
                seed,
                index=index,
                **labels,
            )
            for index in ("1", "2", "3+")
        ]

    controls = events_for(window)
    output = dict(
        window_sensitivity=[],
        training_halves=[],
        same_episodes=[],
        jump_by_cumulative_removed=[],
        retention_by_removed=[],
    )
    for width in (8, 32, 128, 512):
        for kind, events in events_for(width).items():
            output["window_sensitivity"].extend(index_rows(events, window=width, kind=kind))
    batches = sorted({a["batch"] for a, _ in pairs})
    # A fixed checkpoint has no temporal halves.
    if pairs and pairs[0][0]["phase"] == "pre_update":
        split = (len(batches) + 1) // 2
        for label, selected in (("early", batches[:split]), ("late", batches[split:])):
            for kind, events in controls.items():
                output["training_halves"].extend(
                    index_rows(
                        [e for e in events if e["batch"] in selected],
                        half=label,
                        batches=selected,
                        kind=kind,
                    )
                )
    episodes = defaultdict(set)
    for a, _ in pairs:
        episodes[(a["batch"], a["problem_id"], a["episode_id"])].add(a["segment_index"])
    # Require actual first, second and third boundaries, not merely a large index.
    eligible = {key for key, indices in episodes.items() if {0, 1, 2, 3} <= indices}
    for kind, events in controls.items():
        selected = [e for e in events if (e["batch"], e["problem_id"], e["episode_id"]) in eligible]
        # All three boundaries must satisfy the token window for a matched attempt.
        if kind == "compaction":
            valid = defaultdict(set)
            for e in selected:
                valid[(e["batch"], e["problem_id"], e["episode_id"])].add(e["index"])
            eligible = {key for key in eligible if {1, 2, 3} <= valid[key]}
        selected = [e for e in selected if (e["batch"], e["problem_id"], e["episode_id"]) in eligible]
        output["same_episodes"].extend(index_rows(selected, kind=kind, eligible_episodes=len(eligible)))
    boundary = controls["compaction"]
    for field, count, destination, reducer, first_only in (
        ("removed_cumulative", 5, "jump_by_cumulative_removed", jump_row, False),
        ("removed_tokens", 3, "retention_by_removed", retention_row, False),
        ("removed_tokens", 3, "retention_by_removed", retention_row, True),
    ):
        ordered = sorted(
            [e for e in boundary if e.get(field) is not None and (not first_only or e["index"] == 1)],
            key=lambda e: e[field],
        )
        for index, group in enumerate(np.array_split(np.asarray(ordered, dtype=object), min(count, len(ordered)) or 1)):
            if len(group):
                output[destination].append(
                    reducer(
                        list(group),
                        bootstrap,
                        seed,
                        group=index + 1,
                        mean_removed_tokens=float(np.mean([e[field] for e in group])),
                        first_only=first_only,
                        events_in_bin=len(group),
                    )
                )
    return output


def main():
    # Reuse the paired loader and renderer, including all identity checks.
    from .figures import main as figures_main

    figures_main()


if __name__ == "__main__":
    main()
