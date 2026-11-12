#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path


MAX_MOTIF_LEN = 10
MIN_MOTIF_LEN = 2
MAX_MOTIF_TOTAL_LEN = 200000
MAX_WEIGHT = 10**6


@dataclass(frozen=True)
class CaseSpec:
    case_id: int
    seed: int
    n: int
    k: int
    m: int
    min_gap: int


CASE_SPECS = [
    CaseSpec(case_id=1, seed=2001, n=24, k=3, m=8, min_gap=10),
    CaseSpec(case_id=2, seed=2002, n=80, k=4, m=30, min_gap=50),
    CaseSpec(case_id=3, seed=2003, n=300, k=5, m=120, min_gap=500),
    CaseSpec(case_id=4, seed=2004, n=2000, k=6, m=800, min_gap=1000000),
    CaseSpec(case_id=5, seed=2005, n=12000, k=8, m=4000, min_gap=20000),
    CaseSpec(case_id=6, seed=2006, n=50000, k=10, m=12000, min_gap=40000),
    CaseSpec(case_id=7, seed=2007, n=100000, k=12, m=25000, min_gap=80000),
    CaseSpec(case_id=8, seed=2008, n=200000, k=2, m=1500, min_gap=200000),
    CaseSpec(case_id=9, seed=2009, n=200000, k=26, m=20000, min_gap=120000),
    CaseSpec(case_id=10, seed=2010, n=50000, k=6, m=10000, min_gap=50000),
]


def alphabet(k: int) -> str:
    return "".join(chr(ord("a") + i) for i in range(k))


def positive_composition(n: int, k: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(1, n), k - 1))
    values = []
    prev = 0
    for cut in cuts + [n]:
        values.append(cut - prev)
        prev = cut
    return values


def motif_capacity(k: int) -> int:
    return sum(k**length for length in range(MIN_MOTIF_LEN, MAX_MOTIF_LEN + 1))


def enumerate_all_motifs(chars: str) -> list[str]:
    items: list[str] = []
    for length in range(MIN_MOTIF_LEN, MAX_MOTIF_LEN + 1):
        items.extend("".join(p) for p in itertools.product(chars, repeat=length))
    return items


def sample_unique_motifs(k: int, m: int, rng: random.Random) -> list[str]:
    chars = alphabet(k)
    capacity = motif_capacity(k)
    if m > capacity:
        raise ValueError(f"Cannot sample {m} unique motifs with k={k}")

    if capacity <= 100000 and m > capacity // 4:
        pool = enumerate_all_motifs(chars)
        rng.shuffle(pool)
        return pool[:m]

    seen: set[str] = set()
    motifs: list[str] = []
    while len(motifs) < m:
        length = rng.randint(MIN_MOTIF_LEN, MAX_MOTIF_LEN)
        s = "".join(rng.choice(chars) for _ in range(length))
        if s in seen:
            continue
        seen.add(s)
        motifs.append(s)
    return motifs


class AhoCorasick:
    def __init__(self) -> None:
        self.next: list[dict[str, int]] = [{}]
        self.link: list[int] = [0]
        self.out: list[int] = [0]

    def add(self, s: str, weight: int) -> None:
        v = 0
        for ch in s:
            nxt = self.next[v].get(ch)
            if nxt is None:
                nxt = len(self.next)
                self.next[v][ch] = nxt
                self.next.append({})
                self.link.append(0)
                self.out.append(0)
            v = nxt
        self.out[v] += weight

    def build(self) -> None:
        q: deque[int] = deque()
        for nxt in self.next[0].values():
            q.append(nxt)
            self.link[nxt] = 0

        while q:
            v = q.popleft()
            self.out[v] += self.out[self.link[v]]
            for ch, nxt in self.next[v].items():
                q.append(nxt)
                u = self.link[v]
                while u and ch not in self.next[u]:
                    u = self.link[u]
                self.link[nxt] = self.next[u].get(ch, 0)

    def score(self, s: str) -> int:
        v = 0
        total = 0
        for ch in s:
            while v and ch not in self.next[v]:
                v = self.link[v]
            v = self.next[v].get(ch, 0)
            total += self.out[v]
        return total


def compute_penalty(s: str, k: int, weights: list[list[int]], ac: AhoCorasick) -> int:
    total = 0
    for i in range(len(s) - 1):
        a = ord(s[i]) - ord("a")
        b = ord(s[i + 1]) - ord("a")
        if not (0 <= a < k and 0 <= b < k):
            raise ValueError("String uses characters outside alphabet")
        total += weights[a][b]
    total += ac.score(s)
    return total


def build_baseline(counts: list[int]) -> str:
    return "".join(chr(ord("a") + i) * counts[i] for i in range(len(counts)))


def build_reversed(counts: list[int]) -> str:
    return "".join(chr(ord("a") + i) * counts[i] for i in range(len(counts) - 1, -1, -1))


def build_greedy_bigram(n: int, k: int, counts0: list[int], weights: list[list[int]]) -> str:
    counts = counts0[:]
    current = next(i for i, c in enumerate(counts) if c > 0)
    out = [chr(ord("a") + current)]
    counts[current] -= 1

    for _ in range(1, n):
        best = -1
        best_w = None
        for nxt in range(k):
            if counts[nxt] <= 0:
                continue
            val = weights[current][nxt]
            if best_w is None or val < best_w or (val == best_w and nxt < best):
                best_w = val
                best = nxt
        if best == -1:
            break
        out.append(chr(ord("a") + best))
        counts[best] -= 1
        current = best
    return "".join(out)


def generate_case(spec: CaseSpec) -> tuple[str, int, int]:
    for attempt in range(1, 1001):
        rng = random.Random(spec.seed * 10007 + attempt)
        counts = positive_composition(spec.n, spec.k, rng)
        weights = [
            [rng.randint(0, MAX_WEIGHT) for _ in range(spec.k)]
            for _ in range(spec.k)
        ]

        motifs = sample_unique_motifs(spec.k, spec.m, rng)
        motif_weights = [rng.randint(1, MAX_WEIGHT) for _ in range(spec.m)]
        total_motif_len = sum(len(s) for s in motifs)
        if total_motif_len > MAX_MOTIF_TOTAL_LEN:
            continue

        ac = AhoCorasick()
        for motif, weight in zip(motifs, motif_weights):
            ac.add(motif, weight)
        ac.build()

        baseline = build_baseline(counts)
        reversed_blocks = build_reversed(counts)
        greedy = build_greedy_bigram(spec.n, spec.k, counts, weights)

        b = compute_penalty(baseline, spec.k, weights, ac)
        r = min(
            b,
            compute_penalty(reversed_blocks, spec.k, weights, ac),
            compute_penalty(greedy, spec.k, weights, ac),
        )

        if b - r < spec.min_gap:
            continue

        lines = [f"{spec.n} {spec.k}"]
        lines.append(" ".join(map(str, counts)))
        lines.extend(" ".join(map(str, row)) for row in weights)
        lines.append(str(spec.m))
        lines.extend(f"{motif} {weight}" for motif, weight in zip(motifs, motif_weights))
        text = "\n".join(lines) + "\n"
        return text, b, r

    raise RuntimeError(f"Failed to generate case {spec.case_id} after many attempts")


def write_cases(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in CASE_SPECS:
        text, b, r = generate_case(spec)
        in_path = output_dir / f"{spec.case_id}.in"
        ans_path = output_dir / f"{spec.case_id}.ans"
        in_path.write_text(text)
        ans_path.write_text("")
        print(
            f"generated {in_path.name}: n={spec.n} k={spec.k} m={spec.m} gap={b-r} baseline={b} reference={r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate public testdata for synthetic problem 2.")
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent / "testdata",
        type=Path,
        help="Directory where .in/.ans files will be written",
    )
    args = parser.parse_args()
    write_cases(args.output_dir)


if __name__ == "__main__":
    main()
