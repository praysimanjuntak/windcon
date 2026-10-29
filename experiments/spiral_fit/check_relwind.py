#!/usr/bin/env python3
"""Precision of the automatic adjacency against human relative-winding annotations.

Consecutive points of a human relative-winding collection drawn in one slice lie on sheets whose
winding numbers differ by the annotated amount (in this window always one: neighbouring sheets).
Each human point is matched to the traced path passing within --match px; the automatic answer for
the pair is then the signed winding difference between the two paths through the slice's accepted
adjacency pairs (direct pair, else a breadth-first walk of up to --max-steps pairs), or "blocked"
when rays between the two paths saw an untraced sheet. Needs the slices traced (trace_window.py)
and their rays cast (find_adjacency.py).

    uv run python check_relwind.py --relwind data/PHercParis4/relative_windings.json \
        --raw-dir auto_paths_hrw --adj-dir adjacency_hrw
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from windcon.relwind import BLOCKED, VERDICTS


def slice_graph(adj: Path, min_votes: int) -> tuple[dict[int, list[tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Signed neighbour lists of the accepted pairs, the accepted pairs and the blocked pairs."""
    with np.load(adj) as d:
        inner, outer, verdict = d["inner"], d["outer"], d["verdict"]
    clean = Counter(zip(inner[verdict == 0].tolist(), outer[verdict == 0].tolist()))
    blocked = set(zip(inner[verdict == VERDICTS.index(BLOCKED)].tolist(),
                      outer[verdict == VERDICTS.index(BLOCKED)].tolist()))
    accepted = {k for k, n in clean.items() if n >= min_votes and k not in blocked}
    graph: dict[int, list[tuple[int, int]]] = {}
    for p, q in accepted:
        graph.setdefault(p, []).append((q, 1))
        graph.setdefault(q, []).append((p, -1))
    return graph, accepted, blocked


def walk(graph: dict[int, list[tuple[int, int]]], a: int, b: int, max_steps: int) -> int | None:
    """Signed winding difference b - a along the accepted pairs, None if not reachable."""
    seen = {a: 0}
    queue = deque([(a, 0)])
    while queue:
        node, depth = queue.popleft()
        if node == b:
            return seen[node]
        if depth == max_steps:
            continue
        for nxt, w in graph.get(node, []):
            if nxt not in seen:
                seen[nxt] = seen[node] + w
                queue.append((nxt, depth + 1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--relwind", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--adj-dir", required=True, type=Path)
    ap.add_argument("--match", type=float, default=4.0)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    pairs_by_z: dict[int, list[tuple[np.ndarray, np.ndarray, float]]] = {}
    for c in json.loads(args.relwind.read_text())["collections"].values():
        pts = sorted(((int(k), p) for k, p in c["points"].items() if p.get("wind_a") is not None),
                     key=lambda kv: kv[0])
        for (_, a), (_, b) in zip(pts[:-1], pts[1:]):
            if round(a["p"][2]) == round(b["p"][2]):
                pairs_by_z.setdefault(round(a["p"][2]), []).append(
                    (np.array(a["p"][:2]), np.array(b["p"][:2]), b["wind_a"] - a["wind_a"]))

    tally: Counter = Counter()
    errors = []
    for z, pairs in sorted(pairs_by_z.items()):
        raw, adj = args.raw_dir / f"auto_z{z}.npz", args.adj_dir / f"adj_z{z}.npz"
        if not (raw.exists() and adj.exists()):
            continue
        with np.load(raw) as d:
            xy, offsets = d["xy"].astype(np.float64), d["offsets"]
        owner = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
        tree = cKDTree(xy)
        graph, accepted, blocked = slice_graph(adj, args.min_votes)
        for a, b, delta in pairs:
            tally["human pairs"] += 1
            da, ia = tree.query(a)
            db, ib = tree.query(b)
            if da > args.match or db > args.match:
                tally["unmatched"] += 1
                continue
            pa, pb = int(owner[ia]), int(owner[ib])
            if pa == pb:
                tally["same path (wrong)"] += 1
                errors.append({"z": z, "a": a.tolist(), "b": b.tolist(), "human": delta, "auto": 0})
                continue
            if (pa, pb) in blocked or (pb, pa) in blocked:
                tally["blocked"] += 1
            got = walk(graph, pa, pb, args.max_steps)
            if got is None:
                tally["not connected"] += 1
                continue
            direct = (pa, pb) in accepted or (pb, pa) in accepted
            key = "direct" if direct else "via walk"
            tally[f"{key}: correct" if got == delta else f"{key}: wrong"] += 1
            if got != delta:
                errors.append({"z": z, "a": a.tolist(), "b": b.tolist(), "human": delta, "auto": got, "direct": direct})
    report = {"tally": dict(tally), "errors": errors}
    print(json.dumps(report["tally"], indent=1))
    for e in errors:
        print("wrong:", e)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
