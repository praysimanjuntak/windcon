#!/usr/bin/env python3
"""Turn adjacency rays into relative-winding point collections and check them as one graph.

Inputs: build_chains.py's chains_paths.npz (the smoothed, shifted paths and their chains) and
find_adjacency.py's adj_z<Z>.npz for the same raw paths. A pair of paths (inner p, outer q) in one
slice is accepted with at least --min-votes clean rays and no blocked ray. Up to --rays-per-pair
clean rays spread along each accepted pair become one two-point collection each: the ray's point on
p (wind_a 0) and on q (wind_a 1), at the paths' shifted positions.

Graph check: chains are nodes (a chain is one winding by construction), accepted pairs are +1 edges
from the inner to the outer chain. Winding offsets are propagated along a maximum spanning forest
(edges weighted by their clean rays); a pair whose chains do not end up exactly one winding apart,
or that joins a chain to itself, contradicts the rest of the graph. Writes into --out:
  auto_relwind.json     the accepted pairs as relative-winding collections (VC3D format; --name)
  relwind_stats.json    counts, component sizes and the contradiction rate

    uv run python build_relwind.py --chains-dir pcls --adj-dir adjacency --out pcls
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

from windcon.annotations import write_relative_winding
from windcon.relwind import BLOCKED, CLEAN, VERDICTS

CODE_CLEAN, CODE_BLOCKED = VERDICTS.index(CLEAN), VERDICTS.index(BLOCKED)


def spanning_potentials(n_nodes: int, edges: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Winding offset and component of every node from a maximum spanning forest of the +1 edges
    (u -> v: v is one winding outward of u); nodes without edges get component -1."""
    parent = np.arange(n_nodes)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree: dict[int, list[tuple[int, int]]] = {}
    for e in np.argsort(-weight, kind="stable").tolist():
        u, v = int(edges[e, 0]), int(edges[e, 1])
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            tree.setdefault(u, []).append((v, 1))
            tree.setdefault(v, []).append((u, -1))
    pot = np.zeros(n_nodes, dtype=np.int64)
    comp = np.full(n_nodes, -1, dtype=np.int64)
    n_comp = 0
    for root in tree:
        if comp[root] >= 0:
            continue
        comp[root] = n_comp
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v, w in tree[u]:
                if comp[v] < 0:
                    comp[v], pot[v] = n_comp, pot[u] + w
                    queue.append(v)
        n_comp += 1
    return pot, comp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chains-dir", required=True, type=Path)
    ap.add_argument("--adj-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--rays-per-pair", type=int, default=3)
    ap.add_argument("--outer-wind", type=float, default=1.0,
                    help="winding annotation of the outer point; 0 or 2 make control sets for agree_eval.sh")
    ap.add_argument("--name", default="auto_relwind.json")
    args = ap.parse_args()
    with np.load(args.chains_dir / "chains_paths.npz") as d:
        xy, offsets, zs = d["xy"].astype(np.float64), d["offsets"], d["zs"]
        path_slice, path_index, path_chain = d["path_slice"], d["path_index"], d["path_chain"]
    flat = {(int(s), int(i)): k for k, (s, i) in enumerate(zip(path_slice, path_index))}

    collections, edges, edge_votes = [], [], []
    tally: Counter = Counter()
    for s, z in enumerate(zs.tolist()):
        f = args.adj_dir / f"adj_z{z}.npz"
        if not f.exists():
            tally["slices without rays"] += 1
            continue
        with np.load(f) as d:
            inner, outer, iv, ov, code = (d[k] for k in ("inner", "outer", "inner_vertex", "outer_vertex", "verdict"))
        tally["rays"] += len(code)
        tally["clean rays"] += int((code == CODE_CLEAN).sum())
        blocked = set(zip(inner[code == CODE_BLOCKED].tolist(), outer[code == CODE_BLOCKED].tolist()))
        by_pair: dict[tuple[int, int], list[int]] = {}
        for r in np.flatnonzero(code == CODE_CLEAN).tolist():
            by_pair.setdefault((int(inner[r]), int(outer[r])), []).append(r)
        for (p, q), rays in by_pair.items():
            if len(rays) < args.min_votes or (p, q) in blocked:
                continue
            kp, kq = flat[(s, p)], flat[(s, q)]
            pick = np.linspace(0, len(rays) - 1, min(args.rays_per_pair, len(rays))).round().astype(int)
            # One two-point collection per ray: the fitter thins strips to points 16 voxels apart
            # but always keeps both ends, and neighbouring sheets are often closer than that.
            for j, r in enumerate(rays[i] for i in pick):
                pts = np.array([[*xy[offsets[kp] + iv[r]], float(z)], [*xy[offsets[kq] + ov[r]], float(z)]])
                collections.append((f"relwind_z{z}_{p:05d}_{q:05d}_{j}", pts, np.array([0.0, args.outer_wind])))
            edges.append((int(path_chain[kp]), int(path_chain[kq])))
            edge_votes.append(len(rays))
    edges = np.array(edges, dtype=np.int64).reshape(-1, 2)
    edge_votes = np.array(edge_votes, dtype=np.int64)
    self_loop = edges[:, 0] == edges[:, 1]
    pot, comp = spanning_potentials(int(path_chain.max()) + 1, edges[~self_loop], edge_votes[~self_loop])
    contradicted = self_loop | (pot[edges[:, 1]] - pot[edges[:, 0]] != 1)
    in_comp = comp >= 0
    n_comp = int(comp.max()) + 1 if in_comp.any() else 0
    hi = np.full(n_comp, np.iinfo(np.int64).min)
    lo = np.full(n_comp, np.iinfo(np.int64).max)
    np.maximum.at(hi, comp[in_comp], pot[in_comp])
    np.minimum.at(lo, comp[in_comp], pot[in_comp])
    sizes = np.bincount(comp[in_comp], minlength=n_comp)
    stats = {
        "params": {"min_votes": args.min_votes, "rays_per_pair": args.rays_per_pair},
        **tally,
        "accepted pairs": len(edges),
        "pairs inside one chain": int(self_loop.sum()),
        "pairs contradicting the forest": int((contradicted & ~self_loop).sum()),
        "contradiction rate": round(float(contradicted.mean()) if len(edges) else 0.0, 4),
        "components": n_comp,
        "chains in components": int(in_comp.sum()),
        "largest components (chains)": sorted(sizes.tolist(), reverse=True)[:5],
        "largest winding spans": sorted((hi - lo).tolist(), reverse=True)[:5],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_relative_winding(args.out / args.name, collections)
    if args.outer_wind == 1.0:
        (args.out / "relwind_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
