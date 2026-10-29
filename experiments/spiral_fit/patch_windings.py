#!/usr/bin/env python3
"""How far off its own winding does a fitted spiral put each part of a verified patch?

For every valid quad of each patch (inside the z window) the quad centre is mapped to spiral space
with a fit's checkpoint; its normalised shifted radius (windings) is unwrapped across theta = 0 over
the patch's quad grid (breadth-first over 4-neighbours, villa's crossing rule), and compared with
the patch median. A quad more than 0.45 winding off fails the fitter's radius test. Off-quads near
a whole winding (|offset| ≈ 1) mean the fit puts that part of a continuous sheet on the next
winding: a local winding-count mismatch between the patch and the fitted spiral. Off-quads well
short of a winding mean drift.

Run with the spiral-fitting environment (imports villa's fit_spiral):
    cd $POSTEL/villa/spiral-fitting && .venv/bin/python patch_windings.py --checkpoint runs/X/checkpoint_fitted.ckpt \
        --dataset data/PHercParis4 --umbilicus data/PHercParis4/umbilicus.json --z 9900 10900 \
        --set orac=verified_patches_few10_orac471 --minus verified_patches_few10 --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

SPIRAL = Path(os.environ.get("POSTEL", Path.home() / "Documents/postel/vesuvius_challenge")) / "villa" / "spiral-fitting"
sys.path.insert(0, str(SPIRAL))
import fit_spiral as fs  # noqa: E402
from checkpoint_io import load_checkpoint_cpu  # noqa: E402
from find_inconsistent_windings import build_transform  # noqa: E402
from sample_spiral import get_theta_and_radii  # noqa: E402


def load_patches(ckpt: dict, patches_dir: Path, umbilicus: Path, shell: Path, z0: int, z1: int):
    from fit_session import ScrollSpec, SpiralInputPaths

    cfg = fs.Config().as_dict()
    cfg.update(ckpt["cfg"])
    cfg.update({"input_disable_patches": False, "input_use_verified_patches": True,
                "loss_weight_shell_outer": 0.0, "z_begin": z0, "z_end": z1})
    fit_config = fs.FitConfig(cfg)
    scroll = ScrollSpec(name="analysis", voxel_size_um=9.6,
                        spiral_outward_sense=ckpt.get("spiral_outward_sense") or "CW")
    paths = SpiralInputPaths(umbilicus=str(umbilicus), pcls=(), fibers="", verified_patches=str(patches_dir),
                             outer_shell=str(shell))  # the checkpoint's settings may require it
    context = fs.FitContext(fit_config, scroll=scroll, paths=paths)
    context.load_host_inputs()
    return fit_config, context


def quad_offsets(patch, transform, dr: torch.Tensor, z0: int, z1: int):
    """Per valid in-window quad: offset (windings) from its connected component's median winding,
    offset from that median snapped to the nearest whole winding (villa's target: the fitted sheet),
    and the quad centre (x, y, z), raw winding coordinate and theta for cross-patch comparisons."""
    zyxs = patch.zyxs.float()
    centres = (zyxs[:-1, :-1] + zyxs[1:, :-1] + zyxs[:-1, 1:] + zyxs[1:, 1:]) / 4
    qz = torch.stack([zyxs[:-1, :-1, 0], zyxs[1:, :-1, 0], zyxs[:-1, 1:, 0], zyxs[1:, 1:, 0]])
    mask = patch.valid_quad_mask.cpu() & (qz.amax(0) >= z0) & (qz.amin(0) < z1)
    ii, jj = (t.numpy() for t in torch.where(mask))
    if len(ii) < 2:
        return np.zeros(0), np.zeros(0), np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    pts = centres[ii, jj].to(dr.device)
    spiral = torch.cat([transform(pts[k:k + 65536]) for k in range(0, len(pts), 65536)])
    theta, _, shifted = get_theta_and_radii(spiral[..., 1:], dr)
    theta = theta.double().cpu().numpy()
    s = (shifted / dr).double().cpu().numpy()
    index = -np.ones(mask.shape, dtype=np.int64)
    index[ii, jj] = np.arange(len(ii))
    edges = []
    for di, dj in ((1, 0), (0, 1)):  # 4-neighbour edges between valid quads
        a = index[: index.shape[0] - di, : index.shape[1] - dj]
        b = index[di:, dj:]
        ok = (a >= 0) & (b >= 0)
        edges.append(np.stack([a[ok], b[ok]], axis=1))
    e = np.concatenate(edges)
    n = len(ii)
    graph = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    n_comp, comp = connected_components(graph, directed=False)
    u = np.full(n, np.nan)
    target = np.full(n, np.nan)
    for c in range(n_comp):
        members = np.flatnonzero(comp == c)
        root = members[len(members) // 2]
        order, pred = breadth_first_order(graph, root, directed=False, return_predecessors=True)
        u[root] = s[root]
        for v in order[1:]:
            p = pred[v]
            dtheta = theta[v] - theta[p]
            step = float(dtheta > np.pi) - float(dtheta < -np.pi)  # villa's theta = 0 crossing rule
            u[v] = u[p] + (s[v] - s[p]) + step
        median = np.median(u[members])
        target[members] = u[members] - np.round(median)
        u[members] -= median
    xyz = centres[ii, jj].numpy()[:, ::-1].copy()  # zyx -> xyz
    return u, target, xyz, s, theta


def summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    off = np.concatenate([r["_off"] for r in rows])
    tgt = np.concatenate([r["_target"] for r in rows])
    big = off[np.abs(off) > 0.45]
    phase = np.array([r["phase"] for r in rows])
    out = {
        "patches": len(rows),
        "quads": int(len(off)),
        "quads_off_own_median_fraction": round(float(np.mean(np.abs(off) > 0.45)), 4),
        "quads_off_fitted_sheet_fraction": round(float(np.mean(np.abs(tgt) > 0.45)), 4),
        "patch_phase_abs_p25_50_75": [round(float(v), 3) for v in np.percentile(np.abs(phase), [25, 50, 75])],
        "patches_between_sheets": round(float(np.mean(np.abs(phase) > 0.35)), 4),
        "patches_spanning_a_winding": round(float(np.mean([r["spread"] >= 0.9 for r in rows])), 4),
        "median_spread_windings": round(float(np.median([r["spread"] for r in rows])), 3),
        "off_quads_near_whole_winding": round(float(np.mean(np.abs(np.abs(big) - 1.0) < 0.25)), 4) if len(big) else None,
    }
    sat = [r["villa_satisfied"] for r in rows if r.get("villa_satisfied") is not None]
    if sat:
        sat = np.array(sat)
        ph = np.array([abs(r["phase"]) for r in rows if r.get("villa_satisfied") is not None])
        out["villa_satisfied_mean"] = round(float(sat.mean()), 4)
        out["villa_satisfied_mean_phase_under_0.2"] = round(float(sat[ph < 0.2].mean()), 4) if (ph < 0.2).any() else None
        out["villa_satisfied_mean_phase_over_0.35"] = round(float(sat[ph > 0.35].mean()), 4) if (ph > 0.35).any() else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path, help="folder with verified_patches/ and the set folders")
    ap.add_argument("--umbilicus", required=True, type=Path)
    ap.add_argument("--outer-shell", required=True, type=Path, help="the dataset's outer_shell tifxyz folder")
    ap.add_argument("--z", type=int, nargs=2, default=(9900, 10900))
    ap.add_argument("--set", action="append", default=[], help="label=folder of patch names (in --dataset)")
    ap.add_argument("--minus", help="folder (in --dataset) whose patches are removed from every set")
    ap.add_argument("--satisfied", type=Path, help="the fit's satisfied_fitted.json, to compare with villa's test")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dump-quads", type=Path, help="npz of every quad: xyz, patch index, winding coordinate, "
                                                    "theta, offset from the patch's sheet (for phase_conflicts.py)")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    minus = {p.name for p in (args.dataset / args.minus).iterdir()} if args.minus else set()
    sets = {}
    for spec in args.set:
        label, folder = spec.split("=", 1)
        sets[label] = sorted({p.name for p in (args.dataset / folder).iterdir()} - minus)
    wanted = sorted(set().union(*sets.values()))
    ckpt = load_checkpoint_cpu(str(args.checkpoint))
    with tempfile.TemporaryDirectory() as tmp:  # load only the patches of interest
        for name in wanted:
            (Path(tmp) / name).symlink_to((args.dataset / "verified_patches" / name).resolve())
        fit_config, context = load_patches(ckpt, Path(tmp), args.umbilicus, args.outer_shell, *args.z)
        patches = context.verified_patches
        transform, dr = build_transform(ckpt, fit_config, context, int(ckpt["z_begin"]), int(ckpt["z_end"]))
        villa = {}
        if args.satisfied:
            villa = {e["id"]: e["fraction"] for e in json.loads(args.satisfied.read_text())["patches"] if e["total_area"] > 0}
        per_patch = {}
        dump = {"xyz": [], "patch": [], "s": [], "theta": [], "target": []}
        names_order = []
        for name in wanted:
            if name not in patches:
                continue
            u, target, xyz, s, theta = quad_offsets(patches[name], transform, dr, *args.z)
            if len(u) < 4:
                continue
            per_patch[name] = {"quads": int(len(u)), "spread": float(np.percentile(u, 98) - np.percentile(u, 2)),
                               "phase": float(np.median(target - u)),  # median minus its nearest whole winding
                               "off_fraction": float(np.mean(np.abs(u) > 0.45)),
                               "off_sheet_fraction": float(np.mean(np.abs(target) > 0.45)),
                               "villa_satisfied": villa.get(name), "_off": u, "_target": target}
            if args.dump_quads:
                dump["xyz"].append(xyz.astype(np.float32))
                dump["patch"].append(np.full(len(u), len(names_order), dtype=np.int32))
                dump["s"].append(s.astype(np.float32))
                dump["theta"].append(theta.astype(np.float32))
                dump["target"].append(target.astype(np.float32))
                names_order.append(name)
    if args.dump_quads:
        np.savez_compressed(args.dump_quads, names=np.array(names_order), **{k: np.concatenate(v) for k, v in dump.items()})
    report = {"checkpoint": str(args.checkpoint), "dr_per_winding": float(dr.item()), "sets": {}}
    for label, names in sets.items():
        report["sets"][label] = summary([per_patch[n] for n in names if n in per_patch])
        print(label, json.dumps(report["sets"][label]))
    report["per_patch"] = {n: {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()
                               if not k.startswith("_")} for n, r in per_patch.items()}
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
