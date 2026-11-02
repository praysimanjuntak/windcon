#!/usr/bin/env python3
"""Figure: two fits' sheets in one CT slice, over the human patches (does a fit jump across sheets?).

A fitted spiral puts its sheets where the winding coordinate s (patch_windings.py) is a whole
number, so in one XY slice those level sets are the fit's sheets in cross-section. Each panel shows
the CT, one fit's sheets and the verified patches cut by the slice (vertices within 1.5 slices):
patches the first fit places between its sheets (|phase| > 0.35 in its patch_windings.py --out
report) in orange, the others in yellow. Without --z/--box the crop is the --size square of the
20-slice band (at least --edge slices inside --window) holding the most patches that the first fit
places between sheets and the second on one (|phase| < 0.2).

Run with the spiral-fitting environment (imports villa's fit_spiral):
    cd $POSTEL/villa/spiral-fitting && .venv/bin/python plot_fit_sheets.py \
        --fit default runs/fullnodt_A_s1/checkpoint_fitted.ckpt pw_all_fullnodt_A_s1.json \
        --fit "relative windings" runs/fullnodt_ARW_s1/checkpoint_fitted.ckpt pw_all_fullnodt_ARW_s1.json \
        --dataset data/PHercParis4 --out fit_sheets.png
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import patch_windings as pw  # noqa: E402  (puts villa's spiral-fitting on sys.path)
from patch_surfaces import patch_vertices  # noqa: E402
from windcon.cli import PARIS4_L2  # noqa: E402
from windcon.render import VMAX, VMIN, plt  # noqa: E402
from windcon.zarr_io import ZarrArray  # noqa: E402

BETWEEN, ON = 0.35, 0.2  # |phase| thresholds of patch_windings.py's report
COLOURS = (("red", "#ff3b30"), ("blue", "#2f9bff"))


def phases(report: Path) -> dict[str, float]:
    return {n: r["phase"] for n, r in json.loads(report.read_text())["per_patch"].items()}


def pick_crop(dataset: Path, first: dict, second: dict, window: tuple[int, int], edge: int, size: int):
    """Slice (every 10th) and box crossed by the most patches that the first fit puts between sheets
    and the second on one."""
    moved = [n for n, p in first.items() if abs(p) > BETWEEN and abs(second.get(n, 1.0)) < ON]
    lo, hi = window[0] + edge, window[1] - edge
    rows = []  # candidate slice / 10, cell x, cell y, patch index, x, y
    for i, name in enumerate(moved):
        found = patch_vertices(dataset / "verified_patches" / name, lo, hi)
        if found is None or not len(found[0]):
            continue
        v = found[0][::4]
        k = np.round(v[:, 2] / 10)
        near = np.abs(v[:, 2] - 10 * k) <= 1.5  # what slice_traces draws for slice 10 k
        v, k = v[near], k[near]
        rows.append(np.column_stack([k, v[:, 0] // size, v[:, 1] // size, np.full(len(v), i), v[:, 0], v[:, 1]]))
    rows = np.concatenate(rows)
    cells, counts = np.unique(np.unique(rows[:, :4], axis=0)[:, :3], axis=0, return_counts=True)
    best = cells[np.argmax(counts)]
    sel = (rows[:, :3] == best).all(axis=1)
    cx, cy = rows[sel, 4].mean(), rows[sel, 5].mean()
    z = int(10 * best[0])
    return z, (int(cx - size / 2), int(cx + size / 2), int(cy - size / 2), int(cy + size / 2)), len(moved)


def slice_traces(dataset: Path, z: int, box: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    x0, x1, y0, y1 = box
    traces = {}
    for d in sorted((dataset / "verified_patches").iterdir()):
        found = patch_vertices(d, z - 1.5, z + 1.5)
        if found is None:
            continue
        v = found[0]
        inside = (v[:, 0] >= x0) & (v[:, 0] < x1) & (v[:, 1] >= y0) & (v[:, 1] < y1)
        if inside.sum() >= 5:
            traces[d.name] = v[inside, :2]
    return traces


def load_fit(checkpoint: Path, dataset: Path, patch: str):
    """A fit's checkpoint as villa's spiral transform (zyx -> spiral space) and its winding pitch."""
    ckpt = pw.load_checkpoint_cpu(str(checkpoint))
    z0, z1 = int(ckpt["z_begin"]), int(ckpt["z_end"])
    with tempfile.TemporaryDirectory() as tmp:  # build_transform needs a loaded context; one patch will do
        (Path(tmp) / patch).symlink_to((dataset / "verified_patches" / patch).resolve())
        cfg, context = pw.load_patches(ckpt, Path(tmp), dataset / "umbilicus.json", dataset / "outer_shell", z0, z1)
        return pw.build_transform(ckpt, cfg, context, z0, z1)


def winding_at(transform, dr, zyx: np.ndarray) -> np.ndarray:
    """The fit's winding coordinate s at (N, 3) zyx points (whole numbers lie on its sheets)."""
    out = []
    with torch.no_grad():
        for k in range(0, len(zyx), 65536):
            pts = torch.tensor(np.ascontiguousarray(zyx[k:k + 65536]), dtype=torch.float32, device=dr.device)
            _, _, shifted = pw.get_theta_and_radii(transform(pts)[..., 1:], dr)
            out.append((shifted / dr).detach().double().cpu().numpy())
    return np.concatenate(out)


def winding_grid(checkpoint: Path, dataset: Path, patch: str, z: int, box: tuple[int, int, int, int], step: int):
    """The fit's winding coordinate s on a grid over the box in slice z (zero-step evaluation)."""
    transform, dr = load_fit(checkpoint, dataset, patch)
    x0, x1, y0, y1 = box
    ys, xs = np.mgrid[y0:y1:step, x0:x1:step]
    s = winding_at(transform, dr, np.stack([np.full(xs.size, z), ys.ravel(), xs.ravel()], axis=1)).reshape(xs.shape)
    del transform
    torch.cuda.empty_cache()
    return xs, ys, s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", nargs=3, action="append", required=True, metavar=("LABEL", "CHECKPOINT", "PHASES"),
                    help="label, checkpoint_fitted.ckpt and its patch_windings.py --out report (give two)")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--window", type=int, nargs=2, default=(9900, 10900), help="z range searched for the crop")
    ap.add_argument("--edge", type=int, default=50, help="keep the crop this many slices inside --window")
    ap.add_argument("--z", type=int)
    ap.add_argument("--box", type=int, nargs=4, metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--size", type=int, default=240, help="crop size (px) when choosing it automatically")
    ap.add_argument("--step", type=int, default=1, help="winding-grid spacing (px)")
    ap.add_argument("--volume", default=PARIS4_L2)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if len(args.fit) != 2:
        raise SystemExit("give exactly two --fit")
    D = args.dataset
    (label_a, _, rep_a), (label_b, _, rep_b) = args.fit
    phase_a, phase_b = phases(Path(rep_a)), phases(Path(rep_b))

    if args.z is None or args.box is None:
        z, box, n_moved = pick_crop(D, phase_a, phase_b, tuple(args.window), args.edge, args.size)
        print(f"{n_moved} patches between sheets in {label_a} and on one in {label_b}; crop z {z}, box {box}")
    else:
        z, box = args.z, tuple(args.box)
    traces = slice_traces(D, z, box)
    if not traces:
        raise SystemExit("no verified patch crosses the crop")
    patch = max(traces, key=lambda n: len(traces[n]))
    grids = [(label, *winding_grid(Path(ckpt), D, patch, z, box, args.step), ph)
             for (label, ckpt, _), ph in zip(args.fit, (phase_a, phase_b))]

    x0, x1, y0, y1 = box
    img = ZarrArray(args.volume).read(((z, z + 1), (y0, y1), (x0, x1)))[0]
    between_a = {n for n in traces if abs(phase_a.get(n, 0.0)) > BETWEEN}
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6), dpi=110, layout="constrained")
    for ax, (label, xs, ys, s, ph), (name, colour) in zip(axes, grids, COLOURS):
        ax.imshow(img, cmap="gray", vmin=VMIN, vmax=VMAX, extent=(x0, x1, y1, y0))
        # Whole-number level sets of s; sin/cos keep the 1-winding step at theta = 0 from drawing a line.
        f = np.ma.masked_where(np.cos(2 * np.pi * s) < 0, np.sin(2 * np.pi * s))
        ax.contour(xs, ys, f, levels=[0], colors=[colour], linewidths=1.6)
        for pname, v in traces.items():
            ax.plot(v[:, 0], v[:, 1], "o", ms=3.2, mew=0.4, mec="black",
                    color="orange" if pname in between_a else "yellow")
        n_between = sum(abs(ph.get(n, 0.0)) > BETWEEN for n in traces)
        # Sheet spacing: px per winding, from the gradient of s (median skips the theta = 0 step).
        pitch = 1.0 / np.median(np.hypot(*np.gradient(s, args.step)))
        print(f"{label}: sheets {pitch:.1f} px apart in the crop; {n_between} of {len(traces)} patches between them")
        ax.set_title(f"{label}: its sheets in {name}, {pitch:.0f} px apart; {n_between} of the {len(traces)} "
                     f"patches here lie between them", fontsize=10)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_axis_off()
    fig.suptitle(f"slice z {z}, x {x0}–{x1}, y {y0}–{y1} (9.6 µm voxels): verified patches as dots, orange where "
                 f"the {label_a} fit puts the patch between sheets", fontsize=11)
    fig.savefig(args.out)
    print(f"{len(traces)} patches in the crop ({len(between_a)} between sheets in {label_a}) -> {args.out}")


if __name__ == "__main__":
    main()
