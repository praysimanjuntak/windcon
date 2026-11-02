#!/usr/bin/env python3
"""Figure: the CT on a verified patch's surface and on each fit's nearest sheet, in the patch's grid.

For one verified patch, the region of its tifxyz grid around a slice is upsampled to one sample per
voxel. From every sample a fit's nearest sheet is found along the surface normal (the closest point
within ±16 voxels where the fit's winding coordinate is a whole number), and the CT is sampled
there. Three images on the same grid: the patch surface itself (human), the first fit's nearest
sheet and the second fit's. Where a fit's sheet lies on the patch its image matches the patch's;
where the fit is off by half a winding its image shows the gap or the neighbouring sheet.
Without --patch, the crop of plot_fit_sheets.py is used and its patch with the most points that the
first fit puts between sheets and the second on one.

Run with the spiral-fitting environment (imports villa's fit_spiral):
    cd $POSTEL/villa/spiral-fitting && .venv/bin/python plot_fit_surface.py \
        --fit default runs/fullnodt_A_s1/checkpoint_fitted.ckpt pw_all_fullnodt_A_s1.json \
        --fit "relative windings" runs/fullnodt_ARW_s1/checkpoint_fitted.ckpt pw_all_fullnodt_ARW_s1.json \
        --dataset data/PHercParis4 --out fit_surface.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.ndimage import map_coordinates

from plot_fit_sheets import BETWEEN, ON, PARIS4_L2, VMAX, VMIN, ZarrArray, load_fit, phases, pick_crop, plt, \
    slice_traces, winding_at
from patch_surfaces import imread

OFFSETS = np.arange(-16.0, 16.01, 0.5)  # searched along the normal, voxels


def patch_region(patch_dir: Path, centre: np.ndarray, half: float):
    """The patch's grid cells within `half` voxels of `centre` (xyz), upsampled to one sample per voxel:
    (H, W, 3) points, (H, W, 3) unit normals and the (H, W) valid mask."""
    P = np.stack([imread(patch_dir / f"{c}.tif") for c in "xyz"], axis=-1).astype(np.float64)
    valid = (P != -1).all(axis=-1)
    if (patch_dir / "mask.tif").exists():
        m = imread(patch_dir / "mask.tif")
        valid &= (m[..., 0] if m.ndim == 3 else m) != 0
    near = valid & (np.abs(P - centre) < half).all(axis=-1)
    ii, jj = np.nonzero(near)
    i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
    P, valid = P[i0:i1, j0:j1], valid[i0:i1, j0:j1]
    step = np.median(np.linalg.norm(np.diff(P, axis=1), axis=-1)[valid[:, 1:] & valid[:, :-1]])
    f = max(int(round(step)), 1)
    gi, gj = np.mgrid[0:P.shape[0] - 1:1 / f, 0:P.shape[1] - 1:1 / f]
    fine = np.stack([map_coordinates(P[..., c], [gi, gj], order=1) for c in range(3)], axis=-1)
    ok = map_coordinates(valid.astype(np.float64), [gi, gj], order=1) > 0.999  # all four corners valid
    du = np.gradient(fine, axis=1)
    dv = np.gradient(fine, axis=0)
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    ok &= norm[..., 0] > 0
    return fine, n / np.where(norm > 0, norm, 1.0), ok


def nearest_sheet(transform, dr, pts: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Offset along the normal (voxels) to the nearest whole-number level of the fit's winding
    coordinate, NaN where none lies within the searched range."""
    zyx = (pts[:, None, :] + OFFSETS[None, :, None] * normals[:, None, :])[..., ::-1].reshape(-1, 3)
    s = winding_at(transform, dr, zyx).reshape(len(pts), len(OFFSETS))
    g, c = np.sin(2 * np.pi * s), np.cos(2 * np.pi * s)
    cross = (g[:, :-1] * g[:, 1:] <= 0) & (c[:, :-1] > 0) & (c[:, 1:] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = OFFSETS[:-1] + (OFFSETS[1] - OFFSETS[0]) * g[:, :-1] / (g[:, :-1] - g[:, 1:])
    t = np.where(cross, t, np.inf)
    best = np.take_along_axis(t, np.argmin(np.abs(t), axis=1)[:, None], axis=1)[:, 0]
    return np.where(np.isfinite(best), best, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", nargs=3, action="append", required=True, metavar=("LABEL", "CHECKPOINT", "PHASES"),
                    help="label, checkpoint_fitted.ckpt and its patch_windings.py --out report (give two)")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--window", type=int, nargs=2, default=(9900, 10900))
    ap.add_argument("--edge", type=int, default=50)
    ap.add_argument("--size", type=int, default=240, help="crop size of plot_fit_sheets.py; also the region size")
    ap.add_argument("--patch", help="verified patch name (default: chosen as described above)")
    ap.add_argument("--centre", type=float, nargs=3, metavar=("X", "Y", "Z"), help="region centre (with --patch)")
    ap.add_argument("--volume", default=PARIS4_L2)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if len(args.fit) != 2:
        raise SystemExit("give exactly two --fit")
    D = args.dataset
    (label_a, _, rep_a), (label_b, _, rep_b) = args.fit
    phase_a, phase_b = phases(Path(rep_a)), phases(Path(rep_b))

    if args.patch:
        name, centre = args.patch, np.array(args.centre)
    else:
        z, box, _ = pick_crop(D, phase_a, phase_b, tuple(args.window), args.edge, args.size)
        traces = slice_traces(D, z, box)
        moved = {n: v for n, v in traces.items()
                 if abs(phase_a.get(n, 0.0)) > BETWEEN and abs(phase_b.get(n, 1.0)) < ON}
        if not moved:
            raise SystemExit("no patch in the crop is between the first fit's sheets and on the second's")
        name = max(moved, key=lambda n: len(moved[n]))
        centre = np.array([moved[name][:, 0].mean(), moved[name][:, 1].mean(), z])
    pts, normals, ok = patch_region(D / "verified_patches" / name, centre, args.size / 2)
    flat_pts, flat_n = pts[ok], normals[ok]
    print(f"patch {name}: region {ok.shape[1]} x {ok.shape[0]} samples around {np.round(centre).astype(int).tolist()}")

    offsets = []
    for _, ckpt, _ in args.fit:
        transform, dr = load_fit(Path(ckpt), D, name)
        offsets.append(nearest_sheet(transform, dr, flat_pts, flat_n))
        del transform
        torch.cuda.empty_cache()

    samples = [flat_pts] + [flat_pts + np.nan_to_num(t)[:, None] * flat_n for t in offsets]
    allp = np.concatenate(samples)
    lo = np.floor(allp.min(axis=0)).astype(int) - 2
    hi = np.ceil(allp.max(axis=0)).astype(int) + 3
    vol = ZarrArray(args.volume).read(((lo[2], hi[2]), (lo[1], hi[1]), (lo[0], hi[0]))).astype(np.float32)
    images = []
    for k, p in enumerate(samples):
        img = np.full(ok.shape, np.nan)
        vals = map_coordinates(vol, [p[:, 2] - lo[2], p[:, 1] - lo[1], p[:, 0] - lo[0]], order=1)
        if k:
            vals[np.isnan(offsets[k - 1])] = np.nan
        img[ok] = vals
        images.append(img)

    titles = ["verified patch surface (human)"]
    for label, t in zip((label_a, label_b), offsets):
        med, far = np.nanmedian(np.abs(t)), np.mean(~(np.abs(t) <= 4))  # NaN (no sheet found) counts as off
        titles.append(f"{label} fit's nearest sheet: median {med:.1f} voxels\nfrom the patch, "
                      f"{far:.0%} of it more than 4 voxels off")
        print(f"{label}: nearest sheet a median {med:.1f} voxels from the patch ({far:.0%} of samples more than "
              f"4 voxels off or without a sheet within {OFFSETS[-1]:.0f})")
    h, w = ok.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6 * h / w + 1.2), dpi=110, layout="constrained")
    for k, (ax, img, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(np.ma.masked_invalid(img), cmap="gray", vmin=VMIN, vmax=VMAX)
        if k:  # samples without a fitted sheet within the searched range
            holes = ok & np.isnan(img)
            ax.imshow(np.ma.masked_where(~holes, holes), cmap=matplotlib.colors.ListedColormap(["magenta"]))
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f"patch {name}: {w} x {h} voxels of its surface around x {centre[0]:.0f}, y {centre[1]:.0f}, "
                 f"z {centre[2]:.0f} (9.6 µm voxels)\nthe {label_a} fit puts this patch between sheets; magenta: "
                 f"no fitted sheet within {OFFSETS[-1]:.0f} voxels", fontsize=11)
    fig.savefig(args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
