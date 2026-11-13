#!/usr/bin/env python3
"""Does the winding model count the same sheets in two registered scans of one object?

`winding_model_9um` was trained on 2.4 um scans read at 9.6 um. Whether it works on the 8.64 and
9.36 um scans is not documented. For some objects the open data holds both kinds of scan and a
`transform.json` in one of the two volumes that maps its voxels onto the other. This script reads
the same slabs from both, runs the model on each and compares the winding count along the centre
ray: the windings between two points on it, which is what fit_spiral.py takes from a ray.

Four readings per line:
  fine      the fine scan at the level whose voxels are about 9.6 um (the model's own kind of data)
  view      the fine scan again, the line seen 32 samples off the slab centre (the model's
            disagreement with itself; the yardstick for the next two)
  coarse    the coarse scan at exactly the fine slab's sample positions
  native    the coarse scan at its own voxel pitch, around the same seed and direction

Lines are seeded on a 3 x 3 x 3 grid around each of a few places deep inside the object. All lines of
a place run along its sheet normal: the direction of strongest intensity change in the fine scan
that its 27 seeds share.

    run.sh transfer_check.py --fine <fine zarr URL> --fine-level 2 --coarse <coarse zarr URL> \
        --model <ckpt_final.pth> --out <name>.npz [--png <name>.png]
    run.sh transfer_check.py --report --out <name>.npz     # the report again, without the GPU

The report gives the differences over four window lengths, and over stretches of 64 samples grouped
by how far apart the sheets are there.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from itertools import pairwise
from pathlib import Path

import numpy as np
from build_store import leaves_scan
from scipy import ndimage

from windcon.zarr_io import OPEN_DATA_MIRROR, S3_OPEN_DATA, ZarrArray

OFF_CENTRE = 32  # samples between the slab centre and the line in the `view` reading
WINDOWS = (40, 80, 120, 160)  # half-widths in samples of the fine reading (384 samples per ray)
READINGS = ("fine", "view", "coarse", "native")
STRETCH = 64  # samples of the fine reading per stretch in the table by sheet spacing
SPACINGS = (0, 10, 12, 14, 18, 1000)  # its groups: samples of the fine reading per winding


def registration(url: str, fine_downscale: int, in_fine_volume: bool) -> np.ndarray:
    """4 x 4 map from voxels of the fine level to voxels of the coarse volume's level 0, both (x, y, z).

    A transform.json maps level-0 voxels of the volume it lies in onto level-0 voxels of the other
    (its `fixed_volume`). One in the coarse volume is inverted, as villa's
    VolumeSlabExtractor.load_segment_to_volume_transform does.
    """
    with urllib.request.urlopen(url.replace(S3_OPEN_DATA, OPEN_DATA_MIRROR, 1), timeout=60) as response:
        matrix = np.asarray(json.loads(response.read())["transformation_matrix"], dtype=np.float64)
    transform = np.eye(4)
    if in_fine_volume:
        transform[:3, :3] = matrix[:, :3] * fine_downscale
        transform[:3, 3] = matrix[:, 3]
    else:
        inverse = np.linalg.inv(matrix[:, :3])
        transform[:3, :3] = inverse * fine_downscale
        transform[:3, 3] = -inverse @ matrix[:, 3]
    return transform


def deep_places(overview: ZarrArray, factor: int, depth: float, count: int) -> np.ndarray:
    """`count` points (x, y, z in level-0 voxels) at least `depth` voxels inside the object, far from each other."""
    mask = np.pad(overview.read(tuple((0, n) for n in overview.shape)) > 0, 1)  # the volume's faces count as surface
    inside = ndimage.distance_transform_edt(mask)[1:-1, 1:-1, 1:-1] >= depth / factor
    candidates = np.argwhere(inside)[::7]  # z, y, x
    if not len(candidates):
        raise SystemExit(f"no place of the object is {depth:.0f} voxels from its surface")
    chosen = [candidates[len(candidates) // 2]]
    distance = np.linalg.norm(candidates - chosen[0], axis=1)
    while len(chosen) < count:
        chosen.append(candidates[distance.argmax()])
        distance = np.minimum(distance, np.linalg.norm(candidates - chosen[-1], axis=1))
    return (np.array(chosen)[:, ::-1] + 0.5) * factor


def sheet_normals(volume: ZarrArray, points_xyz: np.ndarray, sigma: float = 1.5, window: float = 10.0) -> np.ndarray:
    """Unit sheet normals (x, y, z) at the points: the principal direction of the intensity gradient around them."""
    margin = int(3 * (sigma + window)) + 2
    low = np.floor(points_xyz.min(0)).astype(int) - margin
    high = np.ceil(points_xyz.max(0)).astype(int) + margin + 1
    crop = volume.read(((low[2], high[2]), (low[1], high[1]), (low[0], high[0]))).astype(np.float32)
    gradient = [ndimage.gaussian_filter(crop, sigma, order=order) for order in ((0, 0, 1), (0, 1, 0), (1, 0, 0))]
    at = tuple(np.rint(points_xyz - low).astype(int)[:, axis] for axis in (2, 1, 0))
    tensor = np.empty((len(points_xyz), 3, 3))
    for i in range(3):
        for j in range(i, 3):
            tensor[:, i, j] = tensor[:, j, i] = ndimage.gaussian_filter(gradient[i] * gradient[j], window)[at]
    return np.linalg.eigh(tensor)[1][:, :, -1]


def windings_between(phase: np.ndarray, centre: float, half_width: float) -> np.ndarray:
    """Windings the model counts from `centre - half_width` to `centre + half_width` on each ray (samples)."""
    t = np.arange(phase.shape[1])
    return np.array([np.interp(centre + half_width, t, p) - np.interp(centre - half_width, t, p) for p in phase])


def summary(phases: dict[str, np.ndarray], pitch: dict[str, float], usable: np.ndarray) -> list[dict]:
    """Per window: the fine count, and how each other reading differs from it over the same stretch of the line."""
    middle = (phases["fine"].shape[1] - 1) / 2
    rows = []
    for half_width in WINDOWS:
        fine = windings_between(phases["fine"][usable], middle, half_width)
        row = {"half_width_samples": half_width, "rays": int(usable.sum()), "fine_windings_mean": float(fine.mean())}
        for name in READINGS[1:]:
            # The same physical stretch is `half_width * pitch_fine / pitch_reading` samples of the reading.
            other = windings_between(phases[name][usable], middle, half_width * pitch["fine"] / pitch[name])
            difference = other - fine
            row[name] = {
                "same_count_share": float(np.mean(np.abs(difference) < 0.5)),
                "abs_difference_median": float(np.median(np.abs(difference))),
                "difference_mean": float(difference.mean()),
            }
        rows.append(row)
    return rows


def by_spacing(phases: dict[str, np.ndarray], pitch: dict[str, float], usable: np.ndarray) -> list[dict]:
    """Stretches of 64 samples grouped by how far apart the sheets are there (the mean of the two fine readings):
    how often each other reading counts as the fine one does, and its windings per winding of the fine one."""
    middle = (phases["fine"].shape[1] - 1) / 2
    centres = np.arange(STRETCH, phases["fine"].shape[1] - STRETCH + 1, STRETCH // 2)  # every reading covers these
    counts = {}
    for name in READINGS:
        ratio = pitch["fine"] / pitch[name]
        counts[name] = np.concatenate(
            [windings_between(phases[name][usable], middle + (c - middle) * ratio, STRETCH / 2 * ratio) for c in centres]
        )
    spacing = STRETCH / np.maximum((counts["fine"] + counts["view"]) / 2, 1e-6)
    rows = []
    for low, high in pairwise(SPACINGS):
        chosen = (spacing >= low) & (spacing < high)
        if chosen.sum() < 10:
            continue
        fine = counts["fine"][chosen]
        row = {"samples_per_winding": [low, high], "stretches": int(chosen.sum())}
        for name in READINGS[1:]:
            other = counts[name][chosen]
            row[name] = {
                "same_count_share": float(np.mean(np.abs(other - fine) < 0.5)),
                "windings_per_fine_winding": float(other.sum() / fine.sum()),
            }
        rows.append(row)
    return rows


def write_report(out: Path, phases: dict, lines: dict, scale: float, png: Path | None) -> None:
    """Print and save the comparison of the readings stored in `out` (.npz), and draw a few lines."""
    # A line is usable when no reading of it leaves the object (four zero voxels in a row, as in build_store.py).
    usable = ~np.any([[leaves_scan(line, 4) for line in value] for value in lines.values()], axis=0)
    pitch = {"fine": 1.0, "view": 1.0, "coarse": 1.0, "native": 1.0 / scale}  # voxels of the fine level per sample
    report = {
        "lines": len(usable),
        "usable": int(usable.sum()),
        "windows": summary(phases, pitch, usable),
        "by_sheet_spacing": by_spacing(phases, pitch, usable),
    }
    print(json.dumps(report, indent=1))
    out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    if png:
        picture(png, phases, lines, pitch, np.flatnonzero(usable)[:: max(1, int(usable.sum()) // 6)][:6])


def picture(path: Path, phases: dict, lines: dict, pitch: dict, rays: np.ndarray) -> None:
    """Intensity along a few lines in each scan, with the places where the phase passes whole numbers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(rays), 1, figsize=(14, 2.2 * len(rays)), squeeze=False)
    middle = (phases["fine"].shape[1] - 1) / 2
    for axis, ray in zip(axes[:, 0], rays):
        for name, colour, offset in (("fine", "tab:blue", 0), ("coarse", "tab:orange", 300), ("native", "tab:green", 600)):
            samples = np.arange(phases[name].shape[1])
            position = (samples - middle) * pitch[name]  # voxels of the fine level from the seed
            axis.plot(position, lines[name][ray].astype(float) + offset, color=colour, lw=0.8, label=name)
            phase = phases[name][ray] - np.interp(middle, samples, phases[name][ray])
            passing = np.flatnonzero(np.diff(np.floor(phase)) > 0)
            axis.vlines(position[passing], offset, offset + 255, color=colour, lw=0.5, alpha=0.6)
        axis.set_yticks([])
        axis.set_xlim(-middle, middle)
    axes[0, 0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("distance from the seed along the line, voxels of the fine level")
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", action="store_true", help="only report again from an existing --out file")
    ap.add_argument("--fine", help="URL of the fine scan's zarr group")
    ap.add_argument("--fine-level", type=int, default=2, help="level of it with voxels of about 9.6 um")
    ap.add_argument("--coarse", help="URL of the coarse scan's zarr group")
    ap.add_argument("--transform-in", choices=("coarse", "fine"), default="coarse",
                    help="the volume whose transform.json registers the two scans")
    ap.add_argument("--overview-level", type=int, default=5, help="level of the coarse scan the places are chosen in")
    ap.add_argument("--places", type=int, default=6)
    ap.add_argument("--grid", type=int, default=64, help="samples between the 3 x 3 x 3 seeds of a place")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--out", type=Path, required=True, help=".npz with the phases and intensities along every line")
    ap.add_argument("--png", type=Path, help="picture of a few lines in the fine and the coarse scan")
    args = ap.parse_args()
    if args.report:
        saved = np.load(args.out)
        phases, lines = ({name: saved[f"{kind}_{name}"] for name in READINGS} for kind in ("phase", "line"))
        write_report(args.out, phases, lines, float(saved["coarse_voxels_per_fine_voxel"]), args.png)
        return
    if not (args.fine and args.coarse and args.model):
        ap.error("--fine, --coarse and --model are needed unless --report is given")

    import torch
    from vesuvius.neural_tracing.winding_models.volume_slab_extractor import VolumeSlabExtractor
    from vesuvius.neural_tracing.winding_models.winding_model import WindingModel

    fine = ZarrArray(f"{args.fine.rstrip('/')}/{args.fine_level}")
    coarse = ZarrArray(f"{args.coarse.rstrip('/')}/0")
    overview = ZarrArray(f"{args.coarse.rstrip('/')}/{args.overview_level}")
    holder = args.fine if args.transform_in == "fine" else args.coarse
    to_coarse = registration(f"{holder.rstrip('/')}/transform.json", 2**args.fine_level, args.transform_in == "fine")
    to_fine = np.linalg.inv(to_coarse)
    scale = float(np.cbrt(np.linalg.det(to_coarse[:3, :3])))  # coarse voxels per voxel of the fine level
    print(f"one voxel of the fine level is {scale:.4f} coarse voxels", flush=True)

    device = torch.device("cuda:0")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = WindingModel(config.get("model"))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    length, size = int(config.get("ray_length", 384)), int(config.get("transverse_size", 128))
    middle, centre = (length - 1) / 2, size // 2
    extractor = VolumeSlabExtractor(
        [fine.cache, coarse.cache, coarse.cache],
        transverse_size=size,
        ray_length=length,
        spacing=1.0,
        sampling="trilinear",
        tile_size=int(config.get("tile_size", 64)),
        segment_to_volume_xyz=[np.eye(4), to_coarse, np.eye(4)],
    )

    def read(volume: int, direction: np.ndarray, seed: np.ndarray, shift: int = 0):
        """Phase and intensity along the line through `seed`, which lies `shift` columns from the slab centre."""
        frame = extractor.slab_frame(direction, seed - middle * direction)
        image, valid, _ = extractor.extract(volume, direction, seed - middle * direction - shift * frame.axis_a)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            phase = model(torch.from_numpy(image[None]).to(device), torch.from_numpy(valid[None]).to(device))["phase"]
        return phase[0, centre + shift, centre].float().cpu().numpy(), image[centre + shift, centre]

    # Seeds: around places deep enough inside the object for every centre line to stay in it.
    places = deep_places(overview, 2**args.overview_level, scale * (middle + args.grid), args.places)
    offsets = np.stack(np.meshgrid(*[[-args.grid, 0, args.grid]] * 3, indexing="ij"), -1).reshape(-1, 3)
    phases = {name: [] for name in READINGS}
    lines = {name: [] for name in READINGS}
    seeds, directions, place_of = [], [], []
    started = time.time()
    for number, place in enumerate(places):
        grid = to_fine[:3, :3] @ place + to_fine[:3, 3] + offsets
        if (grid.min(0) < 0).any() or (grid.max(0) >= np.array(fine.shape[::-1])).any():
            print(f"place {number}: outside the fine level, skipped", flush=True)
            continue
        normals = sheet_normals(fine, grid)
        direction = np.linalg.eigh(normals.T @ normals)[1][:, -1]  # the axis the 27 normals share
        for seed in grid:
            seed_coarse = to_coarse[:3, :3] @ seed + to_coarse[:3, 3]
            direction_coarse = to_coarse[:3, :3] @ direction
            direction_coarse /= np.linalg.norm(direction_coarse)
            readings = (
                read(0, direction, seed),
                read(0, direction, seed, OFF_CENTRE),
                read(1, direction, seed),
                read(2, direction_coarse, seed_coarse),
            )
            for name, (phase, line) in zip(READINGS, readings):
                phases[name].append(phase)
                lines[name].append(line)
            seeds.append(seed)
            directions.append(direction)
            place_of.append(number)
        print(f"place {number} at coarse (x, y, z) {np.round(place).astype(int).tolist()}: {len(grid)} lines, "
              f"{time.time() - started:.0f} s", flush=True)

    phases = {name: np.array(value, dtype=np.float32) for name, value in phases.items()}
    lines = {name: np.array(value, dtype=np.uint8) for name, value in lines.items()}
    np.savez(
        args.out,
        seed_fine_xyz=np.array(seeds), direction_xyz=np.array(directions), place=np.array(place_of),
        coarse_voxels_per_fine_voxel=scale,
        **{f"phase_{name}": value for name, value in phases.items()},
        **{f"line_{name}": value for name, value in lines.items()},
    )  # fmt: skip
    write_report(args.out, phases, lines, scale, args.png)


if __name__ == "__main__":
    main()
