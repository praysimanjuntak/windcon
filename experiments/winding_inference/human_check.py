#!/usr/bin/env python3
"""Does the winding model count the windings a person counted?

The spiral dataset's relative_windings.json holds point collections drawn across the sheets: each
point lies on a sheet and carries its winding number relative to the others (`wind_a`), one
winding between neighbours. This script lays one slab of the model along each collection (its
ray runs from the first point to the last), reads the model's phase at every point, in the column
of the slab the point lies in, and compares the phase difference of neighbouring points with the
one winding the person counted. With --checkpoint it does the same with fitted spirals.

    run.sh human_check.py --relative-windings <dataset>/relative_windings.json \
        --volume <chunk cache folder> --model <ckpt_final.pth> --z-range 9900 10900 \
        [--checkpoint <run>/checkpoint_fitted.ckpt ... --umbilicus <dataset>/umbilicus.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from store_vs_fit import fitted_transform


def collections(path: Path, z_range: tuple[float, float], longest: float) -> list[np.ndarray]:
    """Points (x, y, z) of each collection in the z-range, ordered by winding, one winding apart."""
    kept = []
    for collection in json.loads(path.read_text())["collections"].values():
        points = sorted((p["wind_a"], *p["p"]) for p in collection["points"].values() if "wind_a" in p)
        rows = np.array(points, dtype=np.float64).reshape(-1, 4)
        if len(rows) < 2 or rows[:, 3].min() < z_range[0] or rows[:, 3].max() >= z_range[1]:
            continue
        if not np.allclose(np.diff(rows[:, 0]), 1.0) or np.linalg.norm(rows[-1, 1:] - rows[0, 1:]) > longest:
            continue
        kept.append(rows[:, 1:])
    return kept


def report(steps: list[np.ndarray]) -> dict:
    """Summary of the windings counted between neighbouring points (the person counted one) per collection."""
    every = np.concatenate(steps)
    totals = np.array([step.sum() for step in steps])
    counted = np.array([len(step) for step in steps])
    return {
        "steps": len(every),
        "steps_counted_as_one": float(np.mean(np.abs(every - 1) < 0.5)),
        "step_median": float(np.median(every)),
        "collections": len(steps),
        "collections_with_the_persons_total": float(np.mean(np.abs(totals - counted) < 0.5)),
        "windings_counted_per_winding_of_the_person": float(totals.sum() / counted.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--relative-windings", type=Path, required=True)
    ap.add_argument("--volume", type=Path, required=True, help="windcon chunk cache folder of the fit volume")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--z-range", type=float, nargs=2, required=True, metavar=("BEGIN", "END"))
    ap.add_argument("--checkpoint", type=Path, action="append", default=[], help="fitted spiral to put to the same test")
    ap.add_argument("--umbilicus", type=Path, help="umbilicus.json, needed with --checkpoint")
    ap.add_argument("--out", type=Path, help="write the report here as JSON")
    args = ap.parse_args()

    import torch
    from vesuvius.neural_tracing.winding_models import infer_winding_volume as villa
    from vesuvius.neural_tracing.winding_models.volume_slab_extractor import VolumeSlabExtractor
    from vesuvius.neural_tracing.winding_models.winding_model import WindingModel

    device = torch.device("cuda:0")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = WindingModel(config.get("model"))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    length, size = int(config.get("ray_length", 384)), int(config.get("transverse_size", 128))
    extractor = VolumeSlabExtractor(
        [args.volume], transverse_size=size, ray_length=length, spacing=1.0, sampling="trilinear",
        tile_size=int(config.get("tile_size", 64)),
    )  # fmt: skip
    drawn = collections(args.relative_windings, tuple(args.z_range), length - 24)
    results = {}

    steps = []
    for points in drawn:
        direction = points[-1] - points[0]
        direction /= np.linalg.norm(direction)
        middle = (points[0] + points[-1]) / 2
        image, valid, frame = extractor.extract(0, direction, middle - (length - 1) / 2 * direction)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            phase = model(torch.from_numpy(image[None]).to(device), torch.from_numpy(valid[None]).to(device))["phase"]
        phase = phase[0].float().cpu().numpy()  # [column along a, column along b, sample]
        place = frame.to_slab(points)
        a, b = (np.clip(np.rint(place[:, k]).astype(int), 0, size - 1) for k in (0, 1))
        at = np.array([np.interp(t, np.arange(length), phase[i, j]) for i, j, t in zip(a, b, place[:, 2])])
        steps.append(np.abs(np.diff(at)))
    results["winding model"] = report(steps)
    print("winding model", json.dumps(results["winding model"]), flush=True)

    if args.checkpoint:
        sys.path.insert(0, villa._spiral_scripts_dir())
        from winding_supervision import _component_residual

        for path in args.checkpoint:
            with torch.inference_mode():
                transform, dr_per_winding, (z_begin, z_end) = fitted_transform(path, args.umbilicus, device)
                steps = []
                for points in drawn:
                    zyx = torch.from_numpy(points[:, ::-1].copy()).to(device, torch.float32)
                    pairs = torch.stack([zyx[:-1], zyx[1:]], dim=1)
                    spiral = transform(pairs.reshape(-1, 3)).reshape(-1, 2, 3)
                    target = torch.ones(len(pairs), device=device)
                    inside = torch.ones(len(pairs), dtype=torch.bool, device=device)
                    residual, _ = _component_residual(spiral, pairs, target, inside, dr_per_winding, z_begin, z_end)
                    steps.append((residual + target).abs().float().cpu().numpy())
            name = path.parent.name
            results[name] = report(steps)
            print(name, json.dumps(results[name]), flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
