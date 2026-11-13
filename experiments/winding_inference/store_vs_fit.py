#!/usr/bin/env python3
"""How well do the rays of a winding-inference store agree with a fitted spiral?

A store says how many windings lie between two points of a ray. A fit gives every point a winding
number. This script draws pairs from each store the way fit_spiral.py does (neighbouring crossings,
and crossings 3 to 15 levels apart), reads the fit's winding difference for each pair with villa's
own residual, and reports how far it is from the store's. With a fit that was not trained on the
store, it compares stores without a fit of their own; the reference fit's errors count against
every store alike.

    run.sh store_vs_fit.py --checkpoint <run>/checkpoint_fitted.ckpt --umbilicus <umbilicus.json> \
        --store <store> [--store <another> ...] [--pairs 200000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from build_store import OLD_SETTING_NAMES

CHUNK = 1 << 18  # points per call of the fit's transform


def fitted_transform(checkpoint_path: Path, umbilicus: Path, device):
    """The fit's map from scan to spiral coordinates, its voxels per winding and its z-range, as fit_spiral.py builds them."""
    import torch
    from vesuvius.neural_tracing.winding_models import infer_winding_volume as villa

    sys.path.insert(0, villa._spiral_scripts_dir())
    from transforms import SpiralAndTransform
    from umbilicus import json_umbilicus_z_to_yx

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    settings = dict(checkpoint["cfg"])
    for name in OLD_SETTING_NAMES:  # both spellings, whichever the checkpoint has
        if f"model_{name}" in settings:
            settings.setdefault(name, settings[f"model_{name}"])
        elif name in settings:
            settings.setdefault(f"model_{name}", settings[name])
    z_begin, z_end = int(checkpoint["z_begin"]), int(checkpoint["z_end"])
    z = np.arange(z_begin, z_end)
    centre = json_umbilicus_z_to_yx(str(umbilicus), coordinate_scale=1.0)
    radius, margin = int(settings["model_flow_bounds_radius"]), int(settings["model_flow_bounds_z_margin"])
    model = SpiralAndTransform(
        flow_integration_steps=int(settings["model_num_flow_integration_steps"]),
        flow_integration_solver=str(settings["model_flow_integration_solver"]),
        flow_min_corner_zyx=torch.tensor([z_begin - margin, -radius, -radius], dtype=torch.int64, device=device),
        flow_max_corner_zyx=torch.tensor([z_end + margin, radius, radius], dtype=torch.int64, device=device),
        umbilicus_zyx=torch.from_numpy(np.concatenate([z[:, None], centre(z)], axis=-1).astype(np.float32)).to(device),
        config=settings,
        spiral_outward_sense=str(checkpoint.get("spiral_outward_sense") or "CW"),
    ).to(device)
    model.load_state_dict(checkpoint["spiral_and_transform"])
    model.eval()
    return model.get_slice_to_spiral_transform(), model.get_dr_per_winding(), (z_begin, z_end)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--store", type=Path, action="append", required=True)
    ap.add_argument("--pairs", type=int, default=200_000, help="pairs drawn of each kind from each store")
    ap.add_argument("--out", type=Path, help="write the report here as JSON")
    args = ap.parse_args()

    import torch
    from vesuvius.neural_tracing.winding_models import infer_winding_volume as villa

    sys.path.insert(0, villa._spiral_scripts_dir())
    from winding_supervision import _component_residual, load_winding_inference_store

    device = torch.device("cuda:0")
    report = {}
    with torch.inference_mode():
        transform, dr_per_winding, (z_begin, z_end) = fitted_transform(args.checkpoint, args.umbilicus, device)
        for path in args.store:
            store = load_winding_inference_store(path, device, verify=False, z_range=(z_begin, z_end))
            generator = torch.Generator(device=device).manual_seed(0)
            kinds = {
                "neighbouring crossings": store.sample_adjacent(args.pairs, generator=generator),
                "3 to 15 levels apart": store.sample_relative(args.pairs, 3, 15, generator=generator),
            }
            row = {"rays": store.fingerprint["num_rays"], "rays_in_window": store.num_z_eligible_rays}
            for kind, sample in kinds.items():
                points = sample["points"]
                flat = points.reshape(-1, 3)
                spiral = torch.cat([transform(flat[i : i + CHUNK]) for i in range(0, len(flat), CHUNK)]).reshape(-1, 2, 3)
                inside = torch.ones(len(points), dtype=torch.bool, device=device)
                residual, valid = _component_residual(
                    spiral, points, sample["target"], inside, dr_per_winding, z_begin, z_end
                )
                error = residual[valid].abs().float().cpu().numpy()
                row[kind] = {
                    "pairs_in_window": int(valid.sum()),
                    "within_half_a_winding": float(np.mean(error < 0.5)),
                    "abs_error_median": float(np.median(error)),
                    "abs_error_p95": float(np.percentile(error, 95)),
                    "signed_error_mean": float(residual[valid].float().mean()),
                }
            report[str(path)] = row
            print(path.name, json.dumps(row), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "stores": report}, indent=1) + "\n")


if __name__ == "__main__":
    main()
