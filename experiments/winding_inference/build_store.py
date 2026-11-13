#!/usr/bin/env python3
"""Build the winding-inference store that villa's fit_spiral.py reads, from any fitted checkpoint.

villa makes the store in two steps: infer_winding_volume.py caches the whole phase field of every
slab, and export_spiral_supervision.py keeps one ray per slab. The public model
(scrollprize/winding_model_9um) predicts a phase for every voxel column, so that cache takes
25 MB per slab. This script runs the same model on the same slabs, decodes rays with villa's own
`decode_center_ray` and writes the store directly. `--columns k` keeps a k x k grid of rays from
each slab instead of the centre one.

Rays start on the sheets of the seed fit and run along its normals: the fit decides where the
model looks, the model decides where the sheets are. A stored ray holds the positions where the
model's phase passes whole numbers, as levels relative to the ray's midpoint.

Work is saved every 512 slabs under <out>.partial; a rerun with the same arguments resumes.
Run it with `run.sh build_store.py ...`, which sets up villa's environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FORMAT_NAME = "winding_inference_crossings"  # winding_supervision.ARTIFACT_TYPE in villa/spiral-fitting
FORMAT_VERSION = 1
PART_SLABS = 512
ARRAYS = ("ray_origin_zyx", "ray_step_zyx", "crossing_offsets", "crossing_t", "crossing_level", "seed_winding")
# What a part file and save_store hold per ray (crossing_t and crossing_level: per crossing, ray after ray).
RAY_FIELDS = ("ray_origin_zyx", "ray_step_zyx", "seed_winding", "ray_source", "crossing_counts", "crossing_t",
              "crossing_level")  # fmt: skip
# Settings villa's seeder reads from a checkpoint without the model_ prefix that fit_spiral.py at the
# same commit saves them under (num_flow_timesteps is no longer saved at all).
OLD_SETTING_NAMES = (
    "num_flow_integration_steps", "flow_integration_solver", "num_flow_timesteps", "num_flow_stages",
    "flow_bounds_z_margin", "flow_bounds_radius", "flow_voxel_resolution", "flow_field_type",
    "gap_expander_logit_resolution", "gap_expander_num_windings", "gap_expander_lr_scale",
    "linear_z_resolution", "initial_dr_per_winding",
)  # fmt: skip


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_rays(args) -> dict[str, np.ndarray]:
    """Seed points, ray directions and seed windings: from --rays, or from the checkpoint with villa's seeder."""
    if args.rays:
        loaded = np.load(args.rays)
    else:
        from vesuvius.neural_tracing.winding_models import infer_winding_volume as villa

        load_checkpoint = villa._load_checkpoint_cfg

        def load_with_old_names(path):
            checkpoint = load_checkpoint(path)
            settings = dict(checkpoint["cfg"])
            for name in OLD_SETTING_NAMES:
                settings.setdefault(name, settings.get(f"model_{name}"))
            checkpoint["cfg"] = settings
            return checkpoint

        villa._load_checkpoint_cfg = load_with_old_names
        loaded = villa.build_seed_rays(
            SimpleNamespace(
                fit_checkpoint=str(args.checkpoint),
                umbilicus=str(args.umbilicus),
                z_range=args.z_range,
                winding_range=args.winding_range,
                winding_step=args.winding_step,
                seed_spacing=args.seed_spacing,
            )
        )
    rays = {
        "seed_xyz": np.asarray(loaded["seed_xyz"], dtype=np.float32),
        "direction_xyz": np.asarray(loaded["direction_xyz"], dtype=np.float32),
        "seed_winding": np.asarray(loaded["seed_winding"], dtype=np.int16),
    }
    if args.max_slabs and args.max_slabs < len(rays["seed_xyz"]):
        keep = np.unique(np.linspace(0, len(rays["seed_xyz"]) - 1, args.max_slabs).astype(int))
        rays = {key: value[keep] for key, value in rays.items()}
    return rays


def column_indices(native_columns: int, stride: int, count: int, spacing: int) -> np.ndarray:
    """Native indices of `count` columns per transverse axis, `spacing` voxels apart around the centre one."""
    centre = round((native_columns * stride - 1) / stride / 2)  # as export_spiral_supervision.py
    if spacing % stride:
        raise ValueError(f"--column-spacing {spacing} must be a multiple of the model's column stride {stride}")
    if count % 2 == 0:
        raise ValueError("--columns must be odd, so that the centre column is one of them")
    indices = centre + (np.arange(count) - count // 2) * (spacing // stride)
    if indices.min() < 0 or indices.max() >= native_columns:
        raise ValueError(f"{count} columns {spacing} voxels apart do not fit in the slab")
    return indices


def leaves_scan(values: np.ndarray, run: int) -> bool:
    """True when `run` consecutive samples of a ray read 0, which a masked scan does only outside the scroll."""
    zeros = np.concatenate([[0], np.cumsum(values == 0)])
    return bool((zeros[run:] - zeros[:-run] == run).any())


def decode_part(indices, rays, extractor, model, device, args, geometry) -> dict[str, np.ndarray]:
    """Run the model on the slabs `indices` and decode the selected columns into rays."""
    import torch
    from vesuvius.neural_tracing.winding_models.export_spiral_supervision import decode_center_ray

    ray_length, spacing, stride, columns = geometry
    midpoint = (ray_length - 1) / 2.0
    anchor = round(midpoint)
    columns_gpu = torch.as_tensor(columns, device=device)

    def extract(index):
        began = time.perf_counter()
        direction = rays["direction_xyz"][index].astype(np.float64)
        origin = rays["seed_xyz"][index].astype(np.float64) - midpoint * spacing * direction
        return (index, *extractor.extract(0, direction, origin), time.perf_counter() - began)

    origins, steps, windings, counts, positions, levels, sources = [], [], [], [], [], [], []
    outside = short = 0
    extract_seconds = model_seconds = 0.0

    def forward(batch):
        nonlocal outside, short, model_seconds
        began = time.perf_counter()
        images = torch.from_numpy(np.stack([item[1] for item in batch])).to(device)
        valid = torch.from_numpy(np.stack([item[2] for item in batch])).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            phase = model(images, valid.bool())["phase"]
        phase = phase.float()[:, columns_gpu][:, :, columns_gpu].cpu().numpy()
        model_seconds += time.perf_counter() - began
        for sample, (index, image, slab_valid, frame, _) in enumerate(batch):
            for a, native_a in enumerate(columns):
                for b, native_b in enumerate(columns):
                    row, column = int(native_a) * stride, int(native_b) * stride
                    if leaves_scan(image[row, column], args.outside_run):
                        outside += 1
                        continue
                    t, level = decode_center_ray(
                        phase[sample, a, b],
                        slab_valid[row, column],
                        anchor=anchor,
                        edge_margin=args.edge_margin,
                        edge_trim=args.edge_trim,
                    )
                    if len(t) < 2:
                        short += 1
                        continue
                    start = frame.origin + spacing * (row * frame.axis_a + column * frame.axis_b)
                    origins.append(start[::-1])
                    steps.append((spacing * frame.direction)[::-1])
                    windings.append(rays["seed_winding"][index])
                    sources.append((index, a, b))
                    counts.append(len(t))
                    positions.append(t)
                    levels.append(level)

    with ThreadPoolExecutor(args.extract_threads) as pool, torch.inference_mode():
        todo = iter(indices)
        window = deque()

        def refill():
            while len(window) < args.extract_threads * 2:
                index = next(todo, None)
                if index is None:
                    return
                window.append(pool.submit(extract, int(index)))

        refill()
        batch = []
        while window:
            batch.append(window.popleft().result())
            extract_seconds += batch[-1][-1]
            refill()
            if len(batch) == args.batch_size or not window:
                forward(batch)
                batch = []

    return {
        "ray_origin_zyx": np.asarray(origins, dtype=np.float32).reshape(-1, 3),
        "ray_step_zyx": np.asarray(steps, dtype=np.float32).reshape(-1, 3),
        "seed_winding": np.asarray(windings, dtype=np.int16),
        "ray_source": np.asarray(sources, dtype=np.int32).reshape(-1, 3),
        "crossing_counts": np.asarray(counts, dtype=np.int64),
        "crossing_t": np.concatenate(positions).astype(np.float32) if positions else np.empty(0, np.float32),
        "crossing_level": np.concatenate(levels).astype(np.int16) if levels else np.empty(0, np.int16),
        "rays_leaving_scan": np.int64(outside),
        "rays_under_two_crossings": np.int64(short),
        # Seconds per slab: reading and sampling the scan (summed over the threads), and the model on the GPU.
        "extract_seconds_per_slab": np.float64(extract_seconds / max(len(indices), 1)),
        "model_seconds_per_slab": np.float64(model_seconds / max(len(indices), 1)),
    }


def merge_parts(parts: list[Path]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """The rays of all parts, and their counters summed."""
    loaded = [np.load(path) for path in parts]
    rays = {name: np.concatenate([part[name] for part in loaded]) for name in RAY_FIELDS}
    counters = {name: int(sum(part[name] for part in loaded)) for name in ("rays_leaving_scan", "rays_under_two_crossings")}
    return rays, counters


def save_store(folder: Path, rays: dict[str, np.ndarray], record: dict) -> dict:
    """Write rays as one shard, with the manifest fit_spiral.py checks, into the existing `folder`."""
    arrays = {name: rays[name] for name in ARRAYS if name != "crossing_offsets"}
    arrays["crossing_offsets"] = np.concatenate([[0], np.cumsum(rays["crossing_counts"])]).astype(np.int64)
    shard = folder / "shard_00000"
    shard.mkdir()
    described = {}
    for name in ARRAYS:
        path = shard / f"{name}.npy"
        np.save(path, np.ascontiguousarray(arrays[name]), allow_pickle=False)
        described[name] = {
            "file": path.name,
            "shape": list(arrays[name].shape),
            "dtype": arrays[name].dtype.str,
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        }
    # Which slab and which of its columns each ray came from, for subsets; fit_spiral.py does not read it.
    np.save(folder / "ray_source.npy", rays["ray_source"], allow_pickle=False)
    manifest = {
        "artifact_type": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "coordinate_order": "zyx",
        "coordinate_space": "reference zarr scale-0 voxels",
        **record,
        "num_rays": len(arrays["ray_origin_zyx"]),
        "num_crossings": len(arrays["crossing_t"]),
        "shards": [
            {
                "name": shard.name,
                "num_retained_rays": len(arrays["ray_origin_zyx"]),
                "num_crossings": len(arrays["crossing_t"]),
                "arrays": described,
            }
        ],
    }
    # The loader recomputes the fingerprint without itself and without three keys of villa's exporter,
    # which a subset of a published store carries over.
    manifest["fingerprint"] = canonical_digest(
        {key: value for key, value in manifest.items() if key not in ("elapsed_seconds", "export_workers", "rays_per_task")}
    )
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_store(folder: Path) -> tuple[dict[str, np.ndarray], dict]:
    """The rays of a store and its manifest.

    A store from villa's exporter has several shards and no ray_source.npy; every ray of it is the centre
    column of its own slab, so its source is (ray number, 0, 0).
    """
    manifest = json.loads((folder / "manifest.json").read_text())
    shards = [
        {name: np.load(folder / shard["name"] / shard["arrays"][name]["file"]) for name in ARRAYS}
        for shard in manifest["shards"]
    ]
    rays = {name: np.concatenate([shard[name] for shard in shards]) for name in ARRAYS if name != "crossing_offsets"}
    rays["crossing_counts"] = np.concatenate([np.diff(shard["crossing_offsets"]) for shard in shards])
    source = folder / "ray_source.npy"
    if source.exists():
        rays["ray_source"] = np.load(source)
    else:
        rays["ray_source"] = np.zeros((len(rays["crossing_counts"]), 3), dtype=np.int32)
        rays["ray_source"][:, 0] = np.arange(len(rays["crossing_counts"]))
    return rays, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--volume", type=Path, required=True, help="windcon chunk cache folder of the fit volume")
    parser.add_argument("--model", type=Path, required=True, help="winding model checkpoint (ckpt_final.pth)")
    parser.add_argument("--out", type=Path, required=True, help="new store folder")
    parser.add_argument("--checkpoint", type=Path, help="fitted checkpoint whose sheets seed the rays")
    parser.add_argument("--umbilicus", type=Path, help="umbilicus.json of the fitted scroll")
    parser.add_argument("--rays", type=Path, help="npz with seed_xyz, direction_xyz, seed_winding instead of a checkpoint")
    parser.add_argument("--z-range", type=int, nargs=2, metavar=("LO", "HI"))
    parser.add_argument("--winding-range", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument("--winding-step", type=int, default=8, help="seed every Nth fitted winding")
    parser.add_argument("--seed-spacing", type=int, default=96, help="voxels between seeds along a sheet and in z")
    parser.add_argument("--columns", type=int, default=1, help="rays kept per transverse axis of a slab (odd)")
    parser.add_argument("--column-spacing", type=int, default=32, help="samples between the rays kept from one slab")
    parser.add_argument("--edge-margin", type=int, default=8, help="samples ignored at each end of a ray")
    parser.add_argument("--edge-trim", type=int, default=2, help="crossings dropped at each end of a ray")
    parser.add_argument("--outside-run", type=int, default=4,
                        help="drop a ray that reads this many zero voxels in a row: it leaves the masked scan")
    parser.add_argument("--step", type=float,
                        help="voxels between samples (default: the model's, 1). The model was trained on 9.6 um "
                             "voxels, so an 8.64 um scan is shown to it at its own scale with --step 1.111")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--extract-threads", type=int, default=4)
    parser.add_argument("--max-slabs", type=int, help="use this many evenly spread seeds (for timing)")
    args = parser.parse_args()
    if not args.rays and not (args.checkpoint and args.umbilicus):
        parser.error("give --checkpoint and --umbilicus, or --rays")
    if args.out.exists():
        parser.error(f"{args.out} already exists")

    import torch
    from vesuvius.neural_tracing.winding_models.volume_slab_extractor import VolumeSlabExtractor
    from vesuvius.neural_tracing.winding_models.winding_model import WindingModel

    partial = Path(str(args.out) + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) for key, value in vars(args).items() if key not in ("batch_size", "extract_threads")}
    settings_path = partial / "arguments.json"
    rays_path = partial / "seed_rays.npz"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            parser.error(f"{partial} was started with other arguments; delete it or repeat them")
        rays = dict(np.load(rays_path))
        print(f"resuming with the {len(rays['seed_xyz']):,} seeds of {rays_path}", flush=True)
    else:
        rays = seed_rays(args)
        np.savez(rays_path, **rays)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    total = len(rays["seed_xyz"])

    device = torch.device("cuda:0")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = WindingModel(config.get("model"))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    ray_length = int(config.get("ray_length", 384))
    transverse = int(config.get("transverse_size", 128))
    spacing = float(args.step or config.get("spacing", 1.0))
    stride = int(config.get("column_stride", 4))
    columns = column_indices(transverse // stride, stride, args.columns, args.column_spacing)
    extractor = VolumeSlabExtractor(
        [args.volume],
        transverse_size=transverse,
        ray_length=ray_length,
        spacing=spacing,
        sampling=str(config.get("sampling", "trilinear")),
        tile_size=int(config.get("tile_size", 64)),
    )
    geometry = (ray_length, spacing, stride, columns)

    print(f"{total:,} slabs, {args.columns**2} ray(s) each, batch {args.batch_size}", flush=True)
    parts = []
    for begin in range(0, total, PART_SLABS):
        end = min(begin + PART_SLABS, total)
        path = partial / f"part_{begin:07d}_{end:07d}.npz"
        parts.append(path)
        if path.exists():
            continue
        started = time.monotonic()
        result = decode_part(range(begin, end), rays, extractor, model, device, args, geometry)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, **result)
        temporary.rename(path)
        print(
            f"{time.strftime('%H:%M:%S')} {end:,}/{total:,} slabs  "
            f"{(end - begin) / (time.monotonic() - started):.2f} slabs/s  "
            f"(scan {result['extract_seconds_per_slab']:.2f} s per slab over {args.extract_threads} threads, "
            f"model {result['model_seconds_per_slab']:.2f} s)  "
            f"{len(result['seed_winding']):,} rays, {len(result['crossing_t']):,} crossings in this part  "
            f"GPU peak {torch.cuda.max_memory_allocated(device) / 2**20:,.0f} MiB",
            flush=True,
        )

    merged, counters = merge_parts(parts)
    manifest = save_store(
        partial,
        merged,
        {
            "producer": "windcon experiments/winding_inference/build_store.py",
            "model_checkpoint": args.model.name,
            "model_checkpoint_sha256": file_digest(args.model),
            "volume": (args.volume / "SOURCE").read_text().strip(),
            "seed_checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "seed_rays": str(args.rays) if args.rays else None,
            "z_range": args.z_range,
            "winding_range": args.winding_range,
            "winding_step": args.winding_step,
            "seed_spacing": args.seed_spacing,
            "num_slabs": total,
            "columns_per_axis": args.columns,
            "column_spacing": args.column_spacing,
            "anchor_sample": round((ray_length - 1) / 2.0),
            "edge_margin": args.edge_margin,
            "edge_trim": args.edge_trim,
            "outside_run": args.outside_run,
            "sample_step_voxels": spacing,
            **counters,
        },
    )
    for path in parts:
        path.unlink()
    shutil.move(str(partial), str(args.out))
    print(f"{args.out}: {manifest['num_rays']:,} rays, {manifest['num_crossings']:,} crossings", flush=True)


if __name__ == "__main__":
    main()
