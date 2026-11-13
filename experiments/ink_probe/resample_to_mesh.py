#!/usr/bin/env python3
"""Bring images rendered on one or more tifxyz meshes into the pixel frame of another mesh.

Two meshes of the same sheet have different grids, so their renders and ink maps cannot be laid
over each other. For every node of the target mesh this script finds the nearest place on each
source mesh, with its distance along the source's normal, and keeps the source that is nearest in
that direction. A pixel of the target then reads the source image at the corresponding place.

    uv run python resample_to_mesh.py --target <target.tifxyz> --source <a.tifxyz> <a map> \
        [--source <b.tifxyz> <b map> ...] --out <prefix> [--max-offset 4]

Writes <prefix>.tif, the image in the target's frame (0 where no source lies within --max-offset
voxels along the normal or within one grid cell), and <prefix>_offset.npy, the signed distance
along the normal at the target's nodes (nan where no source is near). Prints how much of the
target each source covers. A map whose size differs from its mesh's render is stretched to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None


def load_mesh(folder: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Node positions [rows, cols, 3] as (x, y, z), the mask of real nodes, and render pixels per node."""
    xyz = np.stack([np.asarray(Image.open(folder / f"{axis}.tif"), dtype=np.float64) for axis in "xyz"], axis=-1)
    scale = json.loads((folder / "meta.json").read_text())["scale"]
    return xyz, (xyz > 0).all(-1), round(1 / scale[0])


def tangents(xyz: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    """Position change per node step along `axis`, from the valid neighbours of each node."""
    ahead, behind = np.roll(xyz, -1, axis), np.roll(xyz, 1, axis)
    has_ahead, has_behind = np.roll(valid, -1, axis), np.roll(valid, 1, axis)
    edge = [slice(None)] * 2
    edge[axis] = -1
    has_ahead[tuple(edge)] = False
    edge[axis] = 0
    has_behind[tuple(edge)] = False
    steps = has_ahead.astype(float) + has_behind
    difference = np.where(has_ahead[..., None], ahead, xyz) - np.where(has_behind[..., None], behind, xyz)
    return difference / np.maximum(steps, 1)[..., None]


def correspond(target: np.ndarray, source: np.ndarray, source_valid: np.ndarray):
    """For each target point: fractional (row, column) on the source grid, signed distance along the source's
    normal, and distance to the nearest source node in grid steps."""
    by_row, by_col = tangents(source, source_valid, 0), tangents(source, source_valid, 1)
    rows, cols = np.nonzero(source_valid)
    _, index = cKDTree(source[source_valid]).query(target)
    row, col = rows[index], cols[index]
    t_row, t_col = by_row[row, col], by_col[row, col]
    normal = np.cross(t_col, t_row)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
    # target - node = a * t_row + b * t_col + d * normal, solved per point
    basis = np.stack([t_row, t_col, normal], axis=-1)
    usable = np.abs(np.linalg.det(basis)) > 1e-6
    solution = np.zeros((len(target), 3))
    solution[usable] = np.linalg.solve(basis[usable], (target - source[row, col])[usable][..., None])[..., 0]
    solution[~usable] = np.nan
    return row + solution[:, 0], col + solution[:, 1], solution[:, 2], np.hypot(solution[:, 0], solution[:, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--source", type=Path, nargs=2, action="append", required=True, metavar=("MESH", "IMAGE"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-offset", type=float, default=4.0, help="voxels along the normal a source may be away")
    args = ap.parse_args()
    target, target_valid, pixels = load_mesh(args.target)
    points = target[target_valid]
    best = np.full(len(points), np.inf)
    chosen = np.full(len(points), -1)
    place = np.full((len(points), 2), np.nan)  # row, column in the chosen source's render, pixels
    signed = np.full(len(points), np.nan)
    images = []
    for number, (mesh_path, image_path) in enumerate(args.source):
        source, source_valid, source_pixels = load_mesh(mesh_path)
        image = Image.open(image_path)
        size = (source.shape[1] * source_pixels, source.shape[0] * source_pixels)
        images.append(np.asarray(image if image.size == size else image.resize(size, Image.BILINEAR), dtype=np.float32))
        row, col, offset, cells = correspond(points, source, source_valid)
        near = (cells <= 1.0) & (np.abs(offset) <= args.max_offset)
        better = near & (np.abs(offset) < best)
        best[better], chosen[better], signed[better] = np.abs(offset[better]), number, offset[better]
        place[better] = np.stack([row[better], col[better]], -1) * source_pixels
        reached = cells <= 1.0
        print(f"{mesh_path.name}: within one grid cell of {reached.mean():.1%} of the target's nodes; along the normal "
              f"there: median {np.nanmedian(np.abs(offset[reached])) if reached.any() else float('nan'):.1f} voxels, "
              f"within {args.max_offset:g}: {np.mean(np.abs(offset[reached]) <= args.max_offset) if reached.any() else 0:.1%}")
    print(f"target nodes with a source within {args.max_offset:g} voxels: {np.mean(chosen >= 0):.1%}")

    # Node fields on the target grid, then one value per target pixel.
    grid = np.full((*target.shape[:2], 3), np.nan)
    grid[target_valid] = np.column_stack([chosen, place])
    offset_grid = np.full(target.shape[:2], np.nan)
    offset_grid[target_valid] = signed
    np.save(f"{args.out}_offset.npy", offset_grid.astype(np.float32))
    height, width = target.shape[0] * pixels, target.shape[1] * pixels
    out = np.zeros((height, width), dtype=np.float32)
    node_rows, node_cols = np.arange(height) / pixels, np.arange(width) / pixels
    for number, image in enumerate(images):
        mine = grid[..., 0] == number
        if not mine.any():
            continue
        # Interpolate the correspondence inside this source's part of the target; nodes of other sources are
        # filled from the nearest of its own so that the interpolation does not mix two grids.
        nearest = ndimage.distance_transform_edt(~mine, return_distances=False, return_indices=True)
        rows_of, cols_of = (grid[..., k][tuple(nearest)] for k in (1, 2))
        sample = np.meshgrid(node_rows, node_cols, indexing="ij")
        covered = ndimage.map_coordinates(mine.astype(np.float32), sample, order=1, mode="nearest") > 0.5
        at = [ndimage.map_coordinates(field, sample, order=1, mode="nearest") for field in (rows_of, cols_of)]
        values = ndimage.map_coordinates(image, at, order=1, mode="constant", cval=0.0)
        out[covered] = values[covered]
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(f"{args.out}.tif")
    print(f"{args.out}.tif: {height} x {width}, {np.mean(out > 0):.1%} of the pixels filled")


if __name__ == "__main__":
    main()
