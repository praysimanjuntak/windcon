"""Stand-in for volume-cartographer's Python module `vc`, as far as villa's winding model uses it.

villa's `VolumeSlabExtractor` reads CT through `vc`, a C++ extension that is built together with
volume-cartographer and is not on PyPI. This module offers the same calls on top of windcon's
chunk cache, so the winding model runs wherever windcon does. `Volume.open` takes the folder that
`windcon.zarr_io.ZarrArray` keeps for one zarr level (its SOURCE file names the level's URL). A
chunk that is not in the folder yet is fetched on first use.

Sampling follows `ChunkedPlaneSampler` at villa 5a4388f: a coordinate is (x, y, z) in voxels of
the level with voxel centres on whole numbers, the value is trilinear between the eight
neighbours and truncated to uint8, a sample outside the level reads 0, and every sample counts as
valid. One difference: within one voxel of a chunk the store does not have, the C++ sampler
returns 0 and this one interpolates towards 0.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from windcon.zarr_io import ZarrArray

__all__ = ["Volume", "set_chunk_cache_budget", "set_chunk_cache_io_threads"]

_OPENED: dict[str, Volume] = {}
_OPENED_LOCK = threading.Lock()


def set_chunk_cache_budget(nbytes: int) -> None:
    """Nothing to set: chunk files stay in the operating system's page cache."""


def set_chunk_cache_io_threads(count: int) -> None:
    """Nothing to set: `ZarrArray` has its own fetch pool."""


class Volume:
    def __init__(self, array: ZarrArray):
        self._array = array
        self._read_lock = threading.Lock()  # two threads must not fetch the same chunk at once

    @staticmethod
    def open(path) -> Volume:
        """The volume of a chunk cache folder; every caller in the process shares one per folder."""
        folder = Path(path).resolve()
        with _OPENED_LOCK:
            volume = _OPENED.get(str(folder))
            if volume is None:
                source = folder / "SOURCE"
                if not source.is_file():
                    raise FileNotFoundError(f"{folder} is not a windcon chunk cache folder (it has no SOURCE file)")
                volume = Volume(ZarrArray(source.read_text().strip(), cache_dir=folder.parent))
                _OPENED[str(folder)] = volume
        return volume

    def sample_planes(
        self, origins_xyz, x_steps_xyz, y_steps_xyz, shape, level=0, sampling="trilinear", tile_size=32
    ):
        """uint8 images [N, H, W], validity [N, H, W] and a stats dict for N planes.

        Pixel (row, column) of plane n lies at origins[n] + row * y_steps[n] + column * x_steps[n].
        """
        if level != 0:
            raise ValueError("open the folder of the level you want; this stand-in has no pyramid")
        height, width = (int(v) for v in shape)
        origins = torch.from_numpy(np.ascontiguousarray(origins_xyz, dtype=np.float32))
        x_steps = torch.from_numpy(np.ascontiguousarray(x_steps_xyz, dtype=np.float32))
        y_steps = torch.from_numpy(np.ascontiguousarray(y_steps_xyz, dtype=np.float32))
        rows = torch.arange(height, dtype=torch.float32)[None, :, None, None]
        columns = torch.arange(width, dtype=torch.float32)[None, None, :, None]
        coords = origins[:, None, None, :] + rows * y_steps[:, None, None, :]
        coords = coords + columns * x_steps[:, None, None, :]
        return self._sample(coords, sampling)

    def sample_coords(self, coords_xyz, mask=None, sampling="trilinear", tile_size=32):
        """uint8 values and validity for an [N, M, 3] array of points; points with a false `mask` read 0."""
        coords = torch.from_numpy(np.ascontiguousarray(coords_xyz, dtype=np.float32))
        values, valid, stats = self._sample(coords, sampling)
        if mask is not None:
            keep = np.asarray(mask, dtype=bool)
            values = np.where(keep, values, 0).astype(np.uint8)
            valid = np.where(keep, valid, 0).astype(np.uint8)
        return values, valid, stats

    def _sample(self, coords: torch.Tensor, sampling: str):
        if sampling != "trilinear":
            raise ValueError(f"only trilinear sampling is implemented, got {sampling!r}")
        # The C++ sampler skips these three kinds of coordinate and marks them invalid.
        sentinel = ~torch.isfinite(coords).all(-1) | (coords == -1.0).any(-1) | (coords == 0.0).all(-1)
        size_xyz = torch.tensor(self._array.shape[::-1], dtype=torch.float32)
        inside = ((coords >= 0) & (coords < size_xyz)).all(-1) & ~sentinel
        values = torch.zeros(coords.shape[:-1], dtype=torch.uint8)
        if bool(inside.any()):
            used = coords[inside]
            low = used.amin(0).floor().long()
            high = used.amax(0).floor().long() + 2  # half-open, with the upper neighbour of the last voxel
            with self._read_lock:
                block = self._array.read(tuple((int(low[axis]), int(high[axis])) for axis in (2, 1, 0)))
            span = (high - low - 1).to(torch.float32)
            grid = (coords - low.to(torch.float32)) / span * 2 - 1
            grid = torch.where(inside[..., None], grid, torch.full_like(grid, -2.0))
            sampled = F.grid_sample(
                torch.from_numpy(block).to(torch.float32)[None, None],
                grid.reshape(1, 1, 1, -1, 3),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            values = sampled.reshape(coords.shape[:-1]).clamp_(0, 255).to(torch.uint8)
            values[~inside] = 0
        return values.numpy(), (~sentinel).to(torch.uint8).numpy(), {}
