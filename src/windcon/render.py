"""Overlay images for visual verification of traced paths."""

from __future__ import annotations

from pathlib import Path as FsPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .evaluate import Arc, ArcScore, arc_box, read_crop  # noqa: E402
from .zarr_io import ZarrArray  # noqa: E402

VMIN, VMAX = 30, 150  # display window for the uint8 Paris 4 volume


def region_png(
    image: np.ndarray,
    origin_xy: tuple[int, int],
    paths_xy: list[np.ndarray],
    out: str | FsPath,
    arcs: list[Arc] = (),
    title: str = "",
) -> None:
    """Slice crop with every traced path (volume coords) in its own colour; GT arcs in yellow."""
    x0, y0 = origin_xy
    h, w = image.shape
    scale = min(1.0, 2400 / max(h, w))
    fig, ax = plt.subplots(figsize=(w * scale / 100, h * scale / 100), dpi=100)
    ax.imshow(image, cmap="gray", vmin=VMIN, vmax=VMAX, extent=(x0, x0 + w, y0 + h, y0))
    rng = np.random.default_rng(0)
    for xy in paths_xy:
        ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.2, color=plt.cm.hsv(rng.random()))
    for arc in arcs:
        ax.plot(arc.xy[:, 0], arc.xy[:, 1], "o", color="yellow", ms=2.0)
    ax.set_xlim(x0, x0 + w)
    ax.set_ylim(y0 + h, y0)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def arc_panels(
    vol: ZarrArray,
    scores: list[ArcScore],
    arcs: list[Arc],
    out: str | FsPath,
    *,
    z_half: int,
    margin: int = 128,
    ncols: int = 4,
) -> None:
    """One panel per GT arc: claiming paths in green, vertices after a departure in red."""
    by_key = {(a.name, a.z): a for a in arcs}
    nrows = int(np.ceil(len(scores) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), dpi=100)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()
    for ax, sc in zip(axes, scores):
        arc = by_key[(sc.arc, sc.z)]
        x0, x1, y0, y1 = arc_box(arc, margin)
        ax.imshow(read_crop(vol, arc.z, x0, x1, y0, y1, z_half), cmap="gray", vmin=VMIN, vmax=VMAX,
                  extent=(x0, x1, y1, y0))
        ax.plot(arc.xy[:, 0], arc.xy[:, 1], "o", color="yellow", ms=2.5)
        for xy, wrong in sc.paths:
            ax.plot(xy[:, 0], xy[:, 1], "-", color="lime", lw=1.2)
            if wrong.any():
                ax.plot(xy[wrong, 0], xy[wrong, 1], ".", color="red", ms=3)
        ax.set_title(f"{sc.arc} z={sc.z}  cov={sc.covered / sc.arc_length:.0%}  wrong={sc.wrong:.0f}px",
                     fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
