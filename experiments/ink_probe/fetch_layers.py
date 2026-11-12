#!/usr/bin/env python3
"""Write a surface volume (the layer stack rendered around a segment) as a (layer, y, x) uint8 .npy.

Two sources:
* a published surface volume: the bucket key of its OME-Zarr group, whose array `0` holds the stack
  (its voxel size is read from the group's attributes);
* a local folder of per-layer TIFFs named 00.tif, 01.tif, ... in ascending normal offset, as in
  the Hugging Face dataset YoussefMoNader/ink-8um-pherc1447-surfaces (give --native-spacing).

With --spacing the stack is also resampled linearly on all three axes to that many micrometres per
voxel: Hecate's 9.6 um checkpoint needs renders sampled at exactly 9.6 um, while the scans are 8.64
or 9.362 um.

    uv run python fetch_layers.py PHerc0800/segments/<segment>/surface-volumes/<name>.zarr \
        --out segments/PHerc0800 [--spacing 9.6]
    uv run python fetch_layers.py pherc1447/w058/layers --name w058 --native-spacing 8.64 \
        --out segments/PHerc1447 --spacing 9.6

Writes <out>/<name>.npy, and <out>/<name>_<spacing>um.npy with --spacing. With --zarr the stack
is also written at its own spacing as <out>/<name>.zarr, the layout villa's ink inference reads;
with --tifs as one TIFF per layer in <out>/<name>_layers, the layout the v8-in model reads.
--crop Y0 Y1 X0 X1 keeps only that window of the surface in every output.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from windcon.zarr_io import ZarrArray  # noqa: E402

# CloudFront mirror of the open-data bucket: same keys, far steadier than direct S3 from here.
MIRROR = "https://data.aws.ash2txt.org/samples"


def published_stack(key: str, base: str) -> tuple[str, np.ndarray, list[float]]:
    with urllib.request.urlopen(f"{base}/{key}/.zattrs", timeout=60) as resp:
        attrs = json.loads(resp.read())
    scale = next(t["scale"] for t in attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
                 if t["type"] == "scale")
    array = ZarrArray(f"{base}/{key}/0")
    return key.split("/segments/")[1].split("/")[0], array.read(tuple((0, n) for n in array.shape)), scale


def tiff_stack(folder: Path) -> np.ndarray:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return np.stack([np.asarray(Image.open(p)) for p in sorted(folder.glob("*.tif"))])


def write_zarr(path: Path, stack: np.ndarray, scale: list[float], tile: int = 128) -> None:
    """Uncompressed zarr v2 group with the stack as array `0`, chunked like the published ones."""
    chunks = (stack.shape[0], tile, tile)
    (path / "0").mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (path / ".zattrs").write_text(json.dumps({"multiscales": [{
        "version": "0.4", "axes": [{"name": n, "type": "space", "unit": "micrometer"} for n in "zyx"],
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": scale}]}]}]}))
    (path / "0" / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": list(stack.shape), "chunks": list(chunks), "dtype": "|u1",
        "compressor": None, "fill_value": 0, "order": "C", "filters": None, "dimension_separator": "/"}))
    for y in range(0, stack.shape[1], tile):
        for x in range(0, stack.shape[2], tile):
            block = stack[:, y:y + tile, x:x + tile]
            if not block.any():
                continue  # an absent chunk reads as fill_value
            chunk = np.zeros(chunks, dtype=np.uint8)
            chunk[:, :block.shape[1], :block.shape[2]] = block
            out = path / "0" / "0" / str(y // tile)
            out.mkdir(parents=True, exist_ok=True)
            (out / str(x // tile)).write_bytes(chunk.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="bucket key of a surface-volume zarr group, or a folder of layer TIFFs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", help="output name (default: the segment id of a bucket key)")
    ap.add_argument("--native-spacing", type=float, help="voxel size of a TIFF stack in micrometres")
    ap.add_argument("--spacing", type=float, help="also write the stack resampled to this voxel size")
    ap.add_argument("--zarr", action="store_true", help="also write the stack as a zarr group")
    ap.add_argument("--tifs", action="store_true",
                    help="also write <name>_layers/00.tif, 01.tif, ... (the input of the v8-in ink model)")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"),
                    help="keep only this window of the surface, in its own pixels")
    ap.add_argument("--base", default=MIRROR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if Path(args.source).is_dir():
        if not (args.name and args.native_spacing):
            raise SystemExit("a TIFF folder needs --name and --native-spacing")
        name, stack, scale = args.name, tiff_stack(Path(args.source)), [args.native_spacing] * 3
    else:
        name, stack, scale = published_stack(args.source.strip("/"), args.base)
        name = args.name or name
    if args.crop:
        y0, y1, x0, x1 = args.crop
        stack = np.ascontiguousarray(stack[:, y0:y1, x0:x1])
    np.save(args.out / f"{name}.npy", stack)
    print(f"{name}: {stack.shape} at {scale} um, {100 * np.mean(stack.max(axis=0) > 0):.0f} % of the canvas filled")
    if args.zarr:
        write_zarr(args.out / f"{name}.zarr", stack, scale)
    if args.tifs:
        from PIL import Image
        (args.out / f"{name}_layers").mkdir(exist_ok=True)
        for i, layer in enumerate(stack):
            Image.fromarray(layer).save(args.out / f"{name}_layers" / f"{i:02d}.tif")
    if args.spacing:
        from scipy.ndimage import zoom
        resampled = zoom(stack, [s / args.spacing for s in scale], order=1)
        np.save(args.out / f"{name}_{args.spacing:g}um.npy", resampled)
        print(f"  -> {name}_{args.spacing:g}um.npy: {resampled.shape}")


if __name__ == "__main__":
    main()
