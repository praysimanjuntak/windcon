"""``windcon`` command line: trace same-winding paths and evaluate them against human arcs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path as FsPath

import numpy as np

from . import evaluate, render
from .annotations import load_collections, write_same_winding
from .tracer import SliceField, TraceParams, trace_region
from .zarr_io import S3_OPEN_DATA, ZarrArray

PARIS4_L2 = f"{S3_OPEN_DATA}/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2"
DEFAULT_GT = FsPath(__file__).resolve().parents[2] / "data" / "paris4" / "same_windings.json"
DEV_SLICES = frozenset({10604, 12418})  # used for development; every other GT slice is held out


def _params(args: argparse.Namespace) -> TraceParams:
    params = TraceParams()
    if args.min_neighbor_dist is not None:
        params = replace(params, min_neighbor_dist=args.min_neighbor_dist)
    return params


def cmd_trace(args: argparse.Namespace) -> None:
    vol = ZarrArray(args.volume)
    x0, x1, y0, y1 = args.box
    params = _params(args)
    image = evaluate.read_crop(vol, args.z, x0, x1, y0, y1, args.z_half)
    paths = trace_region(SliceField(image, params), min_length=args.min_length)
    origin = np.array([x0, y0], dtype=np.float64)
    strips, stats = [], []
    for i, path in enumerate(paths):
        xy = path.resampled(args.spacing) + origin
        name = f"auto_z{args.z}_{i:04d}"
        strips.append((name, np.column_stack([xy, np.full(len(xy), float(args.z))])))
        stats.append({
            "name": name,
            "length_px": round(path.length, 1),
            "stop_reasons": list(path.stop_reasons),
            "min_neighbor_dist": float(np.nanmin(path.neighbor_dist)),
            "min_gap_contrast": float(np.nanmin(path.gap_contrast)),
            "bridged_fraction": float(np.mean(np.isnan(path.gap_contrast))),
        })
    out = FsPath(args.out)
    write_same_winding(out, strips)
    sidecar = {
        "volume": args.volume,
        "z": args.z,
        "box_x0_x1_y0_y1": [x0, x1, y0, y1],
        "z_half": args.z_half,
        "spacing_px": args.spacing,
        "params": asdict(params),
        "paths": stats,
    }
    out.with_suffix(".stats.json").write_text(json.dumps(sidecar, indent=1))
    total = sum(s["length_px"] for s in stats)
    print(f"{len(paths)} same-winding paths, {total:.0f} px total -> {out}")
    if args.png:
        render.region_png(image, (x0, y0), [p.xy + origin for p in paths], args.png,
                          title=f"z={args.z}: {len(paths)} paths")
        print(f"overlay -> {args.png}")


def cmd_eval(args: argparse.Namespace) -> None:
    vol = ZarrArray(args.volume)
    params = _params(args)
    arcs = evaluate.flat_z_arcs(load_collections(args.gt))
    splits = {
        "dev": [a for a in arcs if a.z in DEV_SLICES],
        "test": [a for a in arcs if a.z not in DEV_SLICES],
    }
    selected = list(splits) if args.split == "all" else [args.split]
    out = FsPath(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"params": asdict(params), "z_half": args.z_half, "tau_px": args.tau, "splits": {}}
    for name in selected:
        split_arcs = splits[name]
        auto = evaluate.evaluate_auto(vol, split_arcs, params, tau=args.tau, z_half=args.z_half)
        seeded = evaluate.evaluate_seeded(vol, split_arcs, params, tau=args.tau, z_half=args.z_half)
        per_slice = {
            str(z): evaluate.summarize_auto([s for s in auto if s.z == z])
            for z in sorted({s.z for s in auto})
        }
        report["splits"][name] = {
            "arcs": len(split_arcs),
            "auto": evaluate.summarize_auto(auto),
            "auto_per_slice": per_slice,
            "seeded": evaluate.summarize(seeded),
            "per_arc": [
                {"arc": s.arc, "z": s.z, "length_px": round(s.arc_length, 1), "claims": s.claims,
                 "coverage": round(s.covered / s.arc_length, 3), "correct_px": round(s.correct, 1),
                 "wrong_px": round(s.wrong, 1), "departures": s.departures}
                for s in auto
            ],
        }
        a = report["splits"][name]["auto"]
        print(f"[{name}] {len(split_arcs)} arcs  auto: precision {a['precision']:.4f}  coverage "
              f"{a['coverage']:.3f}  departures/claim {a['departure_rate']:.3f}  "
              f"({a['correct_px']:.0f} px correct, {a['wrong_px']:.0f} px wrong)")
        if args.panels:
            render.arc_panels(vol, auto, split_arcs, out / f"panels_{name}.png", z_half=args.z_half)
    (out / "metrics.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"metrics -> {out / 'metrics.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="windcon", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--volume", default=PARIS4_L2, help="zarr v2 array URL (one pyramid level)")
    common.add_argument("--z-half", type=int, default=2, help="average slices z-h..z+h (default 2)")
    common.add_argument("--min-neighbor-dist", type=float, default=None,
                        help=f"fork/merge guard in px (default {TraceParams.min_neighbor_dist}; 0 disables)")

    t = sub.add_parser("trace", parents=[common], help="trace every clean band in one slice crop")
    t.add_argument("--z", type=int, required=True)
    t.add_argument("--box", type=int, nargs=4, required=True, metavar=("X0", "X1", "Y0", "Y1"))
    t.add_argument("--min-length", type=float, default=40.0, help="drop shorter paths (px)")
    t.add_argument("--spacing", type=float, default=20.0, help="exported point spacing (px)")
    t.add_argument("--out", required=True, help="same-winding point-collection JSON to write")
    t.add_argument("--png", help="optional overlay image")
    t.set_defaults(func=cmd_trace)

    e = sub.add_parser("eval", parents=[common], help="score against human same-winding arcs")
    e.add_argument("--gt", default=str(DEFAULT_GT), help="same_windings.json with ground-truth arcs")
    e.add_argument("--split", choices=("dev", "test", "all"), default="all")
    e.add_argument("--tau", type=float, default=6.0, help="on-arc distance tolerance (px)")
    e.add_argument("--out", default="results", help="output directory")
    e.add_argument("--panels", action="store_true", help="render per-arc panels")
    e.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
