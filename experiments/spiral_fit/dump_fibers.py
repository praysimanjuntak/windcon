#!/usr/bin/env python3
"""Write the eval fibers as the fitter loads them (fitter frame, decimated) for coverage_split.py.

Run with the spiral-fitting environment (it imports villa's fiber loader):
    cd $POSTEL/villa/spiral-fitting && .venv/bin/python dump_fibers.py --fibers <dir> --out fiber_points.json
Output: {"<fiber file name>": [[x, y, z], ...]}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SPIRAL = Path(os.environ.get("POSTEL", Path.home() / "Documents/postel/vesuvius_challenge")) \
    / "villa" / "spiral-fitting"
sys.path.insert(0, str(SPIRAL))
from spiral_helpers import load_fiber_point_collections  # noqa: E402

BASE_SHAPE_ZYX = [18946, 8174, 8174]  # spiral-scroll.json of PHercParis4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fibers", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    pcls, _ = load_fiber_point_collections(str(args.fibers), 0, base_shape_zyx=BASE_SHAPE_ZYX)
    doc = {Path(c["source_file"]).name: [c["points"][k]["p"] for k in sorted(c["points"], key=int)]
           for c in pcls.values()}
    args.out.write_text(json.dumps(doc))
    print(f"{len(doc)} fibers, {sum(map(len, doc.values()))} points -> {args.out}")


if __name__ == "__main__":
    main()
