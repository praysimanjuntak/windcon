#!/usr/bin/env python3
"""Run villa's fit_spiral.py with overridden satisfaction tolerances (diagnostics only).

    SAT_SCAN_TOL=1e9 python eval_relaxed.py --dataset <root>   # winding test only, no scan distance

The fitter's satisfaction of a point requires both |shifted-radius residual| <= SAT_SPIRAL_TOL * dr
(default 0.45: the point sits on its strip's winding) and a scan-space distance <= SAT_SCAN_TOL
voxels (default 6) to that winding's fitted surface. Overriding one separates the two failure
modes. Same environment variables as fit_spiral.py otherwise (FIT_SPIRAL_RESUME_PATH, ...).
"""

import os
import runpy
import sys
from pathlib import Path

SPIRAL = Path(os.environ.get("POSTEL", Path.home() / "Documents/postel/vesuvius_challenge")) \
    / "villa" / "spiral-fitting"
sys.path.insert(0, str(SPIRAL))
os.chdir(SPIRAL)

import satisfaction_metrics  # noqa: E402

for env, key in (("SAT_SCAN_TOL", "satisfaction_distance_tolerance"),
                 ("SAT_SPIRAL_TOL", "satisfaction_radius_tolerance")):
    if env in os.environ:
        satisfaction_metrics.metrics_config[key] = float(os.environ[env])
print("satisfaction tolerances:", {k: v for k, v in satisfaction_metrics.metrics_config.items()
                                   if k.startswith("satisfaction_")}, flush=True)
sys.argv[0] = str(SPIRAL / "fit_spiral.py")
runpy.run_path(str(SPIRAL / "fit_spiral.py"), run_name="__main__")
