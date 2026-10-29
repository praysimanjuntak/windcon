#!/usr/bin/env python3
"""Per-group sampling statistics of the fitter's point-collection losses, and own-budget overrides.

Both losses average over everything drawn in a step: the unattached-strip loss over sampled points
(up to ``sample_count_unattached_pcl_points_per_step`` per drawn strip), the relative-winding loss
over patch pairs (up to ``sample_count_relative_winding_patch_pairs_per_pcl`` per drawn collection,
between consecutive distinct patches). Draws are split equally between source files (groups).

``--emit`` writes overrides that give the automatic paths (group ``drawn_control_points``) a budget
of their own: every group keeps weight 1 in ``pcl_sampling_weights`` (equal shares, as without
weights), each pool's draw count grows so the human groups keep the draws they had without the
paths, and each loss weight grows by (human + automatic) / human expected points (pairs) per step,
so the human terms keep their gradient and an automatic point counts as much as a human one.
Loads the inputs exactly as the fitter does (FitContext.load_host_inputs, CPU only).

Run from villa's spiral-fitting directory with its environment, with the regime's base config:
    FIT_SPIRAL_CONFIG_OVERRIDES="$(cat configs/fullnodt_E_s1_base.json)" \
        .venv/bin/python sample_shares.py --dataset datasets/fullnodt_E --emit configs/own_budget_fullnodt_E.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import Config, FitConfig
from fit_session import conventional_input_paths, load_scroll_spec
from fit_spiral import FitContext, get_env_config_overrides

AUTO = "drawn_control_points"
POOLS = (  # report key, draw-count key, loss-weight key
    ("unattached_points", "sample_count_unattached_pcls_per_step", "loss_weight_unattached_pcl_radius"),
    ("rel_winding_pairs", "sample_count_relative_winding_pcls", "loss_weight_rel_winding"),
)


def group(sampling_group) -> str:
    return os.path.splitext(os.path.basename(str(sampling_group)))[0]


def shares(ctx, config) -> dict:
    points_cap = config["sample_count_unattached_pcl_points_per_step"]
    pairs_cap = config["sample_count_relative_winding_patch_pairs_per_pcl"]
    per = {"unattached_points": defaultdict(list), "rel_winding_pairs": defaultdict(list)}
    for strip, sg in zip(ctx.unattached_pcl_strips, ctx.unattached_strip_sampling_groups):
        per["unattached_points"][group(sg)].append(min(len(strip["zyxs"]), points_cap))
    for pcl in ctx.cross_patch_pcls:
        if len(pcl["points"]) > 1:
            per["rel_winding_pairs"][group(pcl["sampling_group"])].append(
                min(max(len(pcl["points_by_patch"]) - 1, 0), pairs_cap))
    report = {}
    for pool, members in per.items():
        means = {g: float(np.mean(v)) for g, v in members.items()}
        total = sum(means.values())
        report[pool] = {g: {"members": len(members[g]), "per_draw": round(m, 4),
                            "share_equal_draws": round(m / total, 4) if total else 0.0}
                        for g, m in means.items()}
    return report


def own_budget(report: dict, config: dict, auto_weight: float = 1.0) -> dict:
    groups = sorted(set(report["unattached_points"]) | set(report["rel_winding_pairs"]))
    out: dict = {"pcl_sampling_weights": {g: (auto_weight if g == AUTO else 1.0) for g in groups}}
    for pool, count_key, loss_key in POOLS:
        stats = report[pool]
        human = [g for g in stats if g != AUTO]
        if AUTO not in stats or not human:
            continue
        out[count_key] = round(config[count_key] * (len(human) + auto_weight) / len(human))
        h = sum(stats[g]["per_draw"] for g in human)
        out[loss_key] = round(config[loss_key] * (h + auto_weight * stats[AUTO]["per_draw"]) / h, 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--emit", type=Path, help="write own-budget overrides here")
    args = ap.parse_args()
    config = Config().as_dict()
    config.update(get_env_config_overrides())
    scroll = load_scroll_spec(args.dataset)
    ctx = FitContext(FitConfig(config), scroll=scroll, paths=conventional_input_paths(args.dataset, scroll),
                     cache_dir=os.environ.get("FIT_SPIRAL_CACHE_DIR"))
    ctx.load_host_inputs()
    report = shares(ctx, config)
    if args.emit:
        report["own_budget_overrides"] = own_budget(report, config)
        args.emit.write_text(json.dumps(report["own_budget_overrides"], indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
