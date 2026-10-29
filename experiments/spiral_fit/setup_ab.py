#!/usr/bin/env python3
"""Lay out dataset roots and fit configurations for the spiral-fit A/B experiment.

Run on the GPU machine after fetch_dataset.py, trace_window.py, prepare_pcls.py and build_chains.py:
    python3 setup_ab.py --postel ~/Documents/postel/vesuvius_challenge

Regime = <family>_<variant>. Families:
  sparse    no human annotations: umbilicus, outer shell, normals, winding inference only
  full      plus every human input in the window except the held-out arcs
  fullnodt  full without the strip DT snap and with 10x unattached strips per step
  few       fullnodt with a random 10 % of the verified patches; the other 90 % are held out
            (datasets/eval_few scores them)
Variants (automatic paths are fed as drawn_control_points.json, their own sampling group):
  A  no automatic paths
  B  raw paths of 18 slices (pcls/auto.json), fitter defaults
  C  B without the strip DT snap;  D  C with 10x unattached strips per step
  E  D settings with multi-slice chains (smoothed, shifted onto the sheet face, linked through z)
  F  D settings with the same smoothed, shifted paths left unlinked
  G  like E (full family) but without the points that lie on verified patches
  H  like F but not shifted (pcls/auto_singles_noshift.json: build_chains.py --delta 0)
  I  like E (full family) but with the fitter's default sampling budget
  J, K  E with the strip loss weight 3x (6.0) and 10x (20.0)
  L, M  A and E without the neural winding-inference rays
  N  E plus automatic relative windings (pcls/auto_relwind.json, build_relwind.py) in the
     relative-windings slot, each file with E's strip budget
  O  the automatic relative windings alone;  P  N without the winding-inference rays
  Q  O without the winding-inference rays
  Eg, Gg, Ig  E, G and I with the gap-filled path set (pcls_g/: every 5th slice, including the
     20 slices within 25 of the held-out arcs that rounds 2-3 left out)
  few family: A, Ig (gap-filled chains); rand, targ, orac (471 extra patches, select_patches.py);
     AR10, oracR10 (10-voxel deformation lattice); ALR, oracLR (fine lattice at the coarse rate);
     ANW, oracNW (no winding inference); AP, oracP (no human point collections);
     ADT, oracDT (patch snap from step 10,000); ASD, oracSD (regulariser at a quarter);
     ARW, oracRW (automatic relative windings as drawn_control_points.json)
  fullnodt_ADT, fullnodt_IgDT  the early patch snap with every patch (no snap without patches)
  fullnodt_ARW  the automatic relative windings with every patch
RW variants keep the human relative windings in their slot; the automatic pairs go in the drawn slot,
a sampling stratum of their own (pairs on two patches feed the patch relative-winding loss, the
rest the unattached-strip loss with their winding difference).
Full-family E, G, Eg and Gg give the paths a budget of their own (sample_shares.py --emit): run this
script, then sample_shares.py with configs/<run>_base.json, then this script again.
Evaluation root `datasets/eval` holds only what no fit trains on (held-out same-winding arcs,
the published eval_fibers) plus the inputs held out in the sparse family (verified patches,
relative windings); `datasets/eval_few` swaps in the few family's held-out patches;
`datasets/diag` adds the raw automatic paths.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SCROLL_SPEC = {
    "schema_version": 1, "name": "PHercParis4", "voxel_size_um": 9.6, "spiral_outward_sense": "CW",
    "normal_zarr_group": "4", "lasagna_scale": 4, "base_shape_zyx": [18946, 8174, 8174],
}
BASE = {
    "z_begin": 9900, "z_end": 10900, "optimizer_num_training_steps": 30000,
    "input_use_tracks": False, "input_use_fibers": False, "input_use_fiber_directions": False,
    "input_use_normals": True, "input_use_gradient_magnitude": False,
    "input_use_winding_inference": True, "input_use_outer_shell": True,
}
HUMAN = ("input_use_verified_patches", "input_use_pcl_absolute", "input_use_pcl_relative",
         "input_use_pcl_same_winding")
AUTO_FILES = {  # relative to --postel
    "auto": "pcls/auto.json",
    "chains": "pcls/auto_chains.json",
    "singles": "pcls/auto_singles.json",
    "singles_noshift": "pcls/auto_singles_noshift.json",
    "chains_free": "pcls/auto_chains_free.json",
    "relwind": "pcls/auto_relwind.json",
    "chains_g": "pcls_g/auto_chains.json",
    "chains_free_g": "pcls_g/auto_chains_free.json",
    "patches_few": "data/PHercParis4/verified_patches_few10",
    "patches_rand": "data/PHercParis4/verified_patches_few10_rand471",
    "patches_targ": "data/PHercParis4/verified_patches_few10_targ471",
    "patches_orac": "data/PHercParis4/verified_patches_few10_orac471",
}
FEW_FRACTION, FEW_SEED = 0.10, 0
FEW_LINKS = {"verified_patches": "patches_few"}
NO_STRIP_DT = {"loss_weight_unattached_pcl_dt": 0.0}
# The default 84 unattached strips per step at the 9,500-slice reference scales to round(8.8) = 9
# for this 1,000-slice window; 840 scales to 88.
MORE_STRIPS = {"sample_count_unattached_pcls_per_step": 840}
D_SETTINGS = NO_STRIP_DT | MORE_STRIPS
NO_WINDING_INFERENCE = {"input_use_winding_inference": False}
# Automatic relative windings in the relative-windings slot; two strip files share the strip
# draws equally, so 1,680 keeps the chains' 840.
RELWIND = {"input_use_pcl_relative": True}
RELWIND_LINKS = {"relative_windings.json": "relwind"}
TWO_STRIP_FILES = NO_STRIP_DT | {"sample_count_unattached_pcls_per_step": 1680}
FINE_LATTICE = {"model_flow_voxel_resolution": 10}  # default 16 voxels, about one winding pitch
FAST_FINE = {"model_flow_field_high_res_lr_scale_initial": 1.0,  # default 0.2
             "model_flow_field_high_res_lr_scale_final": 1.0}
# Patches as the only human input: no human point collections (relative, absolute, same-winding).
NO_HUMAN_PCLS = {"input_use_pcl_absolute": False, "input_use_pcl_relative": False,
                 "input_use_pcl_same_winding": False}
EARLY_SNAP = {"loss_start_patch_dt": 10000}  # patch snap onto whole windings from step 10,000 (default 25,000)
SOFT_DEFORM = {"loss_weight_sym_dirichlet": 2.5}  # deformation regulariser at a quarter (default 10)
# regime -> (human inputs, automatic path file key or None, extra overrides, own budget,
#            [other automatic files: dataset file name -> AUTO_FILES key])
REGIMES = {
    "sparse_A": (False, None, {}, False),
    "sparse_B": (False, "auto", {}, False),
    "sparse_C": (False, "auto", NO_STRIP_DT, False),
    "sparse_D": (False, "auto", D_SETTINGS, False),
    "sparse_E": (False, "chains", D_SETTINGS, False),
    "sparse_F": (False, "singles", D_SETTINGS, False),
    "sparse_H": (False, "singles_noshift", D_SETTINGS, False),
    "sparse_J": (False, "chains", D_SETTINGS | {"loss_weight_unattached_pcl_radius": 6.0}, False),
    "sparse_K": (False, "chains", D_SETTINGS | {"loss_weight_unattached_pcl_radius": 20.0}, False),
    "sparse_L": (False, None, NO_WINDING_INFERENCE, False),
    "sparse_M": (False, "chains", D_SETTINGS | NO_WINDING_INFERENCE, False),
    "sparse_N": (False, "chains", TWO_STRIP_FILES | RELWIND, False, RELWIND_LINKS),
    "sparse_O": (False, None, D_SETTINGS | RELWIND, False, RELWIND_LINKS),
    "sparse_P": (False, "chains", TWO_STRIP_FILES | RELWIND | NO_WINDING_INFERENCE, False, RELWIND_LINKS),
    "sparse_Q": (False, None, D_SETTINGS | RELWIND | NO_WINDING_INFERENCE, False, RELWIND_LINKS),
    "sparse_Eg": (False, "chains_g", D_SETTINGS, False),
    "full_A": (True, None, {}, False),
    "full_B": (True, "auto", {}, False),
    "fullnodt_A": (True, None, D_SETTINGS, False),
    "fullnodt_D": (True, "auto", D_SETTINGS, False),
    "fullnodt_E": (True, "chains", D_SETTINGS, True),
    "fullnodt_G": (True, "chains_free", D_SETTINGS, True),
    "fullnodt_I": (True, "chains", D_SETTINGS, False),
    "fullnodt_Eg": (True, "chains_g", D_SETTINGS, True),
    "fullnodt_Gg": (True, "chains_free_g", D_SETTINGS, True),
    "fullnodt_Ig": (True, "chains_g", D_SETTINGS, False),
    "fullnodt_ADT": (True, None, D_SETTINGS | EARLY_SNAP, False),
    "fullnodt_IgDT": (True, "chains_g", D_SETTINGS | EARLY_SNAP, False),
    "fullnodt_ARW": (True, "relwind", D_SETTINGS, False),
    "few_A": (True, None, D_SETTINGS, False, FEW_LINKS),
    "few_Ig": (True, "chains_g", D_SETTINGS, False, FEW_LINKS),
    "few_ARW": (True, "relwind", D_SETTINGS, False, FEW_LINKS),
    "few_oracRW": (True, "relwind", D_SETTINGS, False, {"verified_patches": "patches_orac"}),
    # Annotation targeting (select_patches.py): the few patches plus 471 more, chosen at random,
    # where the chains disagree with few_A_s1, or where few_A_s1 fails (oracle); scored on eval_al.
    "few_rand": (True, None, D_SETTINGS, False, {"verified_patches": "patches_rand"}),
    "few_targ": (True, None, D_SETTINGS, False, {"verified_patches": "patches_targ"}),
    "few_orac": (True, None, D_SETTINGS, False, {"verified_patches": "patches_orac"}),
    # Fitter flexibility (does the oracle arm stop hurting?): a 10-voxel instead of a 16-voxel
    # deformation lattice (R10), or the fine lattice learning at the coarse one's rate (LR).
    "few_AR10": (True, None, D_SETTINGS | FINE_LATTICE, False, FEW_LINKS),
    "few_oracR10": (True, None, D_SETTINGS | FINE_LATTICE, False, {"verified_patches": "patches_orac"}),
    "few_ALR": (True, None, D_SETTINGS | FAST_FINE, False, FEW_LINKS),
    "few_oracLR": (True, None, D_SETTINGS | FAST_FINE, False, {"verified_patches": "patches_orac"}),
    # Does the winding inference conflict with the oracle patches? Both fitted without it (NW).
    "few_ANW": (True, None, D_SETTINGS | NO_WINDING_INFERENCE, False, FEW_LINKS),
    "few_oracNW": (True, None, D_SETTINGS | NO_WINDING_INFERENCE, False, {"verified_patches": "patches_orac"}),
    # Do the human point collections conflict with the oracle patches? Both fitted without them (P).
    "few_AP": (True, None, D_SETTINGS | NO_HUMAN_PCLS, False, FEW_LINKS),
    "few_oracP": (True, None, D_SETTINGS | NO_HUMAN_PCLS, False, {"verified_patches": "patches_orac"}),
    # Is the fit too stiff to place neighbouring patches? An earlier patch snap (DT), or a softer
    # deformation regulariser (SD).
    "few_ADT": (True, None, D_SETTINGS | EARLY_SNAP, False, FEW_LINKS),
    "few_oracDT": (True, None, D_SETTINGS | EARLY_SNAP, False, {"verified_patches": "patches_orac"}),
    "few_ASD": (True, None, D_SETTINGS | SOFT_DEFORM, False, FEW_LINKS),
    "few_oracSD": (True, None, D_SETTINGS | SOFT_DEFORM, False, {"verified_patches": "patches_orac"}),
}
SEEDS = (1, 2, 3, 4)


def split_patches(data: Path, fraction: float, seed: int) -> tuple[Path, Path]:
    """Random split of the verified patches into a training subset and the held-out rest, as
    folders of symlinks (created once, so every fit sees the same split)."""
    train = data / f"verified_patches_few{round(100 * fraction)}"
    rest = data / f"verified_patches_rest{round(100 * (1 - fraction))}"
    if not train.exists():
        names = sorted(p.name for p in (data / "verified_patches").iterdir())
        chosen = set(random.Random(seed).sample(names, round(fraction * len(names))))
        for folder, keep in ((train, True), (rest, False)):
            tmp = folder.with_name(folder.name + ".part")
            tmp.mkdir()
            for name in names:
                if (name in chosen) == keep:
                    (tmp / name).symlink_to(data / "verified_patches" / name)
            tmp.rename(folder)
    return train, rest


def toggles(human: bool, auto: bool) -> dict:
    t = {k: human for k in HUMAN}
    t["input_use_pcl_drawn_control_points"] = auto
    return t


EVAL = {
    "patches": {**toggles(False, False), "input_use_verified_patches": True},
    "pcls": {**toggles(False, False), "input_use_pcl_relative": True,
             "input_use_pcl_same_winding": True, "input_use_fibers": True},
}
# Diagnostics (datasets/diag): the eval point collections plus the raw automatic paths.
EVAL["diag"] = EVAL["pcls"] | {"input_use_pcl_drawn_control_points": True}
# Path agreement (datasets/agree_<file key>): only automatic paths, no patches to attach to, so
# every path or chain is scored as one unattached strip.
EVAL["agree"] = toggles(False, True)
AGREE_KEYS = ("singles", "chains", "relwind")


def make_root(root: Path, links: dict[str, Path]) -> bool:
    """Create a dataset root of symlinks; an existing root is kept (a fit may be reading it).
    Returns False when an input does not exist yet."""
    if root.exists():
        return True
    missing = [t for t in links.values() if not t.exists()]
    if missing:
        print(f"skip {root.name}: missing {', '.join(str(m) for m in missing)}")
        return False
    root.mkdir(parents=True)
    for name, target in links.items():
        (root / name).symlink_to(target)
    (root / "spiral-scroll.json").write_text(json.dumps(SCROLL_SPEC))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--postel", required=True, type=Path,
                    help="experiment root: data/, pcls/, datasets/, configs/ (a second window has its own)")
    ap.add_argument("--z-begin", type=int, default=BASE["z_begin"])
    ap.add_argument("--z-end", type=int, default=BASE["z_end"])
    args = ap.parse_args()
    P = args.postel.expanduser()
    base = BASE | {"z_begin": args.z_begin, "z_end": args.z_end}
    data, pcls, cfg_dir = P / "data" / "PHercParis4", P / "pcls", P / "configs"
    cfg_dir.mkdir(exist_ok=True)
    common = {n: data / n for n in ("umbilicus.json", "outer_shell", "lasagna_inputs", "winding_inference")}
    human = {"verified_patches": data / "verified_patches",
             "relative_windings.json": data / "relative_windings.json",
             "abs_winding.json": data / "abs_winding.json",
             "same_windings.json": pcls / "human_train.json"}
    eval_inputs = common | {
        "verified_patches": data / "verified_patches",
        "relative_windings.json": data / "relative_windings.json",
        "same_windings.json": pcls / "heldout_same.json",
        "fibers": data / "eval_fibers",
    }
    make_root(P / "datasets" / "eval", eval_inputs)
    if (data / "verified_patches").is_dir():
        _, few_rest = split_patches(data, FEW_FRACTION, FEW_SEED)
        make_root(P / "datasets" / "eval_few", eval_inputs | {"verified_patches": few_rest})
    if (data / "verified_patches_eval_al").is_dir():  # select_patches.py's common held-out set
        make_root(P / "datasets" / "eval_al", eval_inputs | {"verified_patches": data / "verified_patches_eval_al"})
    make_root(P / "datasets" / "diag", eval_inputs | {"drawn_control_points.json": P / AUTO_FILES["auto"]})
    for key in AGREE_KEYS:
        make_root(P / "datasets" / f"agree_{key}", common | {"drawn_control_points.json": P / AUTO_FILES[key]})

    for regime, (use_human, auto_key, extra, budget, *other) in REGIMES.items():
        links = common | (human if use_human else {})
        if auto_key:
            links["drawn_control_points.json"] = P / AUTO_FILES[auto_key]
        for name, key in (other[0] if other else {}).items():
            links[name] = P / AUTO_FILES[key]
        if not make_root(P / "datasets" / regime, links):
            continue
        budget_file = cfg_dir / f"own_budget_{regime}.json"
        overrides = json.loads(budget_file.read_text()) if budget and budget_file.exists() else {}
        for seed in SEEDS:
            run = f"{regime}_s{seed}"
            # Scoring resumes the checkpoint, so the model's shape settings must match training.
            seeded = base | {"optimizer_random_seed": seed} | {k: v for k, v in extra.items() if k.startswith("model_")}
            config = seeded | toggles(use_human, auto_key is not None) | extra
            if budget:
                (cfg_dir / f"{run}_base.json").write_text(json.dumps(config))
                if not overrides:
                    print(f"{run}: run sample_shares.py --emit {budget_file} with {run}_base.json first")
                    continue
            (cfg_dir / f"{run}.json").write_text(json.dumps(config | overrides))
            for kind, t in EVAL.items():
                (cfg_dir / f"{run}_eval_{kind}.json").write_text(json.dumps(seeded | t))
    print("roots:", sorted(p.name for p in (P / "datasets").iterdir()))


if __name__ == "__main__":
    main()
