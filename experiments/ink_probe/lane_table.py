#!/usr/bin/env python3
"""Every scored window of a lane beside the score of the other face at the same place, best first.

Ink is on one face of a sheet. On PHerc. 343's letters all seven windows at the PHerc. 0306B triage's cut
(row score 4) have the other face of the same window at 3.1 or less; on an unread scroll, windows where both
faces score at one place have so far been cracks and holes. This lists the windows of a lane written by
`lane_layout.py` and scored by `triage_lane.sh`, and marks those where only one face scores.

    lane_table.py <lane folder> [--cut 4] [--other 3.2] [--out table.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FACES = {"rev": "fwd", "fwd": "rev"}


def windows(lane: Path, file: str) -> dict[tuple[str, str, int], float]:
    """(winding, face, x0) -> score, from every <lane>/maps/<winding>/<file>."""
    table = {}
    for path in sorted((lane / "maps").glob(f"*/{file}")):
        report = json.loads(path.read_text())
        for face in FACES:
            for x0, score in report.get(face, []):
                if score is not None:
                    table[(path.parent.name, face, int(x0))] = float(score)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("lane", type=Path)
    ap.add_argument("--cut", type=float, default=4.0, help="row score a window needs to be listed")
    ap.add_argument("--other", type=float, default=3.2, help="the other face must stay under this to count as quiet")
    ap.add_argument("--out", type=Path, help="write the listed windows here as JSON")
    args = ap.parse_args()

    rows, glyphs = windows(args.lane, "spectral_win.json"), windows(args.lane, "glyph_win.json")
    if not rows:
        raise SystemExit(f"no spectral_win.json under {args.lane / 'maps'}")
    names = sorted({name for name, _, _ in rows})
    print(f"{len(names)} windings scored ({names[0]} .. {names[-1]}), {len(rows)} windows")
    for face in FACES:
        scores = sorted(score for (_, f, _), score in rows.items() if f == face)
        if scores:
            print(
                f"  {face}: median {scores[len(scores) // 2]:.2f}, max {scores[-1]:.2f}, "
                f"at {args.cut:g} or more: {sum(s >= args.cut for s in scores)} of {len(scores)}"
            )
    listed = []
    print("row score  winding face x0      other face  glyph (other face)")
    for (name, face, x0), score in sorted(rows.items(), key=lambda item: -item[1]):
        if score < args.cut:
            break
        other = rows.get((name, FACES[face], x0))
        glyph, other_glyph = glyphs.get((name, face, x0)), glyphs.get((name, FACES[face], x0))
        one_face = other is not None and other < args.other
        listed.append(
            {"winding": name, "face": face, "x0": x0, "row_score": score, "other_face": other,
             "glyph": glyph, "other_face_glyph": other_glyph, "one_face_only": one_face}
        )
        print(
            f"  {score:6.2f}  {name} {face} x{x0:<6d} "
            f"{'' if other is None else format(other, '6.2f'):>6}      "
            f"{'' if glyph is None else format(glyph, '.2f')} ({'' if other_glyph is None else format(other_glyph, '.2f')})"
            + ("  <- one face only" if one_face else "")
        )
    print(f"{sum(row['one_face_only'] for row in listed)} of {len(listed)} listed windows score on one face only")
    if args.out:
        args.out.write_text(json.dumps({"cut": args.cut, "other_face_under": args.other, "windows": listed}, indent=1) + "\n")


if __name__ == "__main__":
    main()
