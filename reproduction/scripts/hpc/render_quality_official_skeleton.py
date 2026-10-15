#!/usr/bin/env python3
"""Render saved DiP/FM planner trajectories with CLoSD's own plot_3d_motion."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
from moviepy.editor import clips_array

from closd.diffusion_planner.data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
from closd.diffusion_planner.data_loaders.humanml.utils.plot_script import plot_3d_motion


def inputs(cases_path, source):
    manifest = json.loads(Path(cases_path).read_text())
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    provenance = json.loads((source / "provenance.json").read_text())
    case_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    if len(case_dirs) != provenance["cases_requested"]:
        raise ValueError("source case count does not match its provenance")
    result = []
    for case_dir in case_dirs:
        if case_dir.name not in by_id:
            raise ValueError("case is missing from captured manifest: {}".format(case_dir.name))
        trajectories = {}
        for model in ("dip", "fm"):
            with np.load(case_dir / (model + "_trajectory.npz"), allow_pickle=False) as data:
                xyz = data["xyz_local_m"].copy()
            if xyz.shape != (60, 22, 3) or not np.isfinite(xyz).all():
                raise ValueError("invalid recovered trajectory: {} {}".format(case_dir.name, model))
            trajectories[model] = xyz
        if not np.allclose(trajectories["dip"][:20], trajectories["fm"][:20], atol=1e-4):
            raise ValueError("model prefixes differ: {}".format(case_dir.name))
        with np.load(Path(cases_path).parent / "cases" / (case_dir.name + ".npz"),
                     allow_pickle=False) as data:
            target_cond = data["target_cond"].copy()
        if target_cond.shape != (8, 3):
            raise ValueError("invalid captured goal condition")
        result.append((by_id[case_dir.name], trajectories, target_cond))
    return result


def render_case(meta, trajectories, target_cond, output, frames=60, make_gif=False):
    output.mkdir()
    duration = frames / 20
    clips = []
    try:
        for label in ("dip", "fm"):
            # The official function changes its input condition in place.
            cond = {"target_cond": torch.from_numpy(target_cond.copy()),
                    "joint_names": list(meta["target_joint_names"]),
                    "is_heading": bool(meta["is_heading"])}
            title = "{} | {} | {}".format(label.upper(), meta["task"], meta["text"])
            clip = plot_3d_motion(str(output / (label + "_official.mp4")),
                                  t2m_kinematic_chain,
                                  trajectories[label][:frames].copy(), title,
                                  dataset="humanml", fps=20,
                                  gt_frames=np.arange(min(20, frames - 1)), cond=cond)
            clip.duration = duration
            clips.append(clip)
            clip.write_videofile(str(output / (label + "_official.mp4")), fps=20,
                                 audio=False, threads=2, logger=None)
        pair = clips_array([clips])
        pair.duration = duration
        path = output / "side_by_side_official.mp4"
        pair.write_videofile(str(path), fps=20, audio=False, threads=2, logger=None)
        pair.close()
        if make_gif or meta["difficulty"] == "hard" or frames < 60:
            with imageio.get_reader(str(path)) as reader:
                images = [reader.get_data(index) for index in range(frames)]
            imageio.mimsave(str(output / "side_by_side_official.gif"), images,
                            duration=1 / 20)
    finally:
        for clip in clips:
            clip.close()
    for movie in ("dip_official.mp4", "fm_official.mp4", "side_by_side_official.mp4"):
        path = output / movie
        if path.stat().st_size < 1024:
            raise ValueError("official skeleton movie is empty: {}".format(path))
        with imageio.get_reader(str(path)) as reader:
            if reader.get_data(0).ndim != 3:
                raise ValueError("official skeleton movie cannot be decoded: {}".format(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="completed smoke_<jobid> or results directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="new, separate directory; original videos are preserved")
    parser.add_argument("--frames", type=int, default=60,
                        help="1-60; values below 60 are a preview only")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= 60:
        parser.error("--frames must be between 1 and 60")
    cases = inputs(args.cases, args.source)
    if args.output.exists():
        parser.error("output exists; refusing to overwrite: {}".format(args.output))
    print("OFFICIAL_SKELETON_PREFLIGHT cases={} frames={}".format(
        len(cases), args.frames), flush=True)
    if args.check:
        return 0
    args.output.mkdir(parents=True)
    rows = []
    for meta, trajectories, target_cond in cases:
        case_dir = args.output / meta["case_id"]
        render_case(meta, trajectories, target_cond, case_dir, args.frames,
                    make_gif=len(cases) == 1)
        rows.append({"case_id": meta["case_id"], "task": meta["task"],
                     "frames": args.frames,
                     "dip_mp4": str((case_dir / "dip_official.mp4").relative_to(args.output)),
                     "fm_mp4": str((case_dir / "fm_official.mp4").relative_to(args.output)),
                     "comparison_mp4": str((case_dir / "side_by_side_official.mp4").relative_to(args.output)),
                     "comparison_gif": (str((case_dir / "side_by_side_official.gif").relative_to(args.output))
                                        if (case_dir / "side_by_side_official.gif").exists() else "")})
        print("OFFICIAL_SKELETON_CASE_COMPLETE={}".format(meta["case_id"]), flush=True)
    with (args.output / "index.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "README.md").write_text(
        "# Official CLoSD skeleton renderer\n\n"
        "Rendered from the saved planner trajectories with the repository's "
        "plot_3d_motion; no controller execution or new sampling. "
        "The original fixed-camera diagnostic videos remain in the source directory.\n"
    )
    print("OFFICIAL_SKELETON_COMPLETE=1 OUTPUT={}".format(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
