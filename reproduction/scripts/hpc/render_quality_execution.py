#!/usr/bin/env python3
"""Render real DiP/FM controller executions through CLoSD's record2anim.py."""

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
from moviepy.editor import VideoFileClip, clips_array

from closd.utils.closd_util import STATES
from reproduction.scripts.hpc.prepare_official_sim_record import load_record_cpu, prepare


TASKS = ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT", "GET_UP")


def select_episodes(dip_record, fm_record, tasks):
    common = sorted(set(dip_record) & set(fm_record),
                    key=lambda name: tuple(int(part) for part in name.split("_")))
    selection = {}
    for task in tasks:
        state = int(STATES[task])
        eligible = []
        for key in common:
            samples = (dip_record[key], fm_record[key])
            if not all("task_state" in sample for sample in samples):
                continue
            if not all(state in np.asarray(sample["task_state"]) for sample in samples):
                continue
            initial = [int(np.asarray(sample["init_state"])) for sample in samples]
            if initial[0] != initial[1]:
                continue
            eligible.append(key)
        if not eligible:
            raise ValueError("no matching real simulator episode for {} in both records".format(task))
        selection[task] = eligible[0]
    return selection


def check_movie(path):
    if not path.is_file() or path.stat().st_size < 1024:
        raise ValueError("empty movie: {}".format(path))
    with imageio.get_reader(str(path)) as reader:
        if reader.get_data(0).ndim != 3:
            raise ValueError("movie cannot be decoded: {}".format(path))


def render(args):
    dip_path = args.run_dir / "execution/dip/episodes.pkl"
    fm_path = args.run_dir / "execution/fm/episodes.pkl"
    if not dip_path.is_file() or not fm_path.is_file():
        raise FileNotFoundError("both official simulator recordings are required")
    tasks = tuple(args.tasks.split(","))
    if not tasks or any(task not in TASKS for task in tasks):
        raise ValueError("--tasks must be comma-separated official task names")
    dip_record = load_record_cpu(dip_path)
    fm_record = load_record_cpu(fm_path)
    selected = select_episodes(dip_record, fm_record, tasks)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite: {}".format(args.output))
    print("OFFICIAL_EXECUTION_PREFLIGHT={}".format(json.dumps(selected)), flush=True)
    if args.check:
        return selected
    if not args.blender or not Path(args.blender).is_file():
        raise FileNotFoundError("Blender executable missing; load Blender module")
    args.output.mkdir(parents=True)
    assets = args.output / "assets"
    # Each official pickle can be many GB. Decode each once, prepare only the
    # selected episodes, then release the full records before Blender renders.
    for task, episode in selected.items():
        case_dir = args.output / task
        case_dir.mkdir()
        for model, source, record in (("dip", dip_path, dip_record),
                                      ("fm", fm_path, fm_record)):
            prepare(source, episode, case_dir / (model + "_prepared.npz"),
                    assets, record_data=record)
    del dip_record, fm_record
    gc.collect()
    rows = []
    for task, episode in selected.items():
        case_dir = args.output / task
        clips = []
        try:
            for model in ("dip", "fm"):
                prepared = case_dir / (model + "_prepared.npz")
                movie = case_dir / (model + "_executed.mp4")
                command = [args.blender, "-b", "--python",
                           "closd/blender/record2anim.py", "--",
                           "--record_path", str(prepared), "--prepared_npz", str(prepared),
                           "--assets_dir", str(assets),
                           "--results_dir", str(case_dir / "blends"),
                           "--render_mp4", str(movie)]
                if args.preview_frames:
                    command.extend(("--preview_frames", str(args.preview_frames)))
                with (case_dir / (model + "_blender.log")).open("x") as log:
                    subprocess.run(command, cwd=str(Path(__file__).resolve().parents[3]),
                                   check=True, stdout=log, stderr=subprocess.STDOUT)
                check_movie(movie)
                clips.append(VideoFileClip(str(movie)))
            duration = min(clip.duration for clip in clips)
            pair = clips_array([[clip.subclip(0, duration) for clip in clips]])
            side = case_dir / "side_by_side_executed.mp4"
            pair.write_videofile(str(side), fps=30, audio=False, threads=2, logger=None)
            pair.close()
            check_movie(side)
            gif = ""
            if task == "GET_UP" or args.preview_frames:
                with imageio.get_reader(str(side)) as reader:
                    frames = [frame for index, frame in enumerate(reader)
                              if index % 3 == 0]
                gif_path = case_dir / "side_by_side_executed.gif"
                imageio.mimsave(str(gif_path), frames, duration=0.1)
                gif = str(gif_path.relative_to(args.output))
            rows.append({"task": task, "episode": episode,
                         "dip_mp4": str((case_dir / "dip_executed.mp4").relative_to(args.output)),
                         "fm_mp4": str((case_dir / "fm_executed.mp4").relative_to(args.output)),
                         "side_by_side_mp4": str(side.relative_to(args.output)),
                         "side_by_side_gif": gif,
                         "source_kind": "closed-loop simulator execution"})
            print("OFFICIAL_EXECUTION_CASE_COMPLETE={}".format(task), flush=True)
        finally:
            for clip in clips:
                clip.close()
    with (args.output / "index.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "README.md").write_text(
        "# Official CLoSD executed-motion renderer\n\n"
        "Source: real CLoSD simulator records in execution/dip and execution/fm. "
        "Body meshes and keyframe animation use closd/blender/record2anim.py, "
        "blender_utils.py, the repository MJCF and FBX assets. "
        "These are closed-loop controller motions, distinct from offline planner trajectories. "
        "The two models use the same selected episode key and seed; their later trajectories can diverge.\n")
    print("OFFICIAL_EXECUTION_COMPLETE=1 OUTPUT={}".format(args.output), flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--preview-frames", type=int, default=0)
    parser.add_argument("--blender", default=os.environ.get("BLENDER_EXECUTABLE"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.preview_frames < 0:
        parser.error("preview frames cannot be negative")
    render(args)


if __name__ == "__main__":
    main()
