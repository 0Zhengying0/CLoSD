#!/usr/bin/env python3
"""Summarize official plus per-episode diagnostics and inspect control traces."""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np


MODELS = (("FM-RK4-3", "fm_rk4_3"), ("DiP-original", "dip_original"))
TASKS = ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT")
OFFICIAL_METRICS = ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT", "GET_UP")


def read_diagnostics(directory):
    manifest_path = directory / "manifest.json"
    episodes_path = directory / "episodes.jsonl"
    if not manifest_path.is_file() or not episodes_path.is_file():
        return None, []
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    completed = sum(bool(row["completed"]) for row in rows)
    if completed != manifest["completed_episodes"]:
        raise ValueError("completed episode count differs from manifest: " + str(directory))
    for metric in OFFICIAL_METRICS:
        counted = sum(bool(row["official_metric_flags"][metric]["counted"])
                      for row in rows if row["completed"])
        success = sum(bool(row["official_metric_flags"][metric]["success"])
                      for row in rows if row["completed"])
        if (counted != manifest["official_counters"][metric]["resets"] or
                success != manifest["official_counters"][metric]["success"]):
            raise ValueError("official metric mismatch for {} in {}".format(metric, directory))
    return manifest, rows


def percent(numerator, denominator):
    return "{:.2f}%".format(100 * numerator / denominator) if denominator else "—"


def median(values, digits=2):
    valid = [float(value) for value in values if value is not None]
    return ("{:0." + str(digits) + "f}").format(statistics.median(valid)) if valid else "—"


def slurm_job_status(run_dir, slug):
    """Use the submitted Job ID and exit markers when a manifest is present."""
    submission_path = run_dir / "submissions.tsv"
    if not submission_path.is_file():
        return None
    with submission_path.open(newline="") as handle:
        matches = [row for row in csv.DictReader(handle, delimiter="\t")
                   if row["label"] == ("FM-RK4-3" if slug == "fm_rk4_3" else "DIP-ORIGINAL")]
    if len(matches) != 1:
        return "MISSING_SUBMISSION"
    row = matches[0]
    if row["submit_status"] != "submitted" or not row["job_id"].isdigit():
        return "SUBMIT_FAILED"
    stdout = Path(row["stdout"].replace("%j", row["job_id"]))
    if not stdout.is_file():
        return "MISSING_LOG"
    text = stdout.read_text(errors="replace")
    if all(marker in text for marker in ("DIAG_PROCESS_EXIT=0", "DIAG_JOB_EXIT=0", "Eval plot ended")):
        return "COMPLETE"
    if re.search(r"DIAG_(?:PROCESS|JOB)_EXIT=[1-9][0-9]*", text):
        return "FAILED"
    return "INCOMPLETE"


def render_summary(run_dir):
    lines = [
        "# CLoSD 逐回合诊断：FM RK4 3 步与原始 DiP",
        "",
        "运行目录：`{}`。原始官方计数未修改；诊断只读取其逐回合布尔结果。".format(run_dir),
        "首次达标以物理步后即时目标条件为准；官方 `task_done` 通常在下一个控制步开始时锁存。",
        "同一随机种子不保证两种生成器的每个 env_id/回合严格配对，因此这里只做分组比较。",
        "",
        "## 官方指标（原始口径）",
        "",
        "| 模型 | 状态 | REACH | KICK | PUNCH | SIT | GET_UP | 完成回合 | 未完成回合 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    loaded = []
    for label, slug in MODELS:
        directory = run_dir / slug / "diagnostics"
        manifest, rows = read_diagnostics(directory)
        job_status = slurm_job_status(run_dir, slug)
        status = job_status or ("COMPLETE" if manifest is not None else "MISSING/FAILED")
        if status == "COMPLETE" and manifest is None:
            status = "MISSING_DIAGNOSTICS"
        loaded.append((label, slug, status, manifest, rows))
        if status != "COMPLETE":
            lines.append("| {} | {} | — | — | — | — | — | — | — |".format(label, status))
            continue
        cells = []
        for name in OFFICIAL_METRICS:
            counts = manifest["official_counters"][name]
            cells.append("{}/{} ({})".format(
                counts["success"], counts["resets"],
                percent(counts["success"], counts["resets"]),
            ))
        lines.append("| {} | COMPLETE | {} | {} | {} |".format(
            label, " | ".join(cells), manifest["completed_episodes"],
            manifest["incomplete_episodes"],
        ))

    lines.extend((
        "", "## 按初始任务的逐回合诊断", "",
        "| 模型 | 初始任务 | 已结束回合 | 初次达标率 | 首次达标时间中位数(s) | 最小目标距离中位数(m) | 跌倒率 | 达标后跌倒 | 官方记成功 | 达标但官方未记成功 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for label, _slug, status, manifest, rows in loaded:
        if status != "COMPLETE":
            continue
        for task in TASKS:
            selected = [r for r in rows if r["completed"] and r["initial_task"] == task]
            hits = [r for r in selected if r["first_initial_goal_step"] is not None]
            falls = [r for r in selected if r["fell"]]
            after_hit_falls = [r for r in hits if r["fall_step"] is not None and
                               r["fall_step"] >= r["first_initial_goal_step"]]
            official_success = [r for r in selected if r["official_success_for_initial_task"]]
            missed = [r for r in hits if not r["official_success_for_initial_task"]]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                label, task, len(selected), percent(len(hits), len(selected)),
                median([r["first_initial_goal_time_s"] for r in hits]),
                median([r["min_initial_target_distance_m"] for r in selected], 3),
                percent(len(falls), len(selected)), len(after_hit_falls),
                len(official_success), len(missed),
            ))

    lines.extend((
        "", "## SIT → GET_UP", "",
        "| 模型 | 初始 SIT 回合 | 切换 GET_UP | 切换时间中位数(s) | 起身目标达标 | 切换后跌倒 | 结束时 GET_UP | 官方 GET_UP 成功/分母 | 结束时骨盆高度中位数(m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for label, _slug, status, manifest, rows in loaded:
        if status != "COMPLETE":
            continue
        sit = [r for r in rows if r["completed"] and r["initial_task"] == "SIT"]
        switched = [r for r in sit if r["sit_to_get_up_step"] is not None]
        get_up = manifest["official_counters"]["GET_UP"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {}/{} | {} |".format(
            label, len(sit), len(switched),
            median([r["sit_to_get_up_time_s"] for r in switched]),
            sum(r["get_up_first_goal_step"] is not None for r in switched),
            sum(bool(r["post_get_up_fell"]) for r in switched),
            sum(r["final_state"] == "GET_UP" for r in switched),
            get_up["success"], get_up["resets"],
            median([r["post_get_up_final_pelvis_height_m"] for r in switched], 3),
        ))

    lines.extend((
        "", "## 字段说明", "",
        "- `first_initial_goal_step` 是物理步后首次满足当前初始目标条件；`first_official_task_done_step` 是官方锁存标志首次可见的控制步。",
        "- REACH 用骨盆水平距离，GET_UP 用骨盆三维距离，SIT 用到合法坐姿空间的距离。KICK/PUNCH 的距离是指定脚/腕到箱体中心的诊断距离；真正的击倒条件为 `box_up_dot <= 0.25`。",
        "- KICK 与 PUNCH 的官方成功掩码本来相同；按初始任务分组是新增诊断视角。GET_UP 只统计实际切换进去且在 GET_UP 结束的回合。",
        "- `controller_action` 是策略动作，`pd_target` 是实际送出的 PD 位置目标，`dof_pos` 与 `body_pos_xyz` 是 PhysX 步后的实现状态。详细轨迹在各模型 `diagnostics/trajectories/*.npz`。",
        "- `evaluation_stopped` 是评测停止时未结束的片段，不属于任何官方分母。",
        "",
    ))
    return "\n".join(lines)


def export_episodes_csv(run_dir, output):
    rows = []
    for label, slug in MODELS:
        _manifest, episodes = read_diagnostics(run_dir / slug / "diagnostics")
        for row in episodes:
            flat = dict(row)
            flat["model"] = label
            flat["state_transitions"] = json.dumps(flat["state_transitions"], ensure_ascii=False)
            flat["official_metric_flags"] = json.dumps(flat["official_metric_flags"], ensure_ascii=False)
            flat["initial_target_xyz_m"] = json.dumps(flat["initial_target_xyz_m"])
            rows.append(flat)
    if not rows:
        raise ValueError("no episode diagnostics are available")
    fields = ["model"] + [field for field in rows[0] if field != "model"]
    with output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def export_trajectory(directory, episode_id, output):
    _manifest, episodes = read_diagnostics(directory)
    matches = [row for row in episodes if row["episode_id"] == episode_id]
    if len(matches) != 1:
        raise ValueError("expected exactly one matching episode_id: " + episode_id)
    row = matches[0]
    archive_path = directory / row["trajectory_file"]
    with np.load(str(archive_path), allow_pickle=False) as archive:
        index = int(row["trajectory_index"])
        if str(archive["episode_ids"][index]) != episode_id:
            raise ValueError("trajectory index does not match episode_id")
        start, stop = map(int, archive["offsets"][index:index + 2])
        trace = {key: archive[key][start:stop] for key in archive.files
                 if key not in ("episode_ids", "offsets")}
    action_dim = trace["controller_action"].shape[1]
    pd_target_dim = trace["pd_target"].shape[1]
    dof_dim = trace["dof_pos"].shape[1]
    body_count = trace["body_pos_xyz"].shape[1]
    fields = ["progress_step", "time_s", "state", "goal_distance_m", "box_up_dot",
              "physical_goal_hit", "official_task_done", "terminated"]
    fields.extend("target_{}".format(axis) for axis in "xyz")
    fields.extend("body_{}_{}".format(j, axis)
                  for j in range(body_count) for axis in "xyz")
    fields.extend("action_{}".format(j) for j in range(action_dim))
    fields.extend("pd_target_{}".format(j) for j in range(pd_target_dim))
    fields.extend("dof_pos_{}".format(j) for j in range(dof_dim))
    with output.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for step in range(stop - start):
            progress = int(trace["progress_step"][step])
            writer.writerow([
                progress, progress * float(_manifest["control_dt_s"]),
                int(trace["state"][step]), float(trace["goal_distance_m"][step]),
                float(trace["box_up_dot"][step]),
                bool(trace["physical_goal_hit"][step]),
                bool(trace["official_task_done"][step]),
                bool(trace["terminated"][step]),
                *trace["target_xyz"][step].tolist(),
                *trace["body_pos_xyz"][step].reshape(-1).tolist(),
                *trace["controller_action"][step].tolist(),
                *trace["pd_target"][step].tolist(),
                *trace["dof_pos"][step].tolist(),
            ])
    return stop - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True  # Python 3.6 does not accept required in add_subparsers().
    summary = subparsers.add_parser("summary")
    summary.add_argument("--run-dir", required=True, type=Path)
    summary.add_argument("--output", type=Path)
    summary.add_argument("--episodes-csv", type=Path)
    trajectory = subparsers.add_parser("trajectory")
    trajectory.add_argument("--diagnostics-dir", required=True, type=Path)
    trajectory.add_argument("--episode-id", required=True)
    trajectory.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "summary":
            for destination in (args.output, args.episodes_csv):
                if destination is not None and destination.exists():
                    raise FileExistsError("output already exists: " + str(destination))
            markdown = render_summary(args.run_dir)
            if args.output:
                with args.output.open("x") as handle:
                    handle.write(markdown)
                print("Summary written to {}".format(args.output))
            else:
                sys.stdout.write(markdown)
            if args.episodes_csv:
                count = export_episodes_csv(args.run_dir, args.episodes_csv)
                print("Wrote {} episode rows to {}".format(count, args.episodes_csv))
        else:
            steps = export_trajectory(args.diagnostics_dir, args.episode_id, args.output)
            print("Wrote {} control steps to {}".format(steps, args.output))
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
