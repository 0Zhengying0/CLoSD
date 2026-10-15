#!/usr/bin/env python3
"""Rebuild offline planner quality tables from completed case artifacts."""

import argparse
import csv
import json
import statistics
from pathlib import Path

NUMERIC = ("final_target_error_m", "min_target_distance_m",
           "final_conditioned_target_error_m", "min_conditioned_target_distance_m",
           "min_target_frame",
           "seam_root_jump_m", "seam_relative_joint_jump_m",
           "seam_heading_jump_deg", "root_speed_mean_mps",
           "root_speed_max_mps", "root_accel_mean_mps2", "root_accel_max_mps2",
           "joint_speed_mean_mps", "joint_speed_max_mps",
           "joint_accel_mean_mps2", "joint_accel_max_mps2")
FIELDS = ("case_id", "task", "difficulty", "model", "sampling", "cfg", "seed", "nfe",
          "mdm_forwards", "planner_latency_sec", "case_input_sha256",
          "checkpoint", "args_sha256") + NUMERIC + (
              "target_metric", "target_joint", "target_metric_note",
              "matched_gt_error_m", "trajectory_npz", "distance_curve_csv",
              "distance_curve_png",
              "model_mp4", "comparison_mp4", "comparison_gif")


def summarize(run_dir, output=None, write_csv=True):
    run_dir = Path(run_dir)
    provenance_path = run_dir / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("missing provenance.json")
    provenance = json.loads(provenance_path.read_text())
    expected = int(provenance["cases_requested"])
    rows, problems = [], []
    for case_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        for model in ("dip", "fm"):
            path = case_dir / (model + "_metrics.json")
            if not path.is_file():
                problems.append("{}: {} metrics missing".format(case_dir.name, model))
                continue
            row = json.loads(path.read_text())
            if row["case_id"] != case_dir.name or row["model"] != model.upper():
                problems.append("{}: {} metadata mismatch".format(case_dir.name, model))
                continue
            needed = ("trajectory_npz", "distance_curve_csv", "distance_curve_png",
                      "model_mp4", "comparison_mp4")
            if any(not (run_dir / row[key]).is_file() for key in needed):
                problems.append("{}: {} artifact missing".format(case_dir.name, model))
                continue
            if row.get("comparison_gif") and not (run_dir / row["comparison_gif"]).is_file():
                problems.append("{}: {} GIF missing".format(case_dir.name, model))
                continue
            rows.append(row)
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    for case_id, pair in by_case.items():
        if len(pair) == 2 and (pair[0].get("case_input_sha256") != pair[1].get("case_input_sha256") or
                               pair[0].get("seed") != pair[1].get("seed") or
                               pair[0].get("cfg") != pair[1].get("cfg")):
            problems.append("{}: paired inputs or seed differ".format(case_id))
    if len(rows) != expected * 2:
        problems.append("completed model rows {}/{}".format(len(rows), expected * 2))
    if write_csv:
        with (run_dir / "results.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        group_fields = ["task", "model", "n"] + [
            suffix for key in NUMERIC for suffix in (key + "_mean", key + "_median")]
        with (run_dir / "group_summary.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=group_fields)
            writer.writeheader()
            for task in ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT", "GET_UP"):
                for model in ("DIP", "FM"):
                    group = [row for row in rows if row["task"] == task and row["model"] == model]
                    if not group:
                        continue
                    summary = {"task": task, "model": model, "n": len(group)}
                    for key in NUMERIC:
                        values = [float(row[key]) for row in group]
                        summary[key + "_mean"] = statistics.mean(values)
                        summary[key + "_median"] = statistics.median(values)
                    writer.writerow(summary)

    status = "COMPLETE" if not problems else "PARTIAL"
    kind = "FORMAL" if expected == 20 else "SMOKE"
    lines = ["# DiP vs Flow Matching：规划器直接输出的动作质量", "",
             "状态：**{} {}**；{} 个案例、{} 条模型结果。".format(kind, status, expected, len(rows)),
             "比较使用同一真实 CLoSD 条件快照及相同初始噪声；DiP 后续扩散随机过程与 FM 不同。未运行控制器。",
             "主目标误差对环境原始目标计算；另在 CSV 中报告对规划器截断目标的误差。REACH 为 pelvis 的水平距离；GET_UP 为 pelvis 的三维距离。SIT 与击打类距离不等于官方任务成功。",
             "所有位置在同一局部 HumanML 坐标系，以米计。历史 20 帧，生成 40 帧，20 fps。无严格匹配 GT，因此不报告 GT 误差。", "",
             "| 案例 | 任务 | 难度 | 模型 | 终点误差(m) | 最小距离(m) | 衔接 root 跳变(m) | 衔接关节姿态跳变(m) | 衔接朝向跳变(°) | root 加速度均值(m/s²) | 关节加速度均值(m/s²) | 距离曲线 | MP4 | 并排 MP4 | GIF |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"]
    for row in rows:
        values = [row["case_id"], row["task"], row["difficulty"], row["model"]]
        values.extend("{:.4f}".format(row[key]) for key in (
            "final_target_error_m", "min_target_distance_m", "seam_root_jump_m",
            "seam_relative_joint_jump_m", "seam_heading_jump_deg",
            "root_accel_mean_mps2", "joint_accel_mean_mps2"))
        values.append("[PNG]({})".format(row["distance_curve_png"]))
        values.extend("[video]({})".format(row[key]) for key in ("model_mp4", "comparison_mp4"))
        values.append("[GIF]({})".format(row["comparison_gif"]) if row.get("comparison_gif") else "—")
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## 按任务分组的均值与中位数", "",
                  "| 任务 | 模型 | N | 终点误差 mean/median (m) | 最小距离 mean/median (m) | root 加速度 mean/median (m/s²) | 关节加速度 mean/median (m/s²) |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for task in ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT", "GET_UP"):
        for model in ("DIP", "FM"):
            group = [row for row in rows if row["task"] == task and row["model"] == model]
            if not group:
                continue
            cells = []
            for key in ("final_target_error_m", "min_target_distance_m",
                        "root_accel_mean_mps2", "joint_accel_mean_mps2"):
                values = [float(row[key]) for row in group]
                cells.append("{:.4f} / {:.4f}".format(statistics.mean(values),
                                                    statistics.median(values)))
            lines.append("| {} | {} | {} | {} |".format(task, model, len(group), " | ".join(cells)))
    if problems:
        lines.extend(["", "## 缺失或不完整", ""])
        lines.extend("- " + item for item in problems)
    lines.extend(["", "模型、checkpoint、`args.json` SHA256、Git commit 和 Slurm Job ID 见 `provenance.json`；逐帧距离见各案例的 `*_target_distance.csv`。", ""])
    text = "\n".join(lines)
    markdown_path = Path(output) if output else run_dir / "summary.md"
    with markdown_path.open("x") as handle:
        handle.write(text)
    return markdown_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New Markdown filename; never overwritten")
    parser.add_argument("--no-csv", action="store_true",
                        help="Use when rebuilding Markdown after results.csv already exists")
    args = parser.parse_args(argv)
    print(summarize(args.run_dir, args.output, write_csv=not args.no_csv))


if __name__ == "__main__":
    main()
