#!/usr/bin/env python3
"""Export DiP/FM video cases and separate closed-loop episode diagnostics."""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from export_success_chart_data import ROOT, TASKS, DISPLAY, relative, write_xlsx

QUALITY = ROOT / "reproduction/experiments/dip_fm_quality/runs/dip_fm_quality_20261014_02"
DIAG = ROOT / "reproduction/experiments/episode_diagnostics/runs/rk4_vs_dip_20261014_01"
DEFAULT_OUTPUT = ROOT / "reproduction/experiments/chart_data/dip_fm_episode_video_analysis_20261018.xlsx"
MODEL_ORDER = ("DIP", "FM")


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    if value in (None, ""):
        return None
    return float(value)


def integer(value):
    if value in (None, ""):
        return None
    return int(value)


def flag(value):
    if value in (None, ""):
        return None
    if value not in ("True", "False"):
        raise ValueError("Unexpected boolean value: " + value)
    return int(value == "True")


def video_path(parent, name):
    if not name:
        return ""
    path = parent / name
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Missing or empty video: " + str(path))
    return relative(path)


def quality_sheets():
    source = QUALITY / "results/results.csv"
    rows = csv_rows(source)
    by_case = defaultdict(dict)
    if len(rows) != 40:
        raise ValueError("Expected 40 DiP/FM quality outputs")
    for row in rows:
        case_id = row["case_id"]
        if row["model"] not in MODEL_ORDER or row["model"] in by_case[case_id]:
            raise ValueError("Missing or duplicated model for case " + case_id)
        by_case[case_id][row["model"]] = row
    if len(by_case) != 20 or any(set(pair) != set(MODEL_ORDER) for pair in by_case.values()):
        raise ValueError("Expected 20 matched condition cases")
    provenance = json.loads((QUALITY / "results/provenance.json").read_text())
    if (provenance["cases_requested"] != 20 or not provenance["same_initial_noise"] or
            provenance["not_controller_executed"] is not True):
        raise ValueError("Video case provenance does not describe the expected offline comparison")

    pairs = [["案例ID", "任务", "难度", "原始文字", "目标关节", "Heading目标(rad)",
              "初始噪声Seed", "共同输入SHA256", "DiP终点误差(m)", "FM终点误差(m)",
              "FM-DiP终点误差(m)", "DiP最小距离(m)", "FM最小距离(m)",
              "DiP衔接root跳变(m)", "FM衔接root跳变(m)",
              "DiP平均root加速度(m/s²)", "FM平均root加速度(m/s²)",
              "DiP平均关节加速度(m/s²)", "FM平均关节加速度(m/s²)",
              "DiP规划MP4", "FM规划MP4", "并排规划MP4", "并排GIF", "条件JSON"]]
    metric_keys = ("final_target_error_m", "min_target_distance_m", "seam_root_jump_m",
                   "seam_relative_joint_jump_m", "seam_heading_jump_deg",
                   "root_speed_mean_mps", "root_accel_mean_mps2", "joint_speed_mean_mps",
                   "joint_accel_mean_mps2")
    detail = [["案例ID", "任务", "难度", "模型", "采样方法", "CFG", "Seed", "NFE",
               "终点误差(m)", "最小距离(m)", "衔接root跳变(m)", "衔接关节跳变(m)",
               "衔接朝向跳变(°)", "root平均速度(m/s)", "root平均加速度(m/s²)",
               "关节平均速度(m/s)", "关节平均加速度(m/s²)", "目标误差口径",
               "目标关节", "模型MP4", "并排MP4", "GIF", "距离曲线CSV", "checkpoint"]]
    for case_id, pair in sorted(by_case.items()):
        dip, fm = (pair[name] for name in MODEL_ORDER)
        case_file = QUALITY / "conditions/cases" / (case_id + ".json")
        case = json.loads(case_file.read_text())
        if (dip["task"] != fm["task"] or dip["difficulty"] != fm["difficulty"] or
                dip["case_input_sha256"] != fm["case_input_sha256"] or
                dip["seed"] != fm["seed"] or case["seed"] != int(dip["seed"]) or
                case["task"] != dip["task"] or dip["sampling"] != "Diffusion 10 steps" or
                fm["sampling"] != "RK4 3 steps" or fm["nfe"] != "12" or
                dip["cfg"] != "7.5" or fm["cfg"] != "7.5"):
            raise ValueError("Case condition or sampling mismatch: " + case_id)
        dip_video = video_path(QUALITY / "results", dip["model_mp4"])
        fm_video = video_path(QUALITY / "results", fm["model_mp4"])
        side_video = video_path(QUALITY / "results", dip["comparison_mp4"])
        gif = video_path(QUALITY / "results", dip["comparison_gif"])
        pairs.append([case_id, dip["task"], dip["difficulty"], case["text"],
                      ", ".join(case["target_joint_names"]), case["heading_target_rad"],
                      int(dip["seed"]), dip["case_input_sha256"],
                      number(dip["final_target_error_m"]), number(fm["final_target_error_m"]),
                      number(fm["final_target_error_m"]) - number(dip["final_target_error_m"]),
                      number(dip["min_target_distance_m"]), number(fm["min_target_distance_m"]),
                      number(dip["seam_root_jump_m"]), number(fm["seam_root_jump_m"]),
                      number(dip["root_accel_mean_mps2"]), number(fm["root_accel_mean_mps2"]),
                      number(dip["joint_accel_mean_mps2"]), number(fm["joint_accel_mean_mps2"]),
                      dip_video, fm_video, side_video, gif, relative(case_file)])
        for row in (dip, fm):
            model_video = video_path(QUALITY / "results", row["model_mp4"])
            comparison_video = video_path(QUALITY / "results", row["comparison_mp4"])
            gif_video = video_path(QUALITY / "results", row["comparison_gif"])
            detail.append([row["case_id"], row["task"], row["difficulty"], row["model"],
                           row["sampling"], number(row["cfg"]), integer(row["seed"]),
                           integer(row["nfe"])] + [number(row[key]) for key in metric_keys] +
                          [row["target_metric"], row["target_joint"], model_video,
                           comparison_video, gif_video,
                           relative(QUALITY / "results" / row["distance_curve_csv"]),
                           row["checkpoint"]])

    group_source = csv_rows(QUALITY / "results/group_summary.csv")
    if len(group_source) != 10 or any(int(row["n"]) != 4 for row in group_source):
        raise ValueError("Expected four video cases for each task and model")
    group = [["任务", "模型", "案例数", "终点误差均值(m)", "终点误差中位数(m)",
              "最小距离均值(m)", "最小距离中位数(m)", "衔接root跳变均值(m)",
              "root加速度均值(m/s²)", "root加速度中位数(m/s²)",
              "关节加速度均值(m/s²)", "关节加速度中位数(m/s²)"]]
    for row in group_source:
        task, model = row["task"], row["model"]
        source_rows = [item for item in rows if item["task"] == task and item["model"] == model]
        if len(source_rows) != 4:
            raise ValueError("Group count mismatch")
        for key in ("final_target_error_m", "min_target_distance_m", "root_accel_mean_mps2",
                    "joint_accel_mean_mps2"):
            if abs(statistics.mean(number(item[key]) for item in source_rows) -
                   number(row[key + "_mean"])) > 1e-9:
                raise ValueError("Group mean mismatch: " + task + " " + model + " " + key)
        group.append([task, model, 4, number(row["final_target_error_m_mean"]),
                      number(row["final_target_error_m_median"]),
                      number(row["min_target_distance_m_mean"]),
                      number(row["min_target_distance_m_median"]),
                      number(row["seam_root_jump_m_mean"]),
                      number(row["root_accel_mean_mps2_mean"]),
                      number(row["root_accel_mean_mps2_median"]),
                      number(row["joint_accel_mean_mps2_mean"]),
                      number(row["joint_accel_mean_mps2_median"])])

    skeleton_source = csv_rows(QUALITY / "official_skeleton_formal_5202408/index.csv")
    if {row["case_id"] for row in skeleton_source} != set(by_case):
        raise ValueError("Official skeleton video index differs from the 20 cases")
    skeleton = [["案例ID", "任务", "帧数", "DiP官方骨架MP4", "FM官方骨架MP4", "并排官方骨架MP4", "GIF"]]
    for row in skeleton_source:
        parent = QUALITY / "official_skeleton_formal_5202408"
        skeleton.append([row["case_id"], row["task"], integer(row["frames"]),
                         video_path(parent, row["dip_mp4"]), video_path(parent, row["fm_mp4"]),
                         video_path(parent, row["comparison_mp4"]),
                         video_path(parent, row["comparison_gif"])])

    execution_source = csv_rows(QUALITY / "official_execution_formal_5202422/index.csv")
    if len(execution_source) != 5 or {row["task"] for row in execution_source} != set(TASKS):
        raise ValueError("Expected five simulator-execution videos")
    execution = [["任务", "回合ID", "DiP仿真执行MP4", "FM仿真执行MP4", "并排执行MP4", "GIF", "视频来源"]]
    for row in execution_source:
        parent = QUALITY / "official_execution_formal_5202422"
        execution.append([row["task"], row["episode"], video_path(parent, row["dip_mp4"]),
                          video_path(parent, row["fm_mp4"]), video_path(parent, row["side_by_side_mp4"]),
                          video_path(parent, row["side_by_side_gif"]), row["source_kind"]])
    return pairs, detail, group, skeleton, execution


EPISODE_FIELDS = (
    ("模型", "model", str), ("回合ID", "episode_id", str),
    ("环境ID", "env_id", integer), ("环境中回合序号", "env_episode_index", integer),
    ("完成回合(1/0)", "completed", flag), ("初始任务", "initial_task", str),
    ("初始目标XYZ(m)", "initial_target_xyz_m", str), ("结束状态", "final_state", str),
    ("结束步", "end_step", integer), ("结束时间(s)", "end_time_s", number),
    ("终止原因", "terminal_reason", str), ("跌倒(1/0)", "fell", flag),
    ("跌倒时间(s)", "fall_time_s", number),
    ("首次初始目标达标时间(s)", "first_initial_goal_time_s", number),
    ("首次官方完成标志时间(s)", "first_official_task_done_time_s", number),
    ("初始目标最小距离(m)", "min_initial_target_distance_m", number),
    ("状态转换", "state_transitions", str),
    ("SIT转GET_UP时间(s)", "sit_to_get_up_time_s", number),
    ("GET_UP首次达标时间(s)", "get_up_first_goal_time_s", number),
    ("GET_UP后结束状态", "post_get_up_final_state", str),
    ("GET_UP后跌倒(1/0)", "post_get_up_fell", flag),
    ("初始任务官方记成功(1/0)", "official_success_for_initial_task", flag),
    ("GET_UP官方计分(1/0)", "official_get_up_counted", flag),
    ("GET_UP官方成功(1/0)", "official_get_up_success", flag),
    ("轨迹片段文件", "trajectory_file", str),
    ("轨迹索引", "trajectory_index", integer),
)


def episode_sheets():
    source = csv_rows(DIAG / "episodes.csv")
    if len(source) != 8652 or Counter(row["model"] for row in source) != {
            "FM-RK4-3": 4405, "DiP-original": 4247}:
        raise ValueError("Episode diagnosis has unexpected size or model labels")
    episode_rows = [[title for title, _key, _convert in EPISODE_FIELDS]]
    counts = defaultdict(lambda: [0, 0])
    completed = Counter()
    for row in source:
        if row["completed"] == "True":
            completed[row["model"]] += 1
            for task, flags in json.loads(row["official_metric_flags"]).items():
                if flags["counted"]:
                    counts[(row["model"], task)][0] += int(flags["success"])
                    counts[(row["model"], task)][1] += 1
        converted = []
        for _title, key, convert in EPISODE_FIELDS:
            value = row[key]
            if key == "trajectory_file" and value:
                subdir = "fm_rk4_3" if row["model"] == "FM-RK4-3" else "dip_original"
                value = relative(DIAG / subdir / "diagnostics" / value)
            else:
                value = convert(value) if value not in (None, "") else None
            converted.append(value)
        episode_rows.append(converted)
    expected = {
        ("FM-RK4-3", "REACH"): (968, 1390), ("FM-RK4-3", "STRIKE_KICK"): (1145, 1309),
        ("FM-RK4-3", "STRIKE_PUNCH"): (1145, 1309), ("FM-RK4-3", "SIT"): (1356, 1407),
        ("FM-RK4-3", "GET_UP"): (1037, 1343), ("DiP-original", "REACH"): (1389, 1390),
        ("DiP-original", "STRIKE_KICK"): (1230, 1317),
        ("DiP-original", "STRIKE_PUNCH"): (1230, 1317),
        ("DiP-original", "SIT"): (1238, 1403), ("DiP-original", "GET_UP"): (1196, 1223),
    }
    if {key: tuple(value) for key, value in counts.items()} != expected or completed != {
            "FM-RK4-3": 4106, "DiP-original": 4110}:
        raise ValueError("Episode official counts differ from published comparison")
    official = [["动作 / 指标", "DiP-original (Job 5199598)", "FM-RK4-3 (Job 5199597)"]]
    for task, display in zip(TASKS, DISPLAY):
        official.append([display] + [counts[(model, task)][0] / counts[(model, task)][1]
                                     for model in ("DiP-original", "FM-RK4-3")])
    official.append(["五项等权总体成功率"] + [statistics.mean(
        counts[(model, task)][0] / counts[(model, task)][1] for task in TASKS)
        for model in ("DiP-original", "FM-RK4-3")])
    return official, episode_rows


def build(output):
    pairs, detail, group, skeleton, execution = quality_sheets()
    official, episodes = episode_sheets()
    notes = [
        ["说明", "内容"],
        ["视频案例批次", "20个真实条件快照；DiP与FM分别直接生成规划轨迹，共40条。视频案例不经过CLoSD控制器，不具备官方闭环成功标签。"],
        ["视频案例模型", "DiP原始10步checkpoint: closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt；FM 300k EMA: reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt。"],
        ["共同输入", "每对共用文字、目标、历史20帧、预测40帧、CFG7.5和初始高斯噪声；扩散后续随机过程不要求完全相同。"],
        ["视频指标", "目标误差和速度/加速度来自恢复后的规划器动作；REACH和GET_UP按各自目标定义，SIT和击打的距离不等于官方成功条件。"],
        ["官方骨架视频", "20个案例的官方骨架离屏渲染；仍然展示规划器输出。"],
        ["仿真执行视频", "每项任务一条闭环控制器执行录制，来自视频项目的另一次记录；不与20个离线案例或下面8652个诊断回合逐项配对。"],
        ["闭环逐回合批次", "另一个实验：DiP-original Job 5199598、FM-RK4-3 Job 5199597；8652条记录含已完成和评测停止时未完成的回合。"],
        ["官方成功率", "仅“闭环官方成功率”工作表使用官方成功标签；五项等权平均，不按初始任务直接合并分母。"],
        ["配对限制", "同seed不保证两模型闭环诊断中的相同env_id/episode_id为严格配对，故逐回合仅作分组分析。"],
        ["数据来源", "质量案例: reproduction/experiments/dip_fm_quality/runs/dip_fm_quality_20261014_02/results/results.csv；闭环回合: reproduction/experiments/episode_diagnostics/runs/rk4_vs_dip_20261014_01/episodes.csv。"],
    ]
    sheets = [
        ("视频案例配对20", pairs, ()),
        ("逐案例指标40", detail, ()),
        ("任务组均值中位数", group, ()),
        ("官方骨架视频20", skeleton, ()),
        ("仿真执行视频5", execution, ()),
        ("闭环官方成功率", official, (2, 3)),
        ("闭环逐回合8652", episodes, ()),
        ("说明", notes, ()),
    ]
    write_xlsx(output, sheets)
    print("XLSX={} SHEETS={} VIDEO_CASES=20 QUALITY_TRAJECTORIES=40 CLOSED_LOOP_EPISODES=8652".format(
        output, len(sheets)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
