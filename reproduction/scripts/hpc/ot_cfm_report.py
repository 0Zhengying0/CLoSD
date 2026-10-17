"""Write evidence based VP-CFM versus OT-CFM success and motion report."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

TASKS = ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT", "GET_UP")
QUALITY = ("final_target_error_m", "min_target_distance_m", "seam_root_jump_m",
           "seam_relative_joint_jump_m", "root_accel_mean_mps2",
           "joint_accel_mean_mps2")
RESULT = re.compile(r"^EVAL_RESULT STATE=(\w+) SUCCESS=(\d+) RESETS=(\d+) RATE=([0-9.]+)$", re.M)


def success(path, marker):
    text = path.read_text(errors="replace")
    if marker not in text or "Eval plot ended" not in text:
        raise ValueError("incomplete evaluation: " + str(path))
    values = {}
    for task, wins, total, _ in RESULT.findall(text):
        if task in values:
            raise ValueError("duplicate task result")
        wins, total = int(wins), int(total)
        if not 0 <= wins <= total or total < 1000:
            raise ValueError("invalid success counts")
        values[task] = (wins, total)
    if set(values) != set(TASKS):
        raise ValueError("missing task counts: " + str(path))
    if ("NFE:       12" not in text or "Solver:    rk4" not in text or
            ("CFG:       7.5" not in text and "cfg:7.5" not in text)):
        raise ValueError("evaluation solver, NFE or CFG mismatch")
    calls = re.findall(r"^PLANNER_LATENCY_SEC=.*?NFE=(\d+) MODEL_FORWARD_CALLS=(\d+) BATCH=(\d+)$", text, re.M)
    if len(calls) != 10 or any(tuple(map(int, call)) != (12, 24, 4096) for call in calls):
        raise ValueError("planner call count, NFE, CFG forwards or batch mismatch")
    return values


def read_csv(path, expected_model):
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["model"] == expected_model]
    if len(rows) != 20 or any(sum(r["task"] == task for r in rows) != 4 for task in TASKS):
        raise ValueError("quality results must contain four cases per task: " + str(path))
    return {r["case_id"]: r for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm-log", required=True, type=Path)
    parser.add_argument("--ot-log", required=True, type=Path)
    parser.add_argument("--fm-quality", required=True, type=Path)
    parser.add_argument("--ot-quality", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    fm = success(args.fm_log, "SWEEP_JOB_EXIT=0")
    ot = success(args.ot_log, "SWEEP_JOB_EXIT=0")
    fm_q = read_csv(args.fm_quality, "FM")
    ot_q = read_csv(args.ot_quality, "OT-CFM")
    if set(fm_q) != set(ot_q):
        raise ValueError("quality case IDs differ")
    for case in fm_q:
        for key in ("case_input_sha256", "seed", "cfg", "nfe"):
            if str(fm_q[case][key]) != str(ot_q[case][key]):
                raise ValueError("paired case mismatch: {} {}".format(case, key))
    lines = ["# VP-CFM 与 OT-CFM：CLoSD 300k 对比", "",
             "训练：相同 MDM、条件、CFG、目标位置损失权重与 300k steps；OT-CFM 使用精确 minibatch OT 噪声配对和线性路径。",
             "评测：相同 CLoSD 控制器，seed=0，4096 环境，500 帧，CFG=7.5，RK4 三步（12 NFE），每任务至少 1000 个结束回合。", "",
             "| 模型 | REACH | KICK | PUNCH | SIT | GET_UP | 五任务平均 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    data = {}
    for label, counts in (("VP-CFM", fm), ("OT-CFM", ot)):
        rates = [counts[t][0] / counts[t][1] for t in TASKS]
        data[label] = {t: {"success": counts[t][0], "total": counts[t][1],
                           "rate": rates[i]} for i, t in enumerate(TASKS)}
        data[label]["macro"] = statistics.mean(rates)
        data[label]["unique_state_macro"] = statistics.mean(
            counts[t][0] / counts[t][1]
            for t in ("REACH", "STRIKE_KICK", "SIT", "GET_UP"))
        lines.append("| {} | {} | {:.2%} |".format(
            label, " | ".join("{:.2%}".format(r) for r in rates), statistics.mean(rates)))
    delta = [data["OT-CFM"][t]["rate"] - data["VP-CFM"][t]["rate"] for t in TASKS]
    lines.append("| OT-CFM − VP-CFM | {} | {:+.2f} pp |".format(
        " | ".join("{:+.2f} pp".format(100 * d) for d in delta),
        100 * (data["OT-CFM"]["macro"] - data["VP-CFM"]["macro"])))
    lines += ["", "成功率来自完整任务日志中的 success/resets 计数；五任务平均为五个标签等权平均。",
              "**评测器限制：KICK 与 PUNCH 共用同一击打成功/重置计数，不能解释成两项独立成功率。**",
              "合并击打只计一次时，四个独立状态的等权平均：VP-CFM {:.2%}，OT-CFM {:.2%}。".format(
                  data["VP-CFM"]["unique_state_macro"], data["OT-CFM"]["unique_state_macro"]),
              "两次评测来自不同代码修订；已核对固定步 RK4 分支与成功计数公式未变，但代码版本仍是比较限制。", "",
              "## 生成动作质量", "",
              "两模型复用同 20 个真实 CLoSD 条件与初始噪声，每任务 4 个案例。以下为平均值；目标误差越低越好。平滑度指标仅作描述。", "",
              "| 模型 | 终点目标误差(m) | 最近目标距离(m) | 历史衔接 root 跳变(m) | 历史衔接姿态跳变(m) | root 加速度(m/s²) | 关节加速度(m/s²) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    quality = {}
    for label, rows in (("VP-CFM", fm_q), ("OT-CFM", ot_q)):
        quality[label] = {key: statistics.mean(float(row[key]) for row in rows.values()) for key in QUALITY}
        lines.append("| {} | {} |".format(label, " | ".join(
            "{:.4f}".format(quality[label][key]) for key in QUALITY)))
    lines += ["", "### 分任务的动作质量", "",
              "每项四个相同条件案例；距离和动作连续性是描述性指标。", "",
              "| 任务 | 模型 | 终点目标误差(m) | 最近目标距离(m) | root 加速度(m/s²) | 关节加速度(m/s²) |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for task in TASKS:
        for label, rows in (("VP-CFM", fm_q), ("OT-CFM", ot_q)):
            subset = [row for row in rows.values() if row["task"] == task]
            values = [statistics.mean(float(row[key]) for row in subset) for key in (
                "final_target_error_m", "min_target_distance_m",
                "root_accel_mean_mps2", "joint_accel_mean_mps2")]
            lines.append("| {} | {} | {} |".format(task, label,
                " | ".join("{:.4f}".format(value) for value in values)))
    lines += ["", "动作质量是规划器直接输出的描述性指标，不等于控制器执行成功率。SIT 与击打目标距离也不等于官方碰撞判定。", "",
              "## 原始记录", "",
              "- OT-CFM 300k checkpoint：`reproduction/runs/OTCFM_DIMP_FINAL_seed10_300k/model000300000.pt`",
              "- OT-CFM 训练日志：`logs/ot_cfm_300k_5220077.out`",
              "- VP-CFM 评测日志：`{}`".format(args.fm_log),
              "- OT-CFM 评测日志：`{}`".format(args.ot_log),
              "- VP-CFM 20 案例：`{}`".format(args.fm_quality),
              "- OT-CFM 20 案例：`{}`".format(args.ot_quality),
              "- OT-CFM 案例轨迹：`{}`".format(args.ot_quality.parent), ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        parser.error("report already exists")
    args.output.write_text("\n".join(lines))
    args.output.with_suffix(".json").write_text(json.dumps(
        {"success": data, "quality_mean": quality}, indent=2) + "\n")
    print("OT_CFM_REPORT_PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
