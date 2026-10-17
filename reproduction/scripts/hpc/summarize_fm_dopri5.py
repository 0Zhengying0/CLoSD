#!/usr/bin/env python3
"""Summarize four new matched-NFE FM solver jobs without guessing failures."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

from summarize_fm_solver_sweep import inspect_job as inspect_old

ROOT = Path(__file__).resolve().parents[3]
OLD_MANIFEST = ROOT / "reproduction/experiments/fm_solver_sweep/runs/fm_solver_20261012_01/submissions.tsv"
TASKS = ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT", "GET_UP")
DISPLAY = ("REACH", "KICK", "PUNCH", "SIT", "GET_UP")
RESULT = re.compile(r"^EVAL_RESULT STATE=(\w+) SUCCESS=(\d+) RESETS=(\d+) RATE=([0-9.]+)$", re.M)
IDS = ("FM-ME", "FM-MH", "FM-MR", "FM-MD")


def tokens(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def inspect(row):
    result = {"status": "MISSING_LOG", "reason": "", "counts": {}, "nfe": [],
              "forwards": [], "latency": [], "metadata": {}}
    if row["submit_status"] != "submitted" or not row["job_id"].isdigit():
        result["status"] = "NOT_SUBMITTED" if row["submit_status"] == "pending" else "SUBMIT_FAILED"
        return result
    path = Path(row["stdout"].replace("%j", row["job_id"]))
    if not path.is_file():
        return result
    text = path.read_text(errors="replace")
    keys = ("MATCHED_ID", "MATCHED_TARGET_NFE", "MATCHED_JOB_ID", "MATCHED_CHECKPOINT",
            "MATCHED_GIT_COMMIT", "MATCHED_SOURCE_SHA256", "MATCHED_SEED", "MATCHED_CONFIG")
    meta = result["metadata"]
    for line in text.splitlines():
        for key in keys:
            if line.startswith(key + "="):
                meta[key] = line.partition("=")[2]
    if not all(marker in text for marker in ("MATCHED_PROCESS_EXIT=0", "MATCHED_JOB_EXIT=0", "Eval plot ended")):
        result["status"] = "FAILED" if ("Traceback" in text or re.search(r"EXIT=[1-9]", text)) else "INCOMPLETE"
        result["reason"] = "evaluation completion markers missing"
        return result
    if (set(meta) != set(keys) or meta["MATCHED_ID"] != row["experiment_id"] or
            meta["MATCHED_TARGET_NFE"] != row["target_nfe"] or
            meta["MATCHED_JOB_ID"] != row["job_id"] or
            meta["MATCHED_CHECKPOINT"] != row["checkpoint"] or meta["MATCHED_SEED"] != "0"):
        result.update(status="INVALID", reason="run metadata mismatch")
        return result
    for task, success, resets, printed in RESULT.findall(text):
        if task in result["counts"] or task not in TASKS:
            result.update(status="INVALID", reason="duplicate or unknown task")
            return result
        success, resets = int(success), int(resets)
        if success > resets or resets < 1000 or abs(float(printed) - success / resets) > 5e-8:
            result.update(status="INVALID", reason="invalid task count or rate")
            return result
        result["counts"][task] = (success, resets)
    if set(result["counts"]) != set(TASKS):
        result.update(status="INVALID", reason="five complete task counts required")
        return result
    for line in text.splitlines():
        if not line.startswith("PLANNER_LATENCY_SEC="):
            continue
        data = tokens(line)
        try:
            nfe = int(data["NFE"])
            forwards = int(data["MODEL_FORWARD_CALLS"])
            latency = float(data["PLANNER_LATENCY_SEC"])
            if (data["GENERATOR"] != "flow" or data["SOLVER"] != row["solver"] or
                    int(data["ODE_STEPS"]) != int(row["steps"]) or
                    int(data["BATCH"]) != 4096 or nfe < 1 or
                    forwards != 2 * nfe or latency <= 0 or
                    (row["solver"] != "dopri5" and nfe != int(row["target_nfe"]))):
                raise ValueError("invalid NFE, CFG forwards, batch or solver")
        except (KeyError, ValueError) as exc:
            result.update(status="INVALID", reason=str(exc))
            return result
        result["nfe"].append(nfe)
        result["forwards"].append(forwards)
        result["latency"].append(latency)
    profiles = re.findall(r"^EVAL_PROFILE .+$", text, re.M)
    macros = re.findall(r"^EVAL_MACRO_SUCCESS=([0-9.]+)$", text, re.M)
    macro = statistics.mean(s / n for s, n in result["counts"].values())
    if not result["nfe"] or len(profiles) != 1 or len(macros) != 1:
        result.update(status="INVALID", reason="profile or macro missing")
        return result
    profile = tokens(profiles[0])
    try:
        if (int(profile["CALLS"]) != len(result["nfe"]) or
                int(profile["MODEL_FORWARDS_TOTAL"]) != sum(result["forwards"]) or
                abs(float(profile["PLANNER_LATENCY_TOTAL_SEC"]) - sum(result["latency"])) > 1e-4 or
                abs(float(macros[0]) - macro) > 5e-8):
            raise ValueError("profile or macro inconsistent with per-call records")
    except (KeyError, ValueError) as exc:
        result.update(status="INVALID", reason=str(exc))
        return result
    result.update(status="COMPLETE", macro=macro)
    return result


def load_rows(run_dir, manifest=None):
    manifest = manifest or run_dir / "matched_submissions.tsv"
    if not manifest.is_file():
        return [{"experiment_id": name, "solver": solver, "steps": "",
                 "target_nfe": "", "pilot_nfe": "", "job_id": "", "stdout": "",
                 "stderr": "", "submit_status": "pending", "checkpoint": ""}
                for name, solver in zip(IDS, ("euler", "heun", "rk4", "dopri5"))]
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if tuple(row["experiment_id"] for row in rows) != IDS:
        raise ValueError("manifest must have four ordered matched experiments")
    return rows


def old_reference():
    with OLD_MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rk4 = next(row for row in rows if row["experiment_id"] == "FM-R12")
    result = inspect_old(rk4)
    if result["status"] != "COMPLETE":
        return None
    return statistics.mean(result["rates"].values())


def render(run_dir, manifest=None):
    rows = load_rows(run_dir, manifest)
    inspected = [(row, inspect(row)) for row in rows]
    target = int(rows[0]["target_nfe"]) if rows[0]["target_nfe"] else None
    lines = ["# FM 300k：四种求解器的新同预算闭环对照", "",
             "四组共用 FM 300k EMA checkpoint、CFG 7.5、seed 0、4096 环境、episode length 500 与官方成功率口径。",
             "Euler/Heun/RK4 固定每次规划 NFE；DOPRI5 使用 `rtol=1e-3, atol=1e-4`，实际 NFE 随规划调用变化。",
             "公共预算来自 20 个真实条件快照的 DOPRI5 校准，并取最接近的 4 的倍数。",
             "", "| 求解器 | 状态 | 步数 | NFE 均值 [最小,最大] | MDM 前向均值 | Planner 延迟均值/中位数(s) | REACH | KICK | PUNCH | SIT | GET_UP | 等权平均 | Job ID |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    csv_rows = []
    for row, result in inspected:
        complete = result["status"] == "COMPLETE"
        mean_nfe = statistics.mean(result["nfe"]) if complete else None
        matched = complete and (row["solver"] != "dopri5" or
                                abs(mean_nfe - int(row["target_nfe"])) <= max(2, 0.2 * int(row["target_nfe"])))
        status = ("COMPLETE" if matched else "OUT_OF_BAND") if complete else result["status"]
        counts = result["counts"] if complete else {}
        rates = [counts[task][0] / counts[task][1] if complete else None for task in TASKS]
        fields = [row["solver"], status, row["steps"] or "—",
                  "{:.1f} [{},{}]".format(mean_nfe, min(result["nfe"]), max(result["nfe"])) if complete else "—",
                  "{:.1f}".format(statistics.mean(result["forwards"])) if complete else "—",
                  "{:.3f}/{:.3f}".format(statistics.mean(result["latency"]),
                                         statistics.median(result["latency"])) if complete else "—"]
        fields += ["{:.2f}% ({}/{})".format(100 * rate, *counts[task]) if rate is not None else "—"
                   for task, rate in zip(TASKS, rates)]
        fields += ["{:.2f}%".format(100 * result["macro"]) if complete else "—", row["job_id"] or "—"]
        lines.append("| " + " | ".join(fields) + " |")
        csv_rows.append({"experiment_id": row["experiment_id"], "solver": row["solver"],
                         "status": status, "job_id": row["job_id"],
                         "target_nfe": row["target_nfe"] or None, "pilot_nfe": row["pilot_nfe"] or None,
                         "nfe_mean": mean_nfe, "nfe_min": min(result["nfe"]) if complete else None,
                         "nfe_max": max(result["nfe"]) if complete else None,
                         "mdm_forwards_mean": statistics.mean(result["forwards"]) if complete else None,
                         "latency_mean_sec": statistics.mean(result["latency"]) if complete else None,
                         "macro": result.get("macro") if complete else None,
                         **{name: rate for name, rate in zip(DISPLAY, rates)}})
    lines += ["", "## 分析", ""]
    if target is None:
        lines.append("- 校准或正式提交尚未完成，不填写成功率或公共 NFE 预算。")
    else:
        lines.append("- 公共预算：{} NFE/规划；校准时 DOPRI5 为 {} NFE。".format(target, rows[0]["pilot_nfe"]))
    if all(result["status"] == "COMPLETE" for _, result in inspected):
        dopri = inspected[-1][1]
        mean_nfe = statistics.mean(dopri["nfe"])
        if abs(mean_nfe - target) <= max(2, 0.2 * target):
            for row, result in inspected[:-1]:
                lines.append("- DOPRI5 相对 {}：等权成功率 {:+.2f} 个百分点；Planner 平均延迟 {:.3f}s 对 {:.3f}s。".format(
                    row["solver"], 100 * (dopri["macro"] - result["macro"]),
                    statistics.mean(dopri["latency"]), statistics.mean(result["latency"])))
        else:
            lines.append("- DOPRI5 正式实测平均 {:.1f} NFE，偏离公共预算 {}；保留各项结果，但不解释为等计算量比较。".format(mean_nfe, target))
    else:
        lines.append("- 未完成/失败任务保留状态；不从部分日志猜测成功率。")
    old = old_reference()
    if old is not None:
        lines.append("- 历史最佳 RK4 12 NFE 成功率 {:.2f}%；它属于不同计算预算，仅作背景参考。".format(100 * old))
    lines += ["- MDM 使用 `round(t × 999)`；时间索引跳变可能影响 DOPRI5 自适应步长与 NFE。",
              "", "## 运行元数据", ""]
    for row, result in inspected:
        lines.append("- {}：checkpoint `{}`；Git `{}`；seed `{}`；配置 `{}`；源码 SHA256 `{}`。".format(
            row["experiment_id"], row["checkpoint"] or "—",
            result["metadata"].get("MATCHED_GIT_COMMIT", "—"),
            result["metadata"].get("MATCHED_SEED", "—"),
            result["metadata"].get("MATCHED_CONFIG", "—"),
            result["metadata"].get("MATCHED_SOURCE_SHA256", "—")))
    lines.append("")
    return "\n".join(lines), csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Select an original or retry submission manifest")
    parser.add_argument("--output", type=Path, help="New Markdown report; refuses overwrite")
    args = parser.parse_args()
    report, rows = render(args.run_dir, args.manifest)
    if args.output:
        csv_path = args.output.with_suffix(".csv")
        if args.output.exists() or csv_path.exists():
            parser.error("report output already exists")
        with args.output.open("x") as handle:
            handle.write(report)
        with csv_path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print("REPORT={} CSV={}".format(args.output, csv_path))
    else:
        print(report)


if __name__ == "__main__":
    main()
