#!/usr/bin/env python3
"""Summarize the nine formal FM solver jobs without inventing missing results."""

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path


ORDER = ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT", "GET_UP")
DISPLAY = ("REACH", "KICK", "PUNCH", "SIT", "GET_UP")
RESULT_RE = re.compile(
    r"^EVAL_RESULT STATE=(\w+) SUCCESS=(\d+) RESETS=(\d+) RATE=([0-9.]+)$",
    re.MULTILINE,
)
METADATA_KEYS = (
    "SWEEP_EXPERIMENT_ID", "SWEEP_JOB_ID", "SWEEP_CHECKPOINT",
    "SWEEP_GIT_COMMIT", "SWEEP_GIT_DIRTY", "SWEEP_SOURCE_SHA256",
    "SWEEP_SEED", "SWEEP_CONFIG",
)


def key_values(line):
    return dict(item.split("=", 1) for item in line.split() if "=" in item)


def read_metadata(text):
    metadata = {}
    for line in text.splitlines():
        for key in METADATA_KEYS:
            if line.startswith(key + "="):
                metadata[key] = line.split("=", 1)[1].strip()
    return metadata


def classify_incomplete(text, stderr_text):
    if re.search(r"SWEEP_(?:PROCESS|JOB)_EXIT=([1-9][0-9]*)", text):
        return "FAILED"
    if re.search(
        r"Traceback \(most recent call last\)|Error executing job|"
        r"out.of.memory|oom.kill|TIME LIMIT|CANCELLED|slurmstepd: error",
        text + "\n" + stderr_text, re.IGNORECASE,
    ):
        return "FAILED"
    return "INCOMPLETE"


def inspect_job(row):
    result = {"status": "MISSING_LOG", "metadata": {}, "rates": {},
              "latencies": [], "forward_calls": [], "reason": ""}
    if row["submit_status"] != "submitted" or not row["job_id"].isdigit():
        result["status"] = "SUBMIT_FAILED"
        return result
    stdout = Path(row["stdout"].replace("%j", row["job_id"]))
    stderr = Path(row["stderr"].replace("%j", row["job_id"]))
    if not stdout.is_file():
        return result
    text = stdout.read_text(errors="replace")
    stderr_text = stderr.read_text(errors="replace") if stderr.is_file() else ""
    result["metadata"] = read_metadata(text)
    if ("SWEEP_PROCESS_EXIT=0" not in text or
            "SWEEP_JOB_EXIT=0" not in text or
            "Eval plot ended" not in text):
        result["status"] = classify_incomplete(text, stderr_text)
        result["reason"] = "process or evaluation completion marker missing"
        return result
    metadata = result["metadata"]
    if any(key not in metadata for key in METADATA_KEYS):
        result["status"] = "INVALID_METRICS"
        result["reason"] = "missing run metadata"
        return result
    if (metadata["SWEEP_EXPERIMENT_ID"] != row["experiment_id"] or
            metadata["SWEEP_JOB_ID"] != row["job_id"] or
            metadata["SWEEP_CHECKPOINT"] != row["checkpoint"] or
            metadata["SWEEP_SEED"] != "0"):
        result["status"] = "INVALID_METRICS"
        result["reason"] = "run metadata does not match submission"
        return result

    counts = {}
    for state, success, resets, _printed_rate in RESULT_RE.findall(text):
        if state in counts:
            result["status"] = "INVALID_METRICS"
            result["reason"] = "duplicate task result"
            return result
        success, resets = int(success), int(resets)
        if not (0 <= success <= resets and resets >= 1000):
            result["status"] = "INVALID_METRICS"
            result["reason"] = "invalid or undersized task sample count"
            return result
        counts[state] = (success, resets)
    if set(counts) != set(ORDER):
        result["status"] = "INVALID_METRICS"
        result["reason"] = "one or more of five task results missing"
        return result

    latencies = []
    forwards = []
    for line in text.splitlines():
        if not line.startswith("PLANNER_LATENCY_SEC="):
            continue
        values = key_values(line)
        try:
            if (values["GENERATOR"] != "flow" or
                    values["SOLVER"] != row["solver"] or
                    int(values["ODE_STEPS"]) != int(row["steps"]) or
                    int(values["NFE"]) != int(row["nfe"]) or
                    int(values["BATCH"]) != 4096):
                raise ValueError("wrong planner configuration")
            n_forwards = int(values["MODEL_FORWARD_CALLS"])
            if n_forwards != 2 * int(row["nfe"]):
                raise ValueError("CFG forward count differs from 2 x NFE")
            latencies.append(float(values["PLANNER_LATENCY_SEC"]))
            forwards.append(n_forwards)
        except (KeyError, ValueError) as exc:
            result["status"] = "INVALID_METRICS"
            result["reason"] = str(exc)
            return result
    if not latencies:
        result["status"] = "INVALID_METRICS"
        result["reason"] = "no timed planner call"
        return result

    macro_matches = re.findall(r"^EVAL_MACRO_SUCCESS=([0-9.]+)$", text, re.MULTILINE)
    profile_matches = re.findall(r"^EVAL_PROFILE .+$", text, re.MULTILINE)
    expected_macro = statistics.mean(counts[k][0] / counts[k][1] for k in ORDER)
    if len(macro_matches) != 1 or abs(float(macro_matches[0]) - expected_macro) > 5e-8:
        result["status"] = "INVALID_METRICS"
        result["reason"] = "missing or inconsistent macro success rate"
        return result
    if len(profile_matches) != 1:
        result["status"] = "INVALID_METRICS"
        result["reason"] = "missing end-of-job planner profile"
        return result
    profile = key_values(profile_matches[0])
    try:
        if (int(profile["CALLS"]) != len(latencies) or
                int(profile["MODEL_FORWARDS_TOTAL"]) != sum(forwards) or
                abs(float(profile["PLANNER_LATENCY_TOTAL_SEC"]) - sum(latencies)) >
                max(1e-4, len(latencies) * 1e-6) or
                abs(float(profile["PLANNER_LATENCY_MEAN_SEC"]) -
                    statistics.mean(latencies)) > 2e-6):
            raise ValueError("end-of-job planner profile differs from per-call logs")
    except (KeyError, ValueError) as exc:
        result["status"] = "INVALID_METRICS"
        result["reason"] = str(exc)
        return result

    result.update(status="COMPLETE", rates={k: counts[k][0] / counts[k][1]
                                            for k in ORDER},
                  counts=counts, latencies=latencies, forward_calls=forwards)
    return result


def cell(value):
    return str(value).replace("|", "\\|") if value not in (None, "") else "—"


def render(rows, run_dir):
    lines = [
        "# Flow Matching 300k：ODE 积分器对照",
        "",
        "运行目录：`{}`。只在日志包含完整五项计数、评测完成标志和正确的 NFE/CFG 前向次数时显示成功率。".format(run_dir),
        "每项成功率由原始成功/结束回合计数计算，macro 是五项等权平均；延迟为日志中所有 planner 调用的均值与中位数。",
        "",
        "| 实验 | 状态 | 求解器 | 步数 | NFE/规划 | 实际前向/规划 | 总前向 | Planner 均值/中位数(s) | REACH | KICK | PUNCH | SIT | GET_UP | Macro | Job ID |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    inspected = []
    for row in rows:
        result = inspect_job(row)
        inspected.append((row, result))
        complete = result["status"] == "COMPLETE"
        rates = ["{:.2f}%".format(100 * result["rates"][name]) for name in ORDER] if complete else ["—"] * 5
        macro = "{:.2f}%".format(100 * statistics.mean(result["rates"].values())) if complete else "—"
        latency = ("{:.3f} / {:.3f}".format(
            statistics.mean(result["latencies"]), statistics.median(result["latencies"])
        ) if complete else "—")
        forward_per_call = str(result["forward_calls"][0]) if complete else "—"
        total_forwards = str(sum(result["forward_calls"])) if complete else "—"
        fields = [row["experiment_id"], result["status"], row["solver"], row["steps"],
                  row["nfe"], forward_per_call, total_forwards, latency]
        fields.extend(rates)
        fields.extend((macro, row["job_id"] or "—"))
        lines.append("| " + " | ".join(map(cell, fields)) + " |")

    lines.extend(("", "## 运行元数据", "",
                  "| 实验 | checkpoint | Git commit | Dirty | 源码 SHA256 | Seed | 配置 | Planner 调用数 |",
                  "| --- | --- | --- | --- | --- | ---: | --- | ---: |"))
    for row, result in inspected:
        metadata = result["metadata"]
        fields = [row["experiment_id"], metadata.get("SWEEP_CHECKPOINT", row["checkpoint"]),
                  metadata.get("SWEEP_GIT_COMMIT"), metadata.get("SWEEP_GIT_DIRTY"),
                  metadata.get("SWEEP_SOURCE_SHA256"), metadata.get("SWEEP_SEED"),
                  metadata.get("SWEEP_CONFIG"),
                  len(result["latencies"]) if result["status"] == "COMPLETE" else None]
        lines.append("| " + " | ".join(map(cell, fields)) + " |")

    problems = [(row["experiment_id"], result["status"], result["reason"])
                for row, result in inspected if result["status"] != "COMPLETE"]
    if problems:
        lines.extend(("", "## 缺失或失败的任务", ""))
        for experiment_id, status, reason in problems:
            lines.append("- {}：{}{}".format(
                experiment_id, status, "；" + reason if reason else "",
            ))
    lines.extend(("", "注：MDM 使用最近整数时间索引；RK4 的中间时刻也经相同取整。速度场随时间索引呈阶梯变化，不能保证经典 RK4 的理论四阶收敛率。", ""))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        help="Write once to this new Markdown file; refuses overwrite")
    args = parser.parse_args(argv)
    manifest = args.run_dir / "submissions.tsv"
    if not manifest.is_file():
        parser.error("missing submission manifest: {}".format(manifest))
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = ("FM-E12", "FM-H12", "FM-R12", "FM-E20", "FM-H20", "FM-R20",
                "FM-E40", "FM-H40", "FM-R40")
    if tuple(row["experiment_id"] for row in rows) != expected:
        parser.error("manifest does not contain the exact ordered nine-task matrix")
    markdown = render(rows, args.run_dir)
    if args.output:
        try:
            with args.output.open("x") as handle:
                handle.write(markdown)
        except FileExistsError:
            parser.error("output already exists: {}".format(args.output))
        print("Summary written to {}".format(args.output))
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
