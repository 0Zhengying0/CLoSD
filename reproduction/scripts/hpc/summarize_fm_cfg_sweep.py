#!/usr/bin/env python3
"""Summarize six formal CFG evaluations; never fill in missing results."""

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

import summarize_fm_solver_sweep as base


IDS = ("FM-CFG1", "FM-CFG2p5", "FM-CFG5", "FM-CFG7p5", "FM-CFG10", "FM-CFG15")
SCALES = ("1", "2.5", "5", "7.5", "10", "15")
ORDER = base.ORDER
DISPLAY = base.DISPLAY
BASELINE_LOG = (Path(__file__).resolve().parents[3] /
                "reproduction/experiments/fm_solver_sweep/runs/fm_solver_20261012_01/"
                "logs/FM-R12_5185912.out")


def inspect_job(row):
    result = {"status": "MISSING_LOG", "reason": "", "metadata": {},
              "counts": {}, "latencies": [], "forwards": []}
    if row["submit_status"] != "submitted" or not row["job_id"].isdigit():
        result["status"] = "SUBMIT_FAILED"
        return result
    stdout = Path(row["stdout"].replace("%j", row["job_id"]))
    stderr = Path(row["stderr"].replace("%j", row["job_id"]))
    if not stdout.is_file():
        return result
    log = stdout.read_text(errors="replace")
    errors = stderr.read_text(errors="replace") if stderr.is_file() else ""
    metadata_keys = ("CFG_EXPERIMENT_ID", "CFG_JOB_ID", "CFG_CHECKPOINT",
                     "CFG_GIT_COMMIT", "CFG_GIT_DIRTY", "CFG_SOURCE_SHA256",
                     "CFG_SCALE", "CFG_GUIDANCE_TYPE", "CFG_SEED", "CFG_CONFIG")
    metadata = {}
    for line in log.splitlines():
        for key in metadata_keys:
            if line.startswith(key + "="):
                metadata[key] = line.split("=", 1)[1]
    result["metadata"] = metadata
    if not all(marker in log for marker in
               ("CFG_PROCESS_EXIT=0", "CFG_JOB_EXIT=0", "Eval plot ended")):
        result["status"] = base.classify_incomplete(log, errors)
        if re.search(r"CFG_(?:PROCESS|JOB)_EXIT=([1-9][0-9]*)", log):
            result["status"] = "FAILED"
        result["reason"] = "evaluation or process completion marker missing"
        return result
    expected = {"CFG_EXPERIMENT_ID": row["experiment_id"],
                "CFG_JOB_ID": row["job_id"], "CFG_CHECKPOINT": row["checkpoint"],
                "CFG_SCALE": row["cfg"], "CFG_GUIDANCE_TYPE": "text", "CFG_SEED": "0"}
    if any(metadata.get(key) != value for key, value in expected.items()) or \
            any(key not in metadata for key in metadata_keys):
        result.update(status="INVALID_METRICS", reason="missing or mismatched run metadata")
        return result
    if ("num_envs:4096" not in metadata["CFG_CONFIG"] or
            "episode_length:500" not in metadata["CFG_CONFIG"] or
            "samples_per_task:1000" not in metadata["CFG_CONFIG"] or
            "nfe:12" not in metadata["CFG_CONFIG"]):
        result.update(status="INVALID_METRICS", reason="formal configuration mismatch")
        return result

    counts = {}
    for state, success, resets, printed_rate in base.RESULT_RE.findall(log):
        if state in counts:
            result.update(status="INVALID_METRICS", reason="duplicate task result")
            return result
        success, resets = int(success), int(resets)
        if (not 0 <= success <= resets or resets < 1000 or
                abs(float(printed_rate) - success / resets) > 5e-8):
            result.update(status="INVALID_METRICS", reason="invalid task count or rate")
            return result
        counts[state] = (success, resets)
    if set(counts) != set(ORDER):
        result.update(status="INVALID_METRICS", reason="missing official task counts")
        return result

    expected_forwards = 12 if row["cfg"] == "1" else 24
    runtime_lines = re.findall(r"^CFG_RUNTIME SCALE=(\S+) WRAPPED=([01]) GUIDANCE=(\w+)$",
                               log, re.MULTILINE)
    latencies, forwards = [], []
    for line in log.splitlines():
        if not line.startswith("PLANNER_LATENCY_SEC="):
            continue
        values = base.key_values(line)
        try:
            if (values["GENERATOR"] != "flow" or values["SOLVER"] != "rk4" or
                    int(values["ODE_STEPS"]) != 3 or int(values["NFE"]) != 12 or
                    int(values["BATCH"]) != 4096 or
                    int(values["MODEL_FORWARD_CALLS"]) != expected_forwards):
                raise ValueError("planner configuration or MDM forward count mismatch")
            latencies.append(float(values["PLANNER_LATENCY_SEC"]))
            forwards.append(int(values["MODEL_FORWARD_CALLS"]))
        except (KeyError, ValueError) as exc:
            result.update(status="INVALID_METRICS", reason=str(exc))
            return result
    if not latencies or len(runtime_lines) != len(latencies) or any(
            scale != row["cfg"] or wrapped != str(int(row["cfg"] != "1")) or
            guidance != "text" for scale, wrapped, guidance in runtime_lines):
        result.update(status="INVALID_METRICS", reason="missing or wrong runtime CFG evidence")
        return result

    macros = re.findall(r"^EVAL_MACRO_SUCCESS=([0-9.]+)$", log, re.MULTILINE)
    macro = statistics.mean(counts[state][0] / counts[state][1] for state in ORDER)
    profiles = re.findall(r"^EVAL_PROFILE .+$", log, re.MULTILINE)
    if len(macros) != 1 or abs(float(macros[0]) - macro) > 5e-8 or len(profiles) != 1:
        result.update(status="INVALID_METRICS", reason="missing or inconsistent final profile")
        return result
    profile = base.key_values(profiles[0])
    try:
        if (int(profile["CALLS"]) != len(latencies) or
                int(profile["MODEL_FORWARDS_TOTAL"]) != sum(forwards) or
                abs(float(profile["PLANNER_LATENCY_TOTAL_SEC"]) - sum(latencies)) >
                max(1e-4, len(latencies) * 1e-6) or
                abs(float(profile["PLANNER_LATENCY_MEAN_SEC"]) -
                    statistics.mean(latencies)) > 2e-6):
            raise ValueError("profile differs from per-call logs")
    except (KeyError, ValueError) as exc:
        result.update(status="INVALID_METRICS", reason=str(exc))
        return result
    result.update(status="COMPLETE", counts=counts, latencies=latencies,
                  forwards=forwards, macro=macro)
    return result


def pct(counts, state):
    success, resets = counts[state]
    return "{:.2f}% ({}/{})".format(100 * success / resets, success, resets)


def render(rows, run_dir):
    baseline_row = {"experiment_id": "FM-R12", "solver": "rk4", "steps": "3",
                    "nfe": "12", "job_id": "5185912", "submit_status": "submitted",
                    "stdout": str(BASELINE_LOG), "stderr": str(BASELINE_LOG.with_suffix(".err")),
                    "checkpoint": rows[0]["checkpoint"]}
    baseline = base.inspect_job(baseline_row)
    baseline_ok = baseline["status"] == "COMPLETE"
    baseline_macro = statistics.mean(baseline["rates"].values()) if baseline_ok else None
    lines = ["# Flow Matching CFG 扫描：RK4 三步、12 NFE", "",
             "运行目录：`{}`。状态为 COMPLETE 时才显示正式成功率；分母是实际结束回合数。".format(run_dir),
             "Planner 延迟为同步 CUDA 后每次规划的均值/中位数；MDM 前向计数来自原始模型 hook。", "",
             "| CFG | 状态 | REACH 成功/结束 | KICK 成功/结束 | PUNCH 成功/结束 | SIT 成功/结束 | GET_UP 成功/结束 | Macro | ΔREACH | ΔGET_UP | ΔMacro | 步数 | NFE/规划 | MDM 前向/规划 | MDM 总前向 | Planner 均值/中位数(s) | Job ID |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    inspected = [(row, inspect_job(row)) for row in rows]
    for row, result in inspected:
        complete = result["status"] == "COMPLETE"
        task_cells = [pct(result["counts"], state) for state in ORDER] if complete else ["—"] * 5
        deltas = ["—"] * 3
        if complete and baseline_ok:
            for index, state in enumerate(("REACH", "GET_UP")):
                success, resets = result["counts"][state]
                deltas[index] = "{:+.2f} pp".format(100 * (success / resets - baseline["rates"][state]))
            deltas[2] = "{:+.2f} pp".format(100 * (result["macro"] - baseline_macro))
        fields = [row["cfg"], result["status"], *task_cells,
                  "{:.2f}%".format(100 * result["macro"]) if complete else "—",
                  *deltas,
                  row["steps"], row["nfe"],
                  str(result["forwards"][0]) if complete else "—",
                  str(sum(result["forwards"])) if complete else "—",
                  "{:.3f} / {:.3f}".format(statistics.mean(result["latencies"]),
                                             statistics.median(result["latencies"])) if complete else "—",
                  row["job_id"] or "—"]
        lines.append("| " + " | ".join(fields) + " |")

    lines.extend(["", "## 已有 CFG=7.5 对照", ""])
    if baseline_ok:
        lines.append("已有积分器扫描 Job 5185912：REACH {}；GET_UP {}；五项等权平均 **{:.2f}%**。"
                     .format(pct(baseline["counts"], "REACH"),
                             pct(baseline["counts"], "GET_UP"),
                             100 * baseline_macro))
        seven = next(result for row, result in inspected if row["cfg"] == "7.5")
        if seven["status"] == "COMPLETE":
            for state, name in (("REACH", "REACH"), ("GET_UP", "GET_UP")):
                delta = 100 * (seven["counts"][state][0] / seven["counts"][state][1] -
                               baseline["rates"][state])
                lines.append("新 CFG=7.5 的 {} 与已有结果差值：{:+.2f} 个百分点。".format(name, delta))
            lines.append("新 CFG=7.5 的 Macro 与已有结果差值：{:+.2f} 个百分点。"
                         .format(100 * (seven["macro"] - baseline_macro)))
    else:
        lines.append("用户提供的已有 RK4 12 NFE、CFG=7.5 对照为 **83.82%**；原始日志状态为 {}，无法核对其逐项计数。".format(baseline["status"]))

    lines.extend(["", "## 运行元数据", "",
                  "| CFG | checkpoint | Git commit | Dirty | 源码 SHA256 | Seed | 配置 | Planner 调用数 |",
                  "| ---: | --- | --- | --- | --- | ---: | --- | ---: |"])
    for row, result in inspected:
        metadata = result["metadata"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["cfg"], metadata.get("CFG_CHECKPOINT", row["checkpoint"]),
            metadata.get("CFG_GIT_COMMIT", "—"), metadata.get("CFG_GIT_DIRTY", "—"),
            metadata.get("CFG_SOURCE_SHA256", "—"), metadata.get("CFG_SEED", "—"),
            metadata.get("CFG_CONFIG", "—"),
            len(result["latencies"]) if result["status"] == "COMPLETE" else "—"))
    problems = [(row["cfg"], result["status"], result["reason"])
                for row, result in inspected if result["status"] != "COMPLETE"]
    if problems:
        lines.extend(["", "## 缺失或失败", ""])
        lines.extend("- CFG={}：{}{}".format(scale, status, "；" + reason if reason else "")
                     for scale, status, reason in problems)
    lines.extend(["", "注：所有组的 MDM 时间编码均使用最近整数索引；RK4 中间时刻也会取整，因此不保证理论四阶收敛。", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write to a new Markdown file; refuse overwrite")
    args = parser.parse_args(argv)
    manifest = args.run_dir / "submissions.tsv"
    if not manifest.is_file():
        parser.error("missing submission manifest: {}".format(manifest))
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if (tuple(row.get("experiment_id") for row in rows) != IDS or
            tuple(row.get("cfg") for row in rows) != SCALES or
            any(row.get("solver") != "rk4" or row.get("steps") != "3" or
                row.get("nfe") != "12" for row in rows)):
        parser.error("manifest does not contain the exact ordered six-task matrix")
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
