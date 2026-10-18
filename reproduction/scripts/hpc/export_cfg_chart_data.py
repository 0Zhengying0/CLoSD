#!/usr/bin/env python3
"""Export the six completed FM CFG evaluations as chart-ready Excel data."""

import argparse
import statistics
from pathlib import Path

from export_success_chart_data import (
    DISPLAY, FM_CHECKPOINT, ROOT, TASKS, chart_rows, read_manifest, relative,
    write_xlsx,
)
from summarize_fm_cfg_sweep import IDS, SCALES, inspect_job as inspect_cfg
from summarize_fm_solver_sweep import inspect_job as inspect_fixed

MANIFEST = ROOT / "reproduction/experiments/fm_cfg_sweep/runs/fm_cfg_20261014_01/submissions.tsv"
DEFAULT_OUTPUT = ROOT / "reproduction/experiments/chart_data/closd_cfg_success_20261018.xlsx"
BASELINE_LOG = ROOT / "reproduction/experiments/fm_solver_sweep/runs/fm_solver_20261012_01/logs/FM-R12_5185912.out"


def build(output):
    rows = read_manifest(MANIFEST)
    if tuple(row["experiment_id"] for row in rows) != IDS or tuple(row["cfg"] for row in rows) != SCALES:
        raise ValueError("CFG manifest must contain all six ordered experiments")
    records = []
    for row in rows:
        if (row["checkpoint"] != str(FM_CHECKPOINT) or row["solver"] != "rk4" or
                row["steps"] != "3" or row["nfe"] != "12"):
            raise ValueError("CFG experiment model or solver configuration mismatch: " + row["experiment_id"])
        result = inspect_cfg(row)
        if result["status"] != "COMPLETE":
            raise ValueError("{}: {} ({})".format(row["experiment_id"], result["status"], result["reason"]))
        rates = {task: result["counts"][task][0] / result["counts"][task][1] for task in TASKS}
        if abs(statistics.mean(rates.values()) - result["macro"]) > 5e-8:
            raise ValueError("CFG macro success differs from five task rates")
        records.append(dict(label="CFG=" + row["cfg"], cfg=float(row["cfg"]),
                            rates=rates, counts=result["counts"],
                            job_id=row["job_id"], checkpoint=row["checkpoint"],
                            log=relative(row["stdout"].replace("%j", row["job_id"])),
                            mdm_forwards=result["forwards"][0],
                            planner_latency=statistics.mean(result["latencies"]),
                            git=result["metadata"]["CFG_GIT_COMMIT"]))

    chart = chart_rows([(record["label"], record) for record in records])
    metadata = [["CFG", "Job ID", "模型", "积分器", "积分步数", "NFE/规划",
                 "MDM前向/规划", "Planner平均延迟(秒)", "Seed", "环境数", "Episode length",
                 "Checkpoint", "Git commit", "日志（仓库相对路径）"]]
    counts = [["CFG", "动作", "成功数", "结束回合数", "成功率"]]
    for record in records:
        metadata.append([record["cfg"], record["job_id"], "FM 300k EMA", "RK4", 3, 12,
                         record["mdm_forwards"], record["planner_latency"], 0, 4096, 500,
                         record["checkpoint"], record["git"], record["log"]])
        for task, display in zip(TASKS, DISPLAY):
            success, resets = record["counts"][task]
            counts.append([record["cfg"], display, success, resets, success / resets])

    baseline_row = {"experiment_id": "FM-R12", "solver": "rk4", "steps": "3",
                    "nfe": "12", "job_id": "5185912", "submit_status": "submitted",
                    "stdout": str(BASELINE_LOG), "stderr": str(BASELINE_LOG.with_suffix(".err")),
                    "checkpoint": str(FM_CHECKPOINT)}
    baseline = inspect_fixed(baseline_row)
    if baseline["status"] != "COMPLETE":
        raise ValueError("Historical CFG 7.5 reference is incomplete: " + baseline["status"])
    seven = next(record for record in records if record["cfg"] == 7.5)
    reference = [["动作 / 指标", "历史 CFG=7.5 (Job 5185912)", "CFG 扫描 7.5 (Job 5199642)"]]
    for task, display in zip(TASKS, DISPLAY):
        reference.append([display, baseline["rates"][task], seven["rates"][task]])
    reference.append(["总体成功率（五项等权）", statistics.mean(baseline["rates"].values()),
                      statistics.mean(seven["rates"].values())])
    notes = [
        ["说明", "内容"],
        ["主图数据", "六组 CFG 扫描的单次正式闭环评测；同一 FM 300k EMA checkpoint、RK4 3步、12 NFE、文字引导。"],
        ["共同条件", "Seed=0、4096环境、episode length=500、每类至少1000个结束回合。"],
        ["总体成功率", "REACH、KICK、PUNCH、SIT、GET_UP 五类成功率的等权平均。"],
        ["数值精度", "由各日志成功数/结束回合数计算；Excel 显示两位小数百分比。"],
        ["CFG=1 前向次数", "无引导包装，12 NFE 对应 12 次实际 MDM 前向；CFG>1 各为 24 次。"],
        ["历史对照", "历史 CFG=7.5、RK4 12 NFE 的 Job 5185912 为 83.82%；本次 CFG 扫描 Job 5199642 为 83.64%。主图仅使用同一次 CFG 扫描的六组结果。"],
        ["作图", "选择“CFG柱状图数据”工作表 A1 到 G7，插入簇状柱形图；数值单元格已设置百分比格式。"],
    ]
    sheets = [
        ("CFG柱状图数据", chart, tuple(range(2, 8))),
        ("实验元数据", metadata, ()),
        ("原始计数", counts, (5,)),
        ("历史7点5对照", reference, (2, 3)),
        ("说明", notes, ()),
    ]
    write_xlsx(output, sheets)
    print("XLSX={} SHEETS={} CFG_GROUPS={}".format(output, len(sheets), len(records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
