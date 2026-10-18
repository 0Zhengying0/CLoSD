#!/usr/bin/env python3
"""Export verified CLoSD closed-loop success rates to a chart-ready XLSX."""

import argparse
import csv
import json
import re
import statistics
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
from summarize_fm_solver_sweep import inspect_job as inspect_fixed
from summarize_fm_dopri5 import inspect as inspect_matched

TASKS = ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT", "GET_UP")
DISPLAY = ("REACH", "KICK", "PUNCH", "SIT", "GET_UP")
FIXED_MANIFEST = ROOT / "reproduction/experiments/fm_solver_sweep/runs/fm_solver_20261012_01/submissions.tsv"
MATCHED_MANIFEST = ROOT / "reproduction/experiments/fm_dopri5/runs/dopri5_20261017_01/matched_submissions_retry01.tsv"
DEFAULT_OUTPUT = ROOT / "reproduction/experiments/chart_data/closd_success_three_charts_20261018.xlsx"
FM_CHECKPOINT = ROOT / "reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt"
DIP_CHECKPOINT = ROOT / "reproduction/runs/DiP_multi-target_official_recipe_seed10/model000300000.pt"


def read_manifest(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def relative(path):
    return str(Path(path).relative_to(ROOT))


def load_early(label, path, checkpoint, solver, steps, job_id):
    text = path.read_text(errors="replace")
    required = ("Collected enough data (1000 per task) success rates:",
                "Eval plot ended", "Setting seed: 0", "Batch:     4096 environments",
                "Model:     " + str(checkpoint), "Solver:    " + solver)
    if any(marker not in text for marker in required):
        raise ValueError("Early evaluation missing complete/configuration marker: " + str(path))
    if solver == "euler" and "DiP sampling backend: flow | solver=euler | steps=10 | NFE=10" not in text:
        raise ValueError("FM Euler-10 sampling setup differs: " + str(path))
    block = text.split("Collected enough data (1000 per task) success rates:", 1)[1]
    block = block.split("Eval plot ended", 1)[0]
    found = re.findall(r"^STATES\.([A-Z_]+): (0\.\d\d|1\.00)$", block, re.M)
    if len(found) != 5 or set(k for k, _ in found) != set(TASKS):
        raise ValueError("Early log must contain five rounded task rates: " + str(path))
    rates = {key: float(value) for key, value in found}
    return dict(label=label, family="DiP" if solver == "diffusion" else "FM",
                solver=solver, steps=steps, nfe_mean=10, nfe_min=10, nfe_max=10,
                job_id=job_id, checkpoint=str(checkpoint), log=relative(path),
                rates=rates, counts=None, precision="日志仅公布至整数百分点")


def load_fixed():
    rows = read_manifest(FIXED_MANIFEST)
    if len(rows) != 9 or {r["experiment_id"] for r in rows} != {
            "FM-{}{}".format(letter, n) for n in (12, 20, 40) for letter in "EHR"}:
        raise ValueError("Fixed-NFE manifest is incomplete")
    results = {}
    for row in rows:
        result = inspect_fixed(row)
        if result["status"] != "COMPLETE":
            raise ValueError("{}: {} ({})".format(row["experiment_id"], result["status"], result["reason"]))
        if row["checkpoint"] != str(FM_CHECKPOINT):
            raise ValueError("Fixed experiment checkpoint mismatch")
        key = row["experiment_id"]
        results[key] = dict(label=key, family="FM", solver=row["solver"],
                            steps=int(row["steps"]), nfe_mean=int(row["nfe"]),
                            nfe_min=int(row["nfe"]), nfe_max=int(row["nfe"]),
                            job_id=row["job_id"], checkpoint=row["checkpoint"],
                            log=relative(row["stdout"].replace("%j", row["job_id"])),
                            rates=result["rates"], counts=result["counts"],
                            precision="成功数/结束回合数，图中显示两位小数百分比")
    return results


def load_matched():
    rows = read_manifest(MATCHED_MANIFEST)
    if [row["experiment_id"] for row in rows] != ["FM-ME", "FM-MH", "FM-MR", "FM-MD"]:
        raise ValueError("Matched-NFE retry manifest is incomplete")
    results = {}
    for row in rows:
        result = inspect_matched(row)
        if result["status"] != "COMPLETE":
            raise ValueError("{}: {} ({})".format(row["experiment_id"], result["status"], result["reason"]))
        if row["checkpoint"] != str(FM_CHECKPOINT) or row["target_nfe"] != "96":
            raise ValueError("Matched experiment checkpoint or NFE budget mismatch")
        nfe = result["nfe"]
        if row["solver"] == "dopri5" and round(statistics.mean(nfe), 1) != 99.2:
            raise ValueError("DOPRI5 actual mean NFE no longer equals 99.2")
        key = row["experiment_id"]
        results[key] = dict(label=key, family="FM", solver=row["solver"],
                            steps="自适应 (ODE_STEPS=0)" if row["solver"] == "dopri5" else int(row["steps"]),
                            nfe_mean=statistics.mean(nfe), nfe_min=min(nfe), nfe_max=max(nfe),
                            job_id=row["job_id"], checkpoint=row["checkpoint"],
                            log=relative(row["stdout"].replace("%j", row["job_id"])),
                            rates={task: result["counts"][task][0] / result["counts"][task][1]
                                   for task in TASKS}, counts=result["counts"],
                            precision="成功数/结束回合数，图中显示两位小数百分比")
    return results


def column_name(number):
    name = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        name = chr(65 + remainder) + name
    return name


def sheet_xml(rows, percent_columns=()):
    widths = []
    for column in range(max(map(len, rows))):
        lengths = [len(str(row[column])) for row in rows if column < len(row) and row[column] is not None]
        widths.append(min(max(max(lengths or [8]) + 2, 12), 65))
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
             '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
             'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>',
             '<sheetFormatPr defaultRowHeight="16"/><cols>']
    for i, width in enumerate(widths, 1):
        parts.append('<col min="{0}" max="{0}" width="{1}" customWidth="1"/>'.format(i, width))
    parts.append('</cols><sheetData>')
    for row_number, row in enumerate(rows, 1):
        parts.append('<row r="{}">'.format(row_number))
        for column_number, value in enumerate(row, 1):
            ref = column_name(column_number) + str(row_number)
            style = 1 if row_number == 1 else (2 if column_number in percent_columns else 0)
            if value is None:
                parts.append('<c r="{}" s="{}"/>'.format(ref, style))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                parts.append('<c r="{}" s="{}"><v>{}</v></c>'.format(ref, style, value))
            else:
                parts.append('<c r="{}" s="{}" t="inlineStr"><is><t xml:space="preserve">{}</t></is></c>'.format(
                    ref, style, escape(str(value))))
        parts.append('</row>')
    parts.append('</sheetData></worksheet>')
    return "".join(parts)


STYLES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2"><font><sz val="10"/><name val="Arial"/></font><font><b/><sz val="10"/><name val="Arial"/></font></fonts>
<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFDCEAF7"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="1" fillId="1" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
<xf numFmtId="10" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def write_xlsx(path, sheets):
    if path.exists():
        raise FileExistsError("Refusing to overwrite existing workbook: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    content_types = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                     '<Default Extension="xml" ContentType="application/xml"/>',
                     '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
                     '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    workbook = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>']
    relationships = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                     '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for index, (name, _rows, _percent_columns) in enumerate(sheets, 1):
        content_types.append('<Override PartName="/xl/worksheets/sheet{0}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'.format(index))
        workbook.append('<sheet name={} sheetId="{}" r:id="rId{}"/>'.format(quoteattr(name), index, index))
        relationships.append('<Relationship Id="rId{0}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{0}.xml"/>'.format(index))
    content_types.append('</Types>')
    workbook.append('</sheets></workbook>')
    relationships.append('<Relationship Id="rId{}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'.format(len(sheets) + 1))
    with zipfile.ZipFile(str(path), 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', ''.join(content_types))
        archive.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                         '</Relationships>')
        archive.writestr('xl/workbook.xml', ''.join(workbook))
        archive.writestr('xl/_rels/workbook.xml.rels', ''.join(relationships))
        archive.writestr('xl/styles.xml', STYLES)
        for index, (_name, rows, percent_columns) in enumerate(sheets, 1):
            archive.writestr('xl/worksheets/sheet{}.xml'.format(index), sheet_xml(rows, percent_columns))


def chart_rows(series):
    rows = [["动作 / 指标"] + [name for name, _ in series]]
    for task, label in zip(TASKS, DISPLAY):
        rows.append([label] + [record["rates"][task] for _name, record in series])
    rows.append(["总体成功率（五项等权）"] + [statistics.mean(record["rates"].values())
                                      for _name, record in series])
    return rows


def build(output):
    for checkpoint, steps in ((FM_CHECKPOINT, 1000), (DIP_CHECKPOINT, 10)):
        if not checkpoint.is_file() or json.loads((checkpoint.parent / "args.json").read_text())["diffusion_steps"] != steps:
            raise ValueError("Checkpoint or training args mismatch: " + str(checkpoint))
    dip = load_early("DiP 10步", ROOT / "logs/dip_latency_300k_5158705.out",
                     DIP_CHECKPOINT, "diffusion", 10, "5158705")
    fm10 = load_early("FM Euler 10步", ROOT / "logs/fm_10step_5153280.out",
                      FM_CHECKPOINT, "euler", 10, "5153280")
    fixed = load_fixed()
    matched = load_matched()
    chart1 = [(dip["label"], dip), (fm10["label"], fm10)]
    chart2 = [("{} · {} NFE".format(fixed[key]["solver"].upper(), nfe), fixed[key])
              for nfe in (12, 20, 40) for key in ("FM-E{}".format(nfe),
                                                   "FM-H{}".format(nfe), "FM-R{}".format(nfe))]
    chart3 = [("{} · {} NFE".format(matched[key]["solver"].upper(),
                                   "平均99.2" if key == "FM-MD" else "96"), matched[key])
              for key in ("FM-ME", "FM-MH", "FM-MR", "FM-MD")]
    all_records = [dip, fm10] + [record for _label, record in chart2] + [record for _label, record in chart3]
    metadata = [["实验", "模型", "积分器/采样", "积分步数", "实际平均NFE", "NFE最小", "NFE最大",
                 "Job ID", "checkpoint", "日志（仓库相对路径）", "数据精度"]]
    count_rows = [["实验", "动作", "成功数", "结束回合数", "成功率", "计数说明"]]
    for record in all_records:
        metadata.append([record[key] for key in ("label", "family", "solver", "steps",
                                                 "nfe_mean", "nfe_min", "nfe_max", "job_id",
                                                 "checkpoint", "log", "precision")])
        for task, display in zip(TASKS, DISPLAY):
            count = record["counts"][task] if record["counts"] else (None, None)
            count_rows.append([record["label"], display, count[0], count[1],
                               record["rates"][task], record["precision"]])
    notes = [
        ["说明", "内容"],
        ["数据范围", "CLoSD 官方闭环成功口径；五类任务为 REACH、KICK、PUNCH、SIT、GET_UP。"],
        ["总体成功率", "五类任务成功率的等权算术平均，不是将五类成功数合并后计算的加权成功率。"],
        ["图1数据精度", "早期 DiP/FM 日志只显示每类成功率至整数百分点；总体由这五个已发布数值等权计算。原始成功数未提供，留空。"],
        ["图1模型", "DiP 是官方设置重训 300k、扩散采样10步；FM 是 300k EMA、Euler10步。使用 5158705 DiP 延迟基准评测，与原项目时间线保持一致。"],
        ["FM训练插值", "FM 训练代码为 cos(pi*t/2)*噪声 + sin(pi*t/2)*数据，即三角函数插值路径。"],
        ["图2和图3精度", "从各正式评测日志的成功数/结束回合数重新计算，图中百分比显示两位小数。"],
        ["DOPRI5 NFE", "ODE_STEPS=0 表示自适应；96 是固定求解器的公共预算。DOPRI5 正式运行的实际平均NFE为99.2，范围86–104。"],
        ["共同条件", "FM 扫描采用 CFG=7.5、seed=0、4096环境、episode length=500、每类至少1000个结束回合。"],
        ["图表操作", "选择任一图表数据工作表的 A1 至末行末列，再在 Excel 插入簇状柱形图；数值单元格已经设为百分比格式。"],
        ["注意", "各行是单次评测；图1仅有整数百分点精度，图2/3有原始计数。跨预算或不同日期运行的差异不等同于统计显著性。"],
    ]
    sheets = [
        ("图1_DiP与FM", chart_rows(chart1), (2, 3)),
        ("图2_固定NFE", chart_rows(chart2), tuple(range(2, 11))),
        ("图3_加入DOPRI5", chart_rows(chart3), (2, 3, 4, 5)),
        ("实验元数据", metadata, ()),
        ("原始计数", count_rows, (5,)),
        ("说明", notes, ()),
    ]
    write_xlsx(output, sheets)
    print("XLSX={} SHEETS={} RUNS={}".format(output, len(sheets), len(all_records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
