# CLoSD 成功率柱状图数据

Excel 文件：`closd_success_three_charts_20261018.xlsx`。前三个工作表分别对应：

1. 官方设置重训 DiP 300k、扩散采样 10 步（Job `5158705`）与三角函数插值训练的 FM 300k、Euler 10 步（Job `5153280`）。
2. FM 300k 在 12、20、40 NFE 下的 Euler、Heun、RK4 九组正式评测（Jobs `5185910`–`5185918`）。
3. FM 300k 在约 96 NFE 预算下的四种求解器（Jobs `5220314`–`5220317`）；DOPRI5 实测平均 99.2 NFE，范围 86–104。

每张图表数据表均以五类动作和“五项等权总体成功率”为行，以待比较的实验为列。数值是 Excel 百分比单元格；可直接选择 A1 到末行末列插入簇状柱形图。另有“实验元数据”“原始计数”“说明”工作表。

第一组的早期日志仅显示整数百分点，没有原始计数。图中的总体值是五个已显示百分比的等权平均。其余两组从正式日志的成功数和结束回合数计算，并显示到小数点后两位百分比。各组都是单次闭环评测。

使用下列命令从原始日志重新生成；已有文件不会被覆盖：

```bash
python3 reproduction/scripts/hpc/export_success_chart_data.py --output /tmp/closd_success_three_charts.xlsx
```

## CFG 对照

`closd_cfg_success_20261018.xlsx` 从同一次 CFG 正式扫描的六个 Job `5199639`–`5199644` 提取 CFG 1、2.5、5、7.5、10、15 的五类动作及五项等权总体成功率。所有组共用 FM 300k EMA、RK4 三步/12 NFE。主图只包含这六组；历史 CFG=7.5、83.82% 的 Job `5185912` 放在独立的“历史7点5对照”工作表。Excel 另附元数据、原始计数和说明。

```bash
python3 reproduction/scripts/hpc/export_cfg_chart_data.py --output /tmp/closd_cfg_success.xlsx
```

## DiP / FM 视频案例与逐回合诊断

`dip_fm_episode_video_analysis_20261018.xlsx` 在不同工作表中整理：20个共用条件的视频案例配对、40条规划器质量指标、各任务均值/中位数、20个官方骨架视频索引、5个闭环仿真录制视频索引，以及**独立实验**的8652条闭环逐回合诊断和官方成功率。视频案例是离线规划器输出；逐回合官方成功率来自另一批闭环作业，不把这些视频案例算作成功回合。

```bash
python3 reproduction/scripts/hpc/export_episode_video_analysis.py --output /tmp/dip_fm_episode_video_analysis.xlsx
```
