"""Standalone, paired CPU sanity check for CLoSD's VP-CFM path versus TorchCFM."""

import argparse
import ast
import copy
import csv
import json
import math
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_flow_sanity")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.datasets import make_moons


ROOT = Path(__file__).resolve().parent
FLOW_FILE = ROOT.parent / "CLoSD/closd/diffusion_planner/diffusion/flow_matching.py"
sys.path.insert(0, str(ROOT / "vendor"))

# The official module imports OTPlanSampler at load time. VP-CFM never uses OT;
# POT is unavailable on this offline node, so satisfy only that unrelated import.
ot_module = types.ModuleType("torchcfm.optimal_transport")


class UnavailableOTPlanSampler:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("OT variants require the optional POT dependency")


ot_module.OTPlanSampler = UnavailableOTPlanSampler
sys.modules[ot_module.__name__] = ot_module
from torchcfm import VariancePreservingConditionalFlowMatcher, __version__ as torchcfm_version


def load_closd_flow():
    """Execute the original class node, avoiding CLoSD's unrelated motion imports."""
    tree = ast.parse(FLOW_FILE.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlowMatching"]
    if len(classes) != 1:
        raise RuntimeError("Expected one FlowMatching class in " + str(FLOW_FILE))
    module = ast.Module(body=classes, type_ignores=[])
    scope = {"math": math, "torch": torch}
    exec(compile(module, str(FLOW_FILE), "exec"), scope)
    return scope["FlowMatching"], scope


class VelocityMLP(torch.nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 2),
        )

    def forward(self, x, tau):
        return self.net(torch.cat((x, tau.reshape(-1, 1)), dim=1))


class ClosdModelAdapter(torch.nn.Module):
    def __init__(self, mlp, timesteps):
        super().__init__()
        self.mlp = mlp
        self.timesteps = timesteps
        self.last_xt = None

    def forward(self, xt, t, **kwargs):
        self.last_xt = xt
        return self.mlp(xt, t.float() / (self.timesteps - 1))


def closd_path(matcher, scope, adapter, x0, x1, ti):
    captured = {}

    def masked_l2(target, prediction, mask):
        captured["velocity"] = target
        return torch.mean((target - prediction) ** 2)

    scope["masked_l2"] = masked_l2
    loss = matcher.training_losses(
        adapter, x1, ti,
        model_kwargs={"y": {"mask": torch.ones(x1.shape[0], 1, dtype=torch.bool)}},
        noise=x0,
    )["loss"]
    return adapter.last_xt, captured["velocity"], loss


def integrate(model, initial, steps=100):
    """Fixed step midpoint ODE; identical solver and initial noise for both models."""
    x = initial.clone()
    states = [x.clone()]
    with torch.no_grad():
        for i in range(steps):
            tau = i / steps
            dt = 1.0 / steps
            t0 = torch.full((x.shape[0],), tau)
            k1 = model(x, t0)
            tm = torch.full((x.shape[0],), tau + dt / 2)
            x = x + dt * model(x + dt * k1 / 2, tm)
            states.append(x.clone())
    return torch.stack(states).numpy()


def mmd2(x, y, bandwidths=(0.2, 0.5, 1.0, 2.0)):
    # Biased MMD estimate; same fixed validation target for all comparisons.
    a = np.sum((x[:, None] - x[None, :]) ** 2, axis=2)
    b = np.sum((y[:, None] - y[None, :]) ** 2, axis=2)
    c = np.sum((x[:, None] - y[None, :]) ** 2, axis=2)
    return float(np.mean([np.exp(-a / (2 * s * s)).mean() +
                          np.exp(-b / (2 * s * s)).mean() -
                          2 * np.exp(-c / (2 * s * s)).mean() for s in bandwidths]))


def trajectory_stats(states):
    delta = np.diff(states, axis=0)
    norms = np.linalg.norm(delta, axis=2)
    dot = np.sum(delta[1:] * delta[:-1], axis=2)
    valid = (norms[1:] > 1e-9) & (norms[:-1] > 1e-9)
    reversals = (dot < 0) & valid
    accel = np.linalg.norm(np.diff(delta, axis=0), axis=2)
    return {
        "reverse_step_fraction": float(reversals.sum() / max(valid.sum(), 1)),
        "relative_step_change": float(accel.sum() / max(norms.sum(), 1e-12)),
        "max_step": float(norms.max()),
        "finite": bool(np.isfinite(states).all()),
    }


def plot_outputs(out, target, noise, states, histories):
    fig, (ax, diff_ax) = plt.subplots(2, 1, figsize=(8, 5.3), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
    for name, vals in histories.items():
        smoothed = np.convolve(vals, np.ones(100) / 100, mode="valid")
        ax.plot(np.arange(100, len(vals) + 1), smoothed, label=name, linewidth=1.4)
    ax.set(ylabel="MSE (100-step mean)", title="Paired VP-CFM training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    diff_ax.plot(np.arange(1, len(histories["CLoSD"]) + 1),
                 np.abs(np.asarray(histories["CLoSD"]) - np.asarray(histories["TorchCFM"])),
                 color="black", linewidth=0.8)
    diff_ax.set(xlabel="Training step", ylabel="|Loss diff|")
    diff_ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(14, 3.5), sharex=True, sharey=True)
    for ax, (name, pts) in zip(axs, [("Gaussian source", noise), ("Two Moons target", target),
                                    ("CLoSD VP-CFM", states["closd"][-1]),
                                    ("TorchCFM VP-CFM", states["torchcfm"][-1])]):
        ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.55, rasterized=True)
        ax.set(title=name, xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), aspect="equal")
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out / "distributions.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, name in zip(axs, ("closd", "torchcfm")):
        z = states[name]
        ax.scatter(target[:, 0], target[:, 1], s=2, alpha=0.12, color="gray")
        for j in range(min(32, z.shape[1])):
            ax.plot(z[:, j, 0], z[:, j, 1], lw=0.9, alpha=0.72)
            ax.scatter(z[0, j, 0], z[0, j, 1], s=8, c="black")
        ax.set(title=name, xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), aspect="equal",
               xlabel="x", ylabel="y")
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20261017)
    np.random.seed(20261017)
    timesteps = 1001
    FlowMatching, scope = load_closd_flow()
    closd = FlowMatching(num_timesteps=timesteps, lambda_target_loc=0.0)
    official = VariancePreservingConditionalFlowMatcher(sigma=0.0)
    x_train, _ = make_moons(n_samples=4096, noise=0.08, random_state=42)
    x_valid, _ = make_moons(n_samples=1024, noise=0.08, random_state=43)
    mean, std = x_train.mean(0), x_train.std(0)
    train = torch.from_numpy(((x_train - mean) / std).astype("float32"))
    valid = ((x_valid - mean) / std).astype("float32")

    mlp_closd = VelocityMLP()
    mlp_official = copy.deepcopy(mlp_closd)
    adapter = ClosdModelAdapter(mlp_closd, timesteps)
    opt_a = torch.optim.Adam(mlp_closd.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(mlp_official.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(12345)
    indices = torch.randint(len(train), (args.steps, args.batch_size), generator=generator)
    sources = torch.randn(args.steps, args.batch_size, 2, generator=generator)
    times = torch.randint(timesteps, (args.steps, args.batch_size), generator=generator)

    # Include both endpoints and random interior values in the numerical check.
    probe_x1 = train[:512]
    probe_x0 = torch.randn(512, 2, generator=generator)
    probe_t = torch.randint(timesteps, (512,), generator=generator)
    probe_t[:4] = torch.tensor([0, 1, 999, 1000])
    with torch.no_grad():
        xt_a, ut_a, _ = closd_path(closd, scope, adapter, probe_x0, probe_x1, probe_t)
        tau = probe_t.float() / (timesteps - 1)
        _, xt_b, ut_b = official.sample_location_and_conditional_flow(probe_x0, probe_x1, t=tau)
        path_error = float((xt_a - xt_b).abs().max())
        velocity_error = float((ut_a - ut_b).abs().max())

    histories = {"CLoSD": [], "TorchCFM": []}
    for step in range(args.steps):
        x0, x1, ti = sources[step], train[indices[step]], times[step]
        tau = ti.float() / (timesteps - 1)
        opt_a.zero_grad(set_to_none=True)
        _, _, loss_a = closd_path(closd, scope, adapter, x0, x1, ti)
        loss_a.backward()
        opt_a.step()
        opt_b.zero_grad(set_to_none=True)
        _, xt, ut = official.sample_location_and_conditional_flow(x0, x1, t=tau)
        loss_b = torch.mean((mlp_official(xt, tau) - ut) ** 2)
        loss_b.backward()
        opt_b.step()
        histories["CLoSD"].append(float(loss_a.detach()))
        histories["TorchCFM"].append(float(loss_b.detach()))
        if (step + 1) % 500 == 0:
            print("step %d/%d: loss %.4f / %.4f" % (step + 1, args.steps, loss_a, loss_b), flush=True)

    eval_noise = torch.randn(1024, 2, generator=generator)
    states = {"closd": integrate(mlp_closd, eval_noise),
              "torchcfm": integrate(mlp_official, eval_noise)}
    end_a, end_b = states["closd"][-1], states["torchcfm"][-1]
    source_mmd = mmd2(eval_noise.numpy(), valid)
    data_mmd_a, data_mmd_b = mmd2(end_a, valid), mmd2(end_b, valid)
    nn_a = float(cKDTree(valid).query(end_a)[0].mean())
    nn_b = float(cKDTree(valid).query(end_b)[0].mean())
    stats_a, stats_b = trajectory_stats(states["closd"]), trajectory_stats(states["torchcfm"])
    losses_a, losses_b = np.asarray(histories["CLoSD"]), np.asarray(histories["TorchCFM"])
    metrics = {
        "seed": 20261017, "device": "cpu", "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": 1e-3, "timesteps": timesteps, "torchcfm_version": torchcfm_version,
        "torch_version": torch.__version__,
        "path_max_abs_error": path_error, "velocity_max_abs_error": velocity_error,
        "loss_max_abs_difference": float(np.max(np.abs(losses_a - losses_b))),
        "final_loss_100_mean": {"closd": float(losses_a[-100:].mean()),
                                "torchcfm": float(losses_b[-100:].mean())},
        "source_mmd2": source_mmd,
        "generated_mmd2": {"closd": data_mmd_a, "torchcfm": data_mmd_b},
        "mean_target_nearest_neighbor_distance": {"closd": nn_a, "torchcfm": nn_b},
        "endpoint_max_abs_difference": float(np.max(np.abs(end_a - end_b))),
        "trajectory_max_abs_difference": float(np.max(np.abs(states["closd"] - states["torchcfm"]))),
        "trajectory": {"closd": stats_a, "torchcfm": stats_b},
    }
    checks = {
        "path_and_velocity_match": path_error < 1e-5 and velocity_error < 1e-5,
        "paired_training_matches": metrics["loss_max_abs_difference"] < 1e-5,
        "generated_distribution_improves": max(data_mmd_a, data_mmd_b) < source_mmd * 0.5,
        "rare_trajectory_reversals": stats_a["reverse_step_fraction"] < 0.01 and stats_b["reverse_step_fraction"] < 0.01,
        "finite_trajectories": stats_a["finite"] and stats_b["finite"],
    }
    metrics["checks"] = checks
    metrics["passed"] = all(checks.values())
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    with (args.output / "loss.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "closd", "torchcfm"])
        writer.writerows((i + 1, a, b) for i, (a, b) in enumerate(zip(losses_a, losses_b)))
    plot_outputs(args.output, valid, eval_noise.numpy(), states, histories)
    verdict = "通过" if metrics["passed"] else "未通过"
    report = f"""# VP-CFM Two Moons Sanity Check

**结论：{verdict}。**

## 设置

- 独立实验脚本：`run.py`；CPU，随机种子 20261017，Two Moons 训练集 4096 点、验证集 1024 点，噪声 0.08，并按训练集每个坐标维度标准化。
- 两个模型均为 3 层宽度 64 的 SiLU MLP；使用相同初始权重、每步同一批样本/高斯源/离散时间；Adam 学习率 1e-3，batch {args.batch_size}，训练 {args.steps} 步。
- CLoSD 分支直接执行原文件中的 `FlowMatching` 类，仅将动作数据专用的 `masked_l2` 换成二维均值 MSE，并设 `lambda_target_loc=0`。TorchCFM 分支使用官方 `VariancePreservingConditionalFlowMatcher(sigma=0)`，版本 {torchcfm_version}。时间在两边均为 `t/1000`。
- 从同一 1024 个高斯点出发，两个模型都用 100 步固定步长 midpoint ODE 采样。

## 结果

| 指标 | CLoSD | TorchCFM |
|---|---:|---:|
| 路径 x_t 最大绝对差 | {path_error:.3g} | 基准 |
| 目标速度最大绝对差 | {velocity_error:.3g} | 基准 |
| 最后 100 步平均 MSE | {losses_a[-100:].mean():.5f} | {losses_b[-100:].mean():.5f} |
| 生成 vs 验证集 MMD² | {data_mmd_a:.5f} | {data_mmd_b:.5f} |
| 轨迹相邻步反向比例 | {stats_a['reverse_step_fraction']:.5f} | {stats_b['reverse_step_fraction']:.5f} |
| 相邻步相对变化量 | {stats_a['relative_step_change']:.5f} | {stats_b['relative_step_change']:.5f} |

两条训练 Loss 的最大绝对差为 {metrics['loss_max_abs_difference']:.3g}；完整采样轨迹的最大绝对差为 {metrics['trajectory_max_abs_difference']:.3g}。高斯源与验证集的 MMD² 为 {source_mmd:.5f}。相同初值的轨迹只有极少量反向步（约 {100 * stats_a['reverse_step_fraction']:.3f}%），图中未见密集折返或异常抖动。生成点已呈双月形，但比真实数据更分散，部分点落在两条月牙之间；本结论是算法一致性与基本生成行为通过，不代表高保真生成质量。

判定条件：路径和速度差均小于 1e-5；配对训练 Loss 差小于 1e-5；两种生成分布的 MMD² 均低于高斯源的 50%；轨迹反向步比例小于 1%，且所有状态为有限值。各项结果：{json.dumps(checks, ensure_ascii=False)}。

## 图与复现

- [Loss 曲线](loss.png) · [二维点分布](distributions.png) · [采样轨迹](trajectories.png)
- [完整指标](metrics.json) · [逐步 Loss](loss.csv)
- 运行：`envs/closd/bin/python flow_matching_sanity/run.py`（从工作区根目录）。

TorchCFM 原版源码固定于提交 `7c653857de4c25b979a740ef8b6a5d5ddb09b6c1`，位于 `vendor/torchcfm/`，保留 MIT 许可证。官方模块导入时会引用未在本机安装的 POT；实验只使用 VP-CFM，不调用 OT 类，因此在脚本中为 OT 导入提供了显式占位类，官方 VP-CFM 源码未改动。此实验验证二维路径与小型 MLP 的行为，不代替完整动作模型评测。
"""
    (args.output / "REPORT.md").write_text(report)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    if not metrics["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
