"""Headless, fixed-camera HumanML skeleton videos for matched planner inputs."""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from closd.diffusion_planner.data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain


def common_bounds(dip_xyz, fm_xyz, target, conditioned_target=None):
    points = np.concatenate((np.asarray(dip_xyz).reshape(-1, 3),
                             np.asarray(fm_xyz).reshape(-1, 3),
                             np.asarray(target).reshape(1, 3),
                             np.asarray(conditioned_target).reshape(1, 3)
                             if conditioned_target is not None else
                             np.asarray(target).reshape(1, 3)))
    if not np.isfinite(points).all():
        raise ValueError("nonfinite rendering coordinates")
    low, high = points.min(0), points.max(0)
    center = (low + high) / 2
    half = max(float((high - low).max() / 2) + 0.25, 1.0)
    return center, half


def draw_pose(ax, xyz, frame, prefix_frames, target, title, center, half, color,
              conditioned_target=None):
    ax.cla()
    coords = xyz[frame]
    for chain in t2m_kinematic_chain:
        segment = coords[chain]
        ax.plot(segment[:, 0], segment[:, 2], segment[:, 1],
                color="#6b7280" if frame < prefix_frames else color,
                linewidth=2.2)
    # Draw observed prefix path and generated root path separately.
    prefix_path = xyz[:min(frame + 1, prefix_frames), 0]
    if len(prefix_path) > 1:
        ax.plot(prefix_path[:, 0], prefix_path[:, 2], prefix_path[:, 1],
                color="#6b7280", linewidth=1.4, alpha=0.9)
    if frame >= prefix_frames:
        generated_path = xyz[prefix_frames - 1:frame + 1, 0]
        ax.plot(generated_path[:, 0], generated_path[:, 2], generated_path[:, 1],
                color=color, linewidth=1.5, alpha=0.9)
    ax.scatter([target[0]], [target[2]], [target[1]],
               color="#d12e8a", s=40, marker="*", depthshade=False)
    ax.text(target[0], target[2], target[1] + 0.08, "env goal", color="#a31468", fontsize=7)
    if conditioned_target is not None:
        ax.scatter([conditioned_target[0]], [conditioned_target[2]],
                   [conditioned_target[1]], color="#2563eb", s=24,
                   marker="D", depthshade=False)
        ax.text(conditioned_target[0], conditioned_target[2],
                conditioned_target[1] + 0.08, "planner goal", color="#1d4ed8", fontsize=7)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[2] - half, center[2] + half)
    ax.set_zlim(center[1] - half, center[1] + half)
    ax.set_xlabel("HML X (m)")
    ax.set_ylabel("HML Z (m)")
    ax.set_zlabel("HML Y (m)")
    ax.view_init(elev=18, azim=-65)
    ax.set_box_aspect((1, 1, 1))
    phase = "observed prefix" if frame < prefix_frames else "generated"
    ax.set_title("{} | {} | t={:.2f}s".format(title, phase, frame / 20), fontsize=8)


def render_video(path, dip_xyz, fm_xyz, target, title, prefix_frames=20,
                 fps=20, side_by_side=False, model="dip", gif_path=None,
                 conditioned_target=None):
    dip_xyz, fm_xyz = np.asarray(dip_xyz), np.asarray(fm_xyz)
    if dip_xyz.shape != fm_xyz.shape or dip_xyz.ndim != 3 or dip_xyz.shape[1:] != (22, 3):
        raise ValueError("both trajectories must have identical [T,22,3] shape")
    center, half = common_bounds(dip_xyz, fm_xyz, target, conditioned_target)
    panels = 2 if side_by_side else 1
    fig = plt.figure(figsize=(10 if panels == 2 else 5, 5), dpi=80)
    axes = [fig.add_subplot(1, panels, index + 1, projection="3d")
            for index in range(panels)]
    if side_by_side:
        items = [(dip_xyz, "DiP | Diffusion 10 | CFG 7.5", "#d97706"),
                 (fm_xyz, "FM | RK4 3 / 12 NFE | CFG 7.5", "#007f86")]
    elif model == "dip":
        items = [(dip_xyz, "DiP | Diffusion 10 | CFG 7.5", "#d97706")]
    else:
        items = [(fm_xyz, "FM | RK4 3 / 12 NFE | CFG 7.5", "#007f86")]
    frames_for_gif = []
    try:
        with imageio.get_writer(str(path), format="FFMPEG", mode="I", fps=fps,
                                codec="libx264", quality=8, macro_block_size=16) as writer:
            for frame in range(len(dip_xyz)):
                for ax, (trajectory, label, color) in zip(axes, items):
                    draw_pose(ax, trajectory, frame, prefix_frames, target,
                              label, center, half, color, conditioned_target)
                fig.suptitle(title, fontsize=9)
                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())
                rgb = rgba[..., :3].copy()
                writer.append_data(rgb)
                if gif_path is not None:
                    frames_for_gif.append(rgb)
    finally:
        plt.close(fig)
    if gif_path is not None:
        imageio.mimsave(str(gif_path), frames_for_gif, format="GIF", duration=1 / fps)
    verify_media(path)
    if gif_path is not None:
        verify_media(gif_path)


def verify_media(path):
    if os.path.getsize(path) < 1024:
        raise ValueError("media file is empty or too small: {}".format(path))
    reader = imageio.get_reader(str(path))
    try:
        first = reader.get_data(0)
        if first.ndim != 3 or first.shape[-1] < 3 or not np.isfinite(first).all():
            raise ValueError("media first frame cannot be decoded: {}".format(path))
    finally:
        reader.close()


def render_distance_curve(path, dip_curve, fm_curve, task, target_joint,
                          dip_conditioned=None, fm_conditioned=None):
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    try:
        for curve, label, color in ((dip_curve, "DiP / Diffusion 10", "#d97706"),
                                    (fm_curve, "FM / RK4 3", "#007f86")):
            curve = np.asarray(curve)
            ax.plot(curve[:, 0], curve[:, 1], label=label, color=color, linewidth=2)
        for curve, label, color in ((dip_conditioned, "DiP / planner goal", "#d97706"),
                                    (fm_conditioned, "FM / planner goal", "#007f86")):
            if curve is not None:
                curve = np.asarray(curve)
                ax.plot(curve[:, 0], curve[:, 1], label=label, color=color,
                        linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_xlabel("Time into generated motion (s)")
        ax.set_ylabel("Environment goal distance (m); dashed = planner goal")
        ax.set_title("{}: {} target distance".format(task, target_joint))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)
