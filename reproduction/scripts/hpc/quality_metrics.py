"""Common, model-neutral metrics for planner output in local HumanML metres."""

import math
import numpy as np
import torch

from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import (
    recover_from_ric, recover_root_rot_pos)
from closd.diffusion_planner.data_loaders.humanml.common.quaternion import qrot
from closd.diffusion_planner.data_loaders.humanml_utils import HML_JOINT_NAMES

GOAL_NAMES = ("pelvis", "left_foot", "right_foot", "left_wrist", "right_wrist", "head", "traj", "heading")


def recover_xyz(normalized, mean, std, prefix_frames=None):
    """Normalized HML -> XYZ metres; optionally align to the prefix end.

    CLoSD target_cond is transformed with recon_data at the last observed
    frame, whereas raw recover_from_ric starts at the first observed frame.
    Use the same subtract-XZ-then-qrot step as RepresentationHandler.
    """
    data = np.asarray(normalized, dtype=np.float32)
    if data.ndim != 3 or data.shape[:2] != (263, 1):
        raise ValueError("expected normalized motion [263,1,T]")
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if mean.shape != (263,) or std.shape != (263,) or np.any(std <= 0):
        raise ValueError("invalid HumanML normalization")
    raw = (data[:, 0].T * std + mean)[None]
    with torch.no_grad():
        motion = torch.from_numpy(raw.copy())
        xyz = recover_from_ric(motion, 22)
        if prefix_frames is not None:
            if not 1 <= prefix_frames <= data.shape[-1]:
                raise ValueError("invalid prefix anchor")
            r_rot, r_pos = recover_root_rot_pos(motion)
            anchor_rot = r_rot[:, prefix_frames - 1]
            anchor_pos = r_pos[:, prefix_frames - 1]
            centered = xyz.clone()
            centered[..., 0] -= anchor_pos[:, None, None, 0]
            centered[..., 2] -= anchor_pos[:, None, None, 2]
            xyz = qrot(anchor_rot[:, None, None, :].expand(
                xyz.shape[:-1] + (4,)), centered)
        xyz = xyz[0].numpy()
    if xyz.shape != (data.shape[-1], 22, 3) or not np.isfinite(xyz).all():
        raise ValueError("recovered coordinates invalid")
    return xyz.astype(np.float32)


def heading(xyz_frame):
    # Same shoulders/hips based forward definition used by motion_process.
    across = (xyz_frame[2] - xyz_frame[1]) + (xyz_frame[17] - xyz_frame[16])
    forward = np.cross(np.array([0., 1., 0.]), across)
    return math.atan2(forward[0], forward[2])


def metrics(xyz, target_cond, task, target_joint_names, prefix_frames=20, fps=20,
            actual_target_local=None):
    xyz = np.asarray(xyz, np.float64)
    target_cond = np.asarray(target_cond, np.float64)
    if xyz.ndim != 3 or xyz.shape[1:] != (22, 3) or not np.isfinite(xyz).all():
        raise ValueError("invalid XYZ trajectory")
    if target_cond.shape != (8, 3) or not np.isfinite(target_cond).all():
        raise ValueError("invalid local target")
    if not 1 < prefix_frames < len(xyz) or fps <= 0:
        raise ValueError("invalid prefix or fps")
    if len(target_joint_names) != 1:
        raise ValueError("expected exactly one primary target joint")
    name = target_joint_names[0]
    if name not in GOAL_NAMES or name == "heading":
        raise ValueError("unknown target joint")
    joint = "pelvis" if name == "traj" else name
    conditioned_point = target_cond[GOAL_NAMES.index(name)]
    point = (np.asarray(actual_target_local, np.float64)
             if actual_target_local is not None else conditioned_point)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("invalid environment target in local HML coordinates")
    joint_xyz = xyz[:, HML_JOINT_NAMES.index(joint)]
    dims = [0, 2] if task == "REACH" or name == "traj" else [0, 1, 2]
    distance = np.linalg.norm(joint_xyz[prefix_frames:, dims] - point[dims], axis=1)
    conditioned_distance = np.linalg.norm(
        joint_xyz[prefix_frames:, dims] - conditioned_point[dims], axis=1)
    if len(distance) != len(xyz) - prefix_frames:
        raise ValueError("distance curve length mismatch")

    prev, first = xyz[prefix_frames - 1], xyz[prefix_frames]
    seam_root = float(np.linalg.norm(first[0] - prev[0]))
    seam_pose = float(np.linalg.norm((first - first[0]) - (prev - prev[0]), axis=1).mean())
    delta_heading = (heading(first) - heading(prev) + math.pi) % (2 * math.pi) - math.pi

    # Smoothness is measured on the generated interval only. Units are m/s
    # and m/s²; values are descriptive, not physical feasibility claims.
    generated = xyz[prefix_frames:]
    root_velocity = np.diff(generated[:, 0], axis=0) * fps
    joint_velocity = np.diff(generated, axis=0) * fps
    root_accel = np.diff(root_velocity, axis=0) * fps
    joint_accel = np.diff(joint_velocity, axis=0) * fps
    out = {
        "target_metric": "horizontal_pelvis_to_environment_goal" if task == "REACH" else "joint_to_environment_goal",
        "target_joint": name,
        "final_target_error_m": float(distance[-1]),
        "min_target_distance_m": float(distance.min()),
        "final_conditioned_target_error_m": float(conditioned_distance[-1]),
        "min_conditioned_target_distance_m": float(conditioned_distance.min()),
        "min_target_frame": int(prefix_frames + distance.argmin()),
        "seam_root_jump_m": seam_root,
        "seam_relative_joint_jump_m": seam_pose,
        "seam_heading_jump_deg": float(abs(math.degrees(delta_heading))),
        "root_speed_mean_mps": float(np.linalg.norm(root_velocity, axis=1).mean()),
        "root_speed_max_mps": float(np.linalg.norm(root_velocity, axis=1).max()),
        "root_accel_mean_mps2": float(np.linalg.norm(root_accel, axis=1).mean()),
        "root_accel_max_mps2": float(np.linalg.norm(root_accel, axis=1).max()),
        "joint_speed_mean_mps": float(np.linalg.norm(joint_velocity, axis=2).mean()),
        "joint_speed_max_mps": float(np.linalg.norm(joint_velocity, axis=2).max()),
        "joint_accel_mean_mps2": float(np.linalg.norm(joint_accel, axis=2).mean()),
        "joint_accel_max_mps2": float(np.linalg.norm(joint_accel, axis=2).max()),
        "matched_gt_error_m": None,
    }
    if task in ("STRIKE_KICK", "STRIKE_PUNCH"):
        out["target_metric_note"] = "limb-to-original-box target proximity, not official box-tip success"
    elif task == "SIT":
        out["target_metric_note"] = "pelvis-to-seat target point, not official seat-volume success"
    elif task == "GET_UP":
        out["target_metric_note"] = "3D pelvis-to-original standing target"
    else:
        out["target_metric_note"] = "horizontal pelvis-to-original environment target"
    times = np.arange(1, len(distance) + 1) / fps
    curve = np.column_stack((times, distance))
    conditioned_curve = np.column_stack((times, conditioned_distance))
    return out, curve, conditioned_curve
