#!/usr/bin/env python3
"""Convert a real CLoSD simulator recording for the repository Blender renderer.

This does only the torch/poselib calculations unavailable in Karolina's Blender
Python. The character meshes and animation are still made by record2anim.py.
"""

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch

from closd.blender.obj_utils import xml2mesh
from closd.utils.closd_util import STATES
from closd.utils.poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree


def quaternion_to_euler(quat):
    # Same quaternion convention and formula as blender_utils; avoids importing bpy.
    x, y, z, w = np.split(quat, 4, axis=-1)
    ysqr = y * y
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + ysqr))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (ysqr + z * z))
    return np.concatenate((roll, pitch, yaw), axis=-1)


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def load_record_cpu(path):
    """Read official joblib records containing CUDA tensors on a CPU login node.

    HumanoidIm writes its ``init_state`` tensor without moving it to CPU. During
    unpickling PyTorch calls this hook for its embedded tensor bytes; restore it
    immediately after loading so the rest of the process is unaffected.
    """
    original = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda blob: torch.load(
            io.BytesIO(blob), map_location=torch.device("cpu"))
        return joblib.load(path)
    finally:
        torch.storage._load_from_bytes = original


def prepare(record_path, sample_name, output, assets_dir, downsample=2, record_data=None):
    if output.exists():
        raise FileExistsError(output)
    record = record_data if record_data is not None else load_record_cpu(record_path)
    if sample_name not in record:
        raise KeyError("episode {} missing; available: {}".format(sample_name, list(record)[:20]))
    sample = record[sample_name]
    pose_quat = as_numpy(sample["body_quat"])[::downsample]
    trans = as_numpy(sample["trans"])[::downsample]
    markers = as_numpy(sample["ref_body_pos_full"])[::downsample]
    tree = SkeletonTree.from_dict(sample["skeleton_tree"])
    state = SkeletonState.from_rotation_and_root_translation(
        tree, torch.from_numpy(pose_quat), torch.from_numpy(trans), is_local=True)
    positions = state.global_translation.cpu().numpy()
    eulers = quaternion_to_euler(as_numpy(state.global_rotation))
    n_frames = len(positions)
    if n_frames < 2 or positions.shape != eulers.shape or positions.shape[1:] != (24, 3):
        raise ValueError("invalid simulator pose shape: {} {}".format(positions.shape, eulers.shape))
    if markers.shape != positions.shape or not np.isfinite(positions).all() or not np.isfinite(eulers).all():
        raise ValueError("invalid simulator marker/pose data")
    target_trans = as_numpy(sample["target_trans"])[::downsample] if "target_trans" in sample else np.empty((0, 3))
    target_quat = as_numpy(sample["target_quat"])[::downsample] if "target_quat" in sample else np.empty((0, 4))
    target_euler = quaternion_to_euler(target_quat) if target_quat.size else np.empty((0, 3))
    task_state = as_numpy(sample["task_state"])[::downsample] if "task_state" in sample else np.empty((0,), dtype=np.int64)
    init_state = int(as_numpy(sample["init_state"]).item()) if "init_state" in sample else 0
    if len(target_trans) not in (0, n_frames) or len(task_state) not in (0, n_frames):
        raise ValueError("simulator record fields have inconsistent lengths")
    assets_dir = Path(assets_dir)
    mesh_dir = assets_dir / "smpl_humanoid"
    if not mesh_dir.exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
        xml2mesh("closd/data/assets/mjcf/smpl_humanoid.xml", str(mesh_dir))
    if len(list(mesh_dir.glob("*.obj"))) != 24:
        raise ValueError("official humanoid mesh conversion is incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, sample_name=sample_name, pos_data=positions,
                        euler_data=eulers, markers_data=markers,
                        target_trans=target_trans, target_euler=target_euler,
                        task_state=task_state, init_state=init_state,
                        fps=int(sample.get("fps", 60)) // downsample)
    metadata = {"source_record": str(Path(record_path).resolve()),
                "episode": sample_name, "initial_task": STATES(init_state).name if init_state else "UNKNOWN",
                "states_seen": sorted({STATES(int(x)).name for x in task_state}),
                "frames": n_frames, "fps": int(sample.get("fps", 60)) // downsample,
                "kind": "closed-loop executed simulator trajectory"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--episode", required=True, help="official recording key, e.g. 0_0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--downsample", type=int, default=2)
    args = parser.parse_args()
    if args.downsample < 1:
        parser.error("--downsample must be positive")
    print(json.dumps(prepare(args.record, args.episode, args.output,
                             args.assets_dir, args.downsample), indent=2))


if __name__ == "__main__":
    main()
