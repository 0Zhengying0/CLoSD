"""Capture replayable, real CLoSD planner inputs before sampling.

Enabled only by CLOSD_QUALITY_CAPTURE_DIR. No synthetic goals or prefixes are
created here. The capture run intentionally exits after four cases per state.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import (
    recover_from_ric, recover_root_rot_pos)
from closd.diffusion_planner.data_loaders.humanml.common.quaternion import qrot
from closd.diffusion_planner.data_loaders.humanml_utils import HML_JOINT_NAMES


TASKS = ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT", "GET_UP")
GOAL_NAMES = ["pelvis", "left_foot", "right_foot", "left_wrist", "right_wrist", "head", "traj", "heading"]


def target_distance(prefix, mean, std, target_cond, joint_names, task):
    """Distance from last observed joint to model target, both in local HML metres."""
    name = joint_names[0]
    if name not in GOAL_NAMES or name == "heading":
        raise ValueError("unsupported target joint: {}".format(name))
    point = torch.from_numpy(np.asarray(prefix).squeeze(1).T.copy()).float()[None]
    point = point * torch.from_numpy(np.asarray(std)).float() + torch.from_numpy(np.asarray(mean)).float()
    xyz = recover_from_ric(point, 22)[0, -1]
    r_rot, r_pos = recover_root_rot_pos(point)
    xyz = xyz.clone()
    xyz[:, 0] -= r_pos[0, -1, 0]
    xyz[:, 2] -= r_pos[0, -1, 2]
    xyz = qrot(r_rot[0, -1][None].expand(22, 4), xyz).numpy()
    joint = "pelvis" if name == "traj" else name
    observed = xyz[HML_JOINT_NAMES.index(joint)]
    target = np.asarray(target_cond)[GOAL_NAMES.index(name)]
    dims = [0, 2] if task == "REACH" or name == "traj" else [0, 1, 2]
    return float(np.linalg.norm(observed[dims] - target[dims]))


class QualitySnapshotCapture:
    def __init__(self, output_dir, per_task=16, selected=4, base_seed=0):
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError("capture output exists: {}".format(self.output_dir))
        self.per_task = int(per_task)
        self.selected = int(selected)
        if self.per_task < self.selected or self.selected not in (3, 4, 5):
            raise ValueError("capture requires 3-5 selected cases per task")
        self.base_seed = int(base_seed)
        self.candidates = {task: {} for task in TASKS}
        self.calls = 0
        self.last_candidate_call = 0

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value).copy()

    def ready_to_finish(self, counts):
        enough_for_selection = all(count >= self.selected for count in counts.values())
        capped = all(count >= self.per_task for count in counts.values())
        stable = self.calls >= 10 and self.calls - self.last_candidate_call >= 5
        return enough_for_selection and (capped or stable)

    def record(self, task, model_kwargs, recon_data):
        self.calls += 1
        previous_counts = {name: len(bucket) for name, bucket in self.candidates.items()}
        if task.context_switch_prob != 0 or task.planning_horizon_multiplyer != 1:
            raise RuntimeError("capture requires simulator prefix without context switching")
        y = model_kwargs["y"]
        states = self._numpy(task.cur_state)
        progress = self._numpy(task.progress_buf)
        mean = self._numpy(task.mean).astype(np.float32)
        std = self._numpy(task.std).astype(np.float32)
        from closd.utils.closd_util import STATES
        for i in range(task.num_envs):
            try:
                name = STATES(int(states[i])).name
            except ValueError:
                continue
            bucket = self.candidates.get(name)
            if bucket is None or len(bucket) >= self.per_task or i in bucket:
                continue
            # At the first planning call CLoSD may replace simulator history
            # with a HumanML dataset prefix. Capture only realised sim history.
            if progress[i] < task.planning_horizon_30fps:
                continue
            joint_names = list(y["target_joint_names"][i])
            if not joint_names:
                continue
            prefix = self._numpy(y["prefix"][i]).astype(np.float32)
            if not np.isfinite(prefix).all() or np.max(np.abs(prefix)) == 0:
                continue
            target = self._numpy(y["target_cond"][i]).astype(np.float32)
            try:
                conditioned_distance = target_distance(
                    prefix, mean, std, target, joint_names, name)
                ranking_target = target.copy()
                ranking_target[GOAL_NAMES.index(joint_names[0])] = self._numpy(
                    task._quality_original_target_local[i])
                distance = target_distance(
                    prefix, mean, std, ranking_target, joint_names, name)
            except (ValueError, IndexError):
                continue
            if not np.isfinite(distance) or not np.isfinite(conditioned_distance):
                continue
            candidate = {
                "task": name, "source_env_id": int(i),
                "source_progress_step": int(progress[i]),
                "history_source": "simulator pose_buffer, after first planning horizon",
                "text": str(y["text"][i]),
                "target_joint_names": joint_names,
                "is_heading": bool(y["is_heading"][i]),
                "heading_target_rad": float(target[-1, 0]) if bool(y["is_heading"][i]) else None,
                "distance_at_capture_m": distance,
                "conditioned_distance_at_capture_m": conditioned_distance,
                "prefix": prefix,
                "target_cond": target,
                "mask": self._numpy(y["mask"][i]).astype(bool),
                "mean": mean,
                "std": std,
                "recon_r_rot": self._numpy(recon_data["r_rot"][i]).astype(np.float32),
                "recon_r_pos": self._numpy(recon_data["r_pos"][i]).astype(np.float32),
                "conditioned_world_target_xyz": self._numpy(task.debug_ends[i, 0]).astype(np.float32),
                "original_world_target_xyz": self._numpy(task._tar_pos[i]).astype(np.float32),
                "original_target_local_hml_xyz": self._numpy(
                    task._quality_original_target_local[i]).astype(np.float32),
                "sim_history_body_xyz": self._numpy(task.pose_buffer[i]).astype(np.float32),
            }
            for extra in ("pred_target_cond", "heading_cond"):
                if extra in y and torch.is_tensor(y[extra]) and y[extra].shape[0] == task.num_envs:
                    candidate[extra] = self._numpy(y[extra][i]).astype(np.float32)
            bucket[i] = candidate
        counts = {key: len(value) for key, value in self.candidates.items()}
        if any(counts[name] > previous_counts[name] for name in TASKS):
            self.last_candidate_call = self.calls
        print("QUALITY_CAPTURE_PROGRESS calls={} counts={}".format(self.calls, counts), flush=True)
        if self.ready_to_finish(counts):
            self.finish(task)
            return True
        max_calls = int(os.environ.get("CLOSD_QUALITY_CAPTURE_MAX_CALLS", "200"))
        if self.calls >= max_calls:
            raise RuntimeError("condition capture incomplete after {} planner calls: {}".format(self.calls, counts))
        return False

    def finish(self, task):
        self.output_dir.mkdir(parents=True, exist_ok=False)
        cases_dir = self.output_dir / "cases"
        cases_dir.mkdir()
        entries = []
        quantiles = np.linspace(0, 1, self.selected)
        labels = ["normal", "moderate", "challenging", "hard"] if self.selected == 4 else ["q{:02d}".format(i) for i in range(self.selected)]
        for task_name in TASKS:
            candidates = sorted(self.candidates[task_name].values(),
                                key=lambda item: item["distance_at_capture_m"])
            indices = np.rint(quantiles * (len(candidates) - 1)).astype(int)
            if len(set(indices)) != self.selected:
                raise RuntimeError("not enough distinct capture distances for {}".format(task_name))
            for j, index in enumerate(indices):
                item = candidates[int(index)]
                case_id = "{}_{:02d}_{}".format(task_name, j, labels[j])
                seed = self.base_seed + len(entries) + 10000
                noise = np.random.default_rng(seed).standard_normal((263, 1, 40)).astype(np.float32)
                npz_path = cases_dir / (case_id + ".npz")
                np.savez_compressed(npz_path,
                                    **{key: value for key, value in item.items() if isinstance(value, np.ndarray)},
                                    initial_noise=noise)
                meta = {key: value for key, value in item.items() if not isinstance(value, np.ndarray)}
                meta.update({"case_id": case_id, "difficulty": labels[j], "seed": seed,
                             "prefix_frames": 20, "pred_frames": 40, "total_frames": 60,
                             "fps": 20, "coordinate_system": "local HumanML3D, Y up, metres",
                             "normalization": "(hml_features - mean) / std",
                             "text_encoding": "raw text; re-encode with each checkpoint's BERT text encoder",
                             "target_coordinate_system": "same local HumanML3D frame as recovered motion",
                             "matched_ground_truth": False,
                             "npz": npz_path.name})
                (cases_dir / (case_id + ".json")).write_text(json.dumps(meta, indent=2) + "\n")
                entries.append(meta)
        manifest = {"schema_version": 1, "source": "real CLoSD planner inputs before sampling",
                    "capture_generator": task.generator_type,
                    "capture_checkpoint": task.mdm_path,
                    "capture_job_id": os.environ.get("SLURM_JOB_ID", ""),
                    "candidate_cap_per_task": self.per_task,
                    "actual_candidate_counts": {name: len(self.candidates[name]) for name in TASKS},
                    "selected_per_task": self.selected,
                    "selection": "distance quantiles within each task; normal to hard are relative labels",
                    "cases": entries}
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("QUALITY_CAPTURE_COMPLETE=1 CASES={} MANIFEST={}".format(
            len(entries), self.output_dir / "manifest.json"), flush=True)
