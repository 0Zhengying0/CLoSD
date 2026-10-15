"""Optional, per-episode CLoSD evaluation diagnostics.

This module observes the existing evaluation. It never changes task state,
reset decisions, or the official success masks computed by the player.
"""

import json
import math
import os
from pathlib import Path

import numpy as np

from closd.utils.closd_util import STATES


TRACE_FIELDS = (
    "progress_step", "state", "body_pos_xyz", "controller_action",
    "pd_target", "dof_pos",
    "target_xyz", "target_actor_pose", "goal_distance_m", "box_up_dot",
    "physical_goal_hit", "official_task_done", "terminated",
)
STRIKE_STATES = (STATES.STRIKE_KICK, STATES.STRIKE_PUNCH)


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _sit_volume_distance(pelvis, corners, height_min, height_max):
    """Distance in metres to the rectangular SIT success volume."""
    center = corners[:, :, :2].mean(axis=1)
    side_u = corners[:, 0, :2] - corners[:, 1, :2]
    side_v = corners[:, 1, :2] - corners[:, 2, :2]
    length_u = np.linalg.norm(side_u, axis=1)
    length_v = np.linalg.norm(side_v, axis=1)
    delta = pelvis[:, :2] - center
    u = np.abs(np.sum(delta * side_u, axis=1) / np.maximum(length_u, 1e-8))
    v = np.abs(np.sum(delta * side_v, axis=1) / np.maximum(length_v, 1e-8))
    du = np.maximum(u - length_u / 2, 0)
    dv = np.maximum(v - length_v / 2, 0)
    dz = np.maximum(np.maximum(height_min - pelvis[:, 2], pelvis[:, 2] - height_max), 0)
    return np.sqrt(du * du + dv * dv + dz * dz)


def snapshot_from_task(task, info, done):
    """Read the post-physics state and the exact current task goal predicate."""
    if not hasattr(task, "_episode_diag_pd_target"):
        raise RuntimeError("PD target was not captured; episode diagnostics require the PD controller")
    body = _numpy(task._rigid_body_pos).astype(np.float32, copy=False)
    target = _numpy(task._tar_pos).astype(np.float32, copy=False)
    actor_pose = _numpy(task._target_states[:, :7]).astype(np.float32, copy=False)
    state = _numpy(info["task_state"]).astype(np.int16, copy=False)
    body_index = _numpy(task.mujoco_joint_idx[:, 0]).astype(np.int64, copy=False)
    tracked = body[np.arange(len(body)), body_index]
    distance = np.full(len(body), np.nan, dtype=np.float32)

    reach = state == STATES.REACH
    distance[reach] = np.linalg.norm(tracked[reach, :2] - target[reach, :2], axis=1)
    get_up = state == STATES.GET_UP
    distance[get_up] = np.linalg.norm(tracked[get_up] - target[get_up], axis=1)
    halt = state == STATES.HALT
    distance[halt] = np.linalg.norm(tracked[halt, :2] - target[halt, :2], axis=1)
    strike = np.isin(state, STRIKE_STATES)
    # This is limb-to-box-centre proximity, NOT the strike success predicate.
    distance[strike] = np.linalg.norm(tracked[strike] - actor_pose[strike, :3], axis=1)
    sit = state == STATES.SIT
    if np.any(sit):
        corners = _numpy(task.bench_corners).astype(np.float32, copy=False)
        distance[sit] = _sit_volume_distance(
            body[sit, 0], corners[sit], float(task.sit_dims[2]),
            float(task.sit_dims[2] + task.max_gap_from_sit),
        )

    box_up_dot = np.full(len(body), np.nan, dtype=np.float32)
    box_states = strike | halt
    # For unit xyzw quaternions, dot(R(q) z, z) = 1 - 2(x² + y²).
    q = actor_pose[box_states, 3:7]
    box_up_dot[box_states] = 1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2)

    return {
        "progress_step": _numpy(task.progress_buf).astype(np.int32, copy=False),
        "state": state,
        "initial_state": _numpy(task.init_state_per_env).astype(np.int16, copy=False),
        "body_pos_xyz": body,
        "controller_action": _numpy(task.actions).astype(np.float32, copy=False),
        "pd_target": _numpy(task._episode_diag_pd_target).astype(np.float32, copy=False),
        "dof_pos": _numpy(task._dof_pos).astype(np.float32, copy=False),
        "target_xyz": target,
        "target_actor_pose": actor_pose,
        "goal_distance_m": distance,
        "box_up_dot": box_up_dot,
        "physical_goal_hit": _numpy(task.get_cur_done()).astype(np.bool_, copy=False),
        "official_task_done": _numpy(info["task_done"]).astype(np.bool_, copy=False),
        "terminated": _numpy(info["terminate"]).astype(np.bool_, copy=False),
        "done": _numpy(done).astype(np.bool_, copy=False),
    }


def _maybe_step(value):
    return None if value < 0 else int(value)


def _maybe_float(value):
    return None if not math.isfinite(float(value)) else float(value)


class EpisodeDiagnosticRecorder:
    """Keep active traces in host memory; write completed episodes in NPZ shards."""

    def __init__(self, output_dir, num_envs, max_steps, dt, checkpoint, generator,
                 metric_names, shard_size=128, configuration=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.trace_dir = self.output_dir / "trajectories"
        self.trace_dir.mkdir()
        self.jsonl_path = self.output_dir / "episodes.jsonl"
        self.jsonl_path.touch(exist_ok=False)
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.checkpoint = str(checkpoint)
        self.generator = str(generator)
        self.metric_names = tuple(metric_names)
        self.configuration = dict(configuration or {})
        self.shard_size = int(shard_size)
        self.length = np.zeros(num_envs, dtype=np.int32)
        self.sequence = np.zeros(num_envs, dtype=np.int32)
        self.traces = None
        self.pending = []
        self.shard_number = 0
        self.completed_count = 0
        self.incomplete_count = 0
        self.metric_counts = {name: {"resets": 0, "success": 0}
                              for name in self.metric_names}
        self._finished = False
        self._reset_tracking(np.arange(num_envs))

    def _reset_tracking(self, ids):
        if not hasattr(self, "initial_state"):
            self.initial_state = np.zeros(self.num_envs, dtype=np.int16)
            self.previous_state = np.zeros(self.num_envs, dtype=np.int16)
            self.first_goal = np.full(self.num_envs, -1, dtype=np.int32)
            self.first_latched_goal = np.full(self.num_envs, -1, dtype=np.int32)
            self.first_get_up_goal = np.full(self.num_envs, -1, dtype=np.int32)
            self.first_get_up_latched_goal = np.full(self.num_envs, -1, dtype=np.int32)
            self.first_fall = np.full(self.num_envs, -1, dtype=np.int32)
            self.sit_to_get_up = np.full(self.num_envs, -1, dtype=np.int32)
            self.minimum_distance = np.full(self.num_envs, np.inf, dtype=np.float32)
            self.minimum_get_up_distance = np.full(self.num_envs, np.inf, dtype=np.float32)
            self.minimum_box_up_dot = np.full(self.num_envs, np.inf, dtype=np.float32)
            self.start_target = np.zeros((self.num_envs, 3), dtype=np.float32)
            self.transitions = [[] for _ in range(self.num_envs)]
        self.first_goal[ids] = -1
        self.first_latched_goal[ids] = -1
        self.first_get_up_goal[ids] = -1
        self.first_get_up_latched_goal[ids] = -1
        self.first_fall[ids] = -1
        self.sit_to_get_up[ids] = -1
        self.minimum_distance[ids] = np.inf
        self.minimum_get_up_distance[ids] = np.inf
        self.minimum_box_up_dot[ids] = np.inf
        for env_id in ids:
            self.transitions[int(env_id)] = []

    def _allocate(self, snapshot):
        n, t = self.num_envs, self.max_steps
        self.traces = {
            "progress_step": np.empty((n, t), dtype=np.int32),
            "state": np.empty((n, t), dtype=np.int16),
            "body_pos_xyz": np.empty((n, t) + snapshot["body_pos_xyz"].shape[1:], dtype=np.float32),
            "controller_action": np.empty((n, t) + snapshot["controller_action"].shape[1:], dtype=np.float32),
            "pd_target": np.empty((n, t) + snapshot["pd_target"].shape[1:], dtype=np.float32),
            "dof_pos": np.empty((n, t) + snapshot["dof_pos"].shape[1:], dtype=np.float32),
            "target_xyz": np.empty((n, t, 3), dtype=np.float32),
            "target_actor_pose": np.empty((n, t, 7), dtype=np.float32),
            "goal_distance_m": np.empty((n, t), dtype=np.float32),
            "box_up_dot": np.empty((n, t), dtype=np.float32),
            "physical_goal_hit": np.empty((n, t), dtype=np.bool_),
            "official_task_done": np.empty((n, t), dtype=np.bool_),
            "terminated": np.empty((n, t), dtype=np.bool_),
        }

    def record_step(self, snapshot, official_masks):
        if self._finished:
            raise RuntimeError("diagnostic recorder already finished")
        if self.traces is None:
            self._allocate(snapshot)
        n = self.num_envs
        if any(len(snapshot[field]) != n for field in TRACE_FIELDS):
            raise ValueError("diagnostic batch shape differs from num_envs")
        if np.any(self.length >= self.max_steps):
            raise RuntimeError("diagnostic trajectory exceeded episode_length")

        fresh = np.flatnonzero(self.length == 0)
        self._reset_tracking(fresh)
        self.initial_state[fresh] = snapshot["initial_state"][fresh]
        self.previous_state[fresh] = snapshot["initial_state"][fresh]
        self.start_target[fresh] = snapshot["target_xyz"][fresh]

        rows = np.arange(n)
        columns = self.length.copy()
        for field in TRACE_FIELDS:
            self.traces[field][rows, columns] = snapshot[field]
        self.length += 1

        initial = snapshot["state"] == self.initial_state
        current_step = snapshot["progress_step"]
        initial_hit = initial & snapshot["physical_goal_hit"] & (self.first_goal < 0)
        self.first_goal[initial_hit] = current_step[initial_hit]
        initial_latched = initial & snapshot["official_task_done"] & (self.first_latched_goal < 0)
        self.first_latched_goal[initial_latched] = current_step[initial_latched]
        get_up = snapshot["state"] == STATES.GET_UP
        get_up_hit = get_up & snapshot["physical_goal_hit"] & (self.first_get_up_goal < 0)
        self.first_get_up_goal[get_up_hit] = current_step[get_up_hit]
        get_up_latched = get_up & snapshot["official_task_done"] & (self.first_get_up_latched_goal < 0)
        self.first_get_up_latched_goal[get_up_latched] = current_step[get_up_latched]
        fall = snapshot["terminated"] & (self.first_fall < 0)
        self.first_fall[fall] = current_step[fall]

        valid_distance = np.isfinite(snapshot["goal_distance_m"])
        initial_distance = initial & valid_distance
        self.minimum_distance[initial_distance] = np.minimum(
            self.minimum_distance[initial_distance], snapshot["goal_distance_m"][initial_distance],
        )
        get_up_distance = get_up & valid_distance
        self.minimum_get_up_distance[get_up_distance] = np.minimum(
            self.minimum_get_up_distance[get_up_distance], snapshot["goal_distance_m"][get_up_distance],
        )
        strike = np.isin(snapshot["state"], STRIKE_STATES)
        self.minimum_box_up_dot[strike] = np.minimum(
            self.minimum_box_up_dot[strike], snapshot["box_up_dot"][strike],
        )

        changed = np.flatnonzero(snapshot["state"] != self.previous_state)
        for env_id in changed:
            before = STATES(int(self.previous_state[env_id])).name
            after = STATES(int(snapshot["state"][env_id])).name
            # The transition happened at the pre-physics update, one step
            # before the post-physics progress index observed here.
            transition_step = max(int(current_step[env_id]) - 1, 0)
            self.transitions[env_id].append({
                "from": before, "to": after, "step": transition_step,
                "time_s": transition_step * self.dt,
            })
            if before == "SIT" and after == "GET_UP":
                self.sit_to_get_up[env_id] = transition_step
        self.previous_state[:] = snapshot["state"]

        for env_id in np.flatnonzero(snapshot["done"]):
            self._close_episode(int(env_id), snapshot, official_masks, completed=True)
            if len(self.pending) >= self.shard_size:
                self._flush()

    def _close_episode(self, env_id, snapshot, official_masks, completed):
        initial_name = STATES(int(self.initial_state[env_id])).name
        final_name = STATES(int(snapshot["state"][env_id])).name
        fall_step = _maybe_step(self.first_fall[env_id])
        switch_step = _maybe_step(self.sit_to_get_up[env_id])
        first_goal = _maybe_step(self.first_goal[env_id])
        first_latched_goal = _maybe_step(self.first_latched_goal[env_id])
        get_up_goal = _maybe_step(self.first_get_up_goal[env_id])
        get_up_latched_goal = _maybe_step(self.first_get_up_latched_goal[env_id])
        progress = int(snapshot["progress_step"][env_id])
        if not completed:
            reason = "evaluation_stopped"
        elif snapshot["terminated"][env_id]:
            reason = "fall"
        elif progress >= self.max_steps - 1:
            reason = "time_limit"
        else:
            reason = "other_reset"

        metrics = {}
        for name in self.metric_names:
            counted = (bool(official_masks[name]["counted"][env_id]) if completed else None)
            success = (bool(official_masks[name]["success"][env_id]) if completed else None)
            metrics[name] = {"counted": counted, "success": success}
            if completed:
                self.metric_counts[name]["resets"] += int(counted)
                self.metric_counts[name]["success"] += int(success)

        row = {
            "episode_id": "env{:04d}-ep{:04d}".format(env_id, int(self.sequence[env_id])),
            "env_id": env_id,
            "env_episode_index": int(self.sequence[env_id]),
            "completed": bool(completed),
            "initial_task": initial_name,
            "initial_target_xyz_m": self.start_target[env_id].tolist(),
            "final_state": final_name,
            "end_step": progress,
            "end_time_s": progress * self.dt,
            "terminal_reason": reason,
            "time_limit_reached": bool(progress >= self.max_steps - 1),
            "fell": fall_step is not None,
            "fall_step": fall_step,
            "fall_time_s": None if fall_step is None else fall_step * self.dt,
            "first_initial_goal_step": first_goal,
            "first_initial_goal_time_s": None if first_goal is None else first_goal * self.dt,
            "first_official_task_done_step": first_latched_goal,
            "first_official_task_done_time_s": None if first_latched_goal is None else first_latched_goal * self.dt,
            "min_initial_target_distance_m": _maybe_float(self.minimum_distance[env_id]),
            "min_box_up_dot": _maybe_float(self.minimum_box_up_dot[env_id]),
            "state_transitions": list(self.transitions[env_id]),
            "sit_to_get_up_step": switch_step,
            "sit_to_get_up_time_s": None if switch_step is None else switch_step * self.dt,
            "get_up_first_goal_step": get_up_goal,
            "get_up_first_goal_time_s": None if get_up_goal is None else get_up_goal * self.dt,
            "get_up_first_official_task_done_step": get_up_latched_goal,
            "get_up_first_official_task_done_time_s": None if get_up_latched_goal is None else get_up_latched_goal * self.dt,
            "min_get_up_target_distance_m": _maybe_float(self.minimum_get_up_distance[env_id]),
            "post_get_up_final_state": final_name if switch_step is not None else None,
            "post_get_up_fell": (fall_step is not None and fall_step >= switch_step)
                                   if switch_step is not None else None,
            "post_get_up_final_pelvis_height_m": float(snapshot["body_pos_xyz"][env_id, 0, 2])
                                                 if switch_step is not None else None,
            "official_metric_flags": metrics,
            "official_counted_for_initial_task": metrics[initial_name]["counted"] if initial_name in metrics else None,
            "official_success_for_initial_task": metrics[initial_name]["success"] if initial_name in metrics else None,
            "official_get_up_counted": metrics.get("GET_UP", {}).get("counted") if initial_name == "SIT" else None,
            "official_get_up_success": metrics.get("GET_UP", {}).get("success") if initial_name == "SIT" else None,
            "trajectory_steps": int(self.length[env_id]),
        }
        trace = {field: self.traces[field][env_id, :self.length[env_id]].copy()
                 for field in TRACE_FIELDS}
        self.pending.append((row, trace))
        if completed:
            self.completed_count += 1
        else:
            self.incomplete_count += 1
        self.length[env_id] = 0
        self.sequence[env_id] += 1

    def _flush(self):
        if not self.pending:
            return
        self.shard_number += 1
        filename = "part-{:05d}.npz".format(self.shard_number)
        output = self.trace_dir / filename
        temporary = self.trace_dir / (filename + ".tmp")
        lengths = [row["trajectory_steps"] for row, _ in self.pending]
        offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        payload = {
            "episode_ids": np.asarray([row["episode_id"] for row, _ in self.pending]),
            "offsets": offsets,
        }
        for field in TRACE_FIELDS:
            payload[field] = np.concatenate([trace[field] for _, trace in self.pending], axis=0)
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, output)
        with self.jsonl_path.open("a") as handle:
            for index, (row, _) in enumerate(self.pending):
                row["trajectory_file"] = "trajectories/" + filename
                row["trajectory_index"] = index
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.pending.clear()

    def finish(self, snapshot, official_counters):
        if self._finished:
            return
        no_masks = {name: {"counted": None, "success": None} for name in self.metric_names}
        for env_id in np.flatnonzero(self.length > 0):
            self._close_episode(int(env_id), snapshot, no_masks, completed=False)
            if len(self.pending) >= self.shard_size:
                self._flush()
        self._flush()
        for name in self.metric_names:
            if self.metric_counts[name] != official_counters[STATES[name]]:
                raise RuntimeError("diagnostic/official counter mismatch for {}: {} != {}".format(
                    name, self.metric_counts[name], official_counters[STATES[name]],
                ))
        manifest = {
            "schema_version": 1,
            "generator": self.generator,
            "checkpoint": self.checkpoint,
            "configuration": self.configuration,
            "slurm_job_id": os.environ.get("CLOSD_DIAG_JOB_ID", os.environ.get("SLURM_JOB_ID")),
            "num_envs": self.num_envs,
            "episode_length": self.max_steps,
            "control_dt_s": self.dt,
            "completed_episodes": self.completed_count,
            "incomplete_episodes": self.incomplete_count,
            "official_counters": self.metric_counts,
            "trace_fields": list(TRACE_FIELDS),
            "control_trace_note": "controller_action and pd_target are pre-physics commands; dof_pos and body_pos_xyz are post-physics realised states",
            "trajectory_format": "NPZ flat arrays; episode i spans offsets[i]:offsets[i+1]",
            "time_convention": "post-physics progress_step; transition_step is pre-physics progress_step",
            "goal_distance_definition": {
                "REACH": "horizontal pelvis-to-goal distance in metres",
                "GET_UP": "3D pelvis-to-goal distance in metres",
                "SIT": "3D pelvis-to-valid-seat-volume distance in metres; zero inside",
                "STRIKE_KICK": "conditioned foot-to-box-centre distance in metres; diagnostic only",
                "STRIKE_PUNCH": "conditioned wrist-to-box-centre distance in metres; diagnostic only",
            },
            "strike_success_definition": "box_up_dot <= 0.25, independent of limb-to-box distance",
            "official_metric_note": "KICK and PUNCH use the original shared strike mask; GET_UP is conditional on entering GET_UP",
        }
        with (self.output_dir / "manifest.json").open("x") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        self._finished = True
        print("EPISODE_DIAGNOSTICS_DIR={} COMPLETED={} INCOMPLETE={}".format(
            self.output_dir, self.completed_count, self.incomplete_count,
        ), flush=True)
