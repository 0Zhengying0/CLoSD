"""Headless opt-in recording using CLoSD's existing simulator state writer.

The generated pickle is an actual controller execution record accepted by the
official Blender record2anim.py. It is separate from offline planner motions.
"""

import json
from pathlib import Path


class QualityExecutionRecorder:
    def __init__(self, output_dir, steps=500):
        self.output_dir = Path(output_dir)
        self.steps = int(steps)
        self.calls = 0
        if self.steps < 2:
            raise ValueError("execution recording needs at least two simulation frames")
        if self.output_dir.exists():
            raise FileExistsError("execution recording output already exists: {}".format(
                self.output_dir))

    def record(self, task):
        # Reuse the exact simulator state serialization used by GUI recording.
        task._record_states()
        task.state_record["task_state"].append(task.cur_state.detach().cpu().clone())
        task.state_record["task_done"].append(task.is_done.detach().cpu().clone())
        self.calls += 1
        if self.calls < self.steps:
            return False
        self.output_dir.mkdir(parents=True, exist_ok=False)
        path = self.output_dir / "episodes.pkl"
        task._write_states_to_file(str(path))
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("official simulator recording was not written")
        metadata = {"schema_version": 1,
                    "kind": "CLoSD closed-loop controller execution",
                    "recording_method": "HumanoidIm._record_states + _write_states_to_file",
                    "frames_recorded": self.calls,
                    "num_envs": task.num_envs,
                    "generator": task.generator_type,
                    "planner_checkpoint": str(task.mdm_path),
                    "cfg": float(task.mdm_cfg_param),
                    "ode_solver": getattr(task, "ode_solver", None),
                    "ode_steps": getattr(task, "ode_steps", None)}
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("QUALITY_EXEC_RECORD_COMPLETE=1 PATH={}".format(path), flush=True)
        raise SystemExit(0)
