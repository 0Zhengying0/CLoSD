"""CPU-only CFG sweep checks for guidance, NFE, reporting and preflight."""

import contextlib
import io
import os
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from closd.diffusion_planner.diffusion.ode_sampler import VelocityODESampler
from closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel
import summarize_fm_cfg_sweep as summary


class FakeMDM(torch.nn.Module):
    cond_mask_prob = 0.1
    cond_mode = "text"

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, timesteps, y=None):
        self.calls.append((int(timesteps[0]), y.get("text_uncond", False),
                           y["text"], y["prefix"].clone(), y["target_cond"].clone()))
        y["mask"] = torch.cat((y["mask"], y["mask"]), -1)
        return torch.full_like(x, 0 if y.get("text_uncond", False) else 1)


class CfgSweepTests(unittest.TestCase):
    def test_all_scales_use_same_text_guidance_and_correct_forward_count(self):
        initial = torch.zeros((2, 3, 1, 4))
        prefix = torch.ones((2, 3, 1, 2))
        target = torch.ones((2, 2, 3))
        for scale in (1, 2.5, 5, 7.5, 10, 15):
            raw = FakeMDM()
            model = raw if scale == 1 else ClassifierFreeSampleModel(raw)
            kwargs = {"y": {"text": ["reach", "get up"],
                            "text_embed": torch.ones((2, 4)),
                            "prefix": prefix.clone(), "target_cond": target.clone(),
                            "mask": torch.ones((2, 1, 1, 4), dtype=torch.bool)}}
            if scale != 1:
                kwargs["y"]["scale"] = torch.full((2,), scale)
                self.assertEqual(model.guidance_type, "text")
            sampler = VelocityODESampler(num_steps=3, solver="rk4")
            output = sampler.p_sample_loop(model, initial.shape, noise=initial,
                                           model_kwargs=kwargs, device=initial.device)
            self.assertEqual(sampler.nfe, 12)
            self.assertEqual(len(raw.calls), 12 if scale == 1 else 24)
            self.assertEqual(output.shape, (2, 3, 1, 6))
            self.assertTrue(torch.equal(output[..., :2], prefix))
            self.assertTrue(torch.allclose(output[..., 2:], torch.full_like(initial, scale)))
            self.assertEqual(raw.calls[0][0], 0)
            self.assertEqual(raw.calls[-1][0], 999)
            self.assertEqual(sum(call[1] for call in raw.calls), 0 if scale == 1 else 12)
            self.assertTrue(all(call[2] == ["reach", "get up"] for call in raw.calls))
            self.assertTrue(all(torch.equal(call[3], prefix) and torch.equal(call[4], target)
                                for call in raw.calls))
            self.assertEqual(kwargs["y"]["mask"].shape[-1], 4)
            self.assertEqual("scale" in kwargs["y"], scale != 1)

    def test_slurm_preflight_rejects_wrong_cfg_and_solver(self):
        script = ROOT / "reproduction/scripts/hpc/closd_velocity_eval.slurm"
        checkpoint = ROOT / "reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt"
        environment = dict(os.environ, GENERATOR="flow", MODEL=str(checkpoint),
                           ODE_SOLVER="rk4", ODE_STEPS="3", CFG_SWEEP_ID="FM-CFG1",
                           CFG_SWEEP_RUN_ROOT=str(ROOT / "reproduction/experiments/fm_cfg_sweep/runs/test"),
                           CFG_SCALE="1")
        good = subprocess.run(["bash", str(script), "--check"], env=environment,
                              capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertIn("NFE:       12", good.stdout)
        environment["CFG_SCALE"] = "7.5"
        bad = subprocess.run(["bash", str(script), "--check"], env=environment,
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("requires CFG_SCALE=1", bad.stderr)
        environment["CFG_SCALE"] = "1"
        environment["ODE_SOLVER"] = "euler"
        bad = subprocess.run(["bash", str(script), "--check"], env=environment,
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)

    def test_summary_uses_counts_and_checks_cfg_forwards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = "/model000300000.pt"
            counts = [(981, 1390), (1152, 1311), (1152, 1311),
                      (1354, 1404), (1026, 1344)]
            macro = statistics.mean(success / resets for success, resets in counts)

            def make_row(scale, job_id):
                return {"experiment_id": "FM-CFG" + scale.replace(".", "p"),
                        "cfg": scale, "solver": "rk4", "steps": "3", "nfe": "12",
                        "job_id": job_id, "stdout": str(root / (scale + "_%j.out")),
                        "stderr": str(root / (scale + "_%j.err")),
                        "submit_status": "submitted", "checkpoint": checkpoint}

            def write_log(row, forward_count):
                scale = row["cfg"]
                lines = ["CFG_EXPERIMENT_ID=" + row["experiment_id"],
                         "CFG_JOB_ID=" + row["job_id"], "CFG_CHECKPOINT=" + checkpoint,
                         "CFG_GIT_COMMIT=abc123", "CFG_GIT_DIRTY=1",
                         "CFG_SOURCE_SHA256=abcdef", "CFG_SCALE=" + scale,
                         "CFG_GUIDANCE_TYPE=text", "CFG_SEED=0",
                         "CFG_CONFIG=num_envs:4096,episode_length:500,samples_per_task:1000,nfe:12",
                         "CFG_RUNTIME SCALE={} WRAPPED={} GUIDANCE=text".format(
                             scale, int(scale != "1")),
                         "PLANNER_LATENCY_SEC=2.000000 GENERATOR=flow ODE_STEPS=3 SOLVER=rk4 NFE=12 MODEL_FORWARD_CALLS={} BATCH=4096".format(forward_count)]
                lines.extend("EVAL_RESULT STATE={} SUCCESS={} RESETS={} RATE={:.8f}".format(
                    state, success, resets, success / resets)
                    for state, (success, resets) in zip(summary.ORDER, counts))
                lines.extend(["EVAL_MACRO_SUCCESS={:.8f}".format(macro),
                              "EVAL_PROFILE CALLS=1 MODEL_FORWARDS_TOTAL={} PLANNER_LATENCY_TOTAL_SEC=2.000000 PLANNER_LATENCY_MEAN_SEC=2.000000".format(forward_count),
                              "Eval plot ended", "CFG_PROCESS_EXIT=0", "CFG_JOB_EXIT=0"])
                Path(row["stdout"].replace("%j", row["job_id"])).write_text("\n".join(lines))

            raw_row = make_row("1", "123")
            write_log(raw_row, 12)
            self.assertEqual(summary.inspect_job(raw_row)["status"], "COMPLETE")
            guided_row = make_row("7.5", "124")
            write_log(guided_row, 24)
            self.assertEqual(summary.inspect_job(guided_row)["status"], "COMPLETE")
            write_log(guided_row, 12)
            self.assertEqual(summary.inspect_job(guided_row)["status"], "INVALID_METRICS")
            missing = make_row("15", "125")
            self.assertEqual(summary.inspect_job(missing)["status"], "MISSING_LOG")
            self.assertIn("981/1390", summary.render([raw_row, guided_row], root))

    def test_submission_manifest_matrix_and_summary_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "submissions.tsv"
            manifest.write_text(
                "experiment_id\tcfg\tsolver\tsteps\tnfe\tjob_id\tstdout\tstderr\tsubmit_status\tcheckpoint\n" +
                "".join("{}\t{}\trk4\t3\t12\t\tunused\tunused\tsubmit_failed\tckpt\n".format(id, scale)
                        for id, scale in zip(summary.IDS, summary.SCALES)))
            output = root / "summary.md"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(summary.main(["--run-dir", str(root), "--output", str(output)]), 0)
            self.assertIn("SUBMIT_FAILED", output.read_text())
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    summary.main(["--run-dir", str(root), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
