#!/usr/bin/env python3
"""CPU checks for adaptive FM sampling and existing fixed solver behavior."""

import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from closd.diffusion_planner.diffusion.ode_sampler import VelocityODESampler
from calibrate_fm_dopri5 import select, tolerance_band
from summarize_fm_dopri5 import inspect, load_rows


class ExponentialVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.indices = []

    def forward(self, x, t, y=None):
        self.indices.append(int(t[0]))
        return x


class MutatingCFG(torch.nn.Module):
    """Mimic the two raw MDM forwards and its y['mask'] rebinding."""

    def __init__(self):
        super().__init__()
        self.raw_forwards = 0
        self.seen = []

    def forward(self, x, t, y=None):
        assert y is not None
        self.seen.append((y["text"], y["target_cond"].clone(), y["prefix"].clone()))
        y["mask"] = torch.zeros_like(y["mask"])
        self.raw_forwards += 2
        return torch.ones_like(x) * y["scale"].item()


class TestDopri5(unittest.TestCase):
    def test_analytic_solution_and_adaptive_nfe(self):
        x0 = torch.ones(2, 3, 1, 4)
        for solver, steps, expected_nfe, max_error in (
            ("euler", 12, 12, 0.12),
            ("heun", 6, 12, 0.03),
            ("rk4", 3, 12, 0.001),
            ("dopri5", 0, None, 0.001),
        ):
            with self.subTest(solver=solver):
                model = ExponentialVelocity()
                sampler = VelocityODESampler(num_steps=steps, solver=solver)
                result = sampler.p_sample_loop(model, x0.shape, noise=x0,
                                               model_kwargs={"y": {}}, device="cpu")
                self.assertEqual(result.shape, x0.shape)
                self.assertTrue(torch.isfinite(result).all())
                self.assertLess(torch.max(torch.abs(result - math.e)).item(), max_error)
                self.assertEqual(len(model.indices), sampler.nfe)
                if expected_nfe is not None:
                    self.assertEqual(sampler.nfe, expected_nfe)
                else:
                    self.assertGreater(sampler.nfe, 0)
                self.assertEqual(model.indices[0], 0)
                if solver != "euler":
                    self.assertEqual(model.indices[-1], 999)
                self.assertTrue(all(0 <= index <= 999 for index in model.indices))

    def test_time_rounding_shared_with_fixed_solvers(self):
        for solver in ("euler", "heun", "rk4", "dopri5"):
            sampler = VelocityODESampler(num_steps=0 if solver == "dopri5" else 3,
                                         solver=solver)
            self.assertEqual(int(sampler._time_to_index(0, 1, "cpu")[0]), 0)
            self.assertEqual(int(sampler._time_to_index(1, 1, "cpu")[0]), 999)
            self.assertEqual(int(sampler._time_to_index(0.5, 1, "cpu")[0]), 500)

    def test_cfg_conditions_and_prefix_survive_rejected_stages(self):
        prefix = torch.randn(1, 3, 1, 2)
        target = torch.randn(1, 8, 3)
        mask = torch.ones(1, 1, 1, 4)
        kwargs = {"y": {"text": ["A person is walking."], "scale": torch.tensor([7.5]),
                        "target_cond": target, "prefix": prefix, "mask": mask}}
        model = MutatingCFG()
        sampler = VelocityODESampler(num_steps=0, solver="dopri5")
        result = sampler.p_sample_loop(model, (1, 3, 1, 4),
                                       noise=torch.zeros(1, 3, 1, 4),
                                       model_kwargs=kwargs, device="cpu")
        self.assertEqual(result.shape, (1, 3, 1, 6))
        self.assertTrue(torch.equal(result[..., :2], prefix))
        self.assertTrue(torch.allclose(result[..., 2:], torch.full((1, 3, 1, 4), 7.5), atol=1e-5))
        self.assertEqual(model.raw_forwards, 2 * sampler.nfe)
        self.assertTrue(torch.equal(kwargs["y"]["mask"], mask))
        self.assertTrue(torch.equal(kwargs["y"]["target_cond"], target))
        for text, seen_target, seen_prefix in model.seen:
            self.assertEqual(text, kwargs["y"]["text"])
            self.assertTrue(torch.equal(seen_target, target))
            self.assertTrue(torch.equal(seen_prefix, prefix))

    def test_existing_fixed_solver_values(self):
        x0 = torch.ones(1, 1, 1, 1)
        for solver, steps, expected in (("euler", 2, 2.25),
                                        ("heun", 2, 2.640625),
                                        ("rk4", 1, 2.708333333333333)):
            sampler = VelocityODESampler(num_steps=steps, solver=solver)
            result = sampler.p_sample_loop(ExponentialVelocity(), x0.shape,
                                           noise=x0, device="cpu")
            self.assertAlmostEqual(result.item(), expected, places=5)

    def test_bounded_cost_and_invalid_options(self):
        with self.assertRaisesRegex(ValueError, "DOPRI5"):
            VelocityODESampler(num_steps=0, solver="dopri5", rtol=0)
        sampler = VelocityODESampler(num_steps=0, solver="dopri5", max_nfe=1)
        with self.assertRaisesRegex(RuntimeError, "max_nfe"):
            sampler.p_sample_loop(ExponentialVelocity(), (1, 1, 1, 1),
                                  noise=torch.ones(1, 1, 1, 1), device="cpu")
        self.assertEqual(sampler.nfe, 1)

    def test_formal_report_accepts_variable_nfe_and_rejects_missing_task(self):
        metadata = "\n".join("{}={}".format(key, value) for key, value in {
            "MATCHED_ID": "FM-MD", "MATCHED_TARGET_NFE": "72", "MATCHED_JOB_ID": "123",
            "MATCHED_CHECKPOINT": "/example/model.pt", "MATCHED_GIT_COMMIT": "abc",
            "MATCHED_SOURCE_SHA256": "def", "MATCHED_SEED": "0",
            "MATCHED_CONFIG": "seed:0",
        }.items())
        counts = "\n".join("EVAL_RESULT STATE={} SUCCESS=800 RESETS=1000 RATE=0.80000000".format(
            task) for task in ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT", "GET_UP"))
        planners = "\n".join("PLANNER_LATENCY_SEC=1.000000 GENERATOR=flow ODE_STEPS=0 "
                             "SOLVER=dopri5 NFE={} MODEL_FORWARD_CALLS={} BATCH=4096".format(
                                 nfe, 2 * nfe) for nfe in (68, 80))
        content = "\n".join((metadata, counts, planners, "EVAL_MACRO_SUCCESS=0.80000000",
                             "EVAL_PROFILE CALLS=2 MODEL_FORWARDS_TOTAL=296 "
                             "PLANNER_LATENCY_TOTAL_SEC=2.000000 "
                             "PLANNER_LATENCY_MEAN_SEC=1.000000",
                             "Eval plot ended", "MATCHED_PROCESS_EXIT=0", "MATCHED_JOB_EXIT=0"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.out"
            path.write_text(content)
            row = {"submit_status": "submitted", "job_id": "123", "stdout": str(path),
                   "experiment_id": "FM-MD", "target_nfe": "72", "checkpoint": "/example/model.pt",
                   "solver": "dopri5", "steps": "0"}
            complete = inspect(row)
            self.assertEqual(complete["status"], "COMPLETE")
            self.assertEqual(complete["nfe"], [68, 80])
            path.write_text(content.replace("EVAL_RESULT STATE=GET_UP", "MISSING STATE=GET_UP"))
            self.assertEqual(inspect(row)["status"], "INVALID")

    def test_calibration_budget_selection(self):
        self.assertEqual(tolerance_band(12), 3)
        rows = [{"rtol": 0.1, "atol": 0.01, "status": "COMPLETE", "nfe": 14},
                {"rtol": 0.01, "atol": 0.001, "status": "COMPLETE", "nfe": 22},
                {"rtol": 0.001, "atol": 0.0001, "status": "COMPLETE", "nfe": 41}]
        chosen = select(rows)
        self.assertEqual([chosen[str(n)]["nfe"] for n in (12, 20, 40)], [14, 22, 41])

    def test_formal_job_creates_missing_work_parent_without_gpu(self):
        root = Path(__file__).resolve().parents[3]
        run_base = root / "reproduction/experiments/fm_dopri5/runs"
        model = root / "reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt"
        script = root / "reproduction/scripts/hpc/closd_velocity_eval.slurm"
        with tempfile.TemporaryDirectory(prefix="test_work_parent_", dir=run_base) as temp:
            run_dir = Path(temp)
            fake_bin = run_dir / "bin"
            fake_bin.mkdir()
            fake_apptainer = fake_bin / "apptainer"
            fake_apptainer.write_text("#!/bin/sh\nexit 0\n")
            fake_apptainer.chmod(0o755)
            env = os.environ.copy()
            for key in ("EXPERIMENT_ID", "SWEEP_RUN_ROOT", "DIAG_LABEL", "DIAG_RUN_ROOT",
                        "CFG_SWEEP_ID", "CFG_SWEEP_RUN_ROOT", "DOPRI5_ID", "DOPRI5_RUN_ROOT"):
                env.pop(key, None)
            env.update({"PATH": str(fake_bin) + os.pathsep + env["PATH"],
                        "SLURM_JOB_ID": "99999999", "GENERATOR": "flow", "MODEL": str(model),
                        "ODE_STEPS": "96", "ODE_SOLVER": "euler", "CFG_SCALE": "7.5",
                        "MATCHED_ID": "FM-ME", "MATCHED_RUN_ROOT": str(run_dir),
                        "MATCHED_NFE": "96"})
            self.assertFalse((run_dir / "work").exists())
            job = subprocess.run(["bash", str(script)], cwd=root, env=env,
                                 capture_output=True, text=True, check=False)
            self.assertEqual(job.returncode, 0, job.stdout + job.stderr)
            self.assertTrue((run_dir / "work/FM-ME_99999999/controller/Humanoid.pth").is_symlink())
            self.assertIn("MATCHED_JOB_EXIT=0", job.stdout)

    def test_retry_manifest_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            header = "experiment_id\tsolver\tsteps\ttarget_nfe\tpilot_nfe\tjob_id\tstdout\tstderr\tsubmit_status\tcheckpoint\n"
            rows = "".join(f"{name}\t{solver}\t0\t96\t98\t123\t/tmp/out\t/tmp/err\tsubmitted\t/tmp/model\n"
                           for name, solver in (("FM-ME", "euler"), ("FM-MH", "heun"),
                                                ("FM-MR", "rk4"), ("FM-MD", "dopri5")))
            retry = run_dir / "matched_submissions_retry01.tsv"
            retry.write_text(header + rows)
            self.assertEqual([row["experiment_id"] for row in load_rows(run_dir, retry)],
                             ["FM-ME", "FM-MH", "FM-MR", "FM-MD"])


if __name__ == "__main__":
    unittest.main()
