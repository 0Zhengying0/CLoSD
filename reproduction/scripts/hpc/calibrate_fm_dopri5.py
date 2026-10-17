#!/usr/bin/env python3
"""Choose DOPRI5 tolerances near the existing 12/20/40-NFE FM budgets."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_dip_fm_quality import load_backend, load_cases, validate_models
from closd.diffusion_planner.diffusion.ode_sampler import VelocityODESampler

TARGETS = (12, 20, 40)
GRID = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)


def tolerance_band(target):
    return max(2, math.ceil(target * 0.2))


def select(rows):
    """Return a pilot-matched setting per budget, or None if unavailable."""
    valid = [row for row in rows if row["status"] == "COMPLETE"]
    selected = {}
    for target in TARGETS:
        if not valid:
            selected[str(target)] = None
            continue
        best = min(valid, key=lambda row: (abs(row["nfe"] - target), row["rtol"]))
        selected[str(target)] = best if abs(best["nfe"] - target) <= tolerance_band(target) else None
    return selected


def prepare_batch(cases, device):
    import torch
    metas = [entry[0] for entry in cases]
    arrays = [entry[1] for entry in cases]
    stacked = lambda key: torch.from_numpy(np.stack([array[key] for array in arrays])).to(device)
    y = {"prefix": stacked("prefix"), "mask": stacked("mask"),
         "prefix_len": 20, "target_cond": stacked("target_cond"),
         "pred_target_cond": stacked("pred_target_cond"),
         "heading_cond": stacked("heading_cond"),
         "target_joint_names": [list(meta["target_joint_names"]) for meta in metas],
         "is_heading": torch.tensor([meta["is_heading"] for meta in metas], device=device),
         "text": [meta["text"] for meta in metas],
         "scale": torch.full((len(cases),), 7.5, device=device)}
    return y, stacked("initial_noise"), [meta["case_id"] for meta in metas]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-num-steps", type=int, default=128)
    parser.add_argument("--max-nfe", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    import torch
    if not torch.cuda.is_available():
        parser.error("A100 GPU required for real checkpoint calibration")
    cases = load_cases(args.cases)
    if len(cases) != 20:
        parser.error("expected 20 real captured conditions")
    records = validate_models(check_ema=True)
    device = torch.device("cuda:0")
    model, raw, _, mean, std = load_backend("fm", records["fm"], device)
    for _, arrays, _ in cases:
        np.testing.assert_allclose(mean, arrays["mean"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(std, arrays["std"], rtol=0, atol=1e-6)
    y, noise, case_ids = prepare_batch(cases, device)
    with torch.inference_mode():
        y["text_embed"] = model.encode_text(y["text"])
    rows = []
    tested = set()

    def evaluate(rtol):
        if rtol in tested:
            return
        tested.add(rtol)
        atol = rtol * 0.1
        sampler = VelocityODESampler(
            num_timesteps=records["fm"]["args"]["diffusion_steps"],
            num_steps=0, solver="dopri5", rtol=rtol, atol=atol,
            max_num_steps=args.max_num_steps, max_nfe=args.max_nfe)
        count = [0]
        hook = raw.register_forward_hook(lambda *_: count.__setitem__(0, count[0] + 1))
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                sample = sampler.p_sample_loop(
                    model, tuple(noise.shape), noise=noise,
                    model_kwargs={"y": y}, device=device)
            torch.cuda.synchronize()
            if (tuple(sample.shape) != (20, 263, 1, 60) or
                    not torch.isfinite(sample).all() or
                    not torch.allclose(sample[..., :20], y["prefix"], atol=1e-6) or
                    count[0] != 2 * sampler.nfe):
                raise RuntimeError("invalid motion, prefix or CFG forward count")
            status, reason = "COMPLETE", ""
        except (RuntimeError, AssertionError) as exc:
            status, reason = "FAILED", str(exc)
        finally:
            hook.remove()
        row = {"rtol": rtol, "atol": atol, "status": status,
               "nfe": sampler.nfe, "mdm_forwards": count[0],
               "planner_latency_sec": time.perf_counter() - start,
               "reason": reason}
        rows.append(row)
        print("DOPRI5_CALIBRATION_ROW=" + json.dumps(row), flush=True)

    for rtol in GRID:
        evaluate(rtol)
    # Search between adjacent tested tolerances when a target was missed.
    # The quantized time embedding can make NFE nonmonotonic, so each result
    # is still selected by its measured distance from the target.
    for target in TARGETS:
        for _ in range(4):
            if select(rows)[str(target)] is not None:
                break
            valid = sorted((row for row in rows if row["status"] == "COMPLETE"),
                           key=lambda row: row["rtol"])
            brackets = [(a, b) for a, b in zip(valid, valid[1:])
                        if (a["nfe"] - target) * (b["nfe"] - target) < 0]
            if not brackets:
                break
            a, b = min(brackets, key=lambda pair: abs(math.log(pair[1]["rtol"] / pair[0]["rtol"])))
            evaluate(math.sqrt(a["rtol"] * b["rtol"]))

    chosen = select(rows)
    default_row = next(row for row in rows if row["rtol"] == 0.001)
    common_nfe = (max(4, 4 * round(default_row["nfe"] / 4))
                  if default_row["status"] == "COMPLETE" else None)
    result = {"checkpoint": records["fm"]["checkpoint"], "cfg": 7.5,
              "case_ids": case_ids, "case_count": len(case_ids),
              "max_num_steps": args.max_num_steps, "max_nfe": args.max_nfe,
              "targets": list(TARGETS), "tolerance_band": {str(t): tolerance_band(t) for t in TARGETS},
              "rows": rows, "selected": chosen,
              "default_dopri5": default_row,
              "recommended_common_nfe": common_nfe,
              "warning": "20-case offline NFE predicts but cannot guarantee NFE in 4096-env closed loop"}
    args.output.mkdir(parents=True)
    (args.output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DOPRI5_CALIBRATION_COMPLETE=1 SELECTED=" + json.dumps({
        key: None if value is None else {"rtol": value["rtol"], "atol": value["atol"],
                                          "nfe": value["nfe"]} for key, value in chosen.items()}),
        flush=True)
    print("DOPRI5_RECOMMENDED_COMMON_NFE={}".format(common_nfe), flush=True)


if __name__ == "__main__":
    main()
