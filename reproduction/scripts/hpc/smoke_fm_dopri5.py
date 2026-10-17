#!/usr/bin/env python3
"""Small GPU replay of real captured CLoSD planner conditions with DOPRI5."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_dip_fm_quality import load_backend, load_cases, validate_models
from quality_metrics import recover_xyz
from closd.diffusion_planner.diffusion.ode_sampler import VelocityODESampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--max-num-steps", type=int, default=128)
    parser.add_argument("--max-nfe", type=int, default=256)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    case_ids = args.case_id or ["REACH_00_normal", "GET_UP_00_normal"]
    if args.output.exists():
        parser.error("output already exists: {}".format(args.output))
    records = validate_models(check_ema=True)
    cases = [entry for entry in load_cases(args.cases) if entry[0]["case_id"] in case_ids]
    if len(cases) != len(set(case_ids)):
        parser.error("one or more requested real cases are missing")
    import torch
    from closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required for checkpoint smoke test")
    device = torch.device("cuda:0")
    model, raw, _, mean, std = load_backend("fm", records["fm"], device)
    if not isinstance(model, ClassifierFreeSampleModel):
        raise RuntimeError("CFG 7.5 requires classifier-free guidance wrapper")
    sampler = VelocityODESampler(
        num_timesteps=records["fm"]["args"]["diffusion_steps"],
        num_steps=0, solver="dopri5", rtol=args.rtol, atol=args.atol,
        max_num_steps=args.max_num_steps, max_nfe=args.max_nfe,
    )
    args.output.mkdir(parents=True)
    metrics = []
    for meta, arrays, _ in cases:
        np.testing.assert_allclose(mean, arrays["mean"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(std, arrays["std"], rtol=0, atol=1e-6)
        torch.manual_seed(int(meta["seed"]))
        torch.cuda.manual_seed_all(int(meta["seed"]))
        y = {"prefix": torch.from_numpy(arrays["prefix"][None]).to(device),
             "mask": torch.from_numpy(arrays["mask"][None]).to(device),
             "prefix_len": 20,
             "target_cond": torch.from_numpy(arrays["target_cond"][None]).to(device),
             "target_joint_names": [list(meta["target_joint_names"])],
             "is_heading": torch.tensor([meta["is_heading"]], device=device),
             "text": [meta["text"]],
             "scale": torch.full((1,), 7.5, device=device)}
        for extra in ("pred_target_cond", "heading_cond"):
            if extra in arrays:
                y[extra] = torch.from_numpy(arrays[extra][None]).to(device)
        with torch.inference_mode():
            y["text_embed"] = model.encode_text(y["text"])
        noise = torch.from_numpy(arrays["initial_noise"][None]).to(device)
        calls = [0]
        hook = raw.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                sample = sampler.p_sample_loop(
                    model, (1, 263, 1, 40), noise=noise,
                    model_kwargs={"y": y}, device=device,
                )
            torch.cuda.synchronize()
        finally:
            hook.remove()
        elapsed = time.perf_counter() - start
        if calls[0] != 2 * sampler.nfe:
            raise RuntimeError("CFG MDM forwards do not equal 2 x NFE")
        normalized = sample[0].cpu().numpy()
        if normalized.shape != (263, 1, 60) or not np.isfinite(normalized).all():
            raise RuntimeError("invalid DOPRI5 normalized motion")
        if not np.allclose(normalized[..., :20], arrays["prefix"], atol=1e-6):
            raise RuntimeError("real history prefix was modified")
        xyz = recover_xyz(normalized, mean, std, prefix_frames=20)
        np.savez_compressed(args.output / (meta["case_id"] + ".npz"),
                            normalized_hml=normalized, xyz_local_m=xyz)
        item = {"case_id": meta["case_id"], "task": meta["task"],
                "seed": meta["seed"], "nfe": sampler.nfe,
                "mdm_forwards": calls[0], "planner_latency_sec": elapsed,
                "finite": True, "shape": list(normalized.shape)}
        metrics.append(item)
        print("DOPRI5_SMOKE_CASE=" + json.dumps(item), flush=True)
    (args.output / "smoke.json").write_text(json.dumps({
        "checkpoint": records["fm"]["checkpoint"], "cfg": 7.5,
        "rtol": args.rtol, "atol": args.atol,
        "max_num_steps": args.max_num_steps, "max_nfe": args.max_nfe,
        "cases": metrics,
    }, indent=2) + "\n")
    print("DOPRI5_SMOKE_COMPLETE=1", flush=True)


if __name__ == "__main__":
    main()
