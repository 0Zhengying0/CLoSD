#!/usr/bin/env python3
"""Replay identical real CLoSD conditions through DiP and FM, then compare planners."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quality_metrics import GOAL_NAMES, metrics, recover_xyz

DIP = ROOT / "closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt"
FM = ROOT / "reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt"
TASKS = ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT", "GET_UP")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_models(check_ema=False):
    records = {}
    for label, path, steps, target_names in (("dip", DIP, 10, "DIMP_FULL"),
                                              ("fm", FM, 1000, "DIMP_FINAL")):
        args_path = path.parent / "args.json"
        if not path.is_file() or path.stat().st_size == 0 or not args_path.is_file():
            raise FileNotFoundError("checkpoint or args.json missing: {}".format(path))
        args = json.loads(args_path.read_text())
        expected = {"dataset": "humanml", "diffusion_steps": steps,
                    "context_len": 20, "pred_len": 40, "num_frames": 60,
                    "multi_target_cond": True, "use_ema": True,
                    "target_joint_names": target_names,
                    "arch": "trans_dec", "text_encoder_type": "bert"}
        for key, value in expected.items():
            if args.get(key) != value:
                raise ValueError("{} args.json {}={!r}, expected {!r}".format(
                    label, key, args.get(key), value))
        if check_ema:
            import torch
            checkpoint = torch.load(path, map_location="cpu")
            if "model_avg" not in checkpoint:
                raise ValueError("{} checkpoint has no EMA model_avg".format(label))
            del checkpoint
        records[label] = {"checkpoint": str(path), "args_path": str(args_path),
                          "args_sha256": sha256(args_path), "args": args}
    return records


def load_cases(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source") != "real CLoSD planner inputs before sampling":
        raise ValueError("case manifest is not from real CLoSD capture")
    entries = manifest.get("cases", [])
    if len(entries) != 20 or any(sum(item["task"] == task for item in entries) != 4
                                 for task in TASKS):
        raise ValueError("expected four captured cases for every CLoSD task")
    if len({item["case_id"] for item in entries}) != len(entries):
        raise ValueError("duplicate case IDs")
    cases = []
    for item in entries:
        case_id = item["case_id"]
        case_path = manifest_path.parent / "cases" / (case_id + ".npz")
        meta_path = manifest_path.parent / "cases" / (case_id + ".json")
        if not case_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError("case missing: {}".format(case_id))
        meta = json.loads(meta_path.read_text())
        if meta != item:
            raise ValueError("case metadata differs from manifest: {}".format(case_id))
        with np.load(case_path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        shapes = {"prefix": (263, 1, 20), "target_cond": (8, 3),
                  "mask": (1, 1, 40), "mean": (263,), "std": (263,),
                  "initial_noise": (263, 1, 40), "recon_r_rot": (4,),
                  "recon_r_pos": (3,), "original_target_local_hml_xyz": (3,)}
        for key, shape in shapes.items():
            if key not in arrays or arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError("invalid {} for {}".format(key, case_id))
        if not arrays["mask"].all() or not np.any(arrays["prefix"]):
            raise ValueError("case has invalid mask or empty history: {}".format(case_id))
        if (len(meta["target_joint_names"]) != 1 or
                meta["target_joint_names"][0] not in GOAL_NAMES[:-1] or
                not meta["text"] or meta["prefix_frames"] != 20 or
                meta["pred_frames"] != 40):
            raise ValueError("invalid case condition: {}".format(case_id))
        cases.append((meta, arrays, case_path))
    return cases


def load_backend(label, record, device):
    import torch
    from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
    from closd.diffusion_planner.utils.model_util import create_model_and_diffusion, load_saved_model
    from closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel
    from closd.diffusion_planner.diffusion.ode_sampler import VelocityODESampler

    args = SimpleNamespace(**record["args"])
    data = get_dataset_loader(name=args.dataset, batch_size=1, num_frames=196,
                              split="test", hml_mode="train", drop_last=False,
                              return_keys=True, device=device)
    raw, diffusion = create_model_and_diffusion(args, data)
    # load_saved_model otherwise silently falls back to non-EMA weights.
    checkpoint = torch.load(record["checkpoint"], map_location="cpu")
    if "model_avg" not in checkpoint:
        raise ValueError("{} checkpoint has no EMA model_avg".format(label))
    del checkpoint
    load_saved_model(raw, record["checkpoint"], use_avg=True)
    raw.to(device).eval()
    model = ClassifierFreeSampleModel(raw, "text").to(device).eval()
    sampler = diffusion if label == "dip" else VelocityODESampler(
        num_timesteps=args.diffusion_steps, num_steps=3, solver="rk4")
    norm = data.dataset.t2m_dataset
    return model, raw, sampler, np.asarray(norm.mean), np.asarray(norm.std)


def sample_one(label, backend, meta, arrays, device):
    import torch
    model, raw, sampler, mean, std = backend
    np.testing.assert_allclose(mean, arrays["mean"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(std, arrays["std"], rtol=0, atol=1e-6)
    seed = int(meta["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
            sample = sampler.p_sample_loop(model, (1, 263, 1, 40),
                                           noise=noise, clip_denoised=False,
                                           model_kwargs={"y": y}, device=device,
                                           progress=False)
        torch.cuda.synchronize()
    finally:
        hook.remove()
    elapsed = time.perf_counter() - start
    expected_calls = 20 if label == "dip" else 24
    if calls[0] != expected_calls:
        raise RuntimeError("{} MDM forwards {}, expected {}".format(label, calls[0], expected_calls))
    normalized = sample[0].detach().cpu().numpy()
    if normalized.shape != (263, 1, 60) or not np.isfinite(normalized).all():
        raise ValueError("{} sampled motion shape or values invalid".format(label))
    if not np.allclose(normalized[..., :20], arrays["prefix"], atol=1e-6):
        raise ValueError("{} did not retain the captured prefix".format(label))
    xyz = recover_xyz(normalized, arrays["mean"], arrays["std"], prefix_frames=20)
    stat, curve, conditioned_curve = metrics(
        xyz, arrays["target_cond"], meta["task"],
        meta["target_joint_names"], actual_target_local=arrays["original_target_local_hml_xyz"])
    stat.update({"model": label.upper(), "case_id": meta["case_id"],
                 "task": meta["task"], "difficulty": meta["difficulty"],
                 "sampling": "Diffusion 10 steps" if label == "dip" else "RK4 3 steps",
                 "nfe": 10 if label == "dip" else 12,
                 "mdm_forwards": calls[0], "planner_latency_sec": elapsed,
                 "cfg": 7.5, "seed": seed})
    return normalized, xyz, stat, curve, conditioned_curve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path,
                        help="manifest.json from a real CLoSD condition capture")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=0,
                        help="1 for GPU smoke; 0 for all 20 formal cases")
    parser.add_argument("--case-id", help="Run exactly this one captured case")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    records = validate_models(check_ema=True)
    cases = load_cases(args.cases)
    if args.case_id:
        cases = [case for case in cases if case[0]["case_id"] == args.case_id]
        if len(cases) != 1:
            parser.error("case ID not found")
    elif args.max_cases:
        if args.max_cases < 1:
            parser.error("max-cases must be positive")
        cases = cases[:args.max_cases]
    print("QUALITY_PREFLIGHT cases={} DiP={} FM={}".format(
        len(cases), records["dip"]["checkpoint"], records["fm"]["checkpoint"]), flush=True)
    if args.validate_only:
        return 0
    if args.output.exists():
        parser.error("output path already exists: {}".format(args.output))
    import torch
    if not torch.cuda.is_available():
        parser.error("generation requires CUDA GPU")
    from quality_render import render_distance_curve, render_video
    import summarize_dip_fm_quality as summary
    device = torch.device("cuda:0")
    backends = {key: load_backend(key, record, device) for key, record in records.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {"case_manifest": str(args.cases),
                  "case_manifest_sha256": sha256(args.cases),
                  "models": {key: {k: v for k, v in record.items() if k != "args"}
                             for key, record in records.items()},
                  "git_commit": subprocess.check_output(
                      ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                  "git_dirty": bool(subprocess.check_output(
                      ["git", "-C", str(ROOT), "status", "--porcelain", "--",
                       "closd/env/tasks/closd.py", "closd/learning/quality_snapshot.py",
                       "reproduction/scripts/hpc/run_dip_fm_quality.py",
                       "reproduction/scripts/hpc/quality_metrics.py",
                       "reproduction/scripts/hpc/quality_render.py"], text=True).strip()),
                  "source_sha256": {
                      str(path.relative_to(ROOT)): sha256(path) for path in (
                          ROOT / "closd/env/tasks/closd.py",
                          ROOT / "closd/learning/quality_snapshot.py",
                          ROOT / "reproduction/scripts/hpc/run_dip_fm_quality.py",
                          ROOT / "reproduction/scripts/hpc/quality_metrics.py",
                          ROOT / "reproduction/scripts/hpc/quality_render.py")},
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                  "cases_requested": len(cases), "cfg": 7.5,
                  "same_initial_noise": True,
                  "internal_randomness_identical": False,
                  "not_controller_executed": True}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for index, (meta, arrays, case_path) in enumerate(cases):
        case_dir = args.output / meta["case_id"]
        case_dir.mkdir()
        trajectories = {}
        for label in ("dip", "fm"):
            normalized, xyz, stat, curve, conditioned_curve = sample_one(
                label, backends[label], meta, arrays, device)
            stat["case_input_sha256"] = sha256(case_path)
            stat["checkpoint"] = records[label]["checkpoint"]
            stat["args_sha256"] = records[label]["args_sha256"]
            np.savez_compressed(case_dir / (label + "_trajectory.npz"),
                                normalized_hml=normalized, xyz_local_m=xyz,
                                target_distance_curve=curve,
                                conditioned_target_distance_curve=conditioned_curve)
            with (case_dir / (label + "_target_distance.csv")).open("x") as handle:
                handle.write("prediction_time_s,environment_target_distance_m,conditioned_target_distance_m\n")
                np.savetxt(handle, np.column_stack((curve, conditioned_curve[:, 1])),
                           delimiter=",", fmt="%.7f")
            stat["trajectory_npz"] = str((case_dir / (label + "_trajectory.npz")).relative_to(args.output))
            stat["distance_curve_csv"] = str((case_dir / (label + "_target_distance.csv")).relative_to(args.output))
            trajectories[label] = (xyz, stat)
        target_name = meta["target_joint_names"][0]
        conditioned_target = arrays["target_cond"][GOAL_NAMES.index(target_name)]
        target = arrays["original_target_local_hml_xyz"]
        dip_xyz, fm_xyz = trajectories["dip"][0], trajectories["fm"][0]
        with np.load(case_dir / "dip_trajectory.npz") as dip_data, \
                np.load(case_dir / "fm_trajectory.npz") as fm_data:
            render_distance_curve(case_dir / "target_distance.png",
                                  dip_data["target_distance_curve"],
                                  fm_data["target_distance_curve"],
                                  meta["task"], target_name,
                                  dip_data["conditioned_target_distance_curve"],
                                  fm_data["conditioned_target_distance_curve"])
        heading_label = (" | heading {:.0f}°".format(np.degrees(meta["heading_target_rad"]))
                         if meta.get("heading_target_rad") is not None else "")
        title = "{} | {} | {} goal [{:.2f},{:.2f},{:.2f}] m{} | {}".format(
            meta["case_id"], meta["task"], target_name,
            *[float(value) for value in target], heading_label, meta["text"])
        render_video(case_dir / "dip.mp4", dip_xyz, fm_xyz, target, title,
                     model="dip", conditioned_target=conditioned_target)
        render_video(case_dir / "fm.mp4", dip_xyz, fm_xyz, target, title,
                     model="fm", conditioned_target=conditioned_target)
        gif = case_dir / "side_by_side.gif" if meta["difficulty"] == "hard" or len(cases) == 1 else None
        render_video(case_dir / "side_by_side.mp4", dip_xyz, fm_xyz, target, title,
                     side_by_side=True, gif_path=gif,
                     conditioned_target=conditioned_target)
        for label in ("dip", "fm"):
            stat = trajectories[label][1]
            stat["model_mp4"] = str((case_dir / (label + ".mp4")).relative_to(args.output))
            stat["comparison_mp4"] = str((case_dir / "side_by_side.mp4").relative_to(args.output))
            stat["comparison_gif"] = str(gif.relative_to(args.output)) if gif else None
            stat["distance_curve_png"] = str((case_dir / "target_distance.png").relative_to(args.output))
            (case_dir / (label + "_metrics.json")).write_text(json.dumps(stat, indent=2) + "\n")
        print("QUALITY_CASE_COMPLETE={} {}/{}".format(meta["case_id"], index + 1, len(cases)), flush=True)
    summary.summarize(args.output)
    print("QUALITY_COMPARISON_COMPLETE=1 OUTPUT={}".format(args.output), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
