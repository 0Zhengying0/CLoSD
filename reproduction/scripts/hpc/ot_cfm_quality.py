"""Generate OT-CFM motions on the 20 captured FM quality conditions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from run_dip_fm_quality import load_backend, load_cases, sample_one, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    config = json.loads((args.checkpoint.parent / "args.json").read_text())
    if config.get("num_steps") != 300000 or not config.get("use_ema"):
        parser.error("expected completed 300k EMA checkpoint")
    cases = load_cases(args.cases)
    backend = load_backend("fm", {"args": config, "checkpoint": str(args.checkpoint)},
                           torch.device("cuda:0"))
    args.output.mkdir(parents=True)
    rows = []
    for meta, arrays, case_path in cases:
        normalized, xyz, stat, curve, conditioned = sample_one(
            "fm", backend, meta, arrays, torch.device("cuda:0"))
        stat["model"] = "OT-CFM"
        stat["checkpoint"] = str(args.checkpoint)
        stat["checkpoint_sha256"] = sha256(args.checkpoint)
        stat["case_input_sha256"] = sha256(case_path)
        case_dir = args.output / meta["case_id"]
        case_dir.mkdir()
        np.savez_compressed(case_dir / "trajectory.npz", normalized_hml=normalized,
                            xyz_local_m=xyz, target_distance_curve=curve,
                            conditioned_target_distance_curve=conditioned)
        (case_dir / "metrics.json").write_text(json.dumps(stat, indent=2) + "\n")
        rows.append(stat)
        print("OT_CFM_QUALITY_CASE", meta["case_id"], flush=True)
    with (args.output / "results.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "provenance.json").write_text(json.dumps({
        "case_manifest": str(args.cases), "case_manifest_sha256": sha256(args.cases),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "solver": "rk4", "steps": 3, "nfe": 12, "cfg": 7.5,
        "initial_noise": "identical to captured FM cases", "cases": len(rows),
    }, indent=2) + "\n")
    print("OT_CFM_QUALITY_PASS", len(rows), flush=True)


if __name__ == "__main__":
    main()
