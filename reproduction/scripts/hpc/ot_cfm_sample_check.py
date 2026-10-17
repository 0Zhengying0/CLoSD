"""Sample one captured CLoSD case with a trained OT-CFM checkpoint."""

import argparse
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
        parser.error("output exists")
    config = json.loads((args.checkpoint.parent / "args.json").read_text())
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    backend = load_backend("fm", {"args": config, "checkpoint": str(args.checkpoint)},
                           torch.device("cuda:0"))
    meta, arrays, case_path = load_cases(args.cases)[0]
    normalized, xyz, stat, curve, conditioned = sample_one(
        "fm", backend, meta, arrays, torch.device("cuda:0"))
    stat["checkpoint_sha256"] = sha256(args.checkpoint)
    stat["case_sha256"] = sha256(case_path)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "sample.npz", normalized_hml=normalized,
                        xyz_local_m=xyz, target_distance_curve=curve,
                        conditioned_target_distance_curve=conditioned)
    (args.output / "metrics.json").write_text(json.dumps(stat, indent=2) + "\n")
    print("OT_CFM_SAMPLE_OK", json.dumps(stat, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
