import argparse
import json
import os
from types import SimpleNamespace

import torch
from tqdm import tqdm

from closd.diffusion_planner.utils.fixseed import fixseed
from closd.diffusion_planner.utils import dist_util
from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from closd.diffusion_planner.model.mdm import MDM
from closd.diffusion_planner.utils.model_util import (
    get_model_args,
    load_saved_model,
)
from closd.diffusion_planner.diffusion.ode_sampler import VelocityODESampler
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import (
    get_target_location,
    sample_goal,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--teacher_model", required=True)
    p.add_argument("--output", required=True)

    # -1 = one complete pass over HumanML3D train.
    p.add_argument("--num_pairs", default=128, type=int)

    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--teacher_steps", default=10, type=int)
    p.add_argument(
        "--teacher_solver",
        default="heun",
        choices=["euler", "heun"],
    )
    p.add_argument("--seed", default=10, type=int)

    return p.parse_args()


def main():
    cli = parse_args()

    fixseed(cli.seed)
    dist_util.setup_dist(0)
    device = dist_util.dev()

    teacher_dir = os.path.dirname(cli.teacher_model)
    args_path = os.path.join(teacher_dir, "args.json")

    if not os.path.isfile(args_path):
        raise FileNotFoundError(
            f"Teacher args.json not found: {args_path}"
        )

    with open(args_path, "r") as f:
        teacher_args = SimpleNamespace(**json.load(f))

    print("===== Generate RF-2 reflow pairs =====")
    print("teacher       :", cli.teacher_model)
    print("solver        :", cli.teacher_solver)
    print("steps         :", cli.teacher_steps)
    print(
        "NFE           :",
        cli.teacher_steps
        if cli.teacher_solver == "euler"
        else 2 * cli.teacher_steps,
    )
    print("num_pairs     :", cli.num_pairs)
    print("======================================")

    data = get_dataset_loader(
        name=teacher_args.dataset,
        batch_size=cli.batch_size,
        num_frames=teacher_args.num_frames,
        split="train",
        hml_mode="train",
        fixed_len=(
            teacher_args.context_len
            + teacher_args.pred_len
        ),
        pred_len=teacher_args.pred_len,
        hml_type=teacher_args.hml_type,
        device=device,
        drop_last=False,
        return_keys=True,
    )

    model = MDM(
        **get_model_args(
            teacher_args,
            data,
        )
    )

    model.to(device)

    print("Loading RF-1 EMA teacher...")

    load_saved_model(
        model,
        cli.teacher_model,
        use_avg=True,
    )

    model.eval()

    sampler = VelocityODESampler(
        num_timesteps=1000,
        num_steps=cli.teacher_steps,
        solver=cli.teacher_solver,
    )

    all_z0 = []
    all_z1 = []
    all_prefix = []
    all_lengths = []
    all_target_cond = []
    all_is_heading = []

    all_text = []
    all_target_joint_names = []
    all_db_key = []

    total = 0

    with torch.no_grad():

        for motion, cond in tqdm(
            data,
            desc="RF-1 reflow",
        ):

            if cli.num_pairs > 0 and total >= cli.num_pairs:
                break

            bs = motion.shape[0]

            # Trim final smoke batch if necessary.
            if cli.num_pairs > 0:
                remaining = cli.num_pairs - total

                if remaining < bs:
                    motion = motion[:remaining]

                    for key, value in list(cond["y"].items()):
                        if torch.is_tensor(value):
                            cond["y"][key] = value[:remaining]
                        elif isinstance(value, list):
                            cond["y"][key] = value[:remaining]

                    bs = remaining

            # ---------------------------------------------
            # Use the same target-conditioning mechanism
            # as RF-1 training.
            # ---------------------------------------------
            target_joint_names, is_heading = sample_goal(
                bs,
                motion.device,
                teacher_args.target_joint_names,
            )

            target_cond = get_target_location(
                motion,
                data.dataset.mean[
                    None, :, None, None
                ],
                data.dataset.std[
                    None, :, None, None
                ],
                cond["y"]["lengths"],
                data.dataset.t2m_dataset.opt.joints_num,
                model.all_goal_joint_names,
                target_joint_names,
                is_heading,
            ).detach()

            cond["y"]["target_cond"] = target_cond
            cond["y"]["target_joint_names"] = target_joint_names
            cond["y"]["is_heading"] = is_heading

            # Save all conditions BEFORE the model is called.
            # MDM mutates some entries such as mask internally.
            prefix_cpu = cond["y"]["prefix"].detach().cpu().clone()
            lengths_cpu = cond["y"]["lengths"].detach().cpu().clone()
            target_cpu = target_cond.detach().cpu().clone()
            heading_cpu = is_heading.detach().cpu().clone()

            texts = list(cond["y"]["text"])

            joint_names = [
                list(x)
                for x in target_joint_names
            ]

            keys = list(
                cond["y"].get(
                    "db_key",
                    [None] * bs,
                )
            )

            cond["y"] = {
                k: (
                    v.to(device)
                    if torch.is_tensor(v)
                    else v
                )
                for k, v in cond["y"].items()
            }

            # Encode text once instead of once per ODE step.
            cond["y"]["text_embed"] = model.encode_text(
                cond["y"]["text"]
            )

            suffix_shape = (
                bs,
                model.njoints,
                model.nfeats,
                teacher_args.pred_len,
            )

            # z0: Gaussian source.
            z0 = torch.randn(
                *suffix_shape,
                device=device,
            )

            # RF-1 transport z0 -> z1.
            sample = sampler.p_sample_loop(
                model,
                suffix_shape,
                noise=z0,
                model_kwargs=cond,
                progress=False,
            )

            # p_sample_loop returns prefix + generated suffix.
            z1 = sample[
                ..., -teacher_args.pred_len:
            ]

            all_z0.append(z0.cpu())
            all_z1.append(z1.cpu())

            all_prefix.append(prefix_cpu)
            all_lengths.append(lengths_cpu)
            all_target_cond.append(target_cpu)
            all_is_heading.append(heading_cpu)

            all_text.extend(texts)
            all_target_joint_names.extend(joint_names)
            all_db_key.extend(keys)

            total += bs

            print(f"generated pairs: {total}")

    if total == 0:
        raise RuntimeError("Generated zero pairs.")

    payload = {
        "z0": torch.cat(all_z0, dim=0),
        "z1": torch.cat(all_z1, dim=0),

        "prefix": torch.cat(all_prefix, dim=0),
        "lengths": torch.cat(all_lengths, dim=0),

        "target_cond": torch.cat(
            all_target_cond,
            dim=0,
        ),

        "is_heading": torch.cat(
            all_is_heading,
            dim=0,
        ),

        "text": all_text,
        "target_joint_names": all_target_joint_names,
        "db_key": all_db_key,

        "metadata": {
            "teacher_model": cli.teacher_model,
            "teacher_solver": cli.teacher_solver,
            "teacher_steps": cli.teacher_steps,
            "teacher_nfe": sampler.nfe,
            "seed": cli.seed,
            "num_pairs": total,
            "context_len": teacher_args.context_len,
            "pred_len": teacher_args.pred_len,
            "target_joint_names": (
                teacher_args.target_joint_names
            ),
        },
    }

    os.makedirs(
        os.path.dirname(
            os.path.abspath(cli.output)
        ),
        exist_ok=True,
    )

    torch.save(
        payload,
        cli.output,
    )

    print("===== Reflow pairs finished =====")
    print("z0     :", payload["z0"].shape)
    print("z1     :", payload["z1"].shape)
    print("prefix :", payload["prefix"].shape)
    print("target :", payload["target_cond"].shape)
    print("saved  :", cli.output)


if __name__ == "__main__":
    main()
