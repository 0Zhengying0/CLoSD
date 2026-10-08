import json
import os

from argparse import ArgumentParser

from torch.utils.data import DataLoader

from closd.diffusion_planner.utils.fixseed import fixseed
from closd.diffusion_planner.utils import dist_util

from closd.diffusion_planner.utils.parser_util import (
    add_base_options,
    add_data_options,
    add_model_options,
    add_diffusion_options,
    add_training_options,
    apply_rules,
)

from closd.diffusion_planner.train.train_platforms import (
    ClearmlPlatform,
    TensorboardPlatform,
    NoPlatform,
    WandBPlatform,
)

from closd.diffusion_planner.data_loaders.get_data import (
    get_dataset_loader,
)

from closd.diffusion_planner.data_loaders.reflow_dataset import (
    ReflowPairDataset,
    reflow_collate,
)

from closd.diffusion_planner.model.mdm import MDM

from closd.diffusion_planner.utils.model_util import (
    get_model_args,
    load_saved_model,
)

from closd.diffusion_planner.diffusion.rectified_flow_reflow import (
    RectifiedFlowReflow,
)

from closd.diffusion_planner.train.reflow_training_loop import (
    ReflowTrainLoop,
)


def rf2_train_args():
    parser = ArgumentParser()

    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)

    parser.add_argument(
        "--reflow_pairs",
        required=True,
        type=str,
        help="Path to RF-1 generated reflow pair file.",
    )

    parser.add_argument(
        "--rf1_init_model",
        required=True,
        type=str,
        help="RF-1 checkpoint used to initialize RF-2.",
    )

    return apply_rules(
        parser.parse_args()
    )


def main():

    args = rf2_train_args()

    fixseed(args.seed)

    train_platform_type = eval(
        args.train_platform_type
    )

    train_platform = train_platform_type(
        args.save_dir
    )

    train_platform.report_args(
        args,
        name="Args",
    )

    if args.save_dir is None:
        raise FileNotFoundError(
            "save_dir was not specified."
        )

    elif (
        os.path.exists(args.save_dir)
        and not args.overwrite
    ):
        raise FileExistsError(
            f"save_dir [{args.save_dir}] already exists."
        )

    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args_path = os.path.join(
        args.save_dir,
        "args.json",
    )

    with open(args_path, "w") as fw:
        json.dump(
            vars(args),
            fw,
            indent=4,
            sort_keys=True,
        )

    dist_util.setup_dist(args.device)

    # --------------------------------------------------
    # Load normal HumanML dataset only for:
    #   model metadata
    #   normalization statistics
    #   target-location conversion
    # --------------------------------------------------
    print("Creating HumanML metadata loader...")

    metadata_loader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        fixed_len=(
            args.context_len
            + args.pred_len
        ),
        pred_len=args.pred_len,
        hml_type=args.hml_type,
        device=dist_util.dev(),
        drop_last=False,
    )

    print("Creating MDM...")

    model = MDM(
        **get_model_args(
            args,
            metadata_loader,
        )
    )

    model.to(
        dist_util.dev()
    )

    # --------------------------------------------------
    # IMPORTANT:
    # Official reflow fine-tunes 1-RF rather than
    # training RF-2 from random initialization.
    # --------------------------------------------------
    print(
        f"Initializing RF-2 from RF-1 EMA "
        f"[{args.rf1_init_model}]..."
    )

    load_saved_model(
        model,
        args.rf1_init_model,
        use_avg=True,
    )

    # --------------------------------------------------
    # Reflow pair dataset.
    # --------------------------------------------------
    reflow_dataset = ReflowPairDataset(
        args.reflow_pairs,
        metadata_loader.dataset,
    )

    reflow_loader = DataLoader(
        reflow_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=reflow_collate,
    )

    print("Creating RF-2 objective...")

    rf2 = RectifiedFlowReflow(
        num_timesteps=1000,
        lambda_target_loc=args.lambda_target_loc,
    )

    print(
        "Total params: %.2fM"
        % (
            sum(
                p.numel()
                for p
                in model.parameters_wo_clip()
            )
            / 1000000.0
        )
    )

    print("===== RF-2 training =====")
    print(
        "Reflow pairs:",
        args.reflow_pairs,
    )
    print(
        "RF-1 initialization:",
        args.rf1_init_model,
    )
    print(
        "dataset size:",
        len(reflow_dataset),
    )
    print(
        "time discretization:",
        rf2.num_timesteps,
    )
    print(
        "target lambda:",
        rf2.lambda_target_loc,
    )

    ReflowTrainLoop(
        args,
        train_platform,
        model,
        rf2,
        reflow_loader,
    ).run_loop()

    train_platform.close()


if __name__ == "__main__":
    main()
