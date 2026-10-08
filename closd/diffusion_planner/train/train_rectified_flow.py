import os
import json

from closd.diffusion_planner.utils.fixseed import fixseed
from closd.diffusion_planner.utils.parser_util import train_args
from closd.diffusion_planner.utils import dist_util

from closd.diffusion_planner.train.training_loop import TrainLoop
from closd.diffusion_planner.train.train_platforms import (
    ClearmlPlatform,
    TensorboardPlatform,
    NoPlatform,
    WandBPlatform,
)

from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from closd.diffusion_planner.model.mdm import MDM
from closd.diffusion_planner.utils.model_util import get_model_args
from closd.diffusion_planner.diffusion.rectified_flow import RectifiedFlow


def main():
    args = train_args()
    fixseed(args.seed)

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name="Args")

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError(
            f"save_dir [{args.save_dir}] already exists."
        )
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")

    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        fixed_len=args.pred_len + args.context_len,
        pred_len=args.pred_len,
        hml_type=args.hml_type,
        device=dist_util.dev(),
    )

    print("creating MDM model...")

    model = MDM(**get_model_args(args, data))
    model.to(dist_util.dev())

    print("creating Rectified Flow objective...")

    rectified_flow = RectifiedFlow(
        num_timesteps=1000,
        lambda_target_loc=args.lambda_target_loc,
    )

    print(
        "Total params: %.2fM"
        % (
            sum(p.numel() for p in model.parameters_wo_clip())
            / 1000000.0
        )
    )

    print("Rectified Flow training...")
    print("RF time discretization:", rectified_flow.num_timesteps)
    print("Target location lambda:", rectified_flow.lambda_target_loc)

    TrainLoop(
        args,
        train_platform,
        model,
        rectified_flow,
        data,
    ).run_loop()

    train_platform.close()


if __name__ == "__main__":
    main()
