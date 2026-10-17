"""Independent OT-CFM training entry point; checkpoints use their own save_dir."""

import json
import os

from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from closd.diffusion_planner.diffusion.ot_cfm import OptimalTransportConditionalFlowMatching
from closd.diffusion_planner.model.mdm import MDM
from closd.diffusion_planner.train.train_platforms import NoPlatform, TensorboardPlatform, ClearmlPlatform, WandBPlatform
from closd.diffusion_planner.train.training_loop import TrainLoop
from closd.diffusion_planner.utils import dist_util
from closd.diffusion_planner.utils.fixseed import fixseed
from closd.diffusion_planner.utils.model_util import get_model_args
from closd.diffusion_planner.utils.parser_util import train_args


def main():
    args = train_args()
    fixseed(args.seed)
    if args.save_dir is None:
        raise ValueError("save_dir is required")
    if os.path.exists(args.save_dir) and os.listdir(args.save_dir):
        raise FileExistsError("refusing to reuse a nonempty OT-CFM run: " + args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=4, sort_keys=True)
    platform = globals()[args.train_platform_type](args.save_dir)
    platform.report_args(args, name="Args")
    dist_util.setup_dist(args.device)
    data = get_dataset_loader(
        name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,
        fixed_len=args.pred_len + args.context_len, pred_len=args.pred_len,
        hml_type=args.hml_type, device=dist_util.dev(),
    )
    model = MDM(**get_model_args(args, data)).to(dist_util.dev())
    objective = OptimalTransportConditionalFlowMatching(
        num_timesteps=1000, lambda_target_loc=args.lambda_target_loc
    )
    print("OT-CFM training: exact minibatch assignment, straight path, sigma=0")
    print("Total params: %.2fM" % (sum(p.numel() for p in model.parameters_wo_clip()) / 1e6))
    TrainLoop(args, platform, model, objective, data).run_loop()
    platform.close()


if __name__ == "__main__":
    main()
