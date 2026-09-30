#!/usr/bin/env python3
"""Run the upstream trainer through one complete target checkpoint cycle."""
import argparse
import os
from pathlib import Path
import sys


class _CheckpointCycleComplete(Exception):
    """Internal control flow; training errors must never be caught as success."""


def checkpoint_limited_loop(base_loop, stop_checkpoint):
    class CheckpointLimitedLoop(base_loop):
        def __init__(self, args, *positional, **kwargs):
            if stop_checkpoint <= 0 or args.save_interval <= 0:
                raise ValueError('Checkpoint step and save interval must be positive')
            if stop_checkpoint % args.save_interval:
                raise ValueError('Stop checkpoint must be a multiple of save_interval')
            if args.num_steps < stop_checkpoint:
                raise ValueError('num_steps must cover the requested checkpoint')
            if os.environ.get('DIFFUSION_TRAINING_TEST', ''):
                raise ValueError('Unset DIFFUSION_TRAINING_TEST; use --stop-after-checkpoint')
            super().__init__(args, *positional, **kwargs)
            if self.total_step() > stop_checkpoint:
                raise ValueError('Resume checkpoint is beyond the requested stop checkpoint')

        def generate_during_training(self):
            # Upstream calls save(), evaluate(), then this method synchronously.
            # Returning from super() includes all enabled generation variants.
            super().generate_during_training()
            if self.total_step() == stop_checkpoint:
                raise _CheckpointCycleComplete()

        def run_loop(self):
            try:
                super().run_loop()
            except _CheckpointCycleComplete:
                print('=== CHECKPOINT CYCLE COMPLETE: step {} ==='.format(stop_checkpoint), flush=True)
                return
            raise RuntimeError('Training ended before completing checkpoint {}'.format(stop_checkpoint))

    return CheckpointLimitedLoop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--stop-after-checkpoint', type=int, required=True,
                        help='Checkpoint filename step, using the upstream zero-based numbering')
    control, training_args = parser.parse_known_args(argv)
    if control.stop_after_checkpoint <= 0:
        parser.error('--stop-after-checkpoint must be positive')
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    from closd.diffusion_planner.train import train_mdm

    original_loop, original_argv = train_mdm.TrainLoop, sys.argv
    train_mdm.TrainLoop = checkpoint_limited_loop(original_loop, control.stop_after_checkpoint)
    sys.argv = [str(Path(__file__))] + training_args
    try:
        # The upstream main still parses/writes args, builds the model, and closes
        # the training platform after our run_loop returns normally.
        train_mdm.main()
    finally:
        train_mdm.TrainLoop, sys.argv = original_loop, original_argv


if __name__ == '__main__':
    main()
