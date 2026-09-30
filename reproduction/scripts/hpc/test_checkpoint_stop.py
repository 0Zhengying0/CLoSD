"""CPU tests of checkpoint stopping through the real upstream training loop.

Run with the CLoSD environment. Model computation and save/eval/gen are replaced
with probes; this checks ordering and failure propagation, not GPU training.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from closd.diffusion_planner.train import training_loop
from train_until_checkpoint import checkpoint_limited_loop


class CheckpointStopTests(unittest.TestCase):
    def exercise(self, target=10, start=0, fail=None, batches=None):
        loop_type = checkpoint_limited_loop(training_loop.TrainLoop, target)
        loop = object.__new__(loop_type)
        loop.num_steps = max(target, 20)
        loop.num_epochs = 1
        loop.lr_anneal_steps = 0
        loop.step, loop.resume_step = start, 0
        loop.log_interval = 1
        loop.save_interval = 10 if target == 10 else 50000
        loop.device = torch.device('cpu')
        loop.data = [(torch.zeros(1), {'y': {}}) for _ in range(batches or (target - start + 3))]
        loop.cond_modifiers = lambda *args: None
        steps, events = [], []
        loop.run_step = lambda *args: steps.append(loop.total_step())
        loop.model = SimpleNamespace(eval=lambda: None, train=lambda: None)

        def probe(kind):
            def run(*args):
                events.append((kind, loop.total_step()))
                if kind == fail and loop.total_step() == target:
                    raise OSError('injected ' + kind + ' failure')
            return run

        with mock.patch.dict(os.environ, {'DIFFUSION_TRAINING_TEST': ''}), \
             mock.patch.object(training_loop.logger, 'get_current', return_value=SimpleNamespace(dumpkvs=lambda: {})), \
             mock.patch.object(training_loop.TrainLoop, 'save', probe('save')), \
             mock.patch.object(training_loop.TrainLoop, 'evaluate', probe('eval')), \
             mock.patch.object(training_loop.TrainLoop, 'generate_during_training', probe('gen')):
            loop.run_loop()
        return steps, events

    def test_smoke_waits_for_full_cycle_and_never_runs_next_step(self):
        steps, events = self.exercise()
        self.assertEqual(steps, list(range(11)))
        self.assertEqual(events, [(kind, step) for step in (0, 10) for kind in ('save', 'eval', 'gen')])

    def test_formal_checkpoint_boundary(self):
        steps, events = self.exercise(target=300000, start=299999)
        self.assertEqual(steps, [299999, 300000])
        self.assertEqual(events, [(kind, 300000) for kind in ('save', 'eval', 'gen')])

    def test_errors_are_not_reported_as_success(self):
        for kind in ('save', 'eval', 'gen'):
            with self.subTest(kind=kind), self.assertRaisesRegex(OSError, 'injected ' + kind):
                self.exercise(fail=kind)

    def test_ending_without_target_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, 'before completing checkpoint'):
            self.exercise(batches=2)

    def test_invalid_stop_configuration_is_rejected_before_initialization(self):
        class Base:
            def __init__(self, args):
                raise AssertionError('Should not initialize the model')

        for target, interval, horizon, flag in [(0, 10, 20, ''), (11, 10, 20, ''),
                                               (30, 10, 20, ''), (10, 10, 20, '1')]:
            with self.subTest(target=target, flag=flag), \
                 mock.patch.dict(os.environ, {'DIFFUSION_TRAINING_TEST': flag}), \
                 self.assertRaises(ValueError):
                checkpoint_limited_loop(Base, target)(SimpleNamespace(save_interval=interval, num_steps=horizon))


if __name__ == '__main__':
    unittest.main()
