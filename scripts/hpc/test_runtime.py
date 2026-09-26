"""CPU-only regression checks for job safety and Slurm path handling."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('run_job', str(Path(__file__).with_name('run_job.py')))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeTests(unittest.TestCase):
    def invoke(self, args, overrides=None):
        with mock.patch.dict(os.environ, overrides or {}, clear=False), mock.patch('sys.argv', ['run_job.py'] + args), contextlib.redirect_stdout(io.StringIO()) as output:
            result = runtime.main()
        return result, output.getvalue()

    def test_fresh_check_does_not_write(self):
        with tempfile.TemporaryDirectory() as d:
            save = Path(d) / 'new run'
            _, output = self.invoke(['dip_train', '--check', '--save-dir', str(save)])
            self.assertFalse(save.exists())
            self.assertIn('--save_dir', output)
            self.assertIn('Resume checkpoint: (none)', output)

    def test_existing_training_requires_resume(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            save = Path(d)
            (save / 'args.json').write_text('{}')
            with self.assertRaises(SystemExit):
                self.invoke(['dip_train', '--check', '--save-dir', d])
            with self.assertRaises(SystemExit):
                self.invoke(['dip_train', '--check', '--save-dir', d, '--resume'])
            for step in [10, 100]:
                (save / ('model%09d.pt' % step)).touch()
            with self.assertRaises(SystemExit):
                self.invoke(['dip_train', '--check', '--save-dir', d, '--resume'])
            (save / 'opt000000100.pt').touch()
            _, output = self.invoke(['dip_train', '--check', '--save-dir', d, '--resume'])
            self.assertIn('Resume checkpoint: ' + str(save / 'model000000100.pt'), output)

    def test_submit_explicit_paths_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'logs'
            with mock.patch.object(runtime.subprocess, 'call', return_value=0) as call:
                self.invoke(['dip_train', '--submit', '--save-dir', str(Path(d) / 'new')], {'CLOSD_LOGS': str(log), 'CLOSD_SLURM_TIME': '00:10:00'})
            self.assertTrue(log.is_dir())
            command = call.call_args[0][0]
            self.assertIn('--chdir=' + str(runtime.ROOT), command)
            self.assertIn('--output=' + str(log / '%x_%j.out'), command)
            self.assertIn('--time=00:10:00', command)
            self.assertEqual(call.call_args[1]['env']['CLOSD_ROOT'], str(runtime.ROOT))

    def test_slurm_spool_location_does_not_change_root(self):
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / 'slurm_script'
            script.write_bytes((runtime.ROOT / 'scripts/hpc/dip_train.slurm').read_bytes())
            env = dict(os.environ, CLOSD_ROOT=str(runtime.ROOT), SLURM_SUBMIT_DIR='/tmp')
            output = subprocess.check_output(['bash', str(script), '--check', '--save-dir', str(Path(d) / 'new')], env=env, cwd=d, universal_newlines=True)
            self.assertIn('--chdir=' + str(runtime.ROOT), output)


if __name__ == '__main__':
    unittest.main()
