#!/usr/bin/env python3
"""Inspect, submit, or execute preserved CLoSD/DiP experiment recipes."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(os.environ.get('CLOSD_ROOT', str(Path(__file__).resolve().parents[2]))).resolve()
JOBS = json.loads((ROOT / 'scripts/hpc/jobs.json').read_text())


def paths():
    shared = Path(os.environ.get('CLOSD_SHARED_ROOT', str(ROOT.parent))).resolve()
    def setting(name, default):
        return Path(os.environ.get(name, str(default))).expanduser().resolve()
    return {
        'repo': ROOT, 'shared': shared,
        'env': setting('CLOSD_ENV', shared / 'envs/closd'),
        'isaac': setting('CLOSD_ISAACGYM', shared / 'isaacgym'),
        'container': setting('CLOSD_CONTAINER', shared / 'containers/closd_cuda121_ubuntu20_devel.sif'),
        'cache': setting('CLOSD_CACHE', ROOT / '.local/cache'),
        'logs': setting('CLOSD_LOGS', ROOT / 'reproduction/logs'),
    }


def recipe(args, p):
    job = JOBS[args.job]
    if args.num_envs is not None and args.job != 'dip_eval_envtest':
        raise ValueError('--num-envs 仅用于 dip_eval_envtest；正式评测保留 4096 个环境')
    values = dict((k, str(v)) for k, v in p.items())
    values['num_envs'] = str(args.num_envs or int(os.environ.get('NUM_ENVS', job.get('num_envs', 4096))))
    checkpoint = None
    if job['kind'] == 'train':
        save = Path(args.save_dir).expanduser().resolve() if args.save_dir else ROOT / job['save_dir']
        values['save_dir'] = str(save)
        models = []
        if save.exists():
            for f in save.iterdir():
                match = re.fullmatch(r'model(\d+)\.pt', f.name)
                if match and f.is_file():
                    models.append((int(match.group(1)), f))
        if args.resume:
            if not models:
                raise ValueError('续训目录没有 model*.pt: ' + str(save))
            step, checkpoint = max(models)
            optimizer = save / ('opt%09d.pt' % step)
            if not optimizer.is_file():
                raise ValueError('续训缺少优化器状态: ' + str(optimizer))
            if not (save / 'args.json').is_file():
                raise ValueError('续训缺少 args.json: ' + str(save))
            # The upstream trainer prefers the highest checkpoint in save_dir.
            # Explicitly report and pass that same file; do not select an older step.
        elif save.exists() and any(save.iterdir()):
            raise ValueError('输出目录非空，首次训练不会覆盖。续训请加 --resume，或用 --save-dir 指定新目录: ' + str(save))
    elif args.resume or args.save_dir:
        raise ValueError('--resume / --save-dir 仅用于训练')
    if 'model' in job:
        model = Path(args.model).expanduser().resolve() if args.model else ROOT / job['model']
        values['model'] = str(model)
        if not model.is_file() or not (model.parent / 'args.json').is_file():
            raise ValueError('模型或相邻 args.json 不存在: ' + str(model))
    elif args.model:
        raise ValueError('--model 仅用于 DiP 评测')
    argv = [s.format(**values) for s in job['argv']]
    if checkpoint:
        argv.extend(['--resume_checkpoint', str(checkpoint)])
    return job, values, argv, checkpoint


def options(args):
    out = []
    for key in ['save_dir', 'model', 'num_envs']:
        value = getattr(args, key)
        if value is not None:
            if key in ['save_dir', 'model']:
                value = str(Path(value).expanduser().resolve())
            out.extend(['--' + key.replace('_', '-'), str(value)])
    if args.resume:
        out.append('--resume')
    return out


def shell(argv):
    return ' '.join(shlex.quote(str(x)) for x in argv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', choices=sorted(JOBS))
    parser.add_argument('--check', action='store_true', help='只检查并打印命令，不创建文件或提交作业')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-dir')
    parser.add_argument('--model')
    parser.add_argument('--num-envs', type=int)
    args = parser.parse_args()
    if args.num_envs is not None and args.num_envs <= 0:
        parser.error('--num-envs 必须为正整数')
    p = paths()
    try:
        job, values, argv, checkpoint = recipe(args, p)
        for name in ['env', 'isaac', 'container']:
            if not p[name].exists():
                raise ValueError('缺少资源 %s: %s' % (name, p[name]))
    except ValueError as exc:
        parser.error(str(exc))
    env = os.environ.copy()
    env.update({'CLOSD_ROOT': str(ROOT), 'CLOSD_SHARED_ROOT': str(p['shared']),
                'CLOSD_ENV': str(p['env']), 'CLOSD_ISAACGYM': str(p['isaac']),
                'CLOSD_CONTAINER': str(p['container']), 'CLOSD_CACHE': str(p['cache']),
                'CLOSD_LOGS': str(p['logs'])})
    submit = ['sbatch', '--chdir=' + str(ROOT), '--export=ALL',
              '--output=' + str(p['logs'] / '%x_%j.out'),
              '--error=' + str(p['logs'] / '%x_%j.err')]
    for setting in ['account', 'partition', 'gres', 'cpus-per-task', 'time']:
        override = os.environ.get('CLOSD_SLURM_' + setting.upper().replace('-', '_'))
        if override:
            submit.append('--%s=%s' % (setting, override))
    submit += [str(ROOT / 'scripts/hpc' / (args.job + '.slurm'))] + options(args)
    # Also bind overridden resources outside the default shared root.
    mounts = sorted(set(str(p[k]) for k in ['repo', 'shared', 'env', 'isaac', 'cache', 'logs']))
    for key in ['save_dir', 'model']:
        if key in values:
            parent = Path(values[key]).parent
            while not parent.exists():
                parent = parent.parent
            if not any(str(parent) == m or str(parent).startswith(m + os.sep) for m in mounts):
                mounts.append(str(parent))
    if os.environ.get('SLURM_TMPDIR'):
        mounts.append(os.environ['SLURM_TMPDIR'])
    container = ['apptainer', 'exec', '--nv']
    for mount in mounts:
        container += ['--bind', mount + ':' + mount]
    container += [str(p['container']), 'bash', str(ROOT / 'scripts/hpc/container.sh'), args.job] + options(args)
    print('Job:', args.job, '\nPaths:', json.dumps(dict((k, str(v)) for k, v in p.items()), ensure_ascii=False))
    print('Resume checkpoint:', checkpoint or '(none)')
    print('Python command:', shell(argv))
    print('Container command:', shell(container))
    print('Submission command:', shell(submit), flush=True)
    if args.check:
        return 0
    if args.submit:
        p['logs'].mkdir(parents=True, exist_ok=True)
        return subprocess.call(submit, env=env, cwd=str(ROOT))
    if not args.inside:
        if not os.environ.get('SLURM_JOB_ID'):
            parser.error('GPU 作业请使用 submit.sh；检查请使用 --check')
        for name in ['cache', 'logs']:
            p[name].mkdir(parents=True, exist_ok=True)
        return subprocess.call(container, env=env, cwd=str(ROOT))
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('--inside 只能在 Slurm 分配中使用')
    if job.get('integration_test'):
        env['DIFFUSION_TRAINING_TEST'] = '1'
    else:
        env.pop('DIFFUSION_TRAINING_TEST', None)
    subprocess.call(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT))
    subprocess.call(['git', 'status', '--short'], cwd=str(ROOT))
    subprocess.check_call([sys.executable, '-c', 'import torch; print("Python", __import__("sys").version); print("PyTorch", torch.__version__, "CUDA", torch.version.cuda); print("GPU", torch.cuda.get_device_name(0))'])
    monitor = None
    handle = None
    try:
        if args.job == 'dip_eval_envtest':
            log = p['logs'] / ('gpu_%s_%senv.csv' % (os.environ['SLURM_JOB_ID'], values['num_envs']))
            handle = log.open('w')
            monitor = subprocess.Popen(['nvidia-smi', '-i', os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0], '--query-gpu=timestamp,memory.used,utilization.gpu', '--format=csv,noheader,nounits', '-l', '1'], stdout=handle)
        result = subprocess.call(argv, env=env, cwd=str(ROOT))
    finally:
        if monitor:
            monitor.terminate()
            monitor.wait()
        if handle:
            handle.close()
            samples = []
            for line in log.read_text().splitlines():
                try:
                    samples.append(float(line.split(',')[1]))
                except (IndexError, ValueError):
                    pass
            print('Peak GPU memory MiB:', max(samples) if samples else 'unavailable')
    print('Job finished with exit code:', result, flush=True)
    return result


if __name__ == '__main__':
    sys.exit(main())
