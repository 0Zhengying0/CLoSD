#!/usr/bin/env python3
"""Restore official HF assets at the revisions recorded in this reproduction."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='check cache refs and official links without downloads')
    args = parser.parse_args()
    root = Path(os.environ.get('CLOSD_ROOT', str(Path(__file__).resolve().parents[3]))).resolve()
    hub = Path(os.environ.get('HF_HOME', str(root / '.local/cache/huggingface'))) / 'hub'
    revisions = json.loads((root / 'reproduction/environment/official-resources.json').read_text())
    for model, revision in revisions.items():
        directory = hub / ('models--' + model.replace('/', '--'))
        snapshot = directory / 'snapshots' / revision
        if not args.check:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model, revision=revision, cache_dir=str(hub))
            (directory / 'refs').mkdir(parents=True, exist_ok=True)
            (directory / 'refs/main').write_text(revision)
        if not snapshot.is_dir() or (directory / 'refs/main').read_text().strip() != revision:
            parser.error('Missing or unpinned cache: ' + model)
        print('Pinned:', model, revision)
    snapshot = hub / 'models--guytevet--CLoSD/snapshots' / revisions['guytevet/CLoSD']
    for source, destination in [('checkpoints/dip', 'closd/diffusion_planner/save'), ('checkpoints/closd', 'output/CLoSD'), ('evaluation', 'closd/diffusion_planner/saved_motions')]:
        for child in (snapshot / source).iterdir():
            if not child.is_dir():
                continue
            link = root / destination / child.name
            if not args.check and not link.exists() and not link.is_symlink():
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(os.path.relpath(str(child), str(link.parent)))
            if not link.exists() or link.resolve() != child.resolve():
                parser.error('Official asset path missing or points elsewhere: ' + str(link))
            print('Verified link:', link.relative_to(root))
    for p in hub.rglob('*'):
        if p.is_symlink() and not p.exists():
            parser.error('Broken cache link: ' + str(p))


if __name__ == '__main__':
    main()
