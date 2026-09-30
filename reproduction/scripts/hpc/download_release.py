#!/usr/bin/env python3
"""Download the two evaluated checkpoints and verify against committed SHA-256 hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=root / 'reproduction/runs/DiP_multi-target_repro_seed10')
    parser.add_argument('--check', action='store_true', help='verify local files only')
    args = parser.parse_args()
    manifest = json.loads((root / 'reproduction/experiments/release.json').read_text())
    for asset in manifest['assets']:
        target = args.destination / asset['name']
        if target.exists():
            if sha256(target) != asset['sha256']:
                parser.error('Existing file differs; refusing to overwrite: ' + str(target))
            print('Verified:', target)
            continue
        if args.check:
            parser.error('Missing: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://github.com/{repository}/releases/download/{tag}/{name}'.format(name=asset['name'], **manifest)
        handle = tempfile.NamedTemporaryFile(dir=str(target.parent), prefix='.' + target.name, delete=False)
        temporary = Path(handle.name)
        try:
            with handle, urlopen(url, timeout=120) as response:
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            if sha256(temporary) != asset['sha256']:
                raise RuntimeError('SHA-256 mismatch: ' + asset['name'])
            # Fail safely if another downloader produced the file meanwhile.
            os.link(str(temporary), str(target))
            print('Downloaded and verified:', target)
        finally:
            temporary.unlink()


if __name__ == '__main__':
    main()
