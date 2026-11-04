"""Restore versioned result archives, checking the release and every member hash.

Uses only the standard library. Source measurements are acquired separately.
Existing files with different contents are never overwritten.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def restore(archive, record, destination):
    if digest(archive) != record['sha256']:
        raise ValueError('Release checksum mismatch; nothing extracted.')
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        manifests = [n for n in names if n.endswith('/RELEASE_MANIFEST.json')]
        if len(manifests) != 1:
            raise ValueError('Expected one release manifest.')
        prefix = manifests[0].removesuffix('RELEASE_MANIFEST.json')
        manifest = json.loads(z.read(manifests[0]))
        if manifest['version'] != record['version']:
            raise ValueError('Artifact version mismatch.')
        selected = []
        for rel, info in manifest['files'].items():
            path = PurePosixPath(rel)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive path.')
            if not rel.startswith('results/'):
                continue  # The checkout supplies code, paper and dataset metadata.
            target = destination / rel
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Destination escapes checkout.')
            item = z.getinfo(prefix + rel)
            if item.file_size != info['bytes']:
                raise ValueError('Member size mismatch: ' + rel)
            raw = z.read(item)
            if hashlib.sha256(raw).hexdigest() != info['sha256']:
                raise ValueError('Member checksum mismatch: ' + rel)
            if target.exists() and digest(target) != info['sha256']:
                raise ValueError('Existing file differs; preserve it in a separate checkout: ' + rel)
            selected.append((rel, info))
        if not selected:
            raise ValueError('No result files in release.')
        # Preflight all members/conflicts before writing any result.
        for rel, info in selected:
            target = destination / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(prefix + rel) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
        return dict(version=manifest['version'], files=len(selected), status='passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', choices=['teaching', 'full'], default='teaching')
    parser.add_argument('--archive', type=Path, help='Use a locally obtained release ZIP.')
    parser.add_argument('--destination', type=Path, default=ROOT)
    args = parser.parse_args()
    index = json.loads((ROOT / 'release_assets.json').read_text())
    record = dict(index['bundles'][args.bundle], version=index['version'])
    with tempfile.TemporaryDirectory(prefix='lorawan-artifact-') as temporary:
        archive = args.archive
        if archive is None:
            if not record.get('url'):
                raise SystemExit('Public release is not yet assigned. Supply --archive /path/to/' + record['filename'])
            archive = Path(temporary) / record['filename']
            request = urllib.request.Request(record['url'], headers={'User-Agent': 'LoRaWAN-Tutorial'})
            with urllib.request.urlopen(request, timeout=120) as src, archive.open('wb') as dst:
                shutil.copyfileobj(src, dst)
        report = restore(archive, record, args.destination)
    report.update(bundle=args.bundle, acquisition='local archive' if args.archive else 'public release')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
