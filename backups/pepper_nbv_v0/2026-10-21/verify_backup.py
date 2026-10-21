"""Read-only integrity verification of the backed-up files."""
from pathlib import Path
import argparse
import hashlib
import json

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'BACKUP_MANIFEST.json').read_text(encoding='utf-8'))
workspace = (args.workspace or root.parents[2]).resolve()
failures = []
for record in manifest['files']:
    path = (workspace / record['path']).resolve()
    if not path.is_relative_to(workspace) or not path.is_file():
        failures.append(record['path'] + ': missing or unsafe path')
        continue
    if path.stat().st_size != record['bytes']:
        failures.append(record['path'] + ': size mismatch')
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != record['sha256']:
        failures.append(record['path'] + ': SHA-256 mismatch')
if failures:
    print('\n'.join(failures))
    raise SystemExit(1)
print(f"Verified {len(manifest['files'])} files ({manifest['total_source_bytes']} bytes)")
