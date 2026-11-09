"""Read-only CPU helpers for the layout experiment closeout; no model imports."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_layout_oracle_v1'
OLD = ROOT / 'experiments/r2_bench_four_routes_v1/exposure'
LABELS = ('EXPOSURE', 'PLAIN1000', 'LAYOUT1000', 'LAYOUT_NULL')
NEW = LABELS[1:]
STAGES = ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')
L = 9.208138264587333


def resolve(path):
    p = str(path).replace('\\', '/')
    if os.name == 'nt' and p.startswith('/mnt/'):
        p = p[5].upper() + ':/' + p[7:]
    elif os.name != 'nt' and len(p) > 2 and p[1:3] == ':/':
        p = '/mnt/' + p[0].lower() + '/' + p[3:]
    result = Path(p)
    return result if result.is_absolute() else ROOT / result


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def jsonl(path):
    return [json.loads(line) for line in resolve(path).read_text('utf-8').splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with resolve(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, obj):
    path = resolve(path)
    obj = json.loads(json.dumps(obj, ensure_ascii=False, allow_nan=False))
    if path.exists():
        if read(path) != obj:
            raise ValueError('Existing receipt differs; select a new explicit output: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def csvwrite(path, rows):
    if not rows:
        raise ValueError('No rows: ' + str(path))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def root_for(label):
    return OLD if label == 'EXPOSURE' else OUT


def output_dir(label, request, kind):
    return root_for(label) / 'evaluation' / label / request / kind


def score_dir(label, request, stage):
    if stage == 'RAW':
        return root_for(label) / 'evaluation/scores' / label / request / stage
    return root_for(label) / 'cpu' / label / request / stage / 'scores'


def metric_identity(row):
    return row['label'], row['request_id'], row['stage']


class FileAudit:
    def __init__(self):
        self.files = {}
        self._stats = {}

    def hash(self, path):
        path = resolve(path)
        stat = path.stat()
        identity = stat.st_size, stat.st_mtime_ns
        if str(path) in self.files:
            if self._stats[str(path)] != identity:
                raise ValueError('Input changed during CPU audit: ' + str(path))
            return self.files[str(path)]
        result = sha(path)
        if (path.stat().st_size, path.stat().st_mtime_ns) != identity:
            raise ValueError('Input changed while hashing: ' + str(path))
        self.files[str(path)] = result
        self._stats[str(path)] = identity
        return result

    def verify(self, book):
        for path, expected in book.items():
            if self.hash(path) != expected:
                raise ValueError('Protected hash mismatch: ' + str(path))

    def read(self, path):
        self.hash(path)
        return read(path)

    def jsonl(self, path):
        self.hash(path)
        return jsonl(path)

