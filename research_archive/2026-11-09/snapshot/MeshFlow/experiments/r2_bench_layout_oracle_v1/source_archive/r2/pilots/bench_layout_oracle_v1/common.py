"""Independent identities; reuses pure I/O helpers, never old output globals."""
from pathlib import Path
import hashlib, json, os, time
from r2.pilots.bench_three_mechanisms_v1.common import (
    resolve, read, sha, arrays, npz, verify, csvwrite, write, event, resources)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_layout_oracle_v1'
EXPERIMENT = 'R2_BENCH_LAYOUT_ORACLE_V1'
OLD = ROOT / 'experiments/r2_bench_four_routes_v1/exposure'
PRIOR = ROOT / 'experiments/r2_bench_multilod_training_v1'
SOURCE = OLD / 'training/M_PLUS3000/checkpoint_step3000.pt'
SOURCE_INFERENCE = OLD / 'training/M_PLUS3000/generator_step3000.pt'
BASE_STEP, NATIVE_BASE = 25500, 13500
UPDATES = {'PLAIN':1000, 'LAYOUT':1000}
LABELS = ('EXPOSURE','PLAIN1000','LAYOUT1000','LAYOUT_NULL')
NEW_LABELS = LABELS[1:]
PANEL_PREFIXES = ('a8e987','27eb79','bcb4f5','459f90','f495bd','df6fd3')
REGISTRATION = OUT / 'registration.json'
L = 9.208138264587333

def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,(EXPERIMENT,)+parts)).encode()).digest()[:4], 'little')

def register_resource(phase):
    r = resources()
    event(OUT/'resources/snapshots.jsonl', dict(phase=phase, **r))
    return r
