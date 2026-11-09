"""Bounded resume entry. Existing receipts are verified; no silent model replay."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_geo_only_acceptance_v1'
PKG='r2.pilots.bench_geo_only_acceptance_v1.'
def call(name,*args):subprocess.run([sys.executable,'-B','-m',PKG+name,*map(str,args)],cwd=ROOT,check=True)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def verify():
    record=read(OUT/'COMPLETE.json');book=read(OUT/'artifact_hashes.json')
    assert record['artifact_manifest_sha256']==hashlib.sha256((OUT/'artifact_hashes.json').read_bytes()).hexdigest()
    for rel,digest in book['files'].items():
        path=ROOT/rel
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,rel
    print(json.dumps(dict(status='VERIFIED_COMPLETE',artifacts=len(book['files']),experiment_status=record['experiment_status'],new_calls=0)))
def resume():
    if (OUT/'COMPLETE.json').exists():return verify()
    if not (OUT/'core_complete.json').exists():
        call('resource_check');call('core','run')
    if not (OUT/'scores/COMPLETE.json').exists():
        call('resource_check');call('evaluate','--manifest',OUT/'manifest.json','--workers','3')
    if not (OUT/'gallery/COMPLETE.json').exists():call('render','--manifest',OUT/'manifest.json')
    needed=[OUT/'scores/geometry_gate_audit.json',*[OUT/'visual_review'/n for n in ('review_root.json','review_assets.json','review_reference.json')]]
    if not all(p.exists() for p in needed):
        raise RuntimeError('Actual visual observations and independent geometry gate audit must be completed; no invented ratings')
    if not (OUT/'visual_review/COMPLETE.json').exists():call('review_collect')
    gate=read(OUT/'acceptance_gate.json')
    if gate['run_source_identity']:
        if not (OUT/'identity/COMPLETE.json').exists():call('identity_prepare','run')
        if not (OUT/'identity/gallery/COMPLETE.json').exists():call('identity_render')
    print('Budgeted compute complete. Finish actual identity image review/report, then run closeout; no automatic extra training or sampling.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','resume'],default='verify',nargs='?')
    a=p.parse_args();verify() if a.action=='verify' else resume()
