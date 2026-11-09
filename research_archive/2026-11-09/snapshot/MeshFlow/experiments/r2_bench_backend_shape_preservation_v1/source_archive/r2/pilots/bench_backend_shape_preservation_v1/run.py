"""Finite resume entry: checked completed stages are reused, never replayed."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
from .common import *

def call(mod,*args):
 command=[sys.executable,'-B','-m','r2.pilots.bench_backend_shape_preservation_v1.'+mod,*args]
 r=subprocess.run(command,cwd=ROOT)
 if r.returncode:raise RuntimeError('Stage failed; prior evidence retained: '+repr(command))

def verify_runtime_revision(ready):
 runtime=ROOT/'r2/pilots/bench_backend_shape_preservation_v1/runtime.py'
 current=sha(runtime)
 if current==ready['runtime_sha256']:return 'ORIGINAL_FROZEN_RUNTIME'
 # One reviewed implementation-only amendment. This is not a generic override
 # and does not grant a new training attempt, rollback, or additional update.
 a=read(OUT/'engineering_amendment.json')
 original='fd011dbb01070c9c1bfebb06727d746fd17dc0f975b834d6a90956230227ccd7'
 revised='d034d24a7125af6ff8e65db5e7d8c1582fa756d2ac6b79a122e706247b7c81f9'
 assert a['schema']=='bench_backend_step100_zero_replay_boundary_recovery_v1'
 assert a['status']=='CPU_VERIFIED_READY_FOR_EXPLICIT_RESUME'
 assert ready['runtime_sha256']==a['old_code_sha256']==original
 assert current==a['new_code_sha256']==revised
 assert sha(OUT/'code_before_boundary_fix.py')==original
 assert a['manifest_sha256']==ready['manifest_sha256']==sha(OUT/'data/manifest.json')
 assert sha(OUT/'training/failure.json')==a['preserved_failure_sha256']
 assert sha(OUT/'training/attempt.json')==a['original_attempt_sha256']
 assert a['completed_counts']==dict(forwards=400,backwards=400,updates=100)
 assert a['remaining_counts']==dict(forwards=1600,backwards=1600,updates=400)
 assert a['next_update']==101 and a['terminal_update']==500 and a['settings_changed'] is False
 assert all(a[k]==0 for k in ('replayed_updates','replayed_forwards','replayed_backwards'))
 saved=read(OUT/'training/checkpoints/step0100.json')
 assert saved==a['checkpoint'] and sha(saved['path'])==saved['sha256']
 return 'EXACT_REVIEWED_ZERO_REPLAY_BOUNDARY_AMENDMENT'

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['resume','verify']);a=p.parse_args()
 if (OUT/'COMPLETE.json').exists():
  r=read(OUT/'COMPLETE.json');verify(r['files']);print('COMPLETE_VERIFIED',len(r['files']));return
 if a.command=='verify':raise RuntimeError('No final COMPLETE; experiment still in progress')
 ready=read(OUT/'PRETRAIN_READY.json')
 assert ready['status']=='PASS' and sha(OUT/'data/manifest.json')==ready['manifest_sha256']
 print(verify_runtime_revision(ready))
 if not (OUT/'training/complete.json').exists():call('runtime','train')
 for arm in ('CANDIDATE_RAW','OFFICIAL_SOURCE','CANDIDATE_SOURCE'):
  if not (OUT/'evaluation'/arm/'complete.json').exists():call('runtime','evaluate','--arm',arm)
 if not (OUT/'scores/COMPLETE.json').exists():call('evaluate','--workers','3')
 if not (OUT/'source_analysis/COMPLETE.json').exists():call('source_analysis')
 if not (OUT/'gallery/COMPLETE.json').exists():call('render')
 if not (OUT/'analysis/COMPLETE.json').exists():call('analysis')
 print('COMPUTE_COMPLETE; actual image review, independent audit and report seal remain manual. No further models authorized.')
if __name__=='__main__':main()
