"""Recover completed bounded stages without silently replaying scientific work."""
from __future__ import annotations
import argparse,subprocess,sys
from .common import *

def call(mod,*args):
 result=subprocess.run([sys.executable,'-B','-m','r2.pilots.bench_backend_relative_deformation_v1.'+mod,*args],cwd=ROOT)
 if result.returncode:raise RuntimeError('Stage failed, preserve evidence and inspect before explicit recovery: '+mod)

def freeze():
 path=OUT/'PRETRAIN_READY.json'
 if path.exists():r=read(path);verify(r['files']);return r
 required=['audits/runtime_cpu_contract.json','audits/cpu_math_checks.json','audits/pretrain_independent.json','evaluation/ENGINEERING_PREP.json']
 audits={n:read(OUT/n) for n in required}
 for name,r in audits.items():
  expected='CPU_ENGINEERING_PREP_PASS' if name=='evaluation/ENGINEERING_PREP.json' else 'PASS'
  if r['status']!=expected:raise RuntimeError('Pretraining audit not passed: '+name)
 cached=read(OUT/'data/relative_cache_manifest.json');verify(cached['protected_files'])
 assert cached['train_summary']['parents']==16 and cached['train_summary']['trusted_faces']==2328
 assert cached['train_summary']['eligible_faces']==2328 and cached['train_summary']['clipped_faces']==128 and cached['train_summary']['excluded_rank_faces']==0
 sources=[OUT/'PROTOCOL.md',OUT/'loss_config.json',OUT/'data/relative_cache_manifest.json']+[OUT/x for x in required]
 sources += [ROOT/'r2/pilots/bench_backend_relative_deformation_v1'/x for x in ('common.py','runtime.py','relative_loss.py','prepare.py','evaluate.py','render.py','render_stream.py')]
 files={str(p):sha(p) for p in sources};files.update({p['path']:p['sha256'] for p in cached['parents']})
 verify(files)
 r=dict(status='PASS',manifest_path=cached['source_manifest_path'],manifest_sha256=cached['source_manifest_sha256'],
    optimizer_updates=500,training_logical_forwards=2000,backwards=2000,evaluation_candidate_forwards=60,
    generator_forwards=0,new_noise=0,files=files)
 write(path,r);return r

def guard(phase,training=False):
 r=resource_snapshot(phase)
 text=r['gpu']['stdout'].strip().splitlines()
 if len(text)!=1:raise RuntimeError('Cannot safely identify GPU')
 v=[x.strip() for x in text[0].split(',')];total,free=float(v[1]),float(v[3])
 head=max(2048,.1*total);estimate=(9 if training else 2.5)*1024
 if free-head<1.2*estimate:raise RuntimeError('Resource headroom requires waiting; no worker launched')
 if r['ram']['available']<8*(1<<30):raise RuntimeError('Insufficient observed CPU RAM; wait')
 return r

def main():
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','train','resume','verify']);a=parser.parse_args()
 final=OUT/'COMPLETE.json'
 if final.exists():r=read(final);verify(r['files']);print('COMPLETE_VERIFIED',len(r['files']));return
 if a.command=='verify':raise RuntimeError('Not sealed yet; do not restart computation')
 freeze()
 if a.command=='freeze':print('PRETRAIN_FROZEN');return
 if not (OUT/'training/complete.json').exists():guard('train_dispatch',True);call('runtime','train')
 if a.command=='train':return
 for arm in ('CANDIDATE_RAW','CANDIDATE_SOURCE'):
  if not (OUT/'evaluation'/arm/'complete.json').exists():guard(arm+'_dispatch');call('runtime','evaluate','--arm',arm)
 if not (OUT/'scores/COMPLETE.json').exists():
  r=resource_snapshot('cpu_score_dispatch')
  if r['ram']['available']<4*(1<<30) or r['cpu_count']<4:raise RuntimeError('CPU scoring capacity insufficient; wait')
  call('evaluate','--workers','3')
 if not (OUT/'source_analysis/COMPLETE.json').exists():call('source_analysis')
 if not (OUT/'gallery/COMPLETE.json').exists():call('render')
 if not (OUT/'analysis/COMPLETE.json').exists():call('analysis')
 print('COMPUTE_COMPLETE; actual review, independent audit, report and seal remain. No additional models authorized.')
if __name__=='__main__':main()
