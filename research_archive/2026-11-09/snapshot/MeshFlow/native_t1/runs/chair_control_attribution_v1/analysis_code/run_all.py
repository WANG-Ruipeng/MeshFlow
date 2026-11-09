"""Recoverable CPU-only entry. Run with the existing meshflow Python environment."""
from core import *
import argparse,fcntl
from registration import register
from scalars import width,cross
from coverage_teacher import coverage,teacher

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--through',default='all',choices=['register','scalar','geometry','figures','all']);args=ap.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 # Local attribution controller lock only; never touches the GPU or old controller locks.
 with (OUT/'attribution_cpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  event('entry',argv=sys.argv,model_loads=0,forwards=0,backwards=0,OT=0,new_Gaussians=0,CUDA_initializations=0)
  stage('01_registration',register)
  # Verify immutable source identity on every reentry, even when completed stages are skipped.
  for r in js(OUT/'source_manifest.json')['sources']:
   if digest(r['path'])!=r['sha256']:raise RuntimeError('Source changed: '+r['path'])
  if args.through=='register':return
  for n,fn in [('02_width',width),('03_cross',cross),('04_coverage',coverage),('05_teacher',teacher)]:
   print(n,flush=True);print(stage(n,fn),flush=True)
  if args.through=='scalar':return
  from geometry import fixtures,pullback
  stage('06_cpu_fixtures',fixtures)
  stage('07_pullback',pullback)
  if args.through=='geometry':return
  from figures import make_figures
  stage('08_figures',make_figures)
  if args.through=='figures':return
  from delivery import deliver
  deliver()

if __name__=='__main__':main()
