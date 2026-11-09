"""Recover/check this CPU-only data package. Never dispatches models or training."""
from pathlib import Path
import argparse,subprocess,sys,time,json
from prepare import DEFAULT_OUT,read,sha,event

def child(name,args=()):
 subprocess.run([sys.executable,'-B',str(Path(__file__).parent/name),*map(str,args)],check=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,default=DEFAULT_OUT)
 p.add_argument('--stage',choices=['resume','check'],default='check')
 a=p.parse_args();out=a.out;t0=time.perf_counter()
 # Decisions are reviewed inputs. Never infer them from an absent implementation or missing file.
 for path in ('parent_inventory.json','regions.json','usability_review.json'):
  if not (out/path).exists():raise RuntimeError('BLOCKED: missing reviewed input '+path)
 if a.stage=='resume':
  # Construction is idempotent at existing NPZ byte identity; differing existing JSON is refused.
  if not (out/'conditions.json').exists():child('construct.py',['--out',out])
  if not (out/'quality/condition_summary.json').exists():child('quality.py',['--conditions',out/'conditions.json','--out',out/'quality'])
  if not (out/'active_conditions.json').exists():
   raise RuntimeError('BLOCKED: active selection must be restored from its reviewed evidence; no silent reselection')
  # The final verifier/report files are preserved snapshots. If absent, run the explicit stage script,
  # not an old R2 training entrypoint. README lists each source stage.
 required=['conditions.json','active_conditions.json','dataset_handoff_active.json','FINAL_SUMMARY.json',
  'validation_export.json','validation_active.json','DATA_PREPARATION_REPORT.md','gallery/index.html']
 for name in required:
  if not (out/name).exists():raise RuntimeError('BLOCKED: missing stage artifact '+name+'; see README stage entry')
 active=read(out/'active_conditions.json');all_rows=read(out/'conditions.json')['rows'];checks=0
 for r in read(out/'parent_inventory.json')['rows']:
  assert sha(r['npz'])==r['sha256'];checks+=1
 for r in all_rows:
  assert sha(r['npz'])==r['sha256'];checks+=1
  image=out/'gallery/conditions'/(r['condition_id']+'.png');sidecar=read(image.with_suffix('.png.json'))
  assert sha(image)==sidecar['image_sha256'];checks+=1
 original={r['condition_id']:r for r in all_rows}
 for r in active['rows']:
  assert r==original[r['condition_id']];checks+=1
 assert sum(r['split']=='train' for r in active['rows'])==117
 assert sum(r['split']=='dev' for r in active['rows'])==6
 assert sha(out/'conditions.json')==active['source_all_exported_sha256']
 assert sha(out/'usability_review.json')==active['usability_review_sha256']
 receipt=dict(stage=a.stage,status='PASS',hash_and_row_checks=checks,
  active_train=117,active_dev=6,source_candidates=129,wall_seconds=time.perf_counter()-t0,
  model_calls=0,training_updates=0,OT_calls=0,new_Gaussian_draws=0)
 event(out,'RESUMABLE_ENTRY_CHECK',**receipt)
 print(json.dumps(receipt))
if __name__=='__main__':main()
