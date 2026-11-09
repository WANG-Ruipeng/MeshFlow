"""Portable pure NumPy recheck of a SMALL review archive. No original data/model required."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source_code'))
from r2.tools.full_boundary_attribution_v1.geometry import analyze

def main():
 p=argparse.ArgumentParser();p.add_argument('--cell',default='all');args=p.parse_args()
 reg=json.loads((ROOT/'registration.json').read_text(encoding='utf-8'))
 cells=[c for c in reg['cells'] if args.cell=='all' or args.cell==c['cell_id']]
 if not cells:raise ValueError('Unknown cell')
 for c in cells:
  with np.load(ROOT/'inputs/cells'/c['cell_id']/'geometry.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  e=json.loads((ROOT/'evidence'/(c['cell_id']+'.json')).read_text());a.update(L=reg['L'],pairs=e['pairs'],graph_identity=c['graph_identity'])
  out=analyze(a,reg['geometry_config'])
  with (ROOT/'results'/c['cell_id']/'samples.csv').open(encoding='utf-8-sig',newline='') as f:old=list(csv.DictReader(f))
  assert len(old)==len(out['samples'])
  for before,after in zip(old,out['samples']):
   for domain in ['surface','parallel_surface','edge','open_edge','vertex']:
    key=domain+'_distance';expected=None if before[key]=='' else float(before[key]);actual=after[key]
    if expected is None:assert actual is None
    else:assert abs(expected-actual)<1e-10,(c['cell_id'],key,expected,actual)
    assert int(before[domain+'_id'])==after[domain+'_id']
  print('PASS',c['cell_id'],len(old),'queries',flush=True)
 print('Portable recheck complete; no outputs modified, no neural modules.')
if __name__=='__main__':main()