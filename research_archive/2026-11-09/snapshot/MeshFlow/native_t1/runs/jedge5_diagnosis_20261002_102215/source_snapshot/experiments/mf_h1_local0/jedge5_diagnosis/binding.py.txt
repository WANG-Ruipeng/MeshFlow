"""Same-noise condition binding from existing endpoint scores only."""
from pathlib import Path
import argparse,csv
import numpy as np
from .common import read_json,write_json

def run(out):
 out=Path(out);groups={};sources=[]
 phases={'A_B','A_J','B_B','B_J'}|{'D_'+x for x in ('G0','R_B_raw250','R_J_raw250','R_B_EMA250','R_J_EMA250','R_B_EMA500','R_J_EMA500')}
 for file in sorted((out/'scores').glob('*/results.json')):
  if file.parent.name not in phases:continue
  r=read_json(file)
  if r['status']!='PASS':raise ValueError('Incomplete scoring '+str(file))
  sources.append(str(file.relative_to(out)))
  for row in r['rows']:
   identity=(r['phase'],row['model'],row['patch'],row['seed'],row.get('precision'),row.get('nfe'),row.get('start_step'),row.get('source_model'),row.get('continuation_model'))
   group=groups.setdefault(identity,{})
   if row['parent'] in group:raise ValueError('Duplicate binding row')
   group[row['parent']]=row
 rows=[]
 for identity,pair in groups.items():
  if set(pair)!={0,1}:raise ValueError('Missing matched-context output')
  a,b=pair[0]['cross_remainder'],pair[1]['cross_remainder']
  def value(seq,n):return seq[n]['symmetric_rms_bbox_pct']
  e00,e01,e10,e11=value(a,0),value(b,0),value(a,1),value(b,1)
  d0,d1=e01-e00,e10-e11
  rows.append(dict(zip(('phase','model','patch','seed','precision','nfe','start_step','source_model','continuation_model'),identity),E00=e00,E01=e01,E10=e10,E11=e11,Delta0=d0,Delta1=d1,B=(d0+d1)*.5,both_positive=d0>0 and d1>0))
 folder=out/'analysis';folder.mkdir(exist_ok=True)
 path=folder/'binding.csv'
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 scopes={}
 for row in rows:
  key='|'.join(str(row[k]) for k in ('phase','model','precision','nfe','start_step'))
  scopes.setdefault(key,[]).append(row)
 summary={k:dict(cases=len(v),both_positive=sum(x['both_positive'] for x in v),mean_binding=float(np.mean([x['B'] for x in v])),median_binding=float(np.median([x['B'] for x in v])),minimum_binding=float(min(x['B'] for x in v))) for k,v in scopes.items()}
 write_json(folder/'binding.json',dict(status='PASS',rows=rows,summary=summary,sources=sources,formula='E_ij = GT remainder i versus output conditioned on C_j; Delta0=E01-E00; Delta1=E10-E11; B=(Delta0+Delta1)/2',new_sampling=0,new_scoring=0,OT=0))
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)

