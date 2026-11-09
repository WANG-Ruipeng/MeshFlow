"""CPU-only teacher signal and frozen exposure weighting; no new labels."""
import collections,json
import numpy as np
from . import common as C

def main():
 d=C.read(C.OUT/'manifest/data.json');rows=[]
 for split in ('train_inputs','internal_val','dev_real','dev_synthetic'):
  for r in d[split]:
   with np.load(r['teacher_cache_path'],allow_pickle=False) as z:
    t=np.subtract(z['proposal_T_d'],z['input_d'],dtype=np.float32)[int(r['K']):].astype(np.float64)
   den=t.size;energy=float(np.sum(t*t));length=np.linalg.norm(t,axis=-1).reshape(-1)
   rows.append(dict(sample_id=r['sample_id'],split=split,domain=r['domain'],variant=r['variant'],label=r.get('label'),uid=r['uid'],condition_id=r['condition_id'],N=r['N'],K=r['K'],free_scalar_count=den,teacher_energy_sum=energy,B=energy/den,signal_status='LOW_SIGNAL' if energy/den<=1e-14 else 'OK',teacher_corner_norm_p95_d=float(np.quantile(length,.95)),teacher_corner_norm_max_d=float(length.max())))
 C.csvwrite(C.OUT/'teacher_cache/signal_per_input.csv',rows)
 group=collections.defaultdict(list)
 for r in rows:group[(r['split'],r['domain'])].append(r)
 summary=[]
 for (split,domain),g in sorted(group.items()):
  summary.append(dict(split=split,domain=domain,inputs=len(g),pooled_B=sum(r['teacher_energy_sum'] for r in g)/sum(r['free_scalar_count'] for r in g),mean_B=float(np.mean([r['B'] for r in g])),median_B=float(np.median([r['B'] for r in g])),LOW_SIGNAL=sum(r['signal_status']=='LOW_SIGNAL' for r in g)))
 C.csvwrite(C.OUT/'teacher_cache/signal_summary.csv',summary)
 byid={r['sample_id']:r for r in rows};exposure=collections.defaultdict(lambda:dict(exposures=0,scalars=0,teacher_energy=0.))
 for batch in C.read(d['train_stream_path'])['batches']:
  for sid in batch['sample_ids']:
   r=byid[sid];v=exposure[r['domain']];v['exposures']+=1;v['scalars']+=r['free_scalar_count'];v['teacher_energy']+=r['teacher_energy_sum']
 total=sum(v['teacher_energy'] for v in exposure.values());ss=sum(v['scalars'] for v in exposure.values())
 for v in exposure.values():v.update(scalar_fraction=v['scalars']/ss,identity_loss_energy_fraction=v['teacher_energy']/total,pooled_B=v['teacher_energy']/v['scalars'])
 C.write(C.OUT/'teacher_cache/signal_audit.json',dict(status='PASS',rows=len(rows),summary=summary,frozen_exposure=dict(exposure),models_loaded=0,new_teacher_queries=0,source_sha256=C.sha(__file__),interpretation='Identity teacher-error energy under frozen exposures; describes loss scale, not geometry importance or a change of weighting.'))
 print(json.dumps(dict(summary=summary,exposure=dict(exposure)),indent=2))
if __name__=='__main__':main()