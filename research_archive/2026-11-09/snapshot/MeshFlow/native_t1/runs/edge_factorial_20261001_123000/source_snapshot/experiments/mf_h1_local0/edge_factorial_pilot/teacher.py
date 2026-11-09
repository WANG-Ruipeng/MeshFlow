"""Sixteen new Edge-only forwards; three old endpoint diagnostics reused read-only."""
from .common import *
from .checkpoint import load_endpoint
from native_t1.training import masked_fm_loss
from native_t1.losses import prepare_targets,evaluate
from .legacy_relation_losses import prepare_relations,relation_loss

def evaluate_teacher(r,out=OUT):
 if 'teacher' in r or r['training']['status']!='PASS':raise RuntimeError('Teacher stage requires endpoint and has no automatic retry')
 source_guard(r);tick=time.perf_counter();old=json.loads((PILOT/'results.json').read_text());samples=load_samples()
 bank=TrainingGeometryTargets(archive());labels=[bank.sample(s) for s in samples]
 targets=[prepare_targets(lab) for lab in labels]
 l3=[prepare_relations(lab['free_gt'],lab['known'],lab['free_ids'],lab['known_ids'],lab['bbox_L']) for lab in labels]
 r['teacher']=dict(status='RUNNING',models={},new_forward_cap=16,times=[.5,.75],sample_archive=record(TEACHER),
  normalization='FM per-sample free-coordinate MSE; each geometry original per-sample formula, no gate/ramp/lambda')
 for name in ('L0','L5','JEdge5'):
  original=old['teacher']['models'][name]
  comparable=(original['status']=='PASS' and original['load']['state_sha256']==STATES[name]
   and old['teacher']['sample_archive']['sha256']==TEACHER_SHA and len(original['rows'])==16)
  for i,row in enumerate(original['rows']):
   comparable &= row['actual_xt_sha256']==old['calibration']['states'][i]['actual_xt_sha256']
  r['teacher']['models'][name]=dict(original,reused=True,source=record(PILOT/'results.json')) if comparable else dict(status='NOT_COMPARABLE',reused=False)
 model=None;c=r['counts']['teacher'];entry=dict(status='RUNNING',rows=[]);r['teacher']['models']['Edge_only']=entry
 try:
  model,entry['load']=load_endpoint(r,'Edge_only')
  with torch.no_grad():
   for i,sample in enumerate(samples):
    for t in (.5,.75):
     idx=2*i+int(t==.75);b=native_collate([sample],'cuda',t_override=t);expected=old['calibration']['states'][idx]
     if array_hash(b['xt'].cpu().numpy())!=expected['actual_xt_sha256'] or labels[i]['mapping_sha256']!=expected['source_mapping_sha256']:raise RuntimeError('Teacher state/mapping changed')
     if c.get('forward_attempts',0)>=16:raise RuntimeError('Teacher16 cap')
     v,runtime=forward(model,b,c);scores={'FM':float(masked_fm_loss(v,b['u'],b['valid_mask'],b['known_mask']))}
     pred=endpoint(v,b);diag={}
     with torch.autocast(device_type='cuda',enabled=False):
      for term in ('L3','L5','D_edge'):
       result=relation_loss('L3',pred,l3[i]) if term=='L3' else evaluate(term,pred,targets[i][term])
       inc(c,term+'_loss_calls');scores[term]=float(result['loss'])
       diag[term]=dict(applicable=bool(result['applicable']),counts=plain(result['counts']),diagnostics=plain(result.get('diagnostics',{})))
     if not all(np.isfinite(x) for x in scores.values()):raise FloatingPointError('Nonfinite teacher metric')
     entry['rows'].append(dict(state_index=idx,sample_index=i,parent=int(sample['parent_index']),K=int(sample['K']),t=t,
      **scores,diagnostics=diag,actual_xt_sha256=expected['actual_xt_sha256'],source_mapping_sha256=labels[i]['mapping_sha256'],runtime=runtime))
  if state_sha256(model)!=entry['load']['state_sha256']:raise RuntimeError('Teacher mutated model')
  entry.update(status='PASS',reused=False,means={term:float(np.mean([x[term] for x in entry['rows']])) for term in ('FM','L3','L5','D_edge')})
  r['teacher']['status']='PASS' if all(x['status']=='PASS' for x in r['teacher']['models'].values()) else 'PASS_WITH_NOT_COMPARABLE'
  event('edge_teacher_complete',states=16,means=entry['means'])
 finally:
  model=None;release();r['teacher']['seconds']=time.perf_counter()-tick;write(r,out)
