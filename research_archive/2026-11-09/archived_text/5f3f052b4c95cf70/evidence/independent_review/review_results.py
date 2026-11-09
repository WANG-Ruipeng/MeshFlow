"""Independent saved-result review. No model forward, MC sampling, exact geometry rerun."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json
import numpy as np
import torch
ROOT=Path.cwd().resolve()
RUN=ROOT/'output/a_margin_single_20260925_01'
OUT=Path(__file__).resolve().parent
POLICY=ROOT/'output/MARGIN_JOINT_CANDIDATE_20260925.md'
EXPECTED_POLICY='8d7509ed3e69b8b9fd59acf6203e927b5d357f4e901bc0f4a4423149b522dfc6'
KEYS=('min_triangle_angle_deg','triangle_min_angle_p05_deg')
NAMES=('fixed','old_teacher','margin_teacher','old_final','margin_final')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ah(*args):
 h=hashlib.sha256()
 for v in args:
  v=np.ascontiguousarray(v);h.update(str(v.dtype).encode());h.update(json.dumps(v.shape).encode());h.update(v.tobytes())
 return h.hexdigest()
def mh(state):
 h=hashlib.sha256()
 for key in sorted(state):
  t=state[key].detach().cpu().contiguous();h.update(json.dumps([key,str(t.dtype),list(t.shape)],separators=(',',':')).encode());h.update(t.numpy().tobytes())
 return h.hexdigest()
def loadnp(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k]) for k in z.files}
def close(a,b):assert np.allclose(a,b,rtol=1e-12,atol=1e-12),(a,b)
def angles(v,f):
 t=v[f];vals=[]
 for j in range(3):
  a=t[:,(j+1)%3]-t[:,j];b=t[:,(j+2)%3]-t[:,j]
  vals.append(np.rad2deg(np.arccos(np.clip(np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)),-1.,1.))))
 per=np.min(np.stack(vals),axis=0)
 return dict(zip(KEYS,[float(np.min(per)),float(np.quantile(per,.05))]))
def qgate(q,a,floor,extra=0.):
 fail=[]
 if q['face_count']!=512:fail.append('face_budget')
 if a['valid'] is not True:fail.append('geometry_validity')
 if q['degenerate_count']:fail.append('degenerate')
 for k in KEYS:
  if not np.isfinite(q[k]) or q[k]+1e-6<floor[k]+extra:fail.append(k)
 return {'qualified':not fail,'failure_reasons':fail}
def gain(a,b):return [1-x/y for x,y in zip(a,b)]
def dump(path,obj):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 torch.set_num_threads(1)
 selected=['RUN_PROTOCOL.json','results.json','SUMMARY.json','TRAIN_DIAGNOSTICS.json','PAIRING_AUDIT.json','EXPLORATORY_JOINT_GATE.json']
 bound={name:sha(RUN/name) for name in selected}
 assert sha(POLICY)==EXPECTED_POLICY
 p=read(RUN/'RUN_PROTOCOL.json');res=read(RUN/'results.json');summary=read(RUN/'SUMMARY.json');diags=read(RUN/'TRAIN_DIAGNOSTICS.json');pairing=read(RUN/'PAIRING_AUDIT.json');gate=read(RUN/'EXPLORATORY_JOINT_GATE.json')
 assert res['protocol_sha256']==summary['protocol_sha256']==gate['protocol_sha256']==bound['RUN_PROTOCOL.json']
 assert summary['results_sha256']==gate['results_sha256']==bound['results.json']
 assert gate['policy_sha256']==EXPECTED_POLICY
 assert summary['mechanism_sha256']==bound['TRAIN_DIAGNOSTICS.json']
 assert res['planned_slots']==res['recorded_slots']==res['success_count']==20 and not res['errors'] and not res['unattempted']
 assert diags['planned_slots']==diags['recorded_slots']==diags['success_count']==24 and not diags['errors'] and not diags['unattempted']
 rows={(r['uid'],r['endpoint']):r for r in res['rows']}
 assert len(rows)==20 and set(rows)=={(u,n) for u in p['uids'] for n in NAMES}
 assert res['rows']==res['all_attempts']
 drows={(r['job'],r['update']):r for r in diags['rows']}
 assert len(drows)==24 and set(drows)=={(j,k) for j in p['jobs'] for k in (0,5000,20000)}
 labels=loadnp(RUN/'train_labels.npz');assert sha(RUN/'train_labels.npz')==p['hashes']['training_labels_sha256']
 mesh_readback=[];models=[];traces={};uid_summaries=[]
 for j,job in p['jobs'].items():
  path=RUN/'training'/j/'final.pt';report=read(path.parent/'TRAIN_REPORT.json');s=torch.load(path,map_location='cpu',weights_only=False)
  trace=read(path.parent/'TRAIN_TRACE.json');assert trace==s['trace'] and len(trace)==20000
  assert s['update']==report['update']==20000 and s['is_predeclared_final'] and report['status']=='COMPLETE'
  assert s['protocol_sha256']==report['protocol_sha256']==bound['RUN_PROTOCOL.json']
  assert mh(s['model'])==s['model_sha256']==report['model_sha256']==pairing['rows'][j]['model_sha256']
  assert sha(path)==report['checkpoint_sha256']==pairing['rows'][j]['checkpoint_sha256']
  sequence=[]
  for i,t in enumerate(trace,1):
   assert t['update']==i and t['uids']==[job['uid']]*4
   ks=t['rng_matched_k'];assert len(ks)==4 and all(type(x)is int and 0<=x<32 for x in ks);sequence.extend(ks)
  seq=bytes(sequence);seqsha=hashlib.sha256(seq).hexdigest()
  assert seqsha==s['step_sequence_sha256']==report['step_sequence_sha256']==pairing['rows'][j]['step_sequence_sha256']
  assert report['parent_exposures']=={job['uid']:80000}
  assert s['config']['initial_model_sha256']==p['hashes']['initial_model_sha256']
  assert s['config']['initial_checkpoint_sha256']==p['hashes']['initial_checkpoint_sha256']
  traces[j]=seq
  models.append({'job':j,'checkpoint_sha256':sha(path),'model_sha256':s['model_sha256'],'k_sequence_sha256':seqsha,'updates':20000,'parent_exposures':80000,'elapsed_training_seconds':report['elapsed_training_seconds']})
 for uid in p['uids']:
  assert traces['old_'+uid[:8]]==traces['margin_'+uid[:8]]
  data=loadnp(RUN/'data'/f'{uid}.npz');fixed=rows[(uid,'fixed')];floor={k:fixed['quality'][k] for k in KEYS};idx=p['uids'].index(uid)
  seeds=[b+p['all16_sorted_uids'].index(uid) for b in p['evaluation']['seed_bases']]
  refhash=ah(data['ref_vertices'],data['ref_faces']);fieldhash=ah(data['logr']);sub={}
  for name in NAMES:
   r=rows[(uid,name)];m=loadnp(RUN/r['mesh_file']);v=m['vertices'];f=m['faces']
   assert sha(RUN/r['mesh_file'])==r['mesh_npz_sha256']
   assert ah(v,f)==r['stored_array_hash'] and ah(v,data['faces'])==r['mesh_hash']
   assert f.shape==(512,3) and np.array_equal(f,data['faces']) and f.dtype==np.int64 and data['faces'].dtype==np.int32
   assert np.array_equal(v,m['u']*m['r'][:,None]) and np.isfinite(v).all()
   computed=angles(v,f)
   for k in KEYS:close(computed[k],r['quality'][k]);close(r['fixed_floor'][k],floor[k]);close(r['actual_margin_vs_fixed_deg'][k],r['quality'][k]-floor[k])
   assert qgate(r['quality'],r['audit'],floor)==r['qualified_original']
   assert qgate(r['quality'],r['audit'],floor,.1)==r['qualified_strict']
   assert r['audit']['exact_requested'] and r['audit']['exact_executed'] and r['audit']['valid'] and r['audit']['topology_pass'] and r['audit']['selfintersection_count']==0
   assert r['audit']['exact']['embedded_endpoint_pass'] and r['audit']['exact']['pair_count']==0
   assert r['reference_hash']==refhash and r['mainfield_hash']==fieldhash and r['data_file_sha256']==sha(RUN/'data'/f'{uid}.npz')
   assert r['metric_seeds']==seeds and r['points_each_direction']==8192 and not r['fallback_to_fixed']
   assert [x['seed'] for x in r['metrics']]==seeds and all(x['points_per_direction']==8192 for x in r['metrics'])
   if name.endswith('teacher'):
    kind=name.split('_')[0];assert np.array_equal(m['u'],labels['directions1_'+kind][idx])
   if name.endswith('final'):
    job=name.split('_')[0]+'_'+uid[:8];d=drows[(job,20000)];dm=loadnp(RUN/d['directions_file'])
    assert np.array_equal(m['u'],dm['u']) and np.array_equal(m['alpha'],dm['alpha']) and r['final_equals_corresponding_diagnostic']
    assert r['identity']['checkpoint_file_sha256']==sha(RUN/'training'/job/'final.pt')
   rms=[x['rms_bbox_percent'] for x in r['metrics']]
   gainfixed=gain(rms,[x['rms_bbox_percent'] for x in fixed['metrics']])
   sub[name]={'rms_bbox_percent':rms,'gain_over_fixed_percent':[x*100 for x in gainfixed],'qualified_original':r['qualified_original']['qualified'],'qualified_strict':r['qualified_strict']['qualified'],'quality_failure_reasons':r['qualified_original']['failure_reasons'],'angles_deg':{k:r['quality'][k] for k in KEYS},'margin_vs_fixed_deg':r['actual_margin_vs_fixed_deg'],'exact_valid_F512':True,'joint_success':r['qualified_original']['qualified'] and min(gainfixed)>=.02}
   mesh_readback.append({'uid':uid,'endpoint':name,'mesh_npz_sha256':r['mesh_npz_sha256'],'stored_array_hash':r['stored_array_hash'],'logical_array_hash':r['mesh_hash'],'faces':len(f),'saved_exact_audit_verified_by_bound_mesh':True,'direct_angle_readback':computed})
  gains=gain(sub['margin_final']['rms_bbox_percent'],sub['old_final']['rms_bbox_percent'])
  uid_summaries.append({'uid':uid,'endpoints':sub,'new_vs_old_gain_percent':[x*100 for x in gains],'new_vs_old_both_ge2pct':min(gains)>=.02})
 oldset={x['uid'] for x in uid_summaries if x['endpoints']['old_final']['joint_success']};newset={x['uid'] for x in uid_summaries if x['endpoints']['margin_final']['joint_success']}
 pairedset={x['uid'] for x in uid_summaries if x['new_vs_old_both_ge2pct']}
 conditions={'all_four_new_finals_exact_valid_and_F512':all(x['endpoints']['margin_final']['exact_valid_F512'] for x in uid_summaries),'net_at_least_one_joint_success_and_no_old_success_lost':len(newset)>=len(oldset)+1 and oldset<=newset,'at_least_two_new_vs_old_dual_MC_ge2pct':len(pairedset)>=2}
 assert conditions==gate['conditions'] and all(conditions.values())==gate['candidate_go']
 assert oldset==set(gate['old_joint_success_uids']) and newset==set(gate['new_joint_success_uids']) and pairedset==set(gate['paired_dual_MC_ge2pct_uids'])
 for rec in summary['paired']:
  u=next(x for x in uid_summaries if x['uid']==rec['uid'])
  for n in NAMES:
   close(rec['endpoints'][n]['rms_bbox_percent'],u['endpoints'][n]['rms_bbox_percent'])
   assert rec['endpoints'][n]['qualified_original']['qualified']==u['endpoints'][n]['qualified_original']
 assert all(sha(RUN/n)==h for n,h in bound.items())
 assert not torch.cuda.is_initialized()
 review={'status':'PASS_SAVED_RESULT_REVIEW','input_sha256':bound,'policy_sha256':EXPECTED_POLICY,'source_sha256':sha(__file__),'planned_and_read_endpoints':20,'planned_and_recorded_observations':24,'all_four_paired_k_sequences_byte_equal':True,'actual_updates':160000,'actual_parent_exposures':640000,'models':models,'per_uid':uid_summaries,'conditions':conditions,'candidate_go':all(conditions.values()),'old_joint_success_uids':sorted(oldset),'new_joint_success_uids':sorted(newset),'paired_dual_MC_ge2pct_uids':sorted(pairedset),'mesh_readback':mesh_readback,'scope':{'model_forward':False,'new_MC_samples':False,'exact_geometry_recomputed':False,'exact_status':'saved exact audit bound to verified NPZ and typed-array hashes','angles':'direct saved vertices/faces recomputation','single_training_seed':10,'all_four_are_previous_development_parents':True,'original_run_ledger_unchanged':True,'reference':'same frozen mainfield support mesh; not independent raw-asset GT'},'cuda_initialized':False}
 dump(OUT/'REVIEW.json',review)
 print(json.dumps({'status':review['status'],'conditions':conditions,'candidate_go':review['candidate_go'],'per_uid':[{'uid':x['uid'],'new_vs_old_gain_percent':x['new_vs_old_gain_percent'],'old_qualified':x['endpoints']['old_final']['qualified_original'],'new_qualified':x['endpoints']['margin_final']['qualified_original'],'new_margins':x['endpoints']['margin_final']['margin_vs_fixed_deg']} for x in uid_summaries]},indent=2))
if __name__=='__main__':main()
