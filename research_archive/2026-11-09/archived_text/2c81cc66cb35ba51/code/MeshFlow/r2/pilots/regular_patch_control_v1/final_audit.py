"""Independent offline audit: no model imports, no generation and no geometry rescoring."""
import argparse, ast, csv, datetime, json, math, sys, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from . import common as C
from .audit_inputs import center

CHECKS=0
HASHES={}
def ck(ok,why):
 global CHECKS
 CHECKS+=1
 if not bool(ok):raise AssertionError(why)
def same(a,b,why):ck(C.same(np.asarray(a),np.asarray(b)),why)
def number(a):return 1. if a=='True' else (0. if a=='False' else float(a))
def close(a,b,why):ck(math.isclose(number(a),number(b),rel_tol=1e-10,abs_tol=1e-12),why)
def file(p,h):
 p=C.resolve(p);key=str(p)
 if key not in HASHES:HASHES[key]=C.sha(p)
 ck(HASHES[key]==h,'file hash '+key)
def maps(d):
 for p,h in d.items():file(p,h)
def rows(p):
 with C.resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def events(p):return [json.loads(x) for x in C.resolve(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def utc(t):return datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat()
def triangle_sets(x):
 x=np.asarray(x);idx=np.lexsort((x[:,:,2],x[:,:,1],x[:,:,0]),axis=1);return np.take_along_axis(x,idx[:,:,None],axis=1)
def audit():
 start=time.monotonic();r=C.load_registration();rh=C.sha(C.OUT/'registration.json');ev=C.OUT/'evaluation'
 ck(rh=='cc7f5f21fbaf998a1e7c713526b8703e92dc9003df55ba3cc45f51cecbf40543','frozen registration')
 maps(r['code_hashes']); inp_audit=C.read(C.OUT/'independent_audit/input_audit.json')
 ck(inp_audit['status']=='PASS' and inp_audit['registration_sha256']==rh,'input independent audit')
 qmap={q['request_id']:q for q in r['requests']};qm={q['condition_id']:q for q in r['queries']}
 ck(len(qmap)==24 and len(qm)==6 and r['budget']['training_updates']==0,'registered budget')
 for q in r['queries']:file(q['path'],q['sha256'])
 receipts={};times=defaultdict(list);counts=Counter();peak=defaultdict(list)
 for arm in r['labels']:
  for q in r['requests']:
   rid=q['request_id'];k=q['K'];n=q['N'];new=q['action']=='NEW'
   file(q['input_path'],q['input_sha256']);file(q['condition_source'],q['condition_source_sha256']);file(q['full_noise_path'],q['full_noise_file_sha256'])
   a=C.arrays(q['input_path']);source=C.arrays(q['condition_source']);known=a['C'].reshape(k,3,3)
   rec=C.read(ev/'raw'/arm/rid/'receipt.json');post=C.read(ev/'post'/arm/rid/'receipt.json');receipts[arm,rid,'RAW']=rec;receipts[arm,rid,'POST']=post
   for st,x in [('raw',rec),('post',post)]:
    counts[st]+=1;ck(x['status']=='COMPLETE' and x['label']==arm and x['request_id']==rid,'receipt cell')
    ck(x['checkpoint_sha256']==r['models'][arm]['sha256'] and x['input_sha256']==q['input_sha256'],'model/input binding')
    file(x[st+'_path'],x[st+'_sha256'])
   for name in ['ply','result','velocities']:file(rec[name+'_path'],rec[name+'_sha256'])
   maps(post.get('files',{}));raw=C.arrays(rec['raw_path']);world=raw['trajectory'].reshape(51,n,3,3);v=C.arrays(rec['velocities_path'])['velocity'].reshape(50,n,3,3)
   ck(world.dtype==v.dtype==np.float32 and np.isfinite(world).all() and np.isfinite(v).all(),'finite FP32')
   ck(not np.any(v.view(np.uint32)&0xffff),'lossless BF16 stored velocities')
   same(raw['C'],known,'saved known');same(raw['z'],a['z'],'frozen Gaussian');same(raw['raw'].reshape(n,3,3),world[-1],'terminal')
   same(world[:,:k],np.broadcast_to(known,(51,k,3,3)).copy(),'C held all51')
   if arm=='CREF_C200':
    local=raw['local_trajectory'].reshape(51,n,3,3);c=raw['center'];ck(np.allclose(c,center(known),rtol=1e-6,atol=2e-7),'FP32 C-only area center')
    same(raw['local_C'],known-c,'local C');same(local[:,:k],np.broadcast_to(known-c,(51,k,3,3)).copy(),'local C all51')
    same(local[0,k:],a['z'][0,k:].reshape(n-k,3,3),'untranslated local z')
    same(world[:,k:],local[:,k:]+c,'world restore exactly');state=local
   else:
    same(world[0,k:],a['z'][0,k:].reshape(n-k,3,3),'world start z');state=world
   for j in range(50):same(state[j+1,k:],state[j,k:]+np.float32(1/50)*v[j,k:],'FP32 Euler step')
   pa=C.arrays(post['post_path']);can=C.arrays(post['canonical_path']);pre=C.arrays(post['pre_geo_path'])
   same(pa['output'][:k],source['C_original'],'final source C corner order');same(pa['vertices'][pa['faces']],pa['output'],'indexed POST exact')
   same(can['output'][:k],known,'canonical protected C');same(pre['output'][:k],known,'preGEO protected C');same(pa['vertices'],can['vertices'],'orientation coordinates unchanged')
   same(triangle_sets(pa['output']),triangle_sets(can['output']),'orientation preserves every unordered face')
   ck(post['source_sha256']==q['condition_source_sha256'],'POST original source binding')
   ck(post['orientation']['coordinates_unchanged'] and post['source_restoration']['status']=='PASS','orientation/source metadata')
   run=C.read(rec['result_path']);ck(run['nfe']==50 and run['known_all51_bitwise'] and run['readout_mode']=='none','run scientific contract')
   if new:
    ck(rec['new_rollouts']==1 and rec['new_sampling_F']==50 and post['new_D0_F']==1,'new call budget')
    path=ev/'raw'/arm/rid/'calls.jsonl';ee=events(path);counts['new_rollouts']+=1
    reserves=[x for x in ee if x['event']=='RESERVE' and x.get('kind')=='sampling_forward'];returns=[x for x in ee if x['event']=='RETURN' and x.get('kind')=='sampling_forward']
    ck(len(reserves)==len(returns)==50,'50 reserved and returned F')
    ck([x['context']['euler_step'] for x in reserves]==list(range(50)),'Euler call order')
    ck(sum(x['event']=='RESERVE' and x.get('kind')=='rollout_attempt' for x in ee)==1 and sum(x['event']=='ROLLOUT_COMPLETE' for x in ee)==1,'no rollout retry')
    sequence=[x['event'] for x in ee if x.get('kind')=='sampling_forward'];ck(sequence==['RESERVE','RETURN']*50,'no unresolved generator call')
    for x in reserves:ck(x['context']['label']==arm and x['context']['request_id']==rid and x['context']['checkpoint_sha256']==r['models'][arm]['sha256'],'call identity')
    counts['sampling_forward_reserves']+=len(reserves);counts['sampling_forward_returns']+=len(returns)
    times[arm].append((ee[1]['time_unix'],ee[-1]['time_unix']));peak[arm].append(run['cuda_peak_reserved_bytes'])
    d=C.read(post['proposal_receipt']);ck(d['model_forwards']==1 and d['checkpoint_sha256']==r['d0_checkpoint']['sha256'] and d['raw_sha256']==rec['raw_sha256'],'single official D0')
    file(post['proposal_path'],d['output_sha256']);counts['D0_proposal_returns']+=1
    ck((Path(post['post_path']).parent/'D0_started.json').exists(),'D0 reservation exists')
   else:
    ck(rec['new_rollouts']==rec['new_sampling_F']==post['new_D0_F']==0,'cache has no new call')
    for stage,x in [('raw',rec),('post',post)]:
     old=q['cache'][arm];file(old[stage+'_path'],old[stage+'_sha256']);o=C.read(old[stage+'_path']);ck(x[stage+'_path']==o[stage+'_path'] and x[stage+'_sha256']==o[stage+'_sha256'],'cache original mesh identity');counts['cached_'+stage]+=1
 ck(dict(counts)=={'raw':48,'post':48,'new_rollouts':32,'sampling_forward_reserves':1600,'sampling_forward_returns':1600,'D0_proposal_returns':32,'cached_raw':16,'cached_post':16},'exact actual budget '+str(counts))
 de=events(C.OUT/'execution/worker_0_of_1_events.jsonl');de=[x for x in de if x['event'].startswith('D0_FORWARD_')]
 ck(len(de)==64 and [x['event'] for x in de]==['D0_FORWARD_RESERVED','D0_FORWARD_RETURNED']*32,'32 D0 calls and no unresolved')
 for x,y in zip(de[::2],de[1::2]):ck(x['request_id']==y['request_id'],'D0 reserve-return identity')
 ck(Counter(x['request_id'] for x in de[::2])==Counter({q['request_id']:2 for q in r['requests'] if q['action']=='NEW'}),'two arms D0 identity')
 latest_raw=max(b for tt in times.values() for _,b in tt);first_d0=datetime.datetime.fromisoformat(de[0]['utc']).timestamp();ck(latest_raw<first_d0,'ALL new RAW before first D0')
 completion=C.read(ev/'score_completion.json');maps(completion['definition_hashes']);sc=completion['receipts'];ck(len(sc)==96 and completion['new_stages']==64 and completion['reused_historical_stages']==32,'score budget')
 sr={};scorecounts=Counter()
 for x in sc:
  key=(x['label'],x['request_id'],x['stage']);ck(key not in sr,'score unique');file(x['path'],x['sha256']);d=C.read(x['path']);sr[key]=d;q=qmap[x['request_id']];rec=receipts[key];stage=x['stage'].lower();identity=d['identity'];query=qm[q['condition_id']];w=C.arrays(query['path'])['C_weights']
  ck(identity['C_sha256']==q['C_sha256'] and identity['Gaussian_sha256']==q['Gaussian_sha256'] and identity['query_sha256']==query['sha256'],'score C/noise/query identity')
  ck(identity['mesh_sha256']==rec[stage+'_sha256'] and identity['label']==x['label'] and identity['stage']==x['stage'],'score geometry label/stage')
  maps(d['files']);maps(d['source_hashes']);m=d['metrics'];f=d['free_face_denominator'];ck(f==(q['N']-q['K'] if stage=='raw' else len(C.arrays(rec['post_path'])['output'])-q['K']),'actual free denominator')
  ck(d['CF_tested_pair_denominator']==q['K']*f and d['FF_tested_pair_denominator']==f*(f-1)//2,'pair denominators');ck(d['C_query_count']==2048 and d['C_preserved_exact'],'query and C')
  v=C.arrays(Path(x['path']).parent/'vectors.npz');den=float(w.sum());close(den,d['C_query_weight_denominator'],'query weight')
  for mask,metric in [('conflict_cross','ConflictCov_C_cross'),('conflict_dup','ConflictCov_C_dup'),('conflict_union','ConflictCov_C'),('conflict_unknown','ConflictCov_C_uncertain_fraction'),('conflict_C_interior','ConflictCov_C_interior_weight_fraction')]:close(np.sum(w*v[mask])/den,m[metric],'saved point-mask arithmetic '+metric)
  same(v['conflict_union'],v['conflict_cross']|v['conflict_dup'],'union point masks')
  close(m['ConflictCov_C'],m['ConflictCov_C_cross']+m['ConflictCov_C_dup']-m['ConflictCov_C_cross_and_dup'],'I inclusion-exclusion')
  if q['action']=='NEW':ck(x['cache_action']=='NEW_REGISTERED_SCORE' and identity['registration_sha256']==rh,'new score frozen identity');scorecounts['new']+=1
  else:
   ck(x['cache_action']=='VERIFIED_HISTORICAL_SCORE_REUSE' and x['new_quality_scans_this_call']==0,'no cached rescore');file(x['reuse_receipt_path'],x['reuse_receipt_sha256']);u=C.read(x['reuse_receipt_path']);ck(u['original_score_identity']==identity and u['score_sha256']==x['sha256'] and u['registration_sha256']==rh,'old score exact reuse');file(u['old_completion_path'],u['old_completion_sha256']);scorecounts['cache']+=1
 ck(scorecounts==Counter(new=64,cache=32),'64 new /32 existing stage scores')
 flat=rows(ev/'per_output.csv');ck(len(flat)==96,'96 output rows');flatmap={(x['label'],x['request_id'],x['stage']):x for x in flat};ck(set(flatmap)==set(sr),'flattened identities')
 for key,d in sr.items():
  row=flatmap[key]
  for metric,val in {**d['metrics'],**d['topology']}.items():
   if isinstance(val,(int,float)) and not isinstance(val,bool) and metric in row:close(row[metric],val,'flatten score metric')
 pairs=rows(ev/'paired_per_output.csv')
 for x in pairs:
  if x['status']!='PASS':continue
  b=flatmap['SMALL_C200',x['request_id'],x['stage']];a=flatmap['CREF_C200',x['request_id'],x['stage']];metric=x['metric'];close(x['baseline_value'],b[metric],'paired baseline');close(x['candidate_value'],a[metric],'paired candidate');close(x['delta'],number(a[metric])-number(b[metric]),'paired delta')
 for filename,source,group in [('paired_per_parent.csv',pairs,['panel','input_action','parent_id','condition_id','condition_type','stage','metric']),('paired_summary.csv',rows(ev/'paired_per_parent.csv'),['panel','input_action','stage','metric'])]:
  groups=defaultdict(list)
  for x in source:groups[tuple(x[g] for g in group)].append(x)
  for x in rows(ev/filename):
   vals=groups[tuple(x[g] for g in group)];valid=[v for v in vals if v['status']=='PASS'];ck(int(x['n_registered'])==len(vals) and int(x['n_valid'])==len(valid),'parent/panel denominator')
   if valid:
    for key in ['baseline_value','candidate_value','delta']:close(x[key],np.mean([number(v[key]) for v in valid]),'parent/panel macro '+key)
 ck(len(rows(ev/'AVMR_PMR_all8.csv'))==768,'all8 AVMR/PMR outputs')
 h=C.read(C.OUT/'historical_context/provenance.json');maps(h['source_hashes']);ck(h['old_actual_review_rows']==h['old_score_rows']==96 and h['new_visual_reviews']==0,'historical context exact old review reuse')
 gallery=C.read(ev/'gallery/manifest.json');maps(gallery['definition_hashes']);maps(gallery['images']);maps(gallery['files']);maps(C.read(ev/'gallery/source_hashes.json'));ck(gallery['pages']==24 and gallery['review_rows']==96,'rendering budget')
 # Exercise completed-wave early return with dispatch and resource calls forbidden.
 from . import run
 from . import inference
 def forbidden(*a,**kw):raise AssertionError('Completed-wave recovery attempted dispatch/resources')
 oldp=run.subprocess.Popen;oldr=C.resources
 run.subprocess.Popen=forbidden;C.resources=forbidden
 try:run.gpu_wave('generate');run.gpu_wave('post')
 finally:run.subprocess.Popen=oldp;C.resources=oldr
 ck('torch' not in sys.modules,'auditor and completed-wave recovery imported no model framework')
 text=(C.CODE/'run.py').read_text(encoding='utf-8-sig');ck('if not active:\n            baseline=used;peak_increment=0' in text and 'now-active[0][3]' in text,'recovery fixes retained')
 ck('replace=True' in (C.CODE/'audit_inputs.py').read_text(encoding='utf-8-sig'),'input audit permits refreshed timing')
 interval={arm:{'first_actual_call_utc':utc(min(a for a,b in v)),'last_rollout_complete_utc':utc(max(b for a,b in v)),'elapsed_first_to_last_seconds':max(b for a,b in v)-min(a for a,b in v),'sum_actual_rollout_seconds':sum(b-a for a,b in v),'max_allocator_reserved_bytes':max(peak[arm])} for arm,v in times.items()}
 overlap=max(0,min(max(b for a,b in t) for t in times.values())-max(min(a for a,b in t) for t in times.values()))
 result=dict(status='PASS',scope='Completed execution, immutable identities, saved-array integration and score arithmetic; no new semantic review',registration_sha256=rh,checks=CHECKS,actual_counts=dict(counts),scores=dict(scorecounts),protected_and_artifact_hashes_verified=len(HASHES),all_RAW_before_D0=dict(last_new_RAW_complete_utc=utc(latest_raw),first_D0_reserved_utc=de[0]['utc'],gap_seconds=first_d0-latest_raw),actual_generator_timing=interval,generator_arm_interval_overlap_seconds=overlap,D0_first_reserved_utc=de[0]['utc'],D0_last_returned_utc=de[-1]['utc'],D0_execution='Actual D0 event ledger is serial; 32 reserve/return pairs. No two-worker D0 speedup claimed.',completed_wave_recovery_no_dispatch='EXECUTED_PASS for generate and post with subprocess and resources forbidden',input_audit_idempotence='Source-verified replace=True; no scientific data rewritten',limits=['HALF to FULL changes condition area, shape, K, free count and CREF center; not a pure K intervention.','I and AVMR domains change with C. Historical image labels come from different review/views; two unchanged FULL inputs are review-drift controls.','Saved point-mask aggregation does not independently rerun geometric conflict detection.','Model weights were bound/hash-verified by frozen preparation; this audit does not reopen large checkpoint files or load neural models.','Allocator reservation/device used delta is not complete WDDM per-process VRAM.','Rendering/hash completeness is not actual visual review; review/report checked separately.','At least one valid C/F edge does not certify complete adoption or an entire sealed interface.'],auditor_calls=dict(training=0,model_forward=0,model_loads=0,new_generation=0,quality_rescoring=0,new_queries=0),seconds=time.monotonic()-start,auditor_sha256=C.sha(__file__),implementation_sources={str(C.CODE/f):C.sha(C.CODE/f) for f in ['prepare.py','inference.py','score.py','render.py','run.py','audit_inputs.py']})
 C.write(C.OUT/'independent_audit/final_audit.json',result,replace=True);C.write(C.OUT/'independent_audit/verified_hashes.json',HASHES,replace=True)
 (C.OUT/'independent_audit/final_audit.md').write_text('Independent offline audit: PASS\n\n32 new RAW rollouts, 1600 reserved/returned generator forwards; 32 single D0 proposals; 16 RAW and 16 POST exact cache reuses. All 48 RAW and 48 POST identities checked. All new RAW preceded D0.\n\n96 scores checked: 64 new / 32 original cache. Reaggregated saved 2048-query masks and paired output/parent/panel arithmetic without geometric rescoring. Source/file hashes and source-winding/C preservation validated.\n\nCompleted generate and post recovery paths were executed with dispatch/resource functions forbidden: no model, process, or resource probe was requested. Two earlier orchestration issues were repaired by root with no scientific replay.\n\nLimitations: changed HALF-to-FULL conditions are not a pure K test. Historical image judgments have different views/reviewers; I and A-VMR query domains change. No global mesh validity follows from a local interface. Actual review/report completeness is audited separately.\n',encoding='utf-8')
 return result
if __name__=='__main__':
 print(json.dumps(audit(),indent=2))
