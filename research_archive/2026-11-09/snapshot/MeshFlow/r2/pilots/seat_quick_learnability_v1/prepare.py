"""CPU-only frozen four-seat data/evaluation preparation; no model or stream creation."""
from pathlib import Path
import csv,hashlib,json,os,shutil,time
import numpy as np
ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_seat_quick_learnability_v1/inputs'
PREV=ROOT/'experiments/r2_seat_failure_diagnosis_v1'
SCALING=ROOT/'experiments/r2_user_data_scaling_32_67_v1'
SELECTED=(
 '73b369b9808bc2d0f6ba522682be8978__SEAT_HALF',
 '3f7417590f1bcfded5c89ecb06d1099b__SEAT_HALF',
 '8c9b2023ecdeb7b049d998127ea6baf9__SEAT_FULL',
 '85b16941984902f8facfa12c7d71c89f__SEAT_FULL',
)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def arr(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def ah(x):
 a=np.ascontiguousarray(x);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def csvrows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  assert read(p)==v,'Frozen file differs: '+str(p);return
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 manifest=OUT/'manifest.json'
 if manifest.exists():
  z=read(manifest)
  for p,h in z['source_hashes'].items():assert sha(p)==h,p
  for q in z['requests']:assert sha(q['input_path'])==q['input_sha256']
  return z
 r=read(PREV/'registration.json');m=read(SCALING/'data/train_manifest.json')
 raw={x['request_id']:x for x in read(PREV/'evaluation/raw_index.json')['rows']}
 post={x['request_id']:x for x in read(PREV/'evaluation/post_index.json')['rows']}
 score={(x['request_id'],x['stage']):x for x in csvrows(PREV/'statistics/per_output.csv')}
 audit={x['condition_id']:x for x in csvrows(PREV/'source_audit/per_condition.csv')}
 protected={}
 def protect(p,expected=None):
  p=str(p);h=sha(p)
  if expected is not None:assert h==expected,p
  if p in protected:assert protected[p]==h,p
  protected[p]=h;return h
 for p in [PREV/'registration.json',PREV/'evaluation/raw_index.json',PREV/'evaluation/post_index.json',PREV/'statistics/per_output.csv',PREV/'source_audit/per_condition.csv',SCALING/'data/train_manifest.json',ROOT/'AGENTS.md',ROOT/'WORKSPACE.md',ROOT/'native_t1/docs/control_metrics.md']:
  protect(p)
 conditions=[];tasks=[];requests=[];queries=[];checks=[]
 for pos,cid in enumerate(SELECTED):
  t=m['tasks'][cid];p=m['parents'][t['uid']];a=audit[cid]
  assert t['uid'] in m['pools']['USER_EXPANDED_67']
  assert t['CC_CF_FF']['CC']=={'illegal':0,'unknown':0} and t['CC_CF_FF']['CF']=={'illegal':0,'unknown':0}
  assert int(a['source_interface_inherited_valid_edges'])>0 and int(a['Y_duplicate_groups'])==0
  matches=[c for c in r['conditions'] if c['split']=='TRAIN' and c['variant']=='base' and c['base_condition_id']==cid];assert len(matches)==1
  c=dict(matches[0]);conditions.append(c)
  protect(t['npz'],t['sha256']);protect(p['npz'],p['sha256']);protect(t['quality_path'],t['quality_sha256'])
  source=arr(t['npz']);parent=arr(p['npz']);C=source['C'];N=int(source['N']);K=int(source['K'])
  assert ah(C)==c['C_sha256'] and N==c['N'] and K==c['K']
  assert same(source['C_original'],parent['model_vertices'][parent['source_faces'][source['source_face_ids']]])
  transform=json.loads(str(parent['transform_json'].item())) if 'transform_json' in parent else {'status':'NA_MISSING_TRANSFORM_JSON'}
  before_score=[score[x['request_id'],'POST'] for x in r['requests'] if x['split']=='TRAIN' and x['variant']=='base' and x['base_condition_id']==cid]
  task={'task_id':cid,'condition_id':cid,'evaluation_condition_id':c['condition_id'],'parent_id':t['uid'],'condition_type':t['condition_type'],'N':N,'K':K,
        'condition_source':t['npz'],'condition_source_sha256':t['sha256'],'parent_npz':p['npz'],'parent_npz_sha256':p['sha256'],
        'C_sha256':ah(C),'source_task':t,'source_parent':p,'source_interface_valid_edges':int(a['source_interface_inherited_valid_edges']),
        'source_CC_CF_FF':t['CC_CF_FF'],'source_Y_exact_duplicate_groups':0,'source_global_quality':t['source_quality'],
        'baseline_POST_valid_edge_outputs':sum(float(x['C_F_has_valid_edge'])>0 for x in before_score),'baseline_POST_outputs':len(before_score),
        'historical_parent_coordinate_transform':transform,'coordinate_contract':'Inherited whole-parent MeshFlow normalization/model coordinates. Do not center/scale C separately or normalize again. Retain current source geometry unchanged.'}
  tasks.append(task)
  query=next(x for x in r['queries'] if x['condition_id']==c['condition_id']);protect(query['path'],query['sha256']);queries.append(query)
  qrs=sorted([x for x in r['requests'] if x['split']=='TRAIN' and x['variant']=='base' and x['base_condition_id']==cid],key=lambda x:x['candidate']);assert [x['candidate'] for x in qrs]==[0,1]
  for orig in qrs:
   q=dict(orig);protect(q['input_path'],q['input_sha256']);inp=arr(q['input_path']);assert same(inp['C'],C)
   assert int(inp['N'])==N and int(inp['K'])==K and ah(inp['z'])==q['Gaussian_sha256']
   free=inp['z'][:,K:].reshape(-1,3,3);assert ah(free)==q['free_Gaussian_sha256']
   dest=OUT/(q['request_id']+'.npz')
   if dest.exists():assert sha(dest)==q['input_sha256']
   else:shutil.copyfile(q['input_path'],dest)
   assert sha(dest)==q['input_sha256']
   q['previous_input_path']=q['input_path'];q['input_path']=str(dest);q['baseline_request_id']=q['request_id'];q['panel']='FOCUSED_TRAIN4';q['quick_condition_order']=pos
   q['baseline_raw_record']=raw[q['request_id']];q['baseline_post_record']=post[q['request_id']]
   for stage,rec in [('raw',raw[q['request_id']]),('post',post[q['request_id']])]:
    path=rec[stage+'_path'];h=rec[stage+'_sha256'];protect(path,h);q['baseline_'+stage+'_path']=path;q['baseline_'+stage+'_sha256']=h
    sr=score[q['request_id'],stage.upper()];protect(sr['score_path'],sr['score_sha256']);sc=read(sr['score_path']);assert sc['identity']['mesh_sha256']==h
    q['baseline_score_'+stage.upper()+'_path']=sr['score_path'];q['baseline_score_'+stage.upper()+'_sha256']=sr['score_sha256']
    # Protect all scorer evidence already bound by the completed score; no rescoring.
    for fp,fh in sc['files'].items():protect(fp,fh)
   rt=arr(q['baseline_raw_path'])['raw'].reshape(-1,3,3);pt=arr(q['baseline_post_path'])['output'].reshape(-1,3,3)
   assert same(rt[:K],C) and same(pt[:K],source['C_original'])
   checks.append({'request_id':q['request_id'],'N':N,'K':K,'input_file_bytes_preserved':True,'C_bytes_preserved':True,'full_noise_bytes_preserved':True,'free_noise_bytes_preserved':True,'source_C_RAW_POST_verified':True,'new_noise_draws':0})
   requests.append(q)
 assert len(tasks)==4 and len({t['parent_id'] for t in tasks})==4 and len(requests)==8
 assert sorted(t['condition_type'] for t in tasks)==['SEAT_FULL','SEAT_FULL','SEAT_HALF','SEAT_HALF']
 for path,h in protected.items():assert sha(path)==h,path
 result={'schema':'r2_seat_quick_learnability_inputs_v1','status':'FROZEN','created_unix':time.time(),'training_data_json':str(SCALING/'data/train_manifest.json'),
         'training_data_sha256':protected[str(SCALING/'data/train_manifest.json')],'all_pool_name':'USER_EXPANDED_67','all_pool_parent_ids':m['pools']['USER_EXPANDED_67'],
         'focused_task_ids':list(SELECTED),'focused_parent_ids':[t['parent_id'] for t in tasks],'train_conditions':tasks,'conditions':conditions,'requests':requests,'queries':queries,
         'L':r['L'],'baseline_label':'EXP67_FROZEN','baseline_checkpoint_identity':r['checkpoint'],'baseline_D0_identity':r['D0'],
         'baseline_registration':str(PREV/'registration.json'),'baseline_registration_sha256':protected[str(PREV/'registration.json')],
         'selection_policy':'Four known TRAIN-base conditions, two HALF/two FULL, four distinct parents. Require legal original C/Y interface and zero C/C,C/Y illegal/UNKNOWN; prefer no exact Y duplicates. Deliberately include prior joining positives and negatives; this is a learnability panel, not a held-out or unbiased test.',
         'excluded_previous_seat_conditions':{'e1e7e66978b888703f146f1fbd99cb1a__SEAT_HALF':'Y has two exact duplicate groups; alternative HALF has no detected source pair problem.','6c36a5b107e83d331191025061735ea3__SEAT_FULL':'Y has 27 exact duplicate groups and 93 illegal FF pairs; select nonduplicate FULL instead.'},
         'batch8_policy':{'focused_arm':'Four parents exactly twice per effective batch; eight examples do not mean eight distinct parents. Independent noise/time for repeated parent occurrences.','all_arm':'Original 67-parent sampler retained by controller.','augmentation':'Controller freezes same 4 clean + 4 width schedule per effective batch across arms; preparation creates no training stream.'},
         'coordinate_policy':'Use inherited normalized full-parent model coordinates and saved source mapping. No new centering, scaling, tessellation or source geometry edits.',
         'quality_limit':'Both HALF sources have no detected pair problem; selected FULL sources have 18/54 inherited FF-illegal pairs outside accepted C/CY. They are not certified globally clean. No source repair performed.',
         'model_calls':0,'new_noise_draws':0,'training_stream_created':False,'source_hashes':protected,'entry_source_sha256':sha(__file__)}
 write(manifest,result);write(OUT/'verification.json',{'status':'PASS','requests':checks,'old_protected_files':len(protected),'old_hashes_unchanged':True,'new_input_files':8,'manifest_sha256':sha(manifest),'models':0,'training_streams':0,'new_noise_draws':0})
 return result

if __name__=='__main__':
 r=prepare();print(json.dumps({'status':r['status'],'manifest':str(OUT/'manifest.json'),'conditions':len(r['train_conditions']),'requests':len(r['requests']),'focused_task_ids':r['focused_task_ids'],'models':0,'training_streams':0},ensure_ascii=False))