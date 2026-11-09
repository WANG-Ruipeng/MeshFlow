"""CPU-only packaging preparation. No model/scorer imports or execution."""
from pathlib import Path
import csv, hashlib, json
ROOT=Path('/mnt/e/MeshFlow');OUT=ROOT/'experiments/r2_window_policy_rl_v1';DEST=OUT/'source_snapshot'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if read(p)!=obj:raise ValueError('Existing evidence differs; not overwriting '+str(p))
 else:p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
mp=OUT/'inputs/manifest.json';tp=OUT/'inputs/training_manifest.json';m=read(mp);t=read(tp)
evidence=[];definition=None
for label in ['FULL_DET','S005','S010','S020']:
 for q in m['calibration_requests']:
  p=OUT/'calibration'/label/'scores'/q['request_id']/'POST/score.json';identity=read(p)['identity']
  assert (identity['stage'],identity['label'],identity['request_id'])==('POST',label,q['request_id'])
  d=identity['definition_hashes']
  if definition is None:definition=d
  assert d==definition,'Scorer definitions differ'
  evidence.append({'path':str(p),'package_relative_path':str(p.relative_to(OUT)),'sha256':sha(p),'request_id':q['request_id'],'label':label})
assert len(evidence)==256 and len(definition)==19
sources=[];copies=[]
for original,expected in sorted(definition.items()):
 p=Path(original);assert p.is_file() and p.stat().st_size<4*1024*1024
 b=p.read_bytes();actual=hashlib.sha256(b).hexdigest();assert actual==expected,'Live scorer changed '+original
 rel=p.relative_to('/mnt/e');dest=DEST/'scorers'/rel
 sources.append({'original_path':original,'original_windows_path':original.replace('/mnt/e/','E:/'),
  'source_project':rel.parts[0],'source_relative_path':str(Path(*rel.parts[1:])),
  'score_definition_sha256':expected,'original_sha256':actual,'package_relative_path':str(dest.relative_to(OUT)),
  'copied_sha256':actual,'bytes':len(b),'verification':'LIVE_SOURCE_MATCHES_SAVED_POST_DEFINITION'})
 copies.append((dest,b))
for p,b in copies:
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
 assert sha(p)==hashlib.sha256(b).hexdigest()
write(DEST/'scorers_manifest.json',{'status':'COMPLETE','version':'actual_post_scorer_snapshot_v1',
 'scope':'19 definitions referenced by all 256 completed calibration POST identities; no scorer execution',
 'definition_count':19,'source_bytes':sum(r['bytes'] for r in sources),
 'definition_set_sha256':hashlib.sha256(json.dumps(definition,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 'scored_evidence_count':256,'evidence':evidence,'files':sources,'models_loaded':0,'rescoring_calls':0,
 'frozen_experiment_code_modified':False,'old_source_files_modified':False})
assets={}
def add(path,expected,role,record,rehash=True,extra=None):
 p=Path(path);key=str(p);exists=p.is_file();size=p.stat().st_size if exists else None
 if key in assets:
  assert assets[key]['expected_sha256']==expected
  assets[key]['roles']=sorted(set(assets[key]['roles']+[role]));assets[key]['evidence']=sorted(set(assets[key]['evidence']+[record]))
  if extra:assets[key].update(extra)
  return
 actual=sha(p) if exists and rehash else None
 if rehash:assert exists and actual==expected,'Protected small asset changed '+key
 assets[key]={'path':key,'windows_path':key.replace('/mnt/e/','E:/'),'expected_sha256':expected,
  'actual_sha256_this_packaging_pass':actual,'bytes':size,'exists_at_inventory':exists,'roles':[role],'evidence':[record],
  'verification':'FRESH_SHA256_MATCH' if rehash else 'REUSED_VERIFIED_BINDING_HASH_NO_WEIGHT_REHASH',
  'copied_to_source_snapshot':False}
 if extra:assets[key].update(extra)
for p,h in m['protected_hashes'].items():
 weight=Path(p).suffix.lower() in ['.pt','.pth','.ckpt','.safetensors']
 add(p,h,'protected_source_asset',str(mp)+'#protected_hashes',not weight)
base=m['source_binding']
add(base['path'],base['sha256'],'FULL_S6000_actual_base_generator',str(mp)+'#source_binding',False,
 {'tensor_state_sha256':base['generator_state_sha256'],'EMA':base['EMA'],'local_step':base['local_step'],'global_step':base['global_step']})
dp=OUT/'dispatch/calibration/d0/completion.json';d0=read(dp)['identity']['official_D0']
add(d0['path'],d0['sha256'],'official_D0_actual_single_pass_denoiser',str(dp)+'#identity.official_D0',False,{'registered_bytes':d0['bytes']})
assert assets[d0['path']]['bytes']==d0['bytes']
add(d0['source_registration'],d0['source_registration_sha256'],'retained_D0_source_registration',str(dp)+'#identity.official_D0.source_registration')
for q in m['evaluation_queries']:
 add(q['path'],q['sha256'],'frozen_surface_query',str(mp)+'#evaluation_queries')
 if 'condition_source' in q: add(q['condition_source'],q['condition_source_sha256'],'source_C_and_source_winding_metadata',str(mp)+'#evaluation_queries')
for c in m['train_conditions']:
 st=c['source_task']
 for pk,hk,role in [('annotation_source','annotation_sha256','source_region_annotation'),('quality_path','quality_sha256','source_quality_record'),('image','image_sha256','source_region_review_image')]:
  if pk in st and hk in st:add(st[pk],st[hk],role,str(mp)+'#train_conditions.source_task')
for rel in ['registration.json','inputs/manifest.json','inputs/training_manifest.json','gate_contract.json','calibration_source_lock.json','training_source_lock.json','training_config.json','USER_PROMPT.txt','user_budget_decision.json']:
 p=OUT/rel;add(p,sha(p),'current_frozen_control_record',str(p))
noise=[]
for partition,rows,record in [('calibration',m['calibration_requests'],mp),('final',m['evaluation_requests'],mp),('training',t['training_requests'],tp)]:
 for q in rows:
  item={k:q[k] for k in ['request_id','condition_id','parent_id','N','K','condition_source','condition_source_sha256','input_path','input_sha256','transition_noise_path','transition_noise_sha256','C_sha256','Gaussian_sha256','free_Gaussian_sha256','transition_Gaussian_sha256'] if k in q}
  item.update(partition=partition,hash_evidence=str(record),verification='FROZEN_MANIFEST_HASH_REUSED_NO_TENSOR_REHASH_THIS_PACKAGING_PASS',
   input_exists=Path(q['input_path']).is_file(),transition_noise_exists=Path(q['transition_noise_path']).is_file())
  assert item['input_exists'] and item['transition_noise_exists'],'Missing frozen noise '+q['request_id']
  noise.append(item)
write(DEST/'frozen_request_noise_hashes.json',{'status':'COMPLETE','version':'frozen_request_noise_inventory_v1','rows':noise,'count':len(noise),
 'counts_by_partition':{'calibration':64,'final':44,'training':512},'note':'Expected file and array hashes are copied from frozen manifests. Files checked for presence, not rehashed here. No noise drawn.'})
ar=[assets[k] for k in sorted(assets)]
write(DEST/'protected_assets_manifest.json',{'status':'COMPLETE','version':'protected_asset_hash_inventory_v1',
 'current_manifest':ref(mp),'current_training_manifest':ref(tp),'d0_execution_evidence':ref(dp),'base_binding':base,'official_D0_binding':d0,
 'asset_count':len(ar),'assets':ar,'counts':{'fresh_hash_verified':sum(x['verification']=='FRESH_SHA256_MATCH' for x in ar),
 'reused_large_weight_hashes':sum(x['verification']=='REUSED_VERIFIED_BINDING_HASH_NO_WEIGHT_REHASH' for x in ar)},
 'limitations':'Large D0/base checkpoint hashes reused from frozen bindings and completed execution evidence; only existence/size checked here. Not a new full-file weight hash pass.',
 'official_generator_ancestor':'Not a new runtime input: this pilot starts from FULL_S6000, not fresh official initialization. Historical EXP67/official ancestors are not restored or packaged.',
 'no_model_calls':True,'no_rescoring':True,'no_large_weight_rehash':True,'no_zip_created':True})
cp=DEST/'scorers_manifest.csv'
if not cp.exists():
 with cp.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(sources[0]));w.writeheader();w.writerows(sources)
rp=DEST/'README.md'
if not rp.exists():rp.write_text("""# Actual scoring source and asset provenance

The scorers/ tree contains the 19 exact small sources named in the saved calibration
POST identity.definition_hashes. All 256 saved POST identities have the same set.
Source and copied bytes were checked against those hashes; no scorer/model ran.

scorers_manifest.json and .csv map original absolute/project-relative paths to
copied paths and hashes. The external MeshFlow-Control tree remains distinct.

protected_assets_manifest.json freshly verifies small conditions, surface queries,
region metadata and frozen control records. FULL_S6000 and official D0 hashes
are reused from frozen bindings and completed execution evidence; only existence
and size are checked here, without re-reading the large weights.

frozen_request_noise_hashes.json records 64 calibration, 44 final and 512 training
request identities. Files were checked for presence; hashes are frozen-manifest
values, explicitly not a new tensor rehash.

This is packaging preparation only: no final ZIP, weight copy, model switch,
source edit, scoring, training or GPU calls.
""",encoding='utf-8')
print(json.dumps({'status':'COMPLETE','source_files':19,'source_bytes':sum(x['bytes'] for x in sources),
 'same_definition_POST_scores':256,'asset_count':len(ar),'request_noise_records':len(noise),
 'scorer_manifest_sha256':sha(DEST/'scorers_manifest.json'),'protected_manifest_sha256':sha(DEST/'protected_assets_manifest.json'),
 'request_manifest_sha256':sha(DEST/'frozen_request_noise_hashes.json')}))
