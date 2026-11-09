"""CPU-only portable archive for the immutable 256 learned raw outlet attempts."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
from ..io_utils import array_hash,file_hash,new_dir,save_npz,write_json
from ..exporter import RadialSupport
from ..legacy.common import sample_surface
from ..legacy.networks import point_features,DomainVelocity
from ..layout_student.train import model_sha256
from .ours_package import SOURCE_NAMES,read,require,compare,files_manifest,verify_files
from . import learned_outlets as learned

EXTRA_SOURCES=('system_probe/learned_outlets.py','system_probe/test_learned_outlets.py',
 'system_probe/learned_outlet_figures.py','system_probe/learned_outlets_package.py','system_probe/test_learned_outlets_package.py',
 'system_probe/warm_start.py','layout_student/deployment.py','layout_student/integrity.py','layout_student/evaluate.py',
 'layout_value/optimizer.py','layout_endpoint/__init__.py','layout_endpoint/train.py','layout_endpoint/rollout.py',
 'layout_fitdiag/__init__.py','layout_fitdiag/train.py','layout_longfit/__init__.py','layout_longfit/train.py',
 'layout_balanced/__init__.py','layout_balanced/train.py','layout_balanced/objective.py')


def inside(root,name):
 root=Path(root).resolve();p=(root/name).resolve();require(p.is_relative_to(root),'Path escapes archive root');return p


def copy_files(files,dest):
 before={name:{'sha256':file_hash(src),'bytes':src.stat().st_size} for name,src in files.items()}
 for name,src in files.items():
  target=inside(dest,name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
  require(file_hash(src)==file_hash(target)==before[name]['sha256'],'Source/copy digest differs')
 verify_files(dest,before);return before


def validate_slots(rows):
 expected={(s,f'{i:05d}',a) for s in (10,11) for i in range(64) for a in ('longfit','balanced')}
 keys=[(r['seed'],r['sample_id'],r['arm']) for r in rows]
 require(len(keys)==256 and len(set(keys))==256 and set(keys)==expected,'Missing or duplicate attempt slots')
 failed=[]
 for r in rows:
  require(r['raw_only'] and r['fallback_to_Fixed'] is False,'Raw-only contract')
  if r['metric_status']!='ok':
   require(r.get('metric_error') and r.get('metric_traceback') and not r.get('paired_available') and r.get('rms_gain_percent_two_measurements') is None,'Failed row lacks evidence or invents metric')
   failed.append([r['seed'],r['sample_id'],r['arm']])
 return failed


def verify_mesh(path,row):
 require(file_hash(path)==row['mesh_npz_sha256'],'Mesh file changed')
 with np.load(path,allow_pickle=False) as z:v=z['vertices'];f=z['faces']
 require(array_hash(v,f)==row['stored_array_hash'],'Stored typed mesh hash changed')
 require(array_hash(v.astype(np.float64),f.astype(np.int64))==row['canonical_geometry_hash'],'Canonical mesh hash changed')
 require(len(f)==row['quality']['face_count'],'Actual face count changed')


def portable(root):
 root=Path(root).resolve();run=root/'run';p=read(run/'RUN_PROTOCOL.json');reg=read(run/'MODEL_REGISTRY.json')
 results=read(run/'RESULTS.json');rows=results['rows'];failed=validate_slots(rows)
 require(results['model_registry_sha256']==file_hash(run/'MODEL_REGISTRY.json') and results['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json'),'Portable ledger identity')
 require(failed==read(run/'SUMMARY.json')['failed_slots'],'Failure slots differ')
 success=0
 for r in rows:
  compare(read(run/'attempts'/f"seed{r['seed']}"/r['sample_id']/r['arm']/'RESULT.json'),r,'Attempt row')
  require(r['model_identity']==reg['models'][r['model_name']],'Row model identity differs')
  if r['metric_status']=='ok':verify_mesh(inside(run,r['mesh_file']),r);success+=1
 for name,identity in reg['models'].items():
  kind=name.rsplit('_',1)[0];src=root/'source_models'/kind;path=src/'training'/identity['job']/'final.pt'
  require(file_hash(path)==identity['file_sha256'],'Portable fixed model changed')
  state=torch.load(path,map_location='cpu',weights_only=False)
  require(state['update']==240000 and state['is_predeclared_final'] and state['model_sha256']==model_sha256(state['model'])==identity['model_sha256'],'Portable final identity')
  require(state['protocol_sha256']==file_hash(src/'RUN_PROTOCOL.json')==identity['protocol_sha256'],'Portable source protocol')
  model=DomainVelocity().cpu();model.load_state_dict(state['model'],strict=True);require(sum(x.numel() for x in model.parameters())==29907,'Portable architecture');del model,state
 count=0
 for seed in (10,11):
  folder=root/'generation'/f'seed{seed}'/'samples';sm=read(folder/'SAMPLES.json')
  require(len(sm['records'])==64 and {r['sample_id'] for r in sm['records']}=={f'{i:05d}' for i in range(64)},'Portable fields missing')
  require(sm['fm_steps']==32,'Original8k sampling grid changed')
  for r in sm['records']:
   with np.load(inside(folder,r['path']),allow_pickle=False) as z:
    require(array_hash(z['noise'],z['z_standardized'],z['z'],z['nodes'])==r['array_hash'] and array_hash(z['nodes'])==r['nodes_hash'],'Portable saved field changed')
   count+=1
 inputs=read(root/'INFERENCE_INPUTS.json');require(len(inputs['records'])==128,'Proxy input count')
 for r in inputs['records']:
  path=inside(root,r['file']);require(file_hash(path)==r['file_sha256'],'Proxy input file changed')
  with np.load(path,allow_pickle=False) as z:
   require(array_hash(z['proxy_points'])==r['proxy_hash'] and array_hash(z['point_features'])==r['point_features_hash'],'Proxy input arrays changed')
   require(np.array_equal(point_features(z['proxy_points']),z['point_features']),'Proxy feature transform changed')
   require(z['proxy_points'].shape==(256,3) and z['base_u'].shape==(258,3) and z['faces'].shape==(512,3),'Proxy contract shape')
 return {'primary_fields':count,'model_inputs':128,'attempts':256,'readable_outputs':success,'failed_slots':failed,
  'four_final_model_tensor_hashes_verified':True,'new_model_forward_or_MC_geometry_sampling':False}


def verify_payload(root):
 root=Path(root).resolve();manifest=read(root/'FILE_MANIFEST.json');verify_files(root,manifest['files'],exclude=('FILE_MANIFEST.json',))
 return {'files':len(manifest['files']),'all_SHA256_and_bytes_match':True,**portable(root)}


def extract(archive,target,manifest_sha):
 target=new_dir(Path(target).resolve())
 with zipfile.ZipFile(archive) as z:
  require(len(z.namelist())==len(set(z.namelist())),'Duplicate ZIP member')
  for name in z.namelist():inside(target,name)
  require(z.testzip() is None,'CRC failure');z.extractall(target)
 require(file_hash(target/'FILE_MANIFEST.json')==manifest_sha,'Fresh manifest differs')
 return verify_payload(target)


def make_inputs(payload,rows):
 support=RadialSupport();records=[]
 for seed in (10,11):
  base=payload/'generation'/f'seed{seed}';sm=read(base/'samples/SAMPLES.json')
  fixed={r['sample_id']:r for r in read(base/'exports/fixed_MANIFEST.json')['records']}
  for r in sm['records']:
   sid=r['sample_id'];sf=base/'samples'/r['path']
   with np.load(sf,allow_pickle=False) as z:nodes=z['nodes']
   v,f=support.mesh(nodes);proxy=sample_surface(v,f,256,np.random.default_rng(4501))[0];features=point_features(proxy)
   matching=[x for x in rows if x['seed']==seed and x['sample_id']==sid and x['metric_status']=='ok']
   require(matching and all(x['deployment_metadata_not_formal_timing']['proxy_points_hash']==array_hash(proxy) for x in matching),'Derived proxy differs from production')
   with np.load(base/'exports'/fixed[sid]['mesh_path'],allow_pickle=False) as z:base_u=z['u'];faces=z['faces']
   rel=f'model_inputs/seed{seed}/{sid}.npz';save_npz(payload/rel,proxy_points=proxy,point_features=features,base_u=base_u,faces=faces)
   records.append({'seed':seed,'sample_id':sid,'file':rel,'file_sha256':file_hash(payload/rel),'proxy_hash':array_hash(proxy),
    'point_features_hash':array_hash(features),'saved_mainfield_sha256':file_hash(sf),'mainfield_nodes_hash':array_hash(nodes),
    'production_proxy_hash_matches_readable_arms':len(matching)})
 write_json(payload/'INFERENCE_INPUTS.json',{'records':records,'proxy_sampler_seed':4501,'network_inputs':'proxy256 features and evolving directions/time; no teacher labels',
  'deterministic_proxy_readback_against_production_hash':True,'new_latent_or_metric_MC_sampling':False,'new_network_forward':False})


def package(run,review,out):
 require(not torch.cuda.is_initialized(),'CPU package only');torch.set_num_threads(1)
 run=Path(run).resolve();review=Path(review).resolve();repo=Path(__file__).resolve().parents[3];out=new_dir(Path(out).resolve());started=time.perf_counter()
 try:
  checks=read(run/'CHECKS.json');require(checks['freeze_after_this_record'] and checks['metric_success_count']==254,'Run not final/frozen')
  before=files_manifest(run);write_json(out/'ORIGINAL_RUN_BEFORE.json',before)
  p=learned.verify_contract(run);reg=read(run/'MODEL_REGISTRY.json');result=read(run/'RESULTS.json');rows=result['rows'];failures=validate_slots(rows)
  require(failures==checks['runtime_failures'] and len(failures)==2,'Failure ledger changed')
  review_result=read(review/'REVIEW.json');require(review_result['results_sha256']==file_hash(run/'RESULTS.json') and review_result['source_sha256']==file_hash(review/'review_results.py'),'Independent review stale')
  require(checks['review_sha256']==file_hash(review/'REVIEW.json'),'Independent review changed')
  compare(learned.summarize(rows,learned.baseline_map(read(run/'REFERENCE_METRICS.json')['rows'])),read(run/'SUMMARY.json'),'Summary readback')
  parent=Path(p['generation_run']);old_pack=repo/'output/a_ae_fm_closed16_20260925_01_package_01';old_manifest=read(old_pack/'ORIGINAL_RUN_BEFORE.json')
  verify_files(parent,old_manifest)
  payload=out/'payload';payload.mkdir();tree_records={}
  for rel,source in {'run':run,'evidence/independent_review':review,'data':Path(read(parent/'RUN_PROTOCOL.json')['data'])}.items():
   entries=files_manifest(source);shutil.copytree(source,payload/rel);verify_files(source,entries);verify_files(payload/rel,entries);tree_records[rel]={'source':str(source),'files':entries}
  source_names=set(SOURCE_NAMES)|set(EXTRA_SOURCES);source_files={f'experiments/a_radial/{n}':repo/'experiments/a_radial'/n for n in sorted(source_names)}
  model_records={}
  for kind in ('longfit','balanced'):
   source=Path(p['model_declarations'][kind+'_10']['source_run']);mp=read(source/'RUN_PROTOCOL.json');dm=read(source/'DATA_MANIFEST.json')['files']
   require(file_hash(source/'DATA_MANIFEST.json')==mp['hashes']['data_manifest_sha256'],'Source training data manifest')
   files={n:inside(source,n) for n in dm}
   for n,digest in dm.items():require(file_hash(files[n])==digest,'Source training input changed')
   for n in ['RUN_PROTOCOL.json','DATA_MANIFEST.json','TRAINING_GATE.json']:files[n]=source/n
   if (source/'PARENT_WEIGHTS.json').exists():files['PARENT_WEIGHTS.json']=source/'PARENT_WEIGHTS.json'
   for seed in (10,11):
    ident=reg['models'][f'{kind}_{seed}'];_,actual=learned.validate_final(source,kind,seed);compare(actual,ident,'Registered fixed model')
    for n in [f'training/joint_{seed}/final.pt',f'training/joint_{seed}/TRAIN_REPORT.json',f'training/joint_{seed}/TRAIN_CONFIG.json',f'training/joint_{seed}/IMPORT_RECORD.json']:
     if (source/n).exists():files[n]=source/n
    state=torch.load(ident['path'],map_location='cpu',weights_only=False)
    for n,digest in state['config']['source_sha256'].items():
     local=repo/'experiments/a_radial'/n;require(file_hash(local)==digest,'Frozen parent trainer source changed');source_files['experiments/a_radial/'+n]=local
   model_records[kind]={'source':str(source),'files':copy_files(files,payload/'source_models'/kind)}
  generation_files={'RUN_PROTOCOL.json':parent/'RUN_PROTOCOL.json','TRAINING_COMPLETE.json':parent/'TRAINING_COMPLETE.json'}
  for seed in (10,11):
   for sub in ('samples','exports'):
    for f in (parent/f'seed{seed}'/sub).rglob('*'):
     if f.is_file():generation_files[f.relative_to(parent).as_posix()]=f
   for stage,step in [('ae',3200),('fm',8000)]:
    for name in [f'step{step:08d}.pt','CONFIG.json','COMPLETE.json']:
     n=f'seed{seed}/{stage}/{name}';generation_files[n]=parent/n
  generation_record=copy_files(generation_files,payload/'generation');source_record=copy_files(source_files,payload/'source')
  receipts={n:old_pack/n for n in ['SEALED_AFTER_RECOVERY.json','POSTSEAL_COPY_AUDIT.json','FAILED.json','RECOVERY_INVOCATION_ERRORS.json']}
  receipt_record=copy_files(receipts,payload/'evidence/original8k_seal')
  make_inputs(payload,rows)
  audit={'scope':'PORTABLE_INPUT_IDENTITY_AND_ARCHIVE_READBACK_NOT_NEW_GEOMETRY_EXPERIMENT','attempts':256,'readable_outputs':254,
   'failure_slots':failures,'statistical_denominators_each_arm':{'attempts':128,'continuous_metrics':127},
   'original8k_full_seal_manifest_unchanged':True,'tree_copies':tree_records,'model_copies':model_records,
   'generation_files':generation_record,'source_files':source_record,'original8k_seal_receipts':receipt_record,
   'new_network_forward':False,'new_MC_or_exact_geometry_evaluation':False,'deterministic_proxy_4501_reconstructed_and_production_hash_verified':True}
  write_json(out/'AUDIT.json',audit);write_json(payload/'AUDIT.json',audit)
  write_json(payload/'SOURCE_WHITELIST.json',{'files':source_record,'explicit_current_modules_plus_frozen_checkpoint_dependency_keys_only':True,
   'parallel_formal_margin_control_modules_excluded':True})
  write_json(payload/'PORTABLE_PATHS.json',{'run':'run','original_run':str(run),'generation':'generation','original_generation':str(parent),
   'models':'source_models/{longfit,balanced}','inputs':'model_inputs/seed{seed}/{sample_id}.npz','source':'source','data':'data',
   'original_absolute_JSON_paths_preserved':True,'generation8k_complete_archive_bound_by_receipt_not_duplicated':True})
  (payload/'README_zh.md').write_text('固定原AE3200/FM8000的128主场 × 两个学习出口，共256尝试的独立封存。两个原guard运行失败原样保留；每臂成功/资格分母128，连续误差只含127。原8k完整ZIP不嵌套，完整seal收据绑定原包；本包保留所有128主场noise/z/nodes、原三出口、AE/FM final、四个学习模型fullstate及其冻结训练输入/协议、显式源码。\n\nmodel_inputs保存同场固定4501采样的proxy256/原特征/base方向/连接；已逐一对生产proxy哈希，没有新网络推理、latent抽样、RMS或exact几何重算。绝对原路径不改，PORTABLE_PATHS说明副本位置。全量SHA/CRC/fresh解压验证仅证明归档和已保存数组身份，不是新的数值复现实验。\n\n这是同场支持网格参考下的闭集出口探索，无Fixed回退，没有类别泛化、官方完整后处理或外部baseline胜出资格。\n',encoding='utf-8')
  verify_files(run,before);verify_files(parent,old_manifest);manifest=files_manifest(payload)
  write_json(payload/'FILE_MANIFEST.json',{'schema':'LEARNED-OUTLETS-ARCHIVE-v1','files':manifest,'self_excluded':True});msha=file_hash(payload/'FILE_MANIFEST.json')
  archive=out/'LEARNED_OUTLETS_CLOSED16_FULL.zip'
  with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
   for name in sorted([*manifest,'FILE_MANIFEST.json']):z.write(payload/name,name)
  fresh=extract(archive,out/'fresh_extract',msha)
  code='from pathlib import Path; import experiments.a_radial.system_probe.learned_outlets as a; import experiments.a_radial.system_probe.learned_outlets_package as b; root=Path.cwd()/"source"; assert all(Path(x.__file__).resolve().is_relative_to(root.resolve()) for x in [a,b]); print("FRESH_SOURCE_IMPORT_PASS")'
  done=subprocess.run([sys.executable,'-B','-c',code],cwd=out/'fresh_extract',env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(out/'fresh_extract/source')),capture_output=True,text=True)
  require(done.returncode==0,'Fresh source import failed: '+done.stderr)
  verify_payload(out/'fresh_extract');require(file_hash(out/'fresh_extract/FILE_MANIFEST.json')==msha,'Fresh manifest mutation')
  verify_files(run,before);verify_files(parent,old_manifest)
  for rel,info in tree_records.items():
   verify_files(Path(info['source']),info['files']);verify_files(out/'fresh_extract'/rel,info['files'])
  for rel,record in source_record.items():require(file_hash(source_files[rel])==record['sha256']==file_hash(out/'fresh_extract/source'/rel),'Source copy/fresh mismatch')
  for kind,info in model_records.items():
   for rel,record in info['files'].items():require(file_hash(Path(info['source'])/rel)==record['sha256'],'Original fixed source input changed')
  receipt={'status':'SEALED','archive':archive.name,'archive_sha256':file_hash(archive),'archive_bytes':archive.stat().st_size,
   'file_manifest_sha256':msha,'all_CRC_pass':True,'fresh':fresh,'fresh_source_import':done.stdout.strip(),'run_files':len(before),
   'original_run_unchanged':True,'original8k_full_manifest_unchanged':True,'source_copy_fresh_hashes_match':True,
   'new_GPU_or_model_forward':False,'new_256_geometric_evaluation':False,'elapsed_seconds':time.perf_counter()-started}
  write_json(out/'SEALED.json',receipt);print(json.dumps(receipt),flush=True);return receipt
 except BaseException as exc:
  write_json(out/'FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc(),'original_runs_never_modified':True,'partial_files_retained':True});raise


def main():
 a=argparse.ArgumentParser();s=a.add_subparsers(dest='command',required=True)
 p=s.add_parser('package');p.add_argument('--run',required=True);p.add_argument('--review',required=True);p.add_argument('--out',required=True)
 p=s.add_parser('verify');p.add_argument('--payload',required=True);v=a.parse_args()
 if v.command=='package':package(v.run,v.review,v.out)
 else:print(json.dumps(verify_payload(v.payload)),flush=True)
if __name__=='__main__':main()
