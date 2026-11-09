"""CPU-only descriptive coverage inventory; never changes the active A/B pool."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
import argparse, csv, hashlib, json, time, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/r2_bench_exposure_allocation_v1/data_expansion'
COVERAGE=ROOT/'experiments/r2_bench_seat_coverage_v1/data/manifest.json'
HIGH=ROOT/'experiments/r2_bench_deployment_task_data_v1'
SHAPE=ROOT/'experiments/r2_bench_backend_shape_preservation_v1/data'
SELECTED=('28b9f2908f0cbb179c21c139c0613648','abf250a01f526137a035c800c58595d9')
SUPPORT={
 '35948e':('END_PANELS_WITH_BACK','两端板状支撑及长靠背；板内构造不从缩略图推断'),
 '4cd23a':('BOX_OR_LOOP_BASE','圆角端部与连续侧/底板形成箱形或环框基座'),
 '5f0e28':('END_PANELS','两端宽板支撑，中间悬空，端转角圆滑'),
 '6b894c':('L_FRAME_MULTI_LEGS','L形座面、两翼靠背与多根独立腿'),
 '7b58a3':('LOOP_END_BASE','并列座条由两端连续环框/板状基座承托'),
 '838c48':('TWO_PEDESTALS','有机轮廓座面由两处粗墩承托'),
 '85bfe7':('END_PANELS_WITH_BACK','窄座面、长靠背、两端板状支撑'),
 '8967e6':('SEPARATE_LEGS_WITH_BACK','平座面、高靠背与独立细腿；隐藏腿数不单凭视图认证'),
 '8d4693':('SEPARATE_LEGS','狭长座板与独立细腿'),
 '91e169':('CONTINUOUS_BLOCK_BASE','圆端座面置于连续箱体/实体式基座上'),
 '943dde':('END_PANELS_WITH_BACK','弯曲座背与两端板状支撑'),
 '98fe81':('TWO_END_PANELS','狭长座面与两块端板'),
 '9e429b':('END_PANELS_MULTI_BAY','三段弧形座区/分隔隆起，由两端宽板承托'),
 'ac1b9a':('SEPARATE_LEGS','矩形座面与四角独立腿'),
 'b6b10e':('ARCHED_END_PANELS','宽平座板与两端圆滑弯下的板状支撑'),
 'b77813':('LEGS_WITH_BACK','平座面、靠背与腿/端框支撑'),
 'bf504b':('ASYMMETRIC_PANEL_FRAME','座面端部凸耳、板状/框状支撑与两块抬高横板；精确支撑件分组UNKNOWN'),
 'c89d4c':('END_FRAME_WITH_BACK','内凹座面、背板和端部板/框支撑；底部隐藏连接UNKNOWN'),
 'c97683':('LOOP_END_BASE','上下长板由两端弧形板连续连接'),
 'dbd069':('LEGS_WITH_BACK','平座面、高靠背和独立腿'),
 'e01c28':('CURVED_CONTINUOUS_BASE','弯曲座条/座板与连续板式基座'),
 'ef207b':('BROAD_END_PANELS','宽座面与宽板式端部支撑/抬高横板；隐藏构件UNKNOWN'),
 'a8e987':('THIN_FOUR_LEG_FRAME','七根座条及周边横框由四根细腿承托'),
 '27eb79':('CURVED_SHELL_ARMS_LEGS','弯曲座背壳、两端扶手与独立杆腿'),
}
def resolve(p):
 s=str(p).replace('\\','/')
 if os.name=='nt' and s.startswith('/mnt/e/'):s='E:/'+s[7:]
 elif os.name!='nt' and s.startswith('E:/'):s='/mnt/e/'+s[3:]
 p=Path(s);return p if p.is_absolute() else ROOT/p
def read(p):return json.loads(resolve(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with resolve(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,Path):return str(v)
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 return v
def write(p,v):
 p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def arrays(p):
 with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def csvwrite(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,dict)) else clean(v) for k,v in r.items()})
def components(faces,mode='edge'):
 parent=list(range(len(faces)));index={}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 for i,f in enumerate(faces):
  keys=[tuple(sorted((int(f[j]),int(f[(j+1)%3])))) for j in range(3)] if mode=='edge' else list(map(int,f))
  for key in keys:
   if key in index:parent[find(i)]=find(index[key])
   else:index[key]=i
 groups=defaultdict(list)
 for i in range(len(faces)):groups[find(i)].append(i)
 return sorted(groups.values(),key=lambda g:(-len(g),g[0]))
def describe_case(z):
 C=z['C'];faces=z['source_faces'][z['known_ids']];norm=np.cross(C[:,1]-C[:,0],C[:,2]-C[:,0]);length=np.linalg.norm(norm,axis=1)
 axes=norm/np.maximum(length[:,None],1e-30)
 # Canonical per-face corner ordering can reverse winding; use the principal
 # unsigned normal axis, not a cancelling signed sum of canonical normals.
 _,basis=np.linalg.eigh((axes.T*length)@axes);normal_axis=basis[:,-1]
 angle=np.degrees(np.arccos(np.clip(np.abs(axes@normal_axis),0,1)))
 pts=C.reshape(-1,3).astype(np.float64);center=pts.mean(0);_,_,vh=np.linalg.svd(pts-center,full_matrices=False)
 edges=Counter(tuple(sorted((int(f[j]),int(f[(j+1)%3])))) for f in faces for j in range(3))
 return dict(C_edge_components=len(components(faces)),C_vertex_components=len(components(faces,'vertex')),C_indexed_open_edges=sum(n==1 for n in edges.values()),C_indexed_nonmanifold_edges=sum(n>2 for n in edges.values()),C_closed_indexed=not any(n==1 for n in edges.values()),C_axis_normal_angle_p95_deg=float(np.quantile(angle,.95)),C_axis_normal_angle_max_deg=float(angle.max()),C_best_plane_rms_model=float(np.sqrt(np.mean(((pts-center)@vh[-1])**2))),C_height_range_model=float(np.ptp(pts[:,1])),C_height_range_is_material_thickness=False)
def inventory():
 begin=time.time();m=read(COVERAGE);rows=m['train'];assert len(rows)==24 and len({r['parent_id'] for r in rows})==24
 protected={str(COVERAGE):sha(COVERAGE)};result=[];shapeimages={}
 for stem in ('train','validation'):
  p=SHAPE/'gallery'/f'{stem}.png';shapeimages[stem]={'path':str(p),'sha256':sha(p),'actually_viewed':True,'reviewer':'AI_PROXY_backend_data','nonblind':True,'scope':'Source-only whole-object support description, not generated output rating; no certification of hidden geometry.'}
 for r in rows:
  p=resolve(r['case_path']);h=sha(p);assert h==r['case_sha256'];protected[str(p)]=h;z=arrays(p)
  assert int(z['N'])==r['N'] and int(z['K'])==r['K']
  assert np.array_equal(np.sort(np.r_[z['known_ids'],z['free_ids']]),np.arange(r['N']))
  assert np.array_equal(z['C'],z['full_target'][z['known_ids']]) and np.array_equal(z['Y'],z['full_target'][z['free_ids']])
  qualitypath=r.get('quality_file') or p.parent/'quality.json';q=read(qualitypath);protected[str(resolve(qualitypath))]=sha(qualitypath)
  provenance=read(SHAPE/'sources'/r['parent_id']/'provenance.json');sourcecase=resolve(provenance['case_path'])
  sz=arrays(sourcecase);assert np.array_equal(z['full_target'],sz['full_target']), 'source contact sheet geometry differs'
  image=shapeimages['train' if provenance['split']=='TRAIN' else 'validation'];prefix=r['parent_id'][:6]
  support,notes=SUPPORT.get(prefix,('UNKNOWN','No reliable source support annotation'))
  desc=r.get('extent_description','UNKNOWN');lower=desc.lower()
  family='SLATTED_OR_SEPARATE_BOARDS' if any(s in lower for s in ('slat','木条','座条','strips')) else ('CURVED_SEATING_SURFACE' if any(s in lower for s in ('curv','bowed','bowl','弧','弯曲')) else ('PLANAR_OR_POLYGONAL_SEAT' if any(s in lower for s in ('plane','矩形','l形','polygon','上表面','平')) else 'OTHER_COHERENT_SEAT'))
  geom=describe_case(z);interface='CLOSED_C' if geom['C_closed_indexed'] else ('SOURCE_LEGAL_SHARED_EDGES' if np.asarray(z.get('source_supported_labels',[])).sum() else 'NO_SOURCE_LEGAL_SHARED_EDGE_RECORDED')
  caveats=[]
  if prefix=='a8e987':caveats=['Original nonconforming point/T-junction contacts retained; not manifold-certified.'];interface='MIXED_SHARED_EDGE_AND_NONCONFORMING_CONTACTS'
  result.append(dict(parent_id=r['parent_id'],condition_id=r['condition_id'],replay_group=r['replay_group'],N=r['N'],K=r['K'],case_path=str(p),case_sha256=h,C_shape_family=family,C_shape_description=desc,source_support_family=support,source_support_observation=notes,source_support_evidence=image,source_support_geometry_identity_verified=True,source_interface_status=interface,source_owner_count=int(np.asarray(z.get('occupied_labels',[])).sum()),source_supported_count=int(np.asarray(z.get('source_supported_labels',[])).sum()),source_alias_count=int(np.asarray(z.get('source_alias_labels',[])).sum()),source_null_count=int(len(z.get('occupied_labels',[]))-np.asarray(z.get('source_supported_labels',[])).sum()),source_interface_evidence=str(resolve(qualitypath)),source_limits=caveats,labels_are_model_input=False,semantic_authority='INHERITED_C_REVIEW_PLUS_ACTUAL_SOURCE_ONLY_AI_PROXY_SUPPORT_REVIEW',**geom))
 csvwrite(OUT/'current24_relations.csv',result);write(OUT/'current24_relations.json',{'rows':result,'source_images_actually_viewed':list(shapeimages.values()),'current_pool_mutated':False,'label_scope':'Data curation/analysis only; Y-derived support descriptors never enter model input.'})
 write(OUT/'current24_manifest_check.json',{'status':'PASS','parents':24,'conditions':24,'old_parents':sum(r['replay_group']=='OLD' for r in rows),'new_parents':sum(r['replay_group']=='NEW' for r in rows),'manifest':str(COVERAGE),'manifest_sha256':sha(COVERAGE),'all_case_hashes_matched':True,'C_Y_exact_partition_and_target_coordinates':True,'source_contactsheet_same_full_target':True,'protected_hashes':protected,'elapsed_seconds':time.time()-begin,'model_calls':0,'training_pool_changes':0})
 print(json.dumps({'status':'CURRENT24_READY','rows':len(result),'C_shape_counts':dict(Counter(r['C_shape_family'] for r in result)),'elapsed':time.time()-begin}),flush=True)

def shortlist():
 splitpath=HIGH/'split_audit/new_data_split.json';s=read(splitpath);current={r['parent_id'] for r in read(COVERAGE)['train']}
 rejected={'27eb79ddf26fd7ac7898580cb9d16864':'CURRENT_TRAIN_AND_PREVIOUS_RECONSTRUCTION','9452149917cb2e27f4fb4dee5181bee':'PREVIOUS_FIXED_REBUILD_BLOCKED_NO_REPEAT','5856737e6fc90bbac1a25f36f85c3fd6':'PREVIOUS_BOX_REBUILD_FAILED_FIDELITY_NO_REPEAT'}
 rows=[]
 for r in s['rows']:
  pid=r['parent_id'];decision='NOT_SELECTED_NO_MATURE_LOW_N_RECONSTRUCTION'
  if pid in SELECTED:decision='SELECTED_SOURCE_CY_ONLY_LOW_N_PENDING'
  elif pid in current:decision='EXCLUDE_CURRENT_TRAIN'
  elif 'UNSEEN' in r['split_role']:decision='PROTECT_FROZEN_HOLDOUT'
  elif 'HOLD' in r['split_role']:decision='HOLD_SOURCE_COMPLETENESS'
  elif pid in rejected:decision=rejected[pid]
  rows.append({**r,'this_round_decision':decision,'current_train_overlap':pid in current,'new_LOW_N_model_ready':False,'current_historical_exposure_scope':'Inherited20261022 audit96parent union; not official checkpoint seen/unseen certification','current_source_geometry_mutated':False})
 assert not (set(SELECTED)&current)
 chosen=[r for r in rows if r['parent_id'] in SELECTED];assert len(chosen)==2 and len({r['visual_family_id'] for r in chosen})==2
 write(OUT/'candidate_preregistration.json',{'status':'FROZEN_BEFORE_ANY_NEW_C_Y_CONSTRUCTION','selection_rule':'Trusted existing complete-source review; not current24 or oldDEV/df6 or reserved617/756; one per visual family; at most4independent sourceUIDs; no generated badcase selection; do not replay failed27/945/585 rebuild','selected_parent_ids':list(SELECTED),'selected':chosen,'not_selected': [r for r in rows if r['parent_id'] not in SELECTED],'source_split_path':str(splitpath),'source_split_sha256':sha(splitpath),'proposal':'One deterministic existing-source surface selection per parent, preserving every full-target face and source topology. No lowN reconstruction algorithm presently verified for these; keep N384/N308 source tasks PENDING, never add to A/B.','construction_scope':'Only extract full upper seating faces from already recorded seat components; confirm all components covered by actual view. No simplifier, mesh generation, new noise, target face deletion or model calls.','N_contract':[128,256],'maximum_new_parent_attempts':4,'actual_selected':2,'training_pool_change':False})
 csvwrite(OUT/'candidate_registry.csv',rows)
 print(json.dumps({'selected':list(SELECTED),'lowN_ready':0,'scope':'PREP_ONLY_NOT_A_B_DATA'}),flush=True)

def inspect_sources():
 from r2.data_tools.bench_parts_v1.construct import planar_regions
 old=read(HIGH/'dataset/source_tasks_pending_adapter/manifest.json')['rows'];rows=[]
 for pid in SELECTED:
  r=next(r for r in old if r['parent_id']==pid and r['condition_id'].endswith('__R1'))
  a=arrays(r['source_case_path']);ids=set(r['source_face_ids']);regions=planar_regions(a)
  selected=[]
  for g in regions:
   if set(g['ids']).issubset(ids):selected.append({**g,'normal':g['normal'].tolist(),'center':g['center'].tolist(),'span':g['span'].tolist(),'low':g['low'].tolist(),'high':g['high'].tolist()})
  row=dict(parent_id=pid,source_case_path=r['source_case_path'],source_sha256=sha(r['source_case_path']),seat_component_face_ids=sorted(ids),seat_components=components(a['source_faces'][sorted(ids)]),planar_regions=selected)
  rows.append(row)
  print(json.dumps({'parent':pid,'seat_components':len(row['seat_components']),'planar_regions':len(selected)},ensure_ascii=False))
 write(OUT/'source_inspection.json',{'rows':rows,'scope':'Existing source-seat component planar-region descriptor only; no geometry modification.'})

def construct():
 from r2.data_tools.bench_parts_v1.construct import planar_regions,partition
 from r2.data_tools.bench_parts_v1.quality import SourceAuditor
 begin=time.time();prereg=read(OUT/'candidate_preregistration.json');assert tuple(prereg['selected_parent_ids'])==SELECTED
 manifest=OUT/'candidate_manifest.json'
 if manifest.exists():
  for r in read(manifest)['candidates']:assert sha(r['case_path'])==r['case_sha256']
  print('ALREADY_CONSTRUCTED_NO_OVERWRITE');return
 old=read(HIGH/'dataset/source_tasks_pending_adapter/manifest.json')['rows'];rows=[];protected={str(COVERAGE):sha(COVERAGE)}
 for pid in SELECTED:
  r=next(r for r in old if r['parent_id']==pid and r['condition_id'].endswith('__R1'))
  source=resolve(r['source_case_path']);assert sha(source)==r['source_case_sha256']
  receipt=resolve(r['source_finite_receipt_path']);assert sha(receipt)==r['source_finite_receipt_sha256']
  protected[str(source)]=sha(source);protected[str(receipt)]=sha(receipt);a=arrays(source)
  seat=np.array(r['source_face_ids'],np.int64);groups=[seat[g] for g in components(a['source_faces'][seat])]
  regions=planar_regions(a);picked=[];choices=[]
  # Existing source components define the complete seat assembly. On each
  # component take its broadest upward original planar face group, excluding
  # vertical thickness, underside and bevel strips; do not invent faces.
  for gi,g in enumerate(groups):
   cand=[x for x in regions if set(x['ids']).issubset(set(g)) and x['normal'][1]>1e-5]
   if not cand:raise RuntimeError('NO_UPWARD_SOURCE_REGION:'+pid+':'+str(gi))
   chosen=max(cand,key=lambda x:(x['area'],x['center'][1],-min(x['ids'])))
   picked+=chosen['ids'];choices.append(dict(component_index=gi,component_face_ids=g.tolist(),selected_face_ids=chosen['ids'],normal=chosen['normal'],area_normalized=chosen['area'],center_normalized=chosen['center']))
  ids=np.array(sorted(set(picked)),np.int64);assert len(choices)==(5 if pid.startswith('28b') else 22)
  pair=partition(a,ids);cid=pid+'__ALL_SEAT_BOARD_UPPER_SURFACES__EXPOSURE_PREP_V1'
  q,sup=SourceAuditor(a,protected[str(source)]).candidate(ids,dict(candidate_id=cid,parent_id=pid,group='PREPARATION_ONLY',proposed_part_type='SEAT_SURFACE'))
  case={**a,**pair,**sup};known_vertices=set(case['C_source_vertex_ids'].tolist())
  case['source_third_in_C']=np.array([bool(ok and tr[2] in known_vertices) for ok,tr in zip(case['source_supported_labels'],case['edge_source_triple'])])
  dest=OUT/'source_candidates'/pid;dest.mkdir(parents=True,exist_ok=True)
  with (dest/'case.npz').open('xb') as f:np.savez_compressed(f,**case)
  with (dest/'input.npz').open('xb') as f:np.savez_compressed(f,**{k:case[k] for k in ('C','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_corner_permutations','N','K','target_version_id')})
  write(dest/'quality.json',q);write(dest/'selection.json',dict(rule='Per already-reviewed seat component, broadest original planar region with positive source normal Y; no target edit or simplification',component_choices=choices,selected_source_face_ids=ids,seat_components_covered=len(choices),source_winding_used=True))
  assert np.array_equal(case['full_target'],a['full_target'])
  assert np.array_equal(np.sort(np.r_[case['known_ids'],case['free_ids']]),np.arange(len(a['full_target'])))
  row=dict(parent_id=pid,condition_id=cid,N=int(case['N']),K=int(case['K']),status='PENDING_LOW_N_RECONSTRUCTION',native_model_ready=False,SEAM_model_ready=False,in_AB_training=False,C_scope='Broad upper face of every original seat board; thickness/underside/bevels stay Y',source_target_unchanged=True,source_case_path=str(source),source_case_sha256=sha(source),source_finite_receipt_path=str(receipt),source_finite_receipt_sha256=sha(receipt),source_full_reaudit=False,case_path=str(dest/'case.npz'),case_sha256=sha(dest/'case.npz'),input_path=str(dest/'input.npz'),input_sha256=sha(dest/'input.npz'),quality_path=str(dest/'quality.json'),quality_sha256=sha(dest/'quality.json'),geometry_status=q['geometry_status'],geometry_reasons=q['reasons'],source_owner_count=int(case['occupied_labels'].sum()),source_alias_count=int(case['source_alias_labels'].sum()),source_null_count=int((~case['source_supported_labels']).sum()),target_components=len(components(a['source_faces'])),seat_boards_covered=len(choices),source_semantic_review_status='PENDING_NEW_C_Y_IMAGE_REVIEW',N_contract=[128,256],N_over_budget=int(case['N'])-256,low_N_blocker='Existing boundary-preserving planar method removed no eligible interior vertices; prior generic QEM held for fidelity. No verified mature replacement in this task. Preserve full source and remain PENDING.',**describe_case(case))
  rows.append(row);print(json.dumps({k:row[k] for k in ('parent_id','N','K','geometry_status','source_owner_count','source_alias_count','source_null_count','status')}),flush=True)
 for p,h in protected.items():assert sha(p)==h
 write(manifest,dict(schema='BENCH_RELATION_COVERAGE_PREP_V1',candidates=rows,train=[],ready_parents=0,source_prep_parents=len(rows),preregistration_path=str(OUT/'candidate_preregistration.json'),preregistration_sha256=sha(OUT/'candidate_preregistration.json'),active_AB_pool_unchanged=True,model_calls=0,training_updates=0))
 csvwrite(OUT/'source_candidates.csv',rows)
 write(OUT/'construction_receipt.json',dict(status='SOURCE_C_Y_PREPARED_LOW_N_PENDING',parent_attempts=2,source_C_Y_partitions=2,new_ready_parents=0,full_target_reaudits=0,new_C_interface_audits=2,model_calls=0,training_updates=0,protected_hashes=protected,elapsed_seconds=time.time()-begin,engineering_events=[{'kind':'import_path_failure_before_geometry','details':'First read-only inspect_sources invocation lacked repository root on sys.path; fixed script import path. No generated assets/model call before failure.'}]))

def render():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from mpl_toolkits.mplot3d.art3d import Poly3DCollection
 rows=[]
 for r in read(OUT/'candidate_manifest.json')['candidates']:
  a=arrays(r['case_path']);whole=a['model_vertices'][a['source_faces']];C=whole[a['known_ids']];Y=whole[a['free_ids']]
  fig=plt.figure(figsize=(16,8));p=OUT/'gallery'/(r['parent_id']+'.png');p.parent.mkdir(parents=True,exist_ok=True)
  lo=whole.reshape(-1,3).min(0);hi=whole.reshape(-1,3).max(0);mid=(lo+hi)/2;half=(hi-lo).max()*.54
  views=[(28,-60),(12,50)]
  for vi,(elev,azim) in enumerate(views):
   for j,title in enumerate(('M: unchanged complete source','C: all broad board upper faces','Y: exact source complement','C + Y: exact union')):
    ax=fig.add_subplot(2,4,vi*4+j+1,projection='3d')
    items=[(whole,'#b3b8bf')] if j==0 else ([(C,'#189ac9')] if j==1 else ([(Y,'#b3b8bf')] if j==2 else [(Y,'#b3b8bf'),(C,'#189ac9')]))
    # One collection sorts all triangles together. Separate C/Y collections can
    # hide a top C surface behind an entire Y object by collection z-order.
    triangles=np.concatenate([x[0] for x in items]);colors=[color for tri,color in items for _ in tri]
    poly=Poly3DCollection(triangles[:,:,[0,2,1]],facecolor=colors,edgecolor='#4d545f',linewidth=.16,alpha=1)
    ax.add_collection3d(poly)
    ax.set_xlim(mid[0]-half,mid[0]+half);ax.set_ylim(mid[2]-half,mid[2]+half);ax.set_zlim(mid[1]-half,mid[1]+half)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=elev,azim=azim);ax.set_axis_off();ax.set_title(title,fontsize=10)
  fig.suptitle(f"{r['parent_id'][:8]} | N={r['N']} K={r['K']} | source only | PENDING: N exceeds 256 | {r['seat_boards_covered']} seat boards",fontsize=13)
  fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.92,wspace=0,hspace=-.15);fig.savefig(p,dpi=150,facecolor='white');plt.close(fig)
  rows.append(dict(parent_id=r['parent_id'],image_path=str(p),image_sha256=sha(p),case_sha256=r['case_sha256'],actually_viewed=False))
 write(OUT/'gallery_manifest.json',dict(rows=rows));print(json.dumps({'rendered':len(rows),'model_calls':0}),flush=True)

def lowN_preregister():
 from r2.data_tools.bench_three_family_v1 import boundary_simplify as B
 m=read(OUT/'candidate_manifest.json');old=read(HIGH/'asset_inventory/high_source_candidates.json')['rows'];versions=[]
 for r in m['candidates']:
  s=next(x for x in old if x['parent_id']==r['parent_id'])
  for variant in ('agg2','agg7'):
   p=resolve(s[variant+'_target_case_path']);a=arrays(p);assert sha(p)==s[variant+'_target_case_sha256']
   versions.append(dict(parent_id=r['parent_id'],variant=variant,path=str(p),sha256=sha(p),N=len(a['source_faces']),previous_reasons=s[variant+'_reasons'],previous_fidelity=s[variant+'_fidelity'],rerun=False))
 p=OUT/'lowN_proposal_preregistration.json'
 payload=dict(status='FROZEN_BEFORE_SHARED_BOUNDARY_PROPOSALS',parents=list(SELECTED),attempts_per_parent=1,method=B.VERSION,method_path=str(Path(B.__file__)),method_sha256=sha(B.__file__),constants=dict(normal_dot_min=B.NORMAL_DOT_MIN,plane_tol_L=B.PLANE_TOL_L,line_tol_L=B.LINE_TOL_L),coordinates='unchanged original vertices',labels='current C membership and each original source edge component; no cross-region remeshing',prior_methods_reused_not_repeated=versions,not_previously_applied_to_selected_parents=True,prior_boundary_method_scope=['27eb79','945214','585673'],admission=dict(N=[128,256],area_ratio=[.98,1.02],max_bidirectional_15bary_probe_L=.0025,component_topology_preserved=True,source_geometry_no_illegal_or_unknown=True),no_fallback_or_parameter_search=True,scope='A single mature shared-collinear-boundary method, distinct from previous interior-only planar method and QEM; if N remains outside range retain proposal BLOCKED. No A/B data changes.')
 if p.exists():assert read(p)==payload
 else:write(p,payload)
 print(json.dumps({'preregistered':2,'method':B.VERSION,'prior_lowN_versions':[{k:r[k] for k in ('parent_id','variant','N','previous_reasons')} for r in versions]}),flush=True)

def lowN_build():
 from r2.data_tools.bench_three_family_v1.boundary_simplify import simplify_boundaries
 from r2.data_tools.bench_three_family_v1.targets import check
 from r2.data_tools.bench_user_v2 import build as b
 begin=time.time();frozen=read(OUT/'lowN_proposal_preregistration.json');assert sha(frozen['method_path'])==frozen['method_sha256']
 before=sha(COVERAGE);rows=[]
 for r in read(OUT/'candidate_manifest.json')['candidates']:
  dest=OUT/'lowN_proposals'/r['parent_id'];receipt=dest/'receipt.json'
  if receipt.exists():
   old=read(receipt)
   for p,h in old['hashes'].items():assert sha(p)==h
   rows.append(old);continue
  a=arrays(r['source_case_path']);case=arrays(r['case_path']);labels=np.zeros(len(a['source_faces']),np.int64)
  for ci,g in enumerate(components(a['source_faces'])):labels[g]=2*ci
  labels[case['known_ids']]+=1
  new,lineage,ledger=simplify_boundaries(a,labels,L=b.Q.L)
  target=b.canonical(new,new['source_faces']);target['target_version_id']=np.asarray('EXPOSURE_PREP_SHARED_BOUNDARY_V1')
  dest.mkdir(parents=True,exist_ok=True)
  with (dest/'target_case.npz').open('xb') as f:np.savez_compressed(f,**target)
  write(dest/'lineage.json',dict(method=frozen['method'],source_case_path=r['source_case_path'],source_sha256=r['source_case_sha256'],new_face_to_source_region=lineage,mapping_is_region_not_source_face_identity=True,ledger=ledger))
  unchanged=len(lineage)==len(a['source_faces']) and all(len(x)==1 for x in lineage) and np.array_equal(new['model_vertices'],a['model_vertices']) and np.array_equal(new['source_faces'],a['source_faces'][[x[0] for x in lineage]])
  parent=dict(parent_id=r['parent_id'],source=dict(clean_case_path=r['source_case_path'],source_geometry_path=r['source_finite_receipt_path']))
  quality=check(parent,target,lineage,dest,labels,unchanged=unchanged)
  result=dict(parent_id=r['parent_id'],method=frozen['method'],N_before=len(a['source_faces']),N_after=len(target['source_faces']),method_summary=ledger['summary'],**quality,status='BLOCKED_N' if not quality['current_N_pass'] else ('PENDING_ACTUAL_REVIEW' if quality['geometry_pass'] else 'BLOCKED_GEOMETRY'),source_parent_hash_after=sha(r['source_case_path']),active_training_pool_mutated=False)
  assert result['source_parent_hash_after']==r['source_case_sha256']
  result['hashes']={str(p):sha(p) for p in dest.iterdir() if p.is_file()};write(receipt,result);rows.append(result)
  print(json.dumps({k:result[k] for k in ('parent_id','N_before','N_after','status','geometry_reasons','method_summary')}),flush=True)
 assert sha(COVERAGE)==before
 write(OUT/'lowN_proposal_summary.json',dict(rows=rows,actual_parent_attempts=len(rows),new_ready_parents=sum(r['status']=='READY' for r in rows),elapsed_seconds=time.time()-begin,active_manifest_unchanged=True,model_calls=0,parameters_searched=0))

def finalize():
 inv=read(OUT/'current24_manifest_check.json');m=read(OUT/'candidate_manifest.json');low=read(OUT/'lowN_proposal_summary.json');gallery=read(OUT/'gallery_manifest.json')
 for p,h in inv['protected_hashes'].items():assert sha(p)==h,p
 for p,h in read(OUT/'construction_receipt.json')['protected_hashes'].items():assert sha(p)==h,p
 reviews=[]
 for r in m['candidates']:
  g=next(x for x in gallery['rows'] if x['parent_id']==r['parent_id']);assert sha(g['image_path'])==g['image_sha256']
  note=('两视角能看到五条完整上表面及真实间隙，Y保留各座条厚度/倒角与两端箱墩；属于并列宽座条配双墩，不是a8细腿框的复制。' if r['parent_id'].startswith('28b') else '两视角能看到沿前后边缘弯下的并列窄条和双墩支撑；C覆盖每条的宽上表面，Y保留厚度与支撑。它是弧形座条装配，不是27连续弯曲座背壳。')
  reviews.append(dict(parent_id=r['parent_id'],actually_viewed=True,reviewer='AI_PROXY_backend_data',nonblind=True,image_path=g['image_path'],image_sha256=g['image_sha256'],views=2,columns=['M','C','Y','C+Y'],semantic_decision='ACCEPT_SOURCE_BROAD_UPPER_BOARD_ASSEMBLY',complete_object_visual='PLAUSIBLE',C_full_broad_face_set='YES',source_region_observation=note,uncertainty='材质/真实制造装配方式UNKNOWN；所见不认证隐藏表面无交。精确源分区与几何合法性另由缓存审计记录。',not_model_ready=True))
 write(OUT/'source_visual_review.json',dict(rows=reviews,annotation_source='Actual view_image inspection of both final rendered images; not generated from metric values.',prior_render_fix='Before final review, C/Y switched to one triangle collection for correct joint painter sorting; no geometry changed.'))
 status=[]
 for r in m['candidates']:
  z=arrays(r['case_path']);gid={fi:ci for ci,group in enumerate(components(z['source_faces'])) for fi in group};owners=z['edge_owner_face'];owner_components=set(gid[int(i)] for i in owners if int(i)>=0);Ccomponents={gid[int(i)] for i in z['known_ids']}
  assert owner_components<=Ccomponents
  l=next(x for x in low['rows'] if x['parent_id']==r['parent_id'])
  status.append(dict(parent_id=r['parent_id'],source_N=r['N'],source_K=r['K'],C_face_ids=z['known_ids'].tolist(),source_owner_edges=r['source_owner_count'],source_alias_edges=r['source_alias_count'],source_null_edges=r['source_null_count'],source_owner_scope='All recorded C/Y source-owner edges connect upper board faces to the remainder of the same board component; they do not certify direct shared-index joins to the separate support components.',source_target_components=r['target_components'],distinct_source_components_with_C=len(Ccomponents),distinct_source_components_without_C=r['target_components']-len(Ccomponents),source_prep_status='ACCEPTED_SOURCE_CY_PREPARATION',lowN_status=l['status'],N_after_single_mature_proposal=l['N_after'],model_ready=False,new_training_inclusion=False,lowN_receipt_path=str(OUT/'lowN_proposals'/r['parent_id']/'receipt.json'),further_coverage_gap='No new continuous curved seat-back shell parent prepared.' if r['parent_id'].startswith('abf') else 'Different support structure from a8; not another thin-four-leg example.'))
 write(OUT/'data_status.json',dict(status='PREPARATION_COMPLETE_WITH_EXPLICIT_LOW_N_BLOCKS',active_pool=dict(parents=24,conditions=24,manifest_sha256=sha(COVERAGE),unchanged=True),source_candidates=status,new_native_ready_parents=0,new_SEAM_ready_parents=0,actual_lowN_proposal_attempts=2,model_calls=0,training_updates=0,source_review_path=str(OUT/'source_visual_review.json'),code_path=str(Path(__file__)),code_sha256=sha(__file__)))
 report='''# Bench 关系覆盖 CPU 数据准备

活动训练池仍为 **24 父 / 24 条完整座面 C（22 OLD + a8 + 27）**。本次没有改变 A/B 数据、模型、噪声或评测。24 条的 N/K、case hash、源面 ID 互斥完备分区及 C/Y 坐标一致性全部通过；来源图与当前完整目标逐位匹配。

## 当前覆盖与证据

`current24_relations.csv/json` 保存逐父 C 形态、主体支撑、来源接口、C 连通性和实际图证据。支撑说明来自两张实际看过的31父来源图，不能当成隐藏几何的无交认证。语义标签只用于整理，不输入模型。

按现有区域描述的保守分组：16 条平面/多边形座面、2 条木条/分离板、3 条弯曲座面、3 条其他完整形状（有机轮廓、圆端、多座区等）。这只是整理标签，不是正式结构聚类；部分边界可能重叠。a8保留原T-junction/共点接触，不能称为标准流形源。现有木条/曲面条件不等于已经充分覆盖独立支撑关系。

## 预选与实际制作

从已有完整来源审阅与独立父/家族清单预选两父，各制作一次 C/Y；未按生成坏例挑近邻，未把617/756留出父、旧DEV、df6或当前训练父加入。前史沿用2026-10-22的96父历史清单核查，不能据此声明官方预训练未见。

| 来源父 | 原完整N | 新C K | 结构与条件 | 来源owner | 当前状态 |
|---|---:|---:|---|---:|---|
| 28b9 | 384 | 10 | 五条宽座板的完整宽上表面；双箱墩 | 20 | 来源C/Y合格，低面BLOCKED |
| abf250 | 308 | 44 | 22条前后弯下的窄座条完整宽上表面；双墩 | 88 | 来源C/Y合格，低面BLOCKED |

每条座板厚度、倒角、底面及完整支撑都留在Y，没有造面、删困难部件、改变顶点或重新归一化。C/Y按源面ID精确分区，C几何检查通过，来源完整体的既有有限面对审计（非法和UNKNOWN均0）按hash复用。两张新图已实际AI_PROXY图审；材质与真实装配方法仍UNKNOWN。

**20/88条owner只通向同一座板余下表面，不直接表示座条已与下面的独立支撑共享索引连接。** 两源保留7/24个完整部件。abf250补的是弧形条板组合，不是27的连续座背碗壳；该关系覆盖缺口仍在。

## 低面数没有停在查标签

检查了每父的现成agg2/agg7版本及此前固定全部边界的平面重建。28b9的现成256面版本面积比0.9582，组件7→9；abf250的256面版本面积比0.9230，双向固定探针最大距离约1.49%L。两者已有失败判据，未重复跑QEM。

另找到未对这两父运行过的成熟 `SHARED_COLLINEAR_PATCH_BOUNDARY_V1`：不移动坐标，同步删相邻patch的共享共线边界点，保留接触、转角与语义区域。先登记默认容差（1e-7 L、normal dot ≥ 1-1e-10），再各执行一次，无参数扫描。

| 父 | 实际提案N | 可删顶点 | 不可删原因 | 几何/保结构检查 |
|---|---:|---:|---|---|
| 28b9 | 384→384 | 0 | 200个patch/语义交界、16个真实转角 | PASS，仍超N |
| abf250 | 308→308 | 0 | 192个patch/语义交界、16个真实转角 | PASS，仍超N |

固定方法约4.9秒完成；未减少面数，不能把结果当成可训练扩池。这只说明**当前已核验方法和固定容差没有合格简化动作**，不是宣称任意保形简化数学上不可能。没有继续改阈值、扫简化器、删木条、删墩或放宽128–256合同。

## 交付与边界

- `current24_manifest_check.json`：当前活动数据逐项核验与保护hash。
- `current24_relations.csv/json`：24父关系表。
- `candidate_preregistration.json`、`lowN_proposal_preregistration.json`：先登记后执行。
- `source_candidates/`、`candidate_manifest.json`：两份真实来源C/Y和可逆来源身份。
- `lowN_proposals/`：一次性提案、谱系、几何/拓扑/面积/固定距离检查、失败账本。
- `gallery/`、`source_visual_review.json`：M/C/Y/C+Y两视角与实际图审。
- `data_status.json`：最终状态，**新增可训练父0**，两份来源准备保留PENDING低面构造。

活动24父及所有被读取原资产hash复核不变。此准备与A/B训练完全独立；不改变训练预算，不称作已完成扩规模，也不承诺这两种结构能提高生成泛化。
'''
 (OUT/'REPORT.md').write_text(report,encoding='utf8')
 print(json.dumps({'status':'COMPLETE','current_pool':24,'prepared_source_parents':2,'new_ready_parents':0,'actual_visual_review_images':2,'lowN_attempts':2,'active_hash_unchanged':True}),flush=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['inventory','shortlist','inspect_sources','construct','render','lowN_preregister','lowN_build','finalize']);x=a.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 if x.stage=='inventory':inventory()
 elif x.stage=='shortlist':shortlist()
 else:globals()[x.stage]()
if __name__=='__main__':main()
