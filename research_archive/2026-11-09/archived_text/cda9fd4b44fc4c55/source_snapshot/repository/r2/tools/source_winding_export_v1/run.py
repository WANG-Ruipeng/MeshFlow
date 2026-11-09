"""Restore source C order then orient free on36 cached POSTs; CPU-only."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections, concurrent.futures, multiprocessing, resource, sys, time, traceback
from pathlib import Path
import numpy as np
from . import common as C
from .prepare import prepare,registration
from . import restoration as R
from r2.tools.c_restore_bounded_v1.orientation import orient_fixed_C
from r2.pilots.anchor_aware_denoiser_v1 import evaluation as E
from r2.tools.c_restore_coupling_v1.evaluation import opposite_reach
from native_t1 import postprocess_cli as PC

STAGES=('MODEL_POST','RESTORED_C_ONLY','ORIENTED')
INPUT_CORNER_KEYS=('corner_displacement','corner_displacement_vector','corner_target_input_ids','corner_vertex_ids','corner_status','corner_C_candidate_count')
def meta(r):return dict(E.meta(E.canonical_identity(r)),method=r['method'])
def case_dir(r):return C.OUT/'outputs'/r['method']/r['request_id']

def direction_counts(faces,K):
 def count(fs):
  edges=collections.defaultdict(list)
  for fi,f in enumerate(fs):
   for j in range(3):
    u,v=int(f[j]),int(f[(j+1)%3]);edges[tuple(sorted((u,v)))].append((fi,u,v))
  two=[x for x in edges.values() if len(x)==2 and x[0][0]!=x[1][0]]
  return dict(two_face_edges=len(two),same_direction_edges=sum(x[0][1:]==x[1][1:] for x in two),nonmanifold_edges=sum(len(x)>2 for x in edges.values()))
 cc=count(faces[:K]);whole=count(faces)
 return dict(C_C_local_two_face_edges=cc['two_face_edges'],C_C_local_same_direction_edges=cc['same_direction_edges'],C_C_local_nonmanifold_edges=cc['nonmanifold_edges'],whole_two_face_same_direction_edges=whole['same_direction_edges'])

def topology(a,faces,K,pairs):
 top,edges,reached=E.actual_valid_graph(a['vertices'],faces,K,pairs)
 top.update(opposite_reach(a['vertices'][faces],K,edges));top.update(direction_counts(faces,K))
 return top,edges,reached

def save_export(dest,name,a,faces,perm,rest,flips):
 tri=a['vertices'][faces];K=len(rest['C']);sc=R.permute_corner_data(a['source_corner_ids'],perm);sf=a['source_face_ids']
 assert C.same(faces,R.permute_corner_data(a['faces'],perm))
 assert C.same(tri,R.permute_corner_data(a['output'],perm))
 assert C.same(faces,a['corner_vertex_ids'][sf[:,None],sc])
 out=dict(vertices=a['vertices'],faces=faces,output=tri,C=tri[:K],free=tri[K:],known_faces=np.int64(K),input_faces=a['input_faces'],
  source_face_ids=sf,source_corner_ids=sc,corner_permutation_to_MODEL_POST=perm,free_face_flips=flips,
  C_original_source_vertex_ids=rest['C_original_source_vertex_ids'],C_original_source_face_ids=rest['C_original_source_face_ids'],
  C_canonical_source_vertex_ids=rest['C_canonical_source_vertex_ids'],C_source_corner_permutation=rest['P'],C_inverse_corner_permutation=rest['P_inverse'],
  vertex_source_input_corner_ids=a['vertex_source_input_corner_ids'],source_face_kept=a['source_face_kept'],
  source_face_reason=a['source_face_reason'],duplicate_of_source_face=a['duplicate_of_source_face'])
 for key in INPUT_CORNER_KEYS:
  out['input_'+key]=a[key]
  out['output_'+key]=a[key][sf[:,None],sc]
 C.save_npz(dest/(name+'.npz'),**out)
 PC.write_indexed_ply(C.checked(dest/(name+'.ply')),a['vertices'],faces)
 v,f=PC.read_indexed_ply(dest/(name+'.ply'));assert C.same(v,a['vertices']) and C.same(f,faces)
 return out

def process_one(r):
 tick=time.perf_counter();r=E.canonical_identity(r);dest=case_dir(r);rp=dest/'receipt.json'
 identity=dict(**meta(r),post_sha256=C.sha(r['post_path']),condition_sha256=C.sha(r['condition_source']),parent_sha256=C.sha(r['parent_prepared_path']),registration_sha256=C.sha(C.OUT/'registration.json'))
 if rp.exists():
  q=C.read(rp);assert q['status']=='PASS' and q['identity']==identity;C.verify(q['files']);return dict(path=str(rp),reused=True,seconds=0,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 started=C.OUT/'events'/r['method']/(r['request_id']+'_started.json')
 if dest.exists() or started.exists():raise RuntimeError('Incomplete previously attempted case; automatic replay prohibited')
 C.write(started,dict(identity=identity,utc=time.time()))
 a=C.payload(r['post_path']);cond=C.payload(r['condition_source']);parent=C.payload(r['parent_prepared_path']);d=C.read(r['diagnostic_path']);K=int(r['K'])
 assert C.same(a['output'][:K],cond['C']) and C.same(a['vertices'][a['faces']],a['output'])
 assert d['quality']['identity']['processed_sha256']==identity['post_sha256']
 pairs=C.read(d['quality']['pair_audit_path'])
 rest=R.restore_source_order(a['vertices'],a['faces'],K,cond,parent,source_face_ids=a['source_face_ids'],source_corner_ids=a['source_corner_ids'])
 stage0,edge0,_=topology(a,a['faces'],K,pairs)
 for key in ('C_F_valid_edge_count','C_F_valid_opposite_winding_edge_count','reached_free_faces','graph_valid_edges'):assert stage0[key]==d['topology'][key]
 assert np.isclose(stage0['opposite_winding_reached_free_area_fraction'],d['topology']['opposite_winding_reached_free_area_fraction'],atol=1e-13,rtol=1e-12)
 zero=np.zeros(len(a['faces']),bool)
 first=save_export(dest,'restored_C',a,rest['faces'],rest['face_corner_permutation'],rest,zero)
 C.csvwrite(dest/'source_C_face_mapping.csv',rest['face_rows'])
 C.write(dest/'source_restoration.json',rest['summary'])
 stage1,edge1,_=topology(a,rest['faces'],K,pairs)
 C.write(dest/'orientation_started.json',dict(utc=time.time(),restored_C_sha256=C.sha(dest/'restored_C.npz')))
 oriented=orient_fixed_C(a['vertices'],rest['faces'],K)
 perm=R.compose_corner_permutations(rest['face_corner_permutation'],oriented['face_corner_permutation'])
 final=save_export(dest,'oriented',a,oriented['faces'],perm,rest,oriented['flips'])
 assert C.same(final['C'],rest['C']) and C.same(final['faces'][:K],rest['faces'][:K])
 assert C.same(first['free'],a['output'][K:])
 for state in (first,final):
  assert C.same(state['vertices'],a['vertices']) and C.same(state['source_face_ids'],a['source_face_ids'])
  assert np.array_equal(np.sort(state['faces'],axis=1),np.sort(a['faces'],axis=1))
  assert np.array_equal(state['C_original_source_vertex_ids'],parent['original_source_faces'][cond['original_source_face_ids']])
 stage2,edge2,reached=topology(a,oriented['faces'],K,pairs)
 for stage in (stage1,stage2):
  for key in ('C_F_shared_edge_count','C_F_valid_edge_count','reached_free_faces','graph_valid_edges','boundary_edges','nonmanifold_edges'):assert stage[key]==stage0[key]
  assert np.isclose(stage['reached_free_area_fraction'],stage0['reached_free_area_fraction'],atol=1e-12,rtol=1e-12)
  assert stage['C_C_local_same_direction_edges']==0
 for name,edges in zip(STAGES,(edge0,edge1,edge2)):C.csvwrite(dest/('valid_edges_'+name+'.csv'),edges)
 C.csvwrite(dest/'orientation_components.csv',oriented['component_rows']);C.csvwrite(dest/'orientation_constraints.csv',oriented['edge_rows'])
 C.write(dest/'orientation_details.json',dict(summary=oriented['summary'],components=oriented['component_rows'],edges=oriented['edge_rows']))
 # Preserve nearest-point identity by remapping the old chosen corner witness;
 # this is not a new nearest-neighbor tie-breaking run.
 boundary=[];metric=[]
 for name,p in zip(STAGES,(np.tile(np.arange(3),(len(a['faces']),1)),rest['face_corner_permutation'],perm)):
  inv=np.argsort(p,axis=1)
  for v in d['vertices']:
   j=v.get('nearest_free_face');c=v.get('nearest_free_corner')
   mapped=int(inv[K+int(j),int(c)]) if j is not None and c is not None else None
   boundary.append({**v,'stage':name,'original_MODEL_POST_nearest_free_corner':c,'nearest_free_corner':mapped,'witness_status':'REMAP_OF_ORIGINAL_CHOSEN_WITNESS_NOT_NEW_SEARCH'})
  metric.extend(dict(stage=name,**v,value_status='INHERITED_AFTER_GEOMETRY_EQUIVALENCE') for v in d['metrics'])
 C.csvwrite(dest/'boundary_vertices.csv',boundary);C.csvwrite(dest/'AVMR_PMR.csv',metric)
 eq=dict(vertex_bytes_unchanged=True,unordered_indexed_faces_unchanged=True,face_rows_and_source_face_ids_unchanged=True,
  input_audit_arrays_unchanged=True,output_corner_mapping_verified=True,restored_C_source_order_bytes_exact=True,
  no_vertex_merge_or_split_or_face_deletion=True,physical_coordinates_scale_unchanged=True,
  model_input_not_modified=True,UV_normals_colors='NA_NOT_PRESENT',new_quality_calls=0,
  quality_status='INHERITED_UNORIENTED_GEOMETRY_EQUIVALENCE_NOT_A_NEW_SCORER_RUN',
  quality_original_POST_sha256=d['quality']['identity']['processed_sha256'],quality_original_diagnostic_path=r['diagnostic_path'])
 q=dict(status='PASS',identity=identity,restoration=rest['summary'],orientation=oriented['summary'],
  topology=dict(zip(STAGES,(stage0,stage1,stage2))),geometry_equivalence=eq,quality_reference=d['quality'],
  original_metrics=d['metrics'],original_boundary_vertices=d['vertices'],oriented_reached_free_face_ids=reached,
  seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  orientation_solves=1,model_forwards=0,new_full_quality_scores=0,new_GEO_C=0,
  files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
 C.write(rp,q);C.write(C.OUT/'events'/r['method']/(r['request_id']+'_complete.json'),dict(receipt_sha256=C.sha(rp),seconds=q['seconds']))
 assert 'torch' not in sys.modules
 return dict(path=str(rp),reused=False,seconds=q['seconds'],peak_RSS_KiB=q['peak_RSS_KiB'])

def aggregate(reg):
 topology_rows=[];orientations=[];boundary=[];vm=[];quality=[];mapping=[];components=[];constraints=[];pairs=[];equivalence=[]
 for r in reg['records']:
  dest=case_dir(r);q=C.read(dest/'receipt.json');m=meta(r)
  for stage,top in q['topology'].items():
   topology_rows.append(dict(**m,stage=stage,**top))
   quality.append(dict(**m,stage=stage,value_status='INHERITED_GEOMETRY_EQUIVALENCE',**q['quality_reference']['metrics'],free_face_denominator=q['quality_reference']['free_face_denominator']))
  orientations.append({**m,**{k:v for k,v in q['orientation'].items() if not isinstance(v,dict)},**{'restoration_'+k:v for k,v in q['restoration'].items()}})
  equivalence.append(dict(**m,**q['geometry_equivalence']))
  for destlist,file in ((boundary,'boundary_vertices.csv'),(vm,'AVMR_PMR.csv'),(mapping,'source_C_face_mapping.csv'),(components,'orientation_components.csv'),(constraints,'orientation_constraints.csv')):
   destlist.extend(dict(**m,**v) for v in C.rows(dest/file))
  t0,t1,t2=(q['topology'][x] for x in STAGES)
  pairs.append(dict(**m,C_C_same_before=t0['C_C_local_same_direction_edges'],C_C_same_after=t2['C_C_local_same_direction_edges'],
   valid_edges_unchanged=t2['C_F_valid_edge_count'],opposed_MODEL_POST=t0['C_F_valid_opposite_winding_edge_count'],opposed_RESTORED_C_ONLY=t1['C_F_valid_opposite_winding_edge_count'],opposed_ORIENTED=t2['C_F_valid_opposite_winding_edge_count'],
   unoriented_reach=t0['reached_free_area_fraction'],opposed_reach_MODEL_POST=t0['opposite_winding_reached_free_area_fraction'],opposed_reach_RESTORED_C_ONLY=t1['opposite_winding_reached_free_area_fraction'],opposed_reach_ORIENTED=t2['opposite_winding_reached_free_area_fraction'],
   reach_gain=t2['opposite_winding_reached_free_area_fraction']-t0['opposite_winding_reached_free_area_fraction'],parity_conflict_components=q['orientation']['parity_conflict_components'],fixed_source_C_conflict_components=q['orientation']['fixed_C_anchor_conflict_components'],
   flipped_free_faces=q['orientation']['flipped_free_faces'],nonmanifold_edges=q['orientation']['nonmanifold_edges']))
 for name,rows in [('topology_per_output',topology_rows),('orientation_per_output',orientations),('boundary_vertices',boundary),('AVMR_PMR_per_output',vm),('quality_inherited_per_output',quality),('source_C_face_mapping',mapping),('orientation_components',components),('orientation_constraints',constraints),('paired_per_parent_noise',pairs),('geometry_equivalence',equivalence)]:C.csvwrite(C.OUT/(name+'.csv'),rows)
 summaries=[]
 for kind,key in [('ALL',None),('REGION','region_type'),('PARENT','parent_id'),('NOISE','candidate')]:
  for method in ('D0_BASE','RBF','BLEND'):
   for value in (sorted({r[key] for r in topology_rows}) if key else ['ALL']):
    for stage in STAGES:
     sub=[r for r in topology_rows if r['method']==method and r['stage']==stage and (key is None or r[key]==value)]
     numeric=['C_F_valid_edge_count','C_F_valid_opposite_winding_edge_count','C_C_local_same_direction_edges','reached_free_area_fraction','opposite_winding_reached_free_area_fraction','nonmanifold_edges']
     summaries.append(dict(group=kind,value=value,method=method,stage=stage,n=len(sub),**{k+'_mean':float(np.mean([r[k] for r in sub])) for k in numeric},
      outputs_with_valid_edges=sum(r['C_F_valid_edge_count']>0 for r in sub),outputs_with_opposed_edges=sum(r['C_F_valid_opposite_winding_edge_count']>0 for r in sub),
      valid_edge_total=sum(r['C_F_valid_edge_count'] for r in sub),opposed_edge_total=sum(r['C_F_valid_opposite_winding_edge_count'] for r in sub)))
 C.csvwrite(C.OUT/'topology_summary.csv',summaries)
 C.write(C.OUT/'aggregate_completion.json',dict(status='PASS',outputs=36,topology_rows=108,AVMR_rows=len(vm),full_quality_values_status='INHERITED_NOT_RECOMPUTED',new_full_quality_scores=0))

def main():
 check=C.read(C.OUT/'restoration_checks/completion.json');assert check['status']=='PASS'
 reg=prepare();tick=time.perf_counter();C.write(C.OUT/'resources_before_dispatch.json',C.resources())
 first=process_one(reg['records'][0]);res=C.resources();C.write(C.OUT/'resources_after_first_case.json',res)
 mem={s.split(':')[0]:int(s.split()[1])*1024 for s in res['memory'].splitlines() if s.split()[1].isdigit()}
 workers=2 if (os.cpu_count() or 1)>=4 and mem['MemAvailable']>max(4<<30,3*first['peak_RSS_KiB']*1024) and res['loadavg'][0]<(os.cpu_count() or 1)-2 else 1
 C.write(C.OUT/'CPU_dispatch.json',dict(workers=workers,threads_per_worker=2,first_authorized_case=first,available_RAM_bytes=mem['MemAvailable'],loadavg=res['loadavg'],GPU_workers=0,reason='Measured actual complete first case; independent remaining source restoration/orientation jobs, at most two CPU workers; no model probes',timing_scope='Concurrent wall time, not isolated benchmark'))
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:receipts=[first,*pool.map(process_one,reg['records'][1:],chunksize=1)]
 aggregate(reg);C.verify(reg['protected']);C.verify(reg['runner_hashes'])
 assert len(list((C.OUT/'events').glob('*/*_started.json')))==len(list((C.OUT/'events').glob('*/*_complete.json')))==36
 C.write(C.OUT/'resources_after_experiment.json',C.resources())
 C.write(C.OUT/'execution_completion.json',dict(status='PASS',outputs=36,source_C_restorations=36,orientation_solves=36,CPU_workers=workers,receipts=receipts,phase_seconds=time.perf_counter()-tick,protected_files_unchanged=len(reg['protected']),frozen_sources_unchanged=True,model_loads=0,model_forwards=0,training_updates=0,new_GEO_C=0,new_full_quality_scores=0,new_images=0,scientific_stage_stopped=True))
 C.event('EXECUTION_COMPLETE',outputs=36,orientation_solves=36,new_model_forwards=0,new_quality_scores=0)
 print('SOURCE_WINDING_EXPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
