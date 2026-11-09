"""Independent NumPy/CSV audit of saved coupling artifacts. No model/scorer imports."""
from __future__ import annotations
import argparse,ast,collections,csv,hashlib,json,math,os,time
from pathlib import Path
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_c_restore_coupling_v1'
AUDIT=OUT/'independent_audit'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def arrays(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def norm(a):return np.sqrt(np.sum(np.asarray(a,dtype=np.float64)**2,axis=-1))
def near(a,b,atol=2e-11,rtol=2e-10):
 assert np.shape(a)==np.shape(b) and np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=False),'Numerical mismatch'
def checked(p):
 p=Path(p).absolute();assert not any(q.is_symlink() for q in (p,*p.parents))
 p=p.resolve();assert p!=AUDIT.resolve() and p.is_relative_to(AUDIT.resolve()),'Outside owned audit'
 return p
def atomic(p,fn):
 p=checked(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=checked(p.with_name(p.name+'.tmp'))
 with tmp.open('w',encoding='utf-8',newline='') as f:fn(f);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def write(p,d):atomic(p,lambda f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False))
def csvwrite(p,data):
 def emit(f):
  w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
 atomic(p,emit)
def verify(mapping):
 for p,h in mapping.items():assert sha(p)==h,p
 return len(mapping)
def kernel(x,y,radius):
 q=np.sqrt(((np.asarray(x,dtype=float)[:,None,:]-np.asarray(y,dtype=float)[None,:,:])**2).sum(2))/radius
 result=np.zeros(q.shape,dtype=float);inside=q<1.;t=1.-q[inside]
 result[inside]=t*t*t*t*(4.*q[inside]+1.)
 return result

def check_rbf(c,proposal,ids,output,detail,diag,L):
 k=len(c);anchors=proposal[:k].reshape(-1,3).astype(float);target=c.reshape(-1,3).astype(float)
 multiplicity=collections.Counter(map(int,ids.reshape(-1)))
 weights=np.asarray([1./multiplicity[int(i)] for i in ids.reshape(-1)])
 b=target-anchors;gram=kernel(anchors,anchors,.1*L);system=gram+np.diag(1e-6/weights)
 for key,value in [('anchor_source_vertex_ids',ids.reshape(-1)),('anchor_proposal_model',anchors),('anchor_target_model',target),('anchor_weight',weights),('anchor_displacement_target',b)]:near(detail[key],value)
 coef=detail['coefficients'];normal=system@coef-b
 assert np.max(np.abs(normal))<2e-8,'RBF linear equation residual'
 fit=gram@coef;near(detail['anchor_displacement_fit'],fit,2e-8);near(detail['anchor_fit_residual'],fit-b,2e-8)
 free=proposal[k:].reshape(-1,3);field=kernel(free,anchors,.1*L)@coef
 near(detail['free_displacement_unclipped'],field,2e-8)
 # Use the saved field after independently verifying it; avoids treating BLAS
 # evaluation-order roundoff as a changed scientific output.
 field=detail['free_displacement_unclipped'];length=norm(field);cap=.02*L
 factor=np.ones(len(free));active=length>cap;factor[active]=cap/length[active]
 near(detail['free_clip_factor'],factor);near(detail['free_displacement_clipped'],field*factor[:,None])
 assert np.array_equal(detail['free_clipped'],active)
 clipped=field*factor[:,None];expected=free.copy();moved=np.any(clipped!=0.,axis=1)
 expected[moved]=(free[moved].astype(float)+clipped[moved]).astype(np.float32)
 roundoff=norm(expected.astype(float)-free)>cap
 expected[roundoff]=np.nextafter(expected[roundoff],free[roundoff])
 assert np.array_equal(detail['free_FP32_cap_rounding_adjusted'],roundoff)
 assert same(output[:k],c) and same(output[k:].reshape(-1,3),expected)
 actual=expected.astype(float)-free;near(detail['free_actual_displacement_FP32'],actual)
 assert np.all(norm(actual)<=cap)
 near(diag['anchor_weighted_vector_MSE'],float(np.sum(weights[:,None]*(fit-b)**2)/weights.sum()),2e-8)
 near(diag['system_residual_max'],float(np.max(np.abs(normal))),2e-8)
 assert diag['clipped_free_corners']==int(active.sum()) and diag['FP32_cap_rounding_adjusted_corners']==int(roundoff.sum())
 for vid in multiplicity:
  assert abs(weights[ids.reshape(-1)==vid].sum()-1.)<1e-12
 return dict(anchor_corners=len(anchors),source_vertices=len(multiplicity),system_residual_max=float(np.abs(normal).max()),clipped_free_corners=int(active.sum()),rounding_adjusted_corners=int(roundoff.sum()),C_bitwise=True,output_reconstructed_bitwise=True,max_free_corner_shift=float(norm(actual).max()))

def chain_audit():
 started=time.monotonic();reg=read(OUT/'registration.json');rh=sha(OUT/'registration.json');records=reg['records']
 assert len(records)==12 and len({r['parent_id'] for r in records})==6
 assert reg['budget']['new_D0_real_forwards']==24 and reg['budget']['CPU_geometry_synthetic_cases']==4 and reg['budget']['CPU_analytic_feedback_cases']==3
 assert reg['budget']['new_D0_synthetic_forwards']==reg['budget']['training_updates']==reg['budget']['new_generator_forwards']==reg['budget']['new_noise']==0
 assert reg['methods']['FEEDBACK']['eta']==.5 and reg['methods']['FEEDBACK']['updates']==2
 eta=.5;cap=.02*reg['fixed_L'];scale=np.float32(reg['scale']);queryrows=[];rbfrows=[]
 events=[json.loads(x) for x in (OUT/'ledger.jsonl').read_text().splitlines() if x]
 expected={(r['request_id'],j) for r in records for j in [1,2]}
 for event in ['D0_FEEDBACK_STARTED','D0_FEEDBACK_COMPLETE']:
  a=[(e['request_id'],e['update']) for e in events if e['event']==event]
  assert len(a)==len(set(a))==24 and set(a)==expected,event
 assert not any(e['event']=='D0_FEEDBACK_FAILED' for e in events)
 for event in ['RBF_STARTED','RBF_COMPLETE']:
  a=[e['request_id'] for e in events if e['event']==event];assert len(a)==len(set(a))==12
 for r in records:
  rid=r['request_id'];k=int(r['K']);n=int(r['N']);a=arrays(r['raw_path']);raw=a['raw'].reshape(n,3,3);c=a['C'];base=arrays(r['D0_denoised_path'])['official_unconstrained_proposal']
  assert sha(r['raw_path'])==r['raw_sha256'] and same(c,raw[:k])
  previous=base;delta=np.zeros(c.shape,dtype=np.float64)
  for j in [1,2]:
   p=OUT/'feedback_queries'/rid/f'update{j}/query.npz';saved=arrays(p);proof=read(p.with_name('receipt.json'))
   assert proof['status']=='PASS' and proof['output_sha256']==sha(p) and proof['registration_sha256']==rh and proof['checkpoint_sha256']==reg['checkpoint_sha256']
   assert proof['request_id']==rid and proof['update']==j and proof['raw_sha256']==r['raw_sha256']
   error=c.astype(float)-previous[:k].astype(float);unprojected=delta+eta*error
   length=norm(unprojected);factor=np.minimum(1.,cap/np.maximum(length,np.finfo(float).tiny));delta=unprojected*factor[...,None]
   near(saved['previous_C_error'],error,0,0);near(saved['unprojected_delta'],unprojected,0,0);near(saved['delta'],delta,0,0)
   working=raw.copy();working[:k]=(c.astype(float)+delta).astype(np.float32);adjusted=np.zeros(c.shape[:2],bool)
   for _ in range(32):
    bad=norm(working[:k].astype(float)-c)>cap
    if not bad.any():break
    adjusted|=bad;working[:k][bad]=np.nextafter(working[:k][bad],c[bad])
   assert np.all(norm(working[:k].astype(float)-c)<=cap) and same(working,saved['working_input'])
   assert same(saved['working_input'][k:],raw[k:]) and same(saved['output'][:k],c) and same(saved['C'],c)
   assert same(saved['output'][k:],saved['proposal'][k:]) and saved['proposal'].dtype==np.float32 and np.isfinite(saved['proposal']).all()
   xd=np.multiply(working.reshape(1,n,9),scale,dtype=np.float32);assert same(xd,saved['actual_input_d']),'GPU input multiply mismatch'
   pd=saved['actual_proposal_d'];assert pd.shape==xd.shape and pd.dtype==np.float32 and np.isfinite(pd).all()
   cpu_inverse=np.divide(pd,scale,dtype=np.float32).reshape(n,3,3);difference=np.abs(cpu_inverse.astype(float)-saved['proposal'])
   tolerance=4*np.finfo(np.float32).eps*np.maximum(1.,np.abs(saved['proposal'].astype(float)))
   assert np.all(difference<=tolerance),'GPU inverse-scale differs beyond FP32 rounding allowance'
   near(saved['C_error'],c.astype(float)-saved['proposal'][:k].astype(float),0,0)
   near(saved['free_change_from_D0'],saved['proposal'][k:].astype(float)-base[k:],0,0)
   assert proof['projection_saturated_corners']==int((length>cap).sum()) and proof['FP32_bound_roundoff_corners']==int(adjusted.sum())
   assert proof['no_rng_consumed'] and proof['residual_dtype']=='torch.bfloat16'
   queryrows.append(dict(request_id=rid,update=j,N=n,K=k,status='PASS',input_d_CPU_FP32_multiply_bitwise=True,inverse_scale_max_CPU_GPU_difference=float(difference.max()),cumulative_C_shift_max=float(norm(working[:k].astype(float)-c).max()),saturated_corners=int((length>cap).sum()),FP32_nextafter_corners=int(adjusted.sum()),free_RAW_bitwise=True,final_C_bitwise=True))
   previous=saved['proposal']
  endpoint=OUT/'pre_geo/FEEDBACK'/rid/'denoised.npz';e=arrays(endpoint);ep=read(endpoint.with_name('receipt.json'))
  assert ep['final_update']==2 and ep['query_sha256']==sha(p) and ep['output_sha256']==sha(endpoint)
  assert same(e['output'],saved['output']) and same(e['official_unconstrained_proposal'],saved['proposal'])
  folder=OUT/'pre_geo/RBF'/rid;d=arrays(folder/'denoised.npz');receipt=read(folder/'receipt.json');diagnostics=read(folder/'diagnostics.json')
  assert receipt['status']=='PASS' and receipt['output_sha256']==sha(folder/'denoised.npz') and receipt['registration_sha256']==rh
  assert same(d['official_unconstrained_proposal'],base)
  ids=arrays(r['condition_source'])['C_source_vertex_ids']
  detail={p.stem:arrays(p)['data'] for p in folder.glob('*.npz') if p.name!='denoised.npz'}
  rbfrows.append(dict(request_id=rid,status='PASS',**check_rbf(c,base,ids,d['output'],detail,diagnostics,reg['fixed_L'])))
 # Verify existing finite controls; no additional synthetic scientific cases.
 controls=read(OUT/'geometry_checks/completion.json');assert controls['status']=='PASS' and len(controls['cases'])==4
 verify(controls['files']);verify(controls['source_hashes'])
 for item in controls['cases']:
  folder=OUT/'geometry_checks'/item['case'];d=arrays(folder/'input_and_output.npz');diag=read(folder/'diagnostic.json')
  check_rbf(d['C'],d['proposal'],d['C_source_vertex_ids'],d['output'],d,diag,reg['fixed_L'])
 checks=read(OUT/'feedback_checks.json');assert checks['status']=='PASS' and len(checks['cases'])==3 and checks['model_forwards']==checks['synthetic_GPU_forwards']==0
 analytic={r['case']:r for r in checks['cases']}
 near(analytic['linear_cross_coupled']['C_errors'],[.02,.01,.005]);near(analytic['linear_cross_coupled']['F_changes'],[0.,.004,.006])
 near(analytic['zero_cross_coupling']['F_changes'],[0.,0.,0.]);near(analytic['projection_saturation']['norm'],.02)
 done=read(OUT/'denoise_completion.json');assert done['status']=='PASS' and done['completed_queries']==done['total_forward_attempts']==24
 assert done['training_updates']==done['new_generator_forwards']==done['new_noise']==done['synthetic_D0_forwards']==0
 protected=verify(reg['protected']);verify(reg['runner_hashes'])
 csvwrite(AUDIT/'feedback_chain_per_query.csv',queryrows);csvwrite(AUDIT/'RBF_equation_per_output.csv',rbfrows)
 result=dict(status='PASS',requests=12,queries=24,RBF_outputs=12,geometry_controls=4,analytic_feedback_controls=3,protected_files_rehashed=protected,runner_sources_verified=len(reg['runner_hashes']),fixed_F_RAW_all=True,cumulative_C_ball_all=True,final_C_bytes_all=True,actual_GPU_input_d_CPU_scaling_bitwise_all=True,inverse_scale_CPU_GPU_max_difference=max(r['inverse_scale_max_CPU_GPU_difference'] for r in queryrows),FP32_inverse_scale_tolerance='4 eps * max(1,abs(proposal)); actual GPU values preserved, no claim CPU/GPU division bitwise identity',RBF_equation_and_weights_and_clip_all=True,registration_sha256=rh,audit_source_sha256=sha(__file__),seconds=time.monotonic()-started,model_loads=0,model_forwards=0,training_updates=0,new_GEO=0,new_quality=0)
 write(AUDIT/'chain_audit.json',result);return result

def num(v):
 if v in (None,''):return None
 if isinstance(v,str) and v.lower() in ['true','false']:return float(v.lower()=='true')
 return float(v)
def eq(a,b):
 a=num(a);b=num(b)
 assert (a is None)==(b is None),(a,b)
 if a is not None:assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=2e-10,rel_tol=2e-9),(a,b)
def reduction(table,summary):
 checked_n=0
 for s in summary:
  base=[k for k in s if k not in ['outputs','grouping'] and not k.endswith(('_valid_n','_mean','_median','_p95')) and s[k]!='']
  group=[r for r in table if all(str(r.get(k,''))==s[k] for k in base)];assert len(group)==int(s['outputs']),(base,len(group),s['outputs'])
  for key in s:
   if not key.endswith('_valid_n'):continue
   field=key[:-8];vals=[num(r.get(field)) for r in group if r.get(field) not in (None,'')]
   eq(s[key],len(vals));checked_n+=1
   for suffix,fn in [('mean',np.mean),('median',np.median),('p95',lambda a:np.quantile(a,.95))]:
    eq(s[field+'_'+suffix],float(fn(vals)) if vals else None);checked_n+=1
 return checked_n

def graph_audit(tri,vertices,faces,k,edge_rows,topology,pairs):
 incidence=collections.defaultdict(list);duplicates=collections.defaultdict(list)
 for i,face in enumerate(faces):
  duplicates[tuple(sorted(map(int,face)))].append(i)
  for u,v in zip(face,np.roll(face,-1)):incidence[tuple(sorted((int(u),int(v))))].append((i,int(u),int(v)))
 duplicate_faces={i for group in duplicates.values() if len(group)>1 for i in group}
 expected={e:v for e,v in incidence.items() if any(i>=k for i,_,_ in v)}
 assert len(edge_rows)==len(expected)
 classifications={'CF':{tuple(map(int,p['pair'])):p['classification'] for p in pairs['CF']['contacts']},'FF':{tuple(sorted(map(int,p['pair']))):p['classification'] for p in pairs['FF']['contacts']}}
 legal={'shared_edge_contact','shared_vertex_contact','coplanar_boundary_or_tolerance_contact','noncoplanar_edge_or_tangent_contact','point_or_tangent_contact'}
 adjacent=collections.defaultdict(set);opposite=collections.defaultdict(set);valid_cf=[];cf=[];valid_count=0
 areas=.5*norm(np.cross(tri[:,1].astype(float)-tri[:,0],tri[:,2].astype(float)-tri[:,0]))
 for e in edge_rows:
  u,v=int(e['vertex_u']),int(e['vertex_v']);inc=expected[u,v];assert ast.literal_eval(e['incident_face_ids'])==[i for i,_,_ in inc]
  eq(e['incidence'],len(inc));eq(e['length_model'],norm(vertices[u].astype(float)-vertices[v]));is_cf=any(i<k for i,_,_ in inc) and any(i>=k for i,_,_ in inc)
  assert (e['is_C_F']=='True')==is_cf;valid=False;opposed=False
  if len(inc)==2:
   (i,a,b),(j,c,d)=inc;opposed=(a==d and b==c);assert (e['winding_opposite']=='True')==opposed
   pair=(min(i,j),max(i,j)-k) if is_cf else tuple(sorted((i-k,j-k)))
   classification=classifications['CF' if is_cf else 'FF'].get(pair,'UNKNOWN_MISSING_SHARED_EDGE_PAIR')
   assert e['pair_classification']==classification
   valid=(i!=j and u!=v and float(e['length_model'])>0 and areas[i]>0 and areas[j]>0 and i not in duplicate_faces and j not in duplicate_faces and classification in legal)
   if valid:
    adjacent[i].add(j);adjacent[j].add(i)
    if opposed:opposite[i].add(j);opposite[j].add(i)
  assert (e['valid_edge']=='True')==valid
  valid_count+=int(valid)
  if is_cf:cf.append(e)
  if is_cf and valid:valid_cf.append(e)
 def reach(adjacency):
  seen=set(range(k));pending=list(seen)
  while pending:
   i=pending.pop();fresh=adjacency[i]-seen;seen.update(fresh);pending.extend(fresh)
  return sorted(i for i in seen if i>=k)
 reached=reach(adjacent);opposed_reached=reach(opposite)
 values={'C_F_shared_edge_count':len(cf),'C_F_valid_edge_count':len(valid_cf),'C_F_has_valid_edge':int(bool(valid_cf)),'C_F_valid_opposite_winding_edge_count':sum(e['winding_opposite']=='True' for e in valid_cf),'graph_valid_edges':valid_count,'reached_free_faces':len(reached),'reached_free_area':float(areas[reached].sum()),'free_area':float(areas[k:].sum()),'reached_free_area_fraction':float(areas[reached].sum()/areas[k:].sum()),'opposite_winding_reached_free_faces':len(opposed_reached),'opposite_winding_reached_free_area':float(areas[opposed_reached].sum()),'opposite_winding_reached_free_area_fraction':float(areas[opposed_reached].sum()/areas[k:].sum()),'exact_duplicate_faces':len(duplicate_faces),'nonmanifold_edges':sum(len(v)>2 for v in incidence.values()),'boundary_edges':sum(len(v)==1 for v in incidence.values())}
 for key,value in values.items():eq(topology[key],value)
 return len(edge_rows)

def score_audit():
 started=time.monotonic();reg=read(OUT/'score_registration.json');verify(reg['protected']);verify(reg['quality_source_hashes'])
 assert reg['source_sha256']==sha(ROOT/'r2/tools/c_restore_coupling_v1/evaluation.py')
 index=rows(OUT/'post_index.csv');assert len(index)==48
 evidence=[];boundary_checks=0;graph_edge_checks=0
 quality_csv={(r['method'],r['request_id']):r for r in rows(OUT/'quality_per_output.csv')}
 topology_csv={(r['method'],r['request_id'],r['phase']):r for r in rows(OUT/'topology_per_output.csv')}
 for r in index:
  k=int(r['K']);p=Path(r['post_path']);post=arrays(p);pre=arrays(r['denoised_path'])['output'];raw=arrays(r['raw_path'])['raw'].reshape(-1,3,3)
  assert sha(p)==r['post_sha256'] and same(post['output'],post['vertices'][post['faces']]) and same(post['output'][:k],raw[:k]) and same(pre[:k],raw[:k])
  d=read(r['diagnostic_path']);assert d['status']=='PASS' and d['identity']['post_sha256']==sha(p) and d['pre_GEO_quality_status']=='NA_NOT_BUDGETED'
  verify(d['files']);verify(d['quality']['files'])
  assert int(d['quality']['free_face_denominator'])==len(post['output'])-k
  for key,value in d['quality']['metrics'].items():
   if isinstance(value,(int,float)) or value is None:eq(quality_csv[r['method'],r['request_id']].get(key),value)
  for phase,tri,vkey,mkey in [('POST',post['output'],'vertices','metrics'),('PRE_GEO',pre,'pre_GEO_vertices','pre_GEO_metrics')]:
   top=d['topology' if phase=='POST' else 'pre_GEO_topology']
   for key,value in top.items():
    if isinstance(value,(int,float)) or value is None:eq(topology_csv[r['method'],r['request_id'],phase].get(key),value)
   if phase=='POST':vertices,faces=post['vertices'],post['faces'];pairs=read(d['quality']['pair_audit_path'])
   else:
    flat=np.ascontiguousarray(tri,dtype=np.float32).reshape(-1,3);_,first,inverse=np.unique(flat.view(np.uint32).reshape(-1,3),axis=0,return_index=True,return_inverse=True)
    vertices,faces=flat[first],inverse.reshape(-1,3);pairs=read(Path(r['diagnostic_path']).parent/'PRE_edge_local_pairs.json')
   edge_rows=rows(Path(r['diagnostic_path']).parent/('valid_edge_details_POST.csv' if phase=='POST' else 'valid_edge_details_PRE.csv'))
   graph_edge_checks+=graph_audit(tri,vertices,faces,k,edge_rows,top,pairs)
   points=np.asarray([[v['x'],v['y'],v['z']] for v in d[vkey]],float);free=tri[k:].reshape(-1,3).astype(float)
   distances=np.sqrt(((points[:,None]-free[None,:])**2).sum(2)).min(1)
   for item,distance in zip(d[vkey],distances):eq(item['distance_model'],distance);boundary_checks+=1
   for metric in d[mkey]:
    matched=int((distances<float(metric['threshold_model'])).sum());eq(metric['matched_count'],matched);eq(metric['boundary_vertex_count'],len(distances));eq(metric['VMR'],matched/len(distances));eq(metric['perfect'],int(matched==len(distances)));boundary_checks+=4
  evidence.append(dict(method=r['method'],request_id=r['request_id'],N_POST=len(post['output']),K=k,C_bitwise=True,indexed_NPZ_exact=True,status='PASS'))
 new={m:sum(r['method']==m for r in index) for m in ['RBF','FEEDBACK','D0_BASE','D3']};assert all(v==12 for v in new.values())
 for method in ['RBF','FEEDBACK']:
  for row in [r for r in index if r['method']==method]:
   folder=OUT/'score_events'/method/row['request_id']
   assert {p.stem for p in folder.glob('*.json')}=={'GEO_STARTED','GEO_COMPLETE','QUALITY_STARTED','QUALITY_COMPLETE'}
 combined={(r['method'],r['request_id']):r for r in rows(OUT/'combined_per_output.csv')};delta_checks=0
 for p in rows(OUT/'paired_per_output.csv'):
  left,right=p['comparison'].split('_minus_');a,b=combined[left,p['request_id']],combined[right,p['request_id']]
  for key,val in p.items():
   if key.endswith('_delta'):
    field=key[:-6];x,y=num(a.get(field)),num(b.get(field));eq(val,None if x is None or y is None else x-y);delta_checks+=1
 vm={(r['method'],r['request_id'],r['phase'],r['setting']):r for r in rows(OUT/'per_output.csv')}
 assert len(vm)==48*2*8
 for p in rows(OUT/'AVMR_paired_per_output.csv'):
  left,right=p['comparison'].split('_minus_');a=vm[left,p['request_id'],p['phase'],p['setting']];b=vm[right,p['request_id'],p['phase'],p['setting']]
  eq(p['A_VMR_delta'],num(a['VMR'])-num(b['VMR']));eq(p['PMR_delta'],num(a['perfect'])-num(b['perfect']));delta_checks+=2
 for p in rows(OUT/'GEO_changes_per_output.csv'):
  c=combined[p['method'],p['request_id']]
  for key,value in p.items():
   if key.endswith('_POST_minus_PRE'):
    field=key[:-15];eq(value,num(c['POST_'+field])-num(c['PRE_'+field]));delta_checks+=1
 for p in rows(OUT/'GEO_AVMR_changes_per_output.csv'):
  a=vm[p['method'],p['request_id'],'POST',p['setting']];b=vm[p['method'],p['request_id'],'PRE_GEO',p['setting']]
  eq(p['A_VMR_POST_minus_PRE'],num(a['VMR'])-num(b['VMR']));eq(p['PMR_POST_minus_PRE'],num(a['perfect'])-num(b['perfect']));delta_checks+=2
 table_pairs=[('quality_per_output','quality_summary'),('topology_per_output','topology_summary'),('boundary_distance_per_output','boundary_distance_summary'),('modifications_per_output','modifications_summary'),('pre_GEO_shape_per_output','pre_GEO_shape_summary'),('paired_per_output','paired_summary'),('GEO_changes_per_output','GEO_changes_summary'),('GEO_AVMR_changes_per_output','GEO_AVMR_changes_summary'),('per_output','AVMR_PMR_summary'),('AVMR_paired_per_output','AVMR_paired_summary')]
 summary_checks=0
 for table,summary in table_pairs:summary_checks+=reduction(rows(OUT/(table+'.csv')),rows(OUT/(summary+'.csv')))
 assert all(r['status']=='NA_NOT_BUDGETED' and r['CF']==r['FF']==r['I']=='' for r in rows(OUT/'PRE_full_quality_status.csv'))
 done=read(OUT/'score_completion.json');assert done['status']=='PASS' and done['new_POST_quality']==done['new_GEO']==24 and done['reused_POST_quality']==24 and done['new_model_calls']==done['new_queries']==0
 csvwrite(AUDIT/'POST_contract_per_output.csv',evidence)
 result=dict(status='PASS',POST_outputs=48,PRE_outputs=48,actual_C_bytes_and_indexed_POST_all=True,new_GEO=24,new_POST_quality=24,reused_POST_quality=24,AVMR_rows=len(vm),boundary_and_match_checks=boundary_checks,paired_delta_checks=delta_checks,summary_stat_checks=summary_checks,graph_edges_verified=graph_edge_checks,graph_scope='Actual incidence/winding/areas/reach independently reconstructed; pair classification reused from frozen saved original scorer, no new pair scan.',PRE_full_quality='NA_NOT_BUDGETED',protected_files_verified=len(reg['protected']),score_source_sha256=reg['source_sha256'],audit_source_sha256=sha(__file__),seconds=time.monotonic()-started,model_forwards=0,new_GEO_runs=0,new_quality_scans=0,source_csv_sha256={str(p):sha(p) for p in OUT.glob('*.csv')})
 write(AUDIT/'score_audit.json',result);return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['chain','score','all'],default='all');a=ap.parse_args()
 for p in [ROOT/'WORKSPACE.md',ROOT/'experiments/r2_multistep_denoise_v1/audit.json',OUT/'registration.json']:
  try:checked(p)
  except AssertionError:pass
  else:raise AssertionError('Negative write guard failed')
 result={}
 if a.stage in ['chain','all']:result['chain']=chain_audit()
 if a.stage in ['score','all']:result['score']=score_audit()
 if a.stage=='all':write(AUDIT/'completion.json',dict(status='PASS',results=result,model_forwards=0,training_updates=0,new_GEO=0,new_quality=0))
 print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
