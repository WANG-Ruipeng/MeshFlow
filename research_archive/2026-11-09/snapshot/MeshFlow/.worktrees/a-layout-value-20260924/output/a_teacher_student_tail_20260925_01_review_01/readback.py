"""One saved-array readback; no field query, model, training, exact audit or MC."""
from pathlib import Path
import json,hashlib,math
import numpy as np
ROOT=Path.cwd().resolve();RUN=ROOT/'output/a_teacher_student_tail_20260925_01';OUT=Path(__file__).resolve().parent
KEYS=('min_triangle_angle_deg','triangle_min_angle_p05_deg')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def arr(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k]) for k in z.files}
def ah(*args):
 h=hashlib.sha256()
 for v in args:
  v=np.ascontiguousarray(v);h.update(str(v.dtype).encode());h.update(json.dumps(v.shape).encode());h.update(v.tobytes())
 return h.hexdigest()
def close(a,b):assert np.allclose(a,b,rtol=1e-11,atol=1e-11),(a,b)
def dump(p,o):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,indent=2,allow_nan=False);f.write('\n')
def main():
 before={n:sha(RUN/n) for n in ('RUN_PROTOCOL.json','RESULTS.json','COMPLETE.json','REPORT.md')}
 p=read(RUN/'RUN_PROTOCOL.json');result=read(RUN/'RESULTS.json');complete=read(RUN/'COMPLETE.json');parent=Path(p['parent_run']);ledger=read(parent/'results.json')
 assert complete['results_sha256']==before['RESULTS.json'] and complete['protocol_sha256']==before['RUN_PROTOCOL.json']
 assert complete['status']=='COMPLETE' and result['geometry_checks']==p['geometry_checks']==72
 assert p['lambdas']==[i/8 for i in range(9)] and len(result['pairs'])==8
 assert p['no_MC'] and p['no_network'] and p['no_training'] and p['no_output_or_lambda_selection'] and not p['exact_intersection_or_topology_audit_executed']
 assert not result['exact_audit_executed'] and not result['full_validity_claim'] and not result['output_selection'] and not result['cuda_initialized']
 for name,digest in {**p['source_files'],**p['source_sha256']}.items():assert sha(name)==digest,name
 rows={(x['uid'],x['endpoint']):x for x in ledger['rows']};seen=set();records=[];geometries=[];all_pass=0
 for pair in result['pairs']:
  uid,kind=pair['uid'],pair['kind'];assert (uid,kind) not in seen;seen.add((uid,kind))
  assert pair==read(RUN/Path(pair['geometry_file']).parent/'RESULT.json')
  path=RUN/pair['geometry_file'];assert sha(path)==pair['geometry_file_sha256'];g=arr(path);faces=g['faces'];assert faces.dtype==np.int32 and faces.shape==(512,3)
  assert np.array_equal(g['lambdas'],np.array(p['lambdas'])) and len(pair['rows'])==9
  endpoints=[rows[(uid,k)] for k in ('fixed',kind+'_teacher',kind+'_final')];fixed,teacher,student=endpoints
  data=arr(parent/'data'/f'{uid}.npz');assert np.array_equal(faces,data['faces'])
  assert len({e['mainfield_hash'] for e in endpoints})==1 and teacher['mainfield_hash']==ah(data['logr'])
  assert len({e['reference_hash'] for e in endpoints})==1 and teacher['reference_hash']==ah(data['ref_vertices'],data['ref_faces'])
  assert pair['student_identity']==student['identity'] and student['identity']['update']==20000 and student['identity']['seed']==10
  tm=arr(parent/teacher['mesh_file']);sm=arr(parent/student['mesh_file'])
  for m,e,i in ((tm,teacher,0),(sm,student,8)):
   assert sha(parent/e['mesh_file'])==e['mesh_npz_sha256']
   assert ah(m['vertices'],m['faces'])==e['stored_array_hash'] and ah(m['vertices'],data['faces'])==e['mesh_hash']
   for key in ('u','r','vertices'):assert np.array_equal(g[key][i],m[key])
   assert pair['rows'][i]['endpoint_replay']['u_radius_vertices_bitwise_equal']
  chord=np.mean((sm['u']-tm['u'])**2,axis=1);angular=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(tm['u'],sm['u']),axis=1),np.sum(tm['u']*sm['u'],axis=1)))
  close(chord,g['vertex_chord_coordinate_mse']);close(angular,g['vertex_angular_error_deg'])
  orders=[];first_failed=[];margins=[]
  for i,(lam,row) in enumerate(zip(g['lambdas'],pair['rows'])):
   assert row['lambda']==lam and row['uid']==uid and row['kind']==kind
   u=g['u'][i];v=g['vertices'][i];radius=g['r'][i]
   if i in (0,8):expected=tm['u'] if i==0 else sm['u']
   else:
    expected=(1-lam)*tm['u']+lam*sm['u'];expected=expected/np.linalg.norm(expected,axis=1)[:,None]
   assert np.array_equal(u,expected) and np.array_equal(v,u*radius[:,None]);close(np.linalg.norm(u,axis=1),1.)
   tri=v[faces];area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1);deg=area2<=max(np.linalg.norm(np.ptp(v,axis=0))**2,1e-300)*1e-14
   faceangles=[]
   for j in range(3):
    a=tri[:,(j+1)%3]-tri[:,j];b=tri[:,(j+2)%3]-tri[:,j]
    faceangles.append(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.sum(a*b,axis=1))))
   minimum=np.min(np.stack(faceangles),axis=0);minimum[deg]=0
   dtri=u[faces];det=np.sum(dtri[:,0]*np.cross(dtri[:,1],dtri[:,2]),axis=1)
   close(minimum,g['face_min_angle_deg'][i]);close(area2*.5,g['face_area'][i]);close(det,g['direction_det'][i])
   q=row['quality'];close(q[KEYS[0]],minimum.min());close(q[KEYS[1]],np.quantile(minimum,.05));close(q['area_min'],area2.min()*.5);close(q['area_total'],area2.sum()*.5)
   assert q['face_count']==512 and q['degenerate_count']==int(deg.sum()) and np.flatnonzero(deg).tolist()==row['degenerate_faces']
   close(det.min(),row['direction_det_min']);assert np.flatnonzero(det<=0).tolist()==row['direction_det_nonpositive_faces'] and row['direction_dets_positive']==bool(np.all(det>0))
   order=np.argsort(minimum,kind='stable');orders.append(order);tail=row['tail'];assert tail['bottom_ceil_5pct_faces']==order[:26].tolist()
   assert tail['p05_support_faces']==order[[25,26]].tolist() and tail['p05_support_order_indices_zero_based']==[25,26]
   close(tail['p05_support_weights'],[.45,.55]);close(tail['p05_support_angles_deg'],minimum[order[[25,26]]]);close(tail['p05_linear_deg'],minimum[order[25]]*.45+minimum[order[26]]*.55)
   assert tail['minimum_face']==int(order[0]) and tail['minimum_tied_faces']==np.flatnonzero(minimum==minimum.min()).tolist()
   base=set(orders[0][:26]);current=set(order[:26]);assert row['teacher_tail_retained_count']==len(base&current)
   assert row['teacher_tail_left_faces']==sorted(base-current) and row['new_tail_faces_since_teacher']==sorted(current-base)
   close(row['tail_jaccard_vs_teacher'],len(base&current)/len(base|current))
   passed={k:q[k]+1e-6>=fixed['quality'][k] for k in KEYS}
   assert passed==row['individual_angle_conditions_pass'] and all(passed.values())==row['angle_conditions_pass']
   assert row['minimum_floor_violating_faces']==np.flatnonzero(minimum+1e-6<fixed['quality'][KEYS[0]]).tolist()
   assert row['p05_failure_contributor_faces']==(order[:27].tolist() if not passed[KEYS[1]] else [])
   assert not row['exact_audit_executed'] and row['full_geometry_validity']=='NOT_TESTED'
   ms=[q[k]-fixed['quality'][k] for k in KEYS]
   for j,k in enumerate(KEYS):close(ms[j],row['margin_vs_fixed_deg'][k])
   margins.append(ms);all_pass+=int(all(passed.values()))
   if not all(passed.values()):first_failed.append(float(lam))
   geometries.append({'uid':uid,'kind':kind,'lambda':float(lam),'min_angle_deg':float(minimum.min()),'p05_deg':float(np.quantile(minimum,.05)),'area_min':float(area2.min()*.5),'area_total':float(area2.sum()*.5),'direction_det_min':float(det.min()),'angles_pass':all(passed.values()),'geometry_array_file_sha256':pair['geometry_file_sha256']})
  for subset in pair['error_subsets'].values():
   ids=np.array(subset['vertex_ids'],dtype=np.int64);assert len(ids)==subset['vertex_count'] and np.array_equal(ids,np.unique(ids))
   if not len(ids):assert subset['direction_coordinate_mse'] is None;continue
   mse=float(chord[ids].mean());close(mse,subset['direction_coordinate_mse']);close(np.sqrt(np.mean(angular[ids]**2)),subset['angular_rmse_deg']);close(angular[ids].max(),subset['angular_max_deg']);close(mse/chord.mean(),subset['coordinate_mse_over_global'])
  assert pair['error_subsets']['teacher_bottom26_corner_vertices']['vertex_ids']==np.unique(faces[orders[0][:26]]).tolist()
  assert pair['error_subsets']['teacher_p05_support_corner_vertices']['vertex_ids']==np.unique(faces[orders[0][[25,26]]]).tolist()
  a=np.array(margins);d=np.diff(a,axis=0)
  records.append({'uid':uid,'kind':kind,'global_direction_rmse_deg':pair['error_subsets']['global']['angular_rmse_deg'],'teacher_tail_coordinate_MSE_over_global':pair['error_subsets']['teacher_bottom26_corner_vertices']['coordinate_mse_over_global'],'first_failed_sampled_lambda':first_failed[0] if first_failed else None,'failed_sampled_lambdas':first_failed,'final_min_p05_margins_deg':margins[-1],'teacher_tail_retained_at_student':pair['rows'][-1]['teacher_tail_retained_count'],'minimum_margin_sampled_nonmonotonic':bool(np.any(d[:,0]>1e-12) and np.any(d[:,0]<-1e-12)),'p05_margin_sampled_nonmonotonic':bool(np.any(d[:,1]>1e-12) and np.any(d[:,1]<-1e-12))})
 assert seen=={(u,k) for u in p['uids'] for k in p['kinds']} and len(geometries)==72
 assert all_pass==result['angle_condition_pass_count'] and result['direction_det_nonpositive_geometry_count']==result['degenerate_geometry_count']==0
 assert all(sha(RUN/n)==h for n,h in before.items())
 report={'status':'PASS_ONE_INDEPENDENT_SAVED_ARRAY_READBACK','input_sha256':before,'source_sha256':sha(__file__),'pairs':8,'saved_geometries_checked':72,'fixed_lambdas':[i/8 for i in range(9)],'endpoint_array_bindings_exact':16,'same_frozen_field_identity_checked':True,'angle_area_direction_det_recomputed_from_saved_arrays':True,'angle_condition_pass_count':all_pass,'nonpositive_direction_determinant_geometries':0,'degenerate_geometries':0,'per_pair':records,'per_geometry':geometries,'no_field_radius_query':True,'no_model_network_training_GPU_or_MC':True,'exact_audit_repeated':False,'original_bound_run_files_unchanged':True,'interpretation':'First failure is only the first failed sampled lambda on this normalized-chord path; no universal error threshold, continuous crossing estimate, monotonicity guarantee, causality proof, validity certification, or replacement method output.'}
 dump(OUT/'READBACK.json',report)
 lines=['# Independent saved-array readback','','PASS: 8 fixed pairs x 9 registered lambdas = 72 saved geometries. All 16 teacher/student endpoint arrays match their original saved u/r/vertices exactly. Field identity, face connectivity and file/typed-array hashes agree with the original endpoint ledger.','','Minimum angle, linear p05, all per-face areas, area totals, and direction determinants were independently recomputed from saved arrays. No radius queries, network forward, training, GPU calls, Monte Carlo sampling, exact intersection audit, or new output selection were performed.','',f'All 72 have positive direction determinants and no recorded area degeneracy; {all_pass}/72 pass both angle conditions. These conditions do not certify full geometric validity.','','| UID / target | first failed sampled lambda | failed sampled lambdas | final min / p05 margins (deg) | teacher bottom26 retained |','|---|---:|---|---:|---:|']
 for x in records:lines.append(f"| {x['uid'][:8]} / {x['kind']} | {x['first_failed_sampled_lambda']} | {x['failed_sampled_lambdas']} | {x['final_min_p05_margins_deg'][0]:+.6f} / {x['final_min_p05_margins_deg'][1]:+.6f} | {x['teacher_tail_retained_at_student']} |")
 lines+=['','The p05 definition is the linear interpolation of zero-based sorted ranks25 and26, with weights0.45 and0.55. The bottom26 membership set is recorded separately; it does not include all p05 support faces. Face identities can change along the path.','','The first failed lambda is a statement about the fixed nine-point diagnostic grid, not the exact continuous first crossing or a universal admissible network error. Angle trends can be nonmonotonic. Global RMS and local tail enrichment are descriptive: neither alone certifies quality or identifies causal vertices. The original qualitative conclusion is consistent with the saved data, provided it is stated as lack of a demonstrated quality guarantee from global error, not the mathematical impossibility of any norm/geometry-dependent bound.','','The interpolants are diagnostic geometries, not additional method outputs. Existing endpoint radius identity was verified by exact saved-array comparison; intermediate radii were not independently queried in this review. Full validity is NOT_TESTED for these interpolants. No scientific performance claim follows from their angle pass counts.','','Evidence: READBACK.json and readback.py. Original run files were unchanged.']
 with (OUT/'READBACK.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
 print(json.dumps({'status':report['status'],'checked':72,'angle_pass':all_pass,'per_pair':records},indent=2))
if __name__=='__main__':main()