import json,hashlib,time,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
root=Path('/mnt/e/MeshFlow'); out=root/'native_t1/runs/chair_bounded_attachment_v1'; started=time.perf_counter()
sources={}; failures=[]; checks=Counter(); rows=[]
def path(p):
 s=str(p).replace('\\','/')
 for prefix in ['E:/MeshFlow/','e:/MeshFlow/','/mnt/e/MeshFlow/']:
  if s.startswith(prefix): return root/s[len(prefix):]
 q=Path(s); return q if q.is_absolute() else root/q
def file_bytes(p):
 p=path(p); b=p.read_bytes(); sources[str(p)]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)); return b
def read(p): return json.loads(file_bytes(p))
def arr(p,keys):
 p=path(p); file_bytes(p)
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def check(name,passed,detail=None):
 checks[name]+=1
 if not bool(passed):failures.append(dict(check=name,detail=detail))
def identity(p,expected):
 file_bytes(p);check('recorded_file_hash',sources[str(path(p))]['sha256']==expected,str(p))
def point(p):return tuple(float(x) for x in p)
def edges(t):
 d=Counter()
 for face in t:
  for a,b in [(0,1),(1,2),(2,0)]:d[tuple(sorted([point(face[a]),point(face[b])]))]+=1
 return d
def measure(t):
 t=np.asarray(t,np.float64);n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);a=np.linalg.norm(n,axis=1)*.5;ang=[]
 for j in range(3):
  v=t[:,(j+1)%3]-t[:,j];w=t[:,(j+2)%3]-t[:,j]
  ang.append(np.degrees(np.arctan2(np.linalg.norm(np.cross(v,w),axis=1),np.einsum('ij,ij->i',v,w))))
 return a,n,np.min(np.stack(ang),axis=0)
requests=read(out/'requests.json');req=[r for r in requests if r['label']=='ENDPOINT'];summary=read(out/'endpoint_summary.json');config=read(out/'projection_config.json')
check('sixteen_registered',len(req)==16)
parents=sorted({r['uid'] for r in req});check('eight_registered_parents',len(parents)==8)
for u in parents:check('exact_two_registered_noises',sorted(r['candidate'] for r in req if r['uid']==u)==[0,1],u)
for r in req:
 rid=r['request_id']; result=read(out/'outputs'/rid/'result.json'); rawresult=read(out/'outputs'/('RAW__'+r['base_id'])/'result.json')
 for p,h in [('raw_path','raw_sha256'),('intervention_log_path','intervention_log_sha256'),('geometry_arrays_path','geometry_arrays_sha256')]:identity(result[p],result[h])
 identity(rawresult['raw_path'],rawresult['raw_sha256']);identity(r['input_npz'],r['input_sha256'])
 original=arr(rawresult['raw_path'],['raw'])['raw']; resultarrays=arr(result['raw_path'],['raw','C','noise','known_mask','valid_mask']); current=resultarrays['raw']; bank=arr(r['input_npz'],['C','z','known_mask','valid_mask']); C=bank['C'].reshape(-1,3,3)
 log=read(result['intervention_log_path']);geo=arr(result['geometry_arrays_path'],['corner_displacement','displacement_vector','changed_corners','changed_faces'])
 K=int(r['K']);N=int(r['N']);F=original[K:].reshape(-1,3,3);G=current[K:].reshape(-1,3,3)
 check('status_projected',result['status']=='PROJECTED',rid);check('FP32_N_shape',current.dtype==original.dtype==np.float32 and current.shape==original.shape==(N,9),rid)
 check('C_raw_bytes',current[:K].tobytes()==original[:K].tobytes()==C.reshape(K,9).tobytes(),rid)
 check('C_saved_array_bytes',resultarrays['C'].tobytes()==bank['C'].tobytes(),rid)
 check('finite_result',np.isfinite(current).all(),rid)
 for k,other in [('noise','z'),('known_mask','known_mask'),('valid_mask','valid_mask')]:check('bank_identity_'+k,resultarrays[k].tobytes()==bank[other].tobytes(),rid)
 LC=float(np.linalg.norm(np.ptp(C.astype(np.float64).reshape(-1,3),axis=0)));delta=.01*LC
 change=G.astype(np.float64)-F.astype(np.float64);disp=np.linalg.norm(change,axis=2);changed=np.any(G!=F,axis=2);changedfaces=np.flatnonzero(changed.any(1)).tolist();unchanged=current.tobytes()==original.tobytes()
 check('C_only_scale',LC==log['L_C'] and delta==log['delta']==result['delta'],rid)
 check('actual_budget',np.all(disp<=delta),rid);check('remaining_budget',np.all(disp<=np.asarray(log['remaining_budget_input'])),rid)
 check('saved_corner_displacement',np.array_equal(disp,geo['corner_displacement']) and np.array_equal(disp,np.asarray(log['per_corner_displacement'])),rid)
 check('saved_displacement_vector',np.array_equal(change,geo['displacement_vector']),rid)
 check('saved_changed_mask',np.array_equal(changed,geo['changed_corners']) and np.array_equal(changed.any(1),geo['changed_faces']),rid)
 check('logged_changed_indices',np.argwhere(changed).tolist()==log['changed_corners'] and changedfaces==log['changed_faces'],rid)
 check('actual_changed_status',result['changed']==(not unchanged),rid)
 accepted=log['accepted_constraints'];check('accepted_count',len(accepted)==log['accepted_count']==result['accepted_count'],rid)
 check('accepted_status_ids',set(a['candidate_id'] for a in accepted)==set(a['candidate_id'] for a in log['candidates'] if a['status']=='ACCEPTED'),rid)
 check('one_edge_per_free_face',len({a['free_face'] for a in accepted})==len(accepted),rid)
 check('one_edge_per_C_anchor',len({tuple(map(tuple,a['known_edge'])) for a in accepted})==len(accepted),rid)
 cedges=edges(C);fedges=edges(F);groups=defaultdict(list)
 for i,face in enumerate(F):
  for j,p in enumerate(face):groups[point(p)].append((i,j))
 cp={point(p) for p in C.reshape(-1,3)}
 for p,occ in groups.items():
  check('exact_groups_preserved',all(np.array_equal(G[i,j],G[occ[0]]) for i,j in occ),rid)
  if p in cp:check('fixed_groups_preserved',all(G[i,j].tobytes()==F[i,j].tobytes() for i,j in occ),rid)
 max_residual=0.;local_rows=[]
 for a in accepted:
  e=np.asarray(a['known_edge'],np.float64);g=np.asarray(a['free_edge'],np.float64);target=np.asarray(a['target_points'],np.float32);face=a['free_face'];corners=a['free_corners'];knownface=a['known_face']
  check('accepted_C_incidence_one',cedges[tuple(sorted(map(tuple,e))) ]==1,rid)
  check('accepted_free_incidence_one',fedges[tuple(sorted(map(tuple,g))) ]==1,rid)
  check('original_edge_coordinates',np.array_equal(F[face,corners],g),rid)
  check('C_target_membership',all(any(np.array_equal(p,q) for q in C[knownface]) for p in target),rid)
  check('C_target_edge_endpoints',set(map(tuple,target))==set(map(tuple,e)),rid)
  check('accepted_equation',np.array_equal(G[face,corners],target),rid)
  le=np.linalg.norm(e[1]-e[0]);lg=np.linalg.norm(g[1]-g[0]);local=min(delta,.1*min(le,lg));move=np.linalg.norm(g-target.astype(np.float64),axis=1)
  check('actual_local_edge_budget',np.all(move<=local),rid)
  cosine=abs(np.dot(e[1]-e[0],g[1]-g[0]))/(le*lg);angle=math.degrees(math.acos(float(np.clip(cosine,0.,1.))))
  check('angle_within_30',angle<=30.,rid)
  record=next(x for x in log['candidates'] if x['candidate_id']==a['candidate_id'])
  costs=[sum(np.linalg.norm(g-target.astype(np.float64),axis=1)**2),sum(np.linalg.norm(g-target[::-1].astype(np.float64),axis=1)**2)]
  check('lower_cost_orientation',costs[0]<=costs[1],rid)
  check('recorded_cost_and_radius',float(costs[0])==record['cost'] and local==record['delta_edge'],rid)
  for key in ['known_edge','free_edge']:
   competitors=sorted(x['cost'] for x in log['candidates'] if x[key]==record[key])
   check('mutual_best_'+key,record['cost']==competitors[0],rid)
   if len(competitors)>1:check('unambiguous_margin_'+key,competitors[0]<=.8*competitors[1] and competitors[1]-competitors[0]>64*np.finfo(np.float64).eps*LC**2,rid)
  for gr in a['groups']:
   p=tuple(gr['original_point']);targetp=np.asarray(gr['target_point'],np.float32)
   check('exact_group_occurrence_list',sorted(map(tuple,gr['face_corners']))==sorted(groups[p]),rid)
   for i,j in gr['face_corners']:
    residual=float(np.linalg.norm(G[i,j].astype(np.float64)-targetp.astype(np.float64)));max_residual=max(max_residual,residual)
    check('all_group_endpoint_equations',residual==0.,rid)
  local_rows.append(dict(known_face=knownface,free_face=face,free_corners=corners,delta_edge=local,endpoint_displacements=move.tolist(),angle_degrees=angle))
 check('constraint_residual_recomputed',max_residual==log['maximum_constraint_residual']==0.,rid)
 if not accepted:check('no_accepted_bitwise_RAW',unchanged,rid)
 before=set(map(tuple,log['geometry_events']['before']));after=set(map(tuple,log['geometry_events']['after']));added=set(map(tuple,log['geometry_events']['added']));removed=set(map(tuple,log['geometry_events']['removed']))
 check('typed_event_schema',all(len(e)==4 and e[0] in ['CF','FF'] for e in before|after),rid)
 check('event_difference_arithmetic',added==after-before and removed==before-after,rid)
 check('no_added_actual_event',unchanged or not added,rid)
 aa,an,ang=measure(F);ba,bn,bang=measure(G);safety=[]
 for i in changedfaces:
  ratio=float(ba[i]/aa[i]);dot=float(np.dot(an[i],bn[i]));ok=aa[i]>1e-11*LC**2 and ba[i]>1e-11*LC**2 and .5<=ratio<=2 and dot>0 and bang[i]>=min(ang[i],5)-1e-7
  check('affected_face_safety_recomputed',ok,[rid,i]);safety.append(dict(face=i,area_ratio=ratio,normal_dot=dot,old_minimum_angle=float(ang[i]),new_minimum_angle=float(bang[i])))
 if accepted:
  check('final_reasons_empty',not log['final_checks']['reasons'],rid)
  check('final_events_consistent',set(map(tuple,log['final_checks']['old_events']))==before and set(map(tuple,log['final_checks']['new_events']))==after,rid)
 check('no_model_execution',log['model_forwards']==log['model_backwards']==log['optimizer_updates']==result['model_calls']==0,rid)
 check('fixed_config_unchanged',log['fixed_config']==config,rid)
 rows.append(dict(request_id=rid,uid=r['uid'],candidate=r['candidate'],status=log['status'],changed=not unchanged,accepted_edges=len(accepted),changed_corner_count=int(changed.sum()),changed_face_count=len(changedfaces),free_face_count=len(F),maximum_displacement=float(disp.max(initial=0)),RMS_displacement=float(np.sqrt(np.mean(disp**2))),delta=delta,max_displacement_over_delta=float(disp.max(initial=0)/delta),constraint_residual=max_residual,accepted_local_edges=local_rows,face_safety=safety,events_before=[list(x) for x in sorted(before)],events_after=[list(x) for x in sorted(after)],events_added=[list(x) for x in sorted(added)],events_removed=[list(x) for x in sorted(removed)]))
changed_parents=sorted({r['uid'] for r in rows if r['changed']});changedrows=[r for r in rows if r['changed']]
check('summary_actual_counts',summary['outputs']==16 and summary['modified']==len(changedrows) and summary['accepted_edges']==sum(r['accepted_edges'] for r in rows))
check('fixed_actual_pattern',len(changedrows)==1 and changedrows[0]['uid']=='86ec9c796a6c1789e792598ed58dd1cc' and changedrows[0]['candidate']==0 and changedrows[0]['accepted_edges']==3)
for p,identityrow in sources.items():check('source_unchanged_after_audit',hashlib.sha256(Path(p).read_bytes()).hexdigest()==identityrow['sha256'],p)
report=dict(schema='bounded_attachment_endpoint_independent_audit_v1',status='PASS' if not failures else 'FAIL',scope='Read-only saved arrays/log arithmetic; no project, classifier, scoring or model calls.',checks=sum(checks.values()),check_counts=dict(checks),issues=failures,registered_outputs=16,parent_denominator=8,candidates_per_parent=2,modified_outputs=len(changedrows),modified_parents=changed_parents,accepted_edges=sum(r['accepted_edges'] for r in rows),unchanged_outputs=sum(not r['changed'] for r in rows),rows=rows,necessary_parent_gate_observation=dict(formal_gate_written=False,required_improving_parents=4,maximum_possible_improving_parents=len(changed_parents),necessary_parent_count_cannot_be_met=len(changed_parents)<4,reason='With identical frozen evaluation, the seven entirely unchanged parents have zero differences. Only the single changed parent can improve, so at most 1/8 can improve and the eight-parent paired median cannot be positive. This is a necessary-condition audit, not a replacement gate computation.'),limits=['Saved typed event sets were checked for exact before/after differences; no new intersection classifier calls were made.','Two old CF proper and two old FF proper events remain on affected pairs in the changed output. No claim of a globally intersection-free mesh.','C-only LC and actual saved coordinates determine budget checks; no GT metric was used to accept or reject output.'],consumed_files=len(sources),consumed_bytes=sum(x['bytes'] for x in sources.values()),sources=sources,project_calls=0,classifier_calls=0,scoring_calls=0,model_calls=0,seconds=time.perf_counter()-started)
folder=out/'audit';folder.mkdir(exist_ok=True);p=folder/'endpoint_independent.json';p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ['status','checks','issues','modified_outputs','modified_parents','accepted_edges','unchanged_outputs','necessary_parent_gate_observation','consumed_files','consumed_bytes','seconds']},indent=2));print(json.dumps(changedrows,indent=2))
