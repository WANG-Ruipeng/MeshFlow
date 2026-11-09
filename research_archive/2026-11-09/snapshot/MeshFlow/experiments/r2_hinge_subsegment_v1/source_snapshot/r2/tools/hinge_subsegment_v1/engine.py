"""Finite CPU-only seam-side split trials with explicit child provenance.

Only the eight registered formal census opportunities are tested. No candidate
expansion, face cleanup, model, source completion, new noise or scorer changes.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,csv,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from . import census as S

C,PE,Q,B=S.C,S.PE,S.O.Q,S.O.B
OUT=S.OUT/'trials'
VERSION='seam_subsegment_child_lineage_split_v1'


def initial(base):
    n=len(base['faces']);K=int(base['known_faces'])
    return dict(vertices=base['vertices'].copy(),faces=base['faces'].copy(),
        child_face_id=np.arange(n,dtype=np.int64),parent_post_face_id=np.arange(n,dtype=np.int64),
        parent_source_raw_slot=base['source_face_ids'].copy(),
        child_seam_mask=base['raw_seam_slot_mask'][base['source_face_ids']].copy(),
        corner_parent_barycentric=np.broadcast_to(np.eye(3),(n,3,3)).copy(),
        base_parent_source_corner_ids=base['source_corner_ids'].copy(),
        base_output_face_count=np.asarray(n),base_free_face_count=np.asarray(n-K),
        known_faces=np.asarray(K))


def exported(state,base):
    # Do not duplicate raw-slot IDs as child IDs or reuse a [RAW slot,corner]
    # identity for a new interpolated split vertex.
    drop={'vertices','faces','output','C','free','source_face_ids','source_corner_ids',
          'output_anchor_ids','face_corner_permutation','kept_source_face_ids'}
    historical={'raw_slot_ids','retained_seam_mask','corner_permutation_to_MODEL_POST','free_face_flips',
        'original_POST_vertex_to_hinge_vertex','index_domain','identity_domain'}
    out={('base_'+k if k in historical or k.startswith('output_corner_') or 'displacement' in k else k):v.copy()
         for k,v in base.items() if k not in drop}
    out.update({k:v.copy() for k,v in state.items()})
    v,f=state['vertices'],state['faces'];K=int(state['known_faces'])
    out.update(output=v[f],C=v[f[:K]],free=v[f[K:]],index_domain=np.asarray('HINGE_SUBSEGMENT_CHILD_LINEAGE_V1'))
    anchor=np.full(f.shape,-1,np.int64)
    for i,(p,bb) in enumerate(zip(state['parent_post_face_id'],state['corner_parent_barycentric'])):
        for j,bary in enumerate(bb):
            for k in range(3):
                if np.array_equal(bary,np.eye(3)[k]):anchor[i,j]=base['output_anchor_ids'][p,k]
    out['output_anchor_ids']=anchor
    return out


def pair_records(pairs,parent,K,selected):
    result={kind:set() for kind in ('CF','FF')};details=[]
    for kind in result:
        for row in pairs[kind]['contacts']:
            if row['classification'] not in selected:continue
            a,b=map(int,row['pair']);i,j=(a,K+b) if kind=='CF' else (K+a,K+b)
            key=tuple(sorted((int(parent[i]),int(parent[j]))))
            result[kind].add(key)
            details.append(dict(kind=kind,parent_pair=key,face_rows=[i,j],classification=row['classification']))
    return result,details


def edge_records(state):
    f=state['faces'];parent=state['parent_post_face_id'];result={}
    for key,occ in PE.incidence(f).items():
        ids=tuple(sorted(int(parent[i]) for i,a,b in occ));signature=(tuple(map(int,key)),ids)
        opposite=None
        if len(occ)==2:
            (i,a,b),(j,c,d)=occ;opposite=bool(f[i,a]==f[j,d] and f[i,b]==f[j,c] and key[0]!=key[1])
        result[signature]=dict(vertex_ids=list(key),parent_incidents=list(ids),incidence=len(occ),winding_opposite=opposite,
            child_incidents=[int(state['child_face_id'][i]) for i,a,b in occ])
    return result


def duplicate_relations(state):
    v,f,p=state['vertices'],state['faces'],state['parent_post_face_id'];groups=collections.defaultdict(list)
    for i,tri in enumerate(v[f]):
        key=tuple(sorted(tuple(map(int,c)) for c in np.ascontiguousarray(tri).view(np.uint32)))
        groups[key].append(i)
    return {tuple(sorted((int(p[i]),int(p[j])))) for ids in groups.values() for z,i in enumerate(ids) for j in ids[z+1:]}


def graph(state,pairs):
    _,edges,_=Q.helpers()['G'].actual_valid_graph(state['vertices'],state['faces'],int(state['known_faces']),pairs)
    return edges


def legal_signatures(state,edges):
    pp=state['parent_post_face_id'];out={}
    for row in edges:
        if row['valid_edge']:
            key=((int(row['vertex_u']),int(row['vertex_v'])),tuple(sorted(int(pp[i]) for i in row['incident_face_ids'])))
            out[key]=row
    return out


def build_trial(current,cand,base,pre,delta):
    """Move one existing vertex cluster; split one seam triangle into two."""
    h,far,u=[int(cand[k]) for k in ('hinge_vertex','seam_far_vertex','bulk_endpoint_vertex')]
    parent=int(cand['seam_post_face_id']);bulk_parent=int(cand['bulk_post_face_id'])
    f=current['faces'];v=current['vertices'];pp=current['parent_post_face_id'];inc=PE.incidence(f)
    matches=[i for i in np.flatnonzero(pp==parent) if h in f[i] and far in f[i]]
    info=dict(hinge_vertex=h,seam_far_vertex=far,moved_vertex=u)
    if len(matches)!=1:return None,info,'SEAM_SIDE_NO_LONGER_SINGLE_CHILD'
    i=matches[0];se=tuple(sorted((h,far)));be=tuple(sorted((h,u)))
    if len(inc.get(se,()))!=1 or len(inc.get(be,()))!=1:return None,info,'NO_LONGER_OPEN'
    j,ba,bb=inc[be][0]
    if pp[j]!=bulk_parent:return None,info,'BULK_LINEAGE_CHANGED'
    _,a,b=inc[se][0];third=3-a-b
    if (f[i,a]==h)==(f[j,ba]==h):return None,info,'CURRENT_SAME_WINDING'
    axis=v[far].astype(float)-v[h];ray=v[u].astype(float)-v[h];t=float(ray@axis/(axis@axis))
    proj=v[h].astype(float)+t*axis;target=proj.astype(np.float32)
    ang=float(np.degrees(np.arccos(np.clip(abs(ray@axis)/(np.linalg.norm(ray)*np.linalg.norm(axis)),-1,1))))
    members=np.flatnonzero(base['corner_vertex_ids'].reshape(-1)==u)
    disp=np.linalg.norm(pre.reshape(-1,3)[members].astype(float)-target.astype(float),axis=1)
    fixed=set(map(int,f[:int(current['known_faces'])].reshape(-1)))
    info.update(t_projection=t,axis_angle_deg=ang,projected_FP64=proj.tolist(),projected_FP32=target.tolist(),
        projected_rounding_error=float(np.linalg.norm(target.astype(float)-proj)),original_cluster_members=members.tolist(),
        original_PRE_max_displacement=float(disp.max()),incremental_POST_displacement=float(np.linalg.norm(v[u]-target)),
        current_seam_child_id=int(current['child_face_id'][i]),current_bulk_child_id=int(current['child_face_id'][j]))
    if not .15<=t<=.85:return None,info,'CURRENT_T_OUTSIDE_FORMAL'
    if ang>20+1e-12:return None,info,'CURRENT_AXIS_OVER_20'
    if np.linalg.norm(ray-t*axis)>delta*C.L+1e-12:return None,info,'CURRENT_SEGMENT_OVER_BUDGET'
    if u in fixed and v[u].tobytes()!=target.tobytes():return None,info,'FIXED_ANCHOR_MOVEMENT'
    if disp.max()>delta*C.L+1e-12:return None,info,'ORIGINAL_CLUSTER_OVER_BUDGET'
    trial={k:z.copy() for k,z in current.items()};trial['vertices'][u]=target
    # Directed parent a->b->third; child orientation follows the parent.
    children=np.asarray([[f[i,a],u,f[i,third]],[u,f[i,b],f[i,third]]],np.int64)
    oldb=current['corner_parent_barycentric'][i]
    hc=int(np.flatnonzero(f[i]==h)[0]);fc=int(np.flatnonzero(f[i]==far)[0])
    splitb=(1-t)*oldb[hc]+t*oldb[fc]
    bary=np.asarray([[oldb[a],splitb,oldb[third]],[splitb,oldb[b],oldb[third]]])
    ids=np.asarray([int(current['child_face_id'].max())+1,int(current['child_face_id'].max())+2])
    for key,append in [('faces',children),('child_face_id',ids),('parent_post_face_id',np.asarray([parent,parent])),
                       ('parent_source_raw_slot',np.repeat(current['parent_source_raw_slot'][i],2)),
                       ('child_seam_mask',np.ones(2,bool)),('corner_parent_barycentric',bary)]:
        trial[key]=np.concatenate((np.delete(current[key],i,axis=0),append),axis=0)
    childrows=np.arange(len(trial['faces'])-2,len(trial['faces']))
    current_affected=np.flatnonzero(np.any(f==u,axis=1));childids=set(map(int,current['child_face_id'][current_affected]))
    affected=np.flatnonzero(np.isin(trial['child_face_id'],list(childids))|np.isin(trial['child_face_id'],ids))
    info.update(split_parent_post_face_id=parent,split_parent_source_raw_slot=int(base['source_face_ids'][parent]),
        new_child_face_ids=ids.tolist(),new_child_rows=childrows.tolist(),affected_trial_rows=affected.tolist(),
        affected_parent_post_face_ids=sorted(set(map(int,trial['parent_post_face_id'][affected]))),
        affected_child_ids=trial['child_face_id'][affected].tolist(),intended_seam_bulk_edge=list(be),
        new_internal_child_edge=sorted((u,int(f[i,third]))))
    return trial,info,None


def evaluate(current,trial,info,base,baseline_pairs,current_pairs,baseline_edges,current_edges):
    K=int(current['known_faces']);v,f=trial['vertices'],trial['faces'];tri=v[f];p=trial['parent_post_face_id']
    aff=np.asarray(info['affected_trial_rows']);children=np.asarray(info['new_child_rows']);failed=[];detail={}
    if not np.array_equal(f[:K],base['faces'][:K]) or tri[:K].tobytes()!=base['output'][:K].tobytes():raise AssertionError('C changed')
    fixed=set(map(int,base['faces'][:K].reshape(-1)));assert all(v[z].tobytes()==base['vertices'][z].tobytes() for z in fixed)
    ss=PE.shape(tri);old=PE.shape(current['vertices'][current['faces']]);orig=PE.shape(base['output'])
    lookup={int(z):i for i,z in enumerate(current['child_face_id'])};rows=[]
    for i in aff:
        child=int(trial['child_face_id'][i]);isnew=i in children;parent=int(p[i]);before=lookup.get(child)
        ref_angle=old['angle'][before] if before is not None else old['angle'][np.flatnonzero(current['child_face_id']==info['current_seam_child_id'])[0]]
        ref_q=old['q'][before] if before is not None else orig['q'][parent]
        cos0=float(ss['normal'][i]@orig['normal'][parent]);coscur=float(ss['normal'][i]@old['normal'][before]) if before is not None else float(ss['normal'][i]@orig['normal'][parent])
        bad=[]
        if len(set(map(int,f[i])))<3 or ss['area'][i]<=1e-12*C.L*C.L:bad.append('COLLAPSED_OR_DEGENERATE')
        if isnew:
            if ss['angle'][i]<5-1e-10:bad.append('CHILD_ANGLE_LT5')
            if ss['q'][i]>10+1e-10:bad.append('CHILD_Q_GT10')
        else:
            if ss['angle'][i]<5 and (ss['angle'][i]<ref_angle-1e-10 or ss['angle'][i]<orig['angle'][parent]-1e-10):bad.append('ANGLE_NEW_OR_WORSE')
            if ss['q'][i]>10 and (ss['q'][i]>ref_q+1e-10 or ss['q'][i]>orig['q'][parent]+1e-10):bad.append('Q_NEW_OR_WORSE')
        if min(cos0,coscur)<.5-1e-12:bad.append('NORMAL_ROTATION_GT60')
        failed.extend(bad);rows.append(dict(child_face_id=child,parent_post_face_id=parent,is_new_child=isnew,area=ss['area'][i],
            minimum_angle=ss['angle'][i],q=ss['q'][i],previous_minimum_angle=ref_angle,previous_q=ref_q,
            normal_rotation_original_deg=float(np.degrees(np.arccos(np.clip(cos0,-1,1)))),failures=bad))
    detail['affected_face_shapes']=rows
    parent=info['split_parent_post_face_id'];pt=base['output'][parent].astype(float);n=orig['normal'][parent]
    target=v[info['moved_vertex']].astype(float);edge=base['vertices'][[info['hinge_vertex'],info['seam_far_vertex']]].astype(float)
    detail['split_surface_errors']=dict(children_area=float(ss['area'][children].sum()),original_parent_area=float(orig['area'][parent]),
        area_difference=float(ss['area'][children].sum()-orig['area'][parent]),plane_distance=float(abs((target-pt[0])@n)),
        line_distance=float(np.linalg.norm(np.cross(target-edge[0],edge[1]-edge[0]))/np.linalg.norm(edge[1]-edge[0])))
    ce,te=edge_records(current),edge_records(trial)
    newedge=[val for key,val in te.items() if key not in ce];newnm=[e for e in newedge if e['incidence']>2]
    same=[e for e in newedge if e['incidence']==2 and not e['winding_opposite']]
    dups=duplicate_relations(trial)-duplicate_relations(current)
    detail.update(new_edge_relations=newedge,new_nonmanifold=newnm,new_same_winding=same,new_duplicate_parent_pairs=sorted(dups))
    if newnm:failed.append('NEW_NONMANIFOLD')
    if same:failed.append('NEW_SAME_WINDING')
    if dups:failed.append('NEW_DUPLICATE')
    # Full original detector only; these are collision checks, not 135-field rescoring.
    pairs=dict(CF=Q.pair_scan(tri[:K],tri[K:],C.L),FF=Q.pair_scan(tri[K:],tri[K:],C.L,self_pairs=True))
    bad,bd=pair_records(pairs,p,K,Q.ILLEGAL);ob,_=pair_records(baseline_pairs,np.arange(len(base['faces'])),K,Q.ILLEGAL);cb,_=pair_records(current_pairs,current['parent_post_face_id'],K,Q.ILLEGAL)
    newbad={kind:sorted(bad[kind]-(ob[kind]&cb[kind])) for kind in ('CF','FF')}
    for kind in newbad:
        if newbad[kind]:failed.append('NEW_'+kind+'_ILLEGAL')
    unknown_classes={r['classification'] for kind in pairs for r in pairs[kind]['contacts'] if 'unresolved' in r['classification'] or 'degenerate' in r['classification']}
    unknown,ud=pair_records(pairs,p,K,unknown_classes);oldunknown,_=pair_records(current_pairs,current['parent_post_face_id'],K,unknown_classes)
    newunknown={kind:sorted(unknown[kind]-oldunknown[kind]) for kind in unknown}
    detail.update(new_illegal_parent_pairs=newbad,all_illegal_child_pairs=bd,unresolved_child_pairs=ud,
        new_unresolved_parent_pairs=newunknown)
    if any(newunknown.values()):failed.append('NEW_UNRESOLVED_CONTACT')
    edges=graph(trial,pairs);newlegal=legal_signatures(trial,edges)
    missing=set(legal_signatures(current,current_edges))-set(newlegal)
    detail['lost_legal_edges']=[dict(vertex_ids=e[0],parent_incidents=e[1]) for e in sorted(missing)]
    if missing:failed.append('LOST_LEGAL_EDGE')
    intended=tuple(info['intended_seam_bulk_edge']);parentbulk=int(p[np.flatnonzero(trial['child_face_id']==info['current_bulk_child_id'])[0]])
    target=[r for r in edges if (r['vertex_u'],r['vertex_v'])==intended]
    detail['target_edge_records']=target
    if len(target)!=1 or not target[0]['valid_edge'] or not target[0]['winding_opposite']:failed.append('SPLIT_TARGET_NOT_LEGAL_OPPOSITE')
    detail['open_edges_before']=sum(z['incidence']==1 for z in ce.values());detail['open_edges_after']=sum(z['incidence']==1 for z in te.values())
    detail['all_guard_failures']=list(dict.fromkeys(failed))
    return detail,pairs,edges


def candidates():
    with (S.DEST/'per_candidate.csv').open(encoding='utf-8-sig',newline='') as file:
        rows=[]
        for raw in csv.DictReader(file):
            if raw['formal_geometric_eligible']!='True':continue
            row={}
            for k,v in raw.items():
                try:row[k]=json.loads(v)
                except (ValueError,TypeError):row[k]=v
            rows.append(row)
    assert len(rows)==8
    return rows


def run():
    done=OUT/'completion.json'
    if done.exists():r=C.read(done);B.verify(r['files']);return r
    frozen=S.OUT/'split_protocol.json'
    if frozen.exists():raise RuntimeError('Prior incomplete split execution preserved; no silent retry')
    qs=S.O.OLD.requests();cand=candidates();sources=C.read(S.OUT/'census_protocol.json');B.verify(sources['protected']);B.verify(sources['code'])
    C.resources(S.OUT/'resources/before_split.json')
    code={str(Path(__file__)):C.sha(__file__),str(Path(S.__file__)):C.sha(S.__file__),**Q.definition_hashes()}
    definition=dict(version=VERSION,new_unresolved_policy='Reject any newly unresolved/degenerate parent relation; preserve and explicitly report old unresolved cases.',
        shape_tolerance=1e-10,normal_cosine_tolerance=1e-12,original_displacement_tolerance=1e-12,
        guards='Record all tested guards even if the first one already rejects; never retry with weaker settings.',
        source_hash=code[str(Path(__file__))],parent_relations='Each illegal sibling pair maps to (parent,parent), always new; old legal edge uses vertex-pair plus parent incidence.',
        accepted_identity='Unique child_face_id; original parent_post_face_id/raw slot and barycentric corner lineage; no fabricated original slot.')
    C.write(S.OUT/'execution_definition.json',definition)
    C.write(frozen,dict(status='FROZEN',version=VERSION,utc=C.utc(),workers=1,threads=1,GPU_workers=0,
        registered_candidates=cand,protected=sources['protected'],code=code,
        unknown_policy=definition['new_unresolved_policy'],
        budget='56 fixed independent cells; eight formal candidate attempts, no expansion',base='HINGE_OPP015'))
    tick=time.perf_counter();summaries=[];allrecords=[]
    for q in qs:
        paths=S.paths(q);base=C.arrays(paths['mesh']);pre=C.arrays(paths['pre'])['output'];bp=C.read(paths['mesh'].parent/'pairs.json');be=S.T.edges_file(paths['edges'])
        for delta in S.DELTAS:
            dest=OUT/('SPLIT'+str(round(delta*1000)).zfill(3))/q['request_id'];C.write(dest/'started.json',dict(utc=C.utc(),request_id=q['request_id'],delta=delta))
            start=time.perf_counter();state=initial(base);curpairs=bp;curedges=be;records=[]
            queue=[r for r in cand if r['request_id']==q['request_id'] and r['delta_fraction']==delta]
            queue.sort(key=lambda r:(r['max_original_PRE_GEO_displacement_if_FP32_target'],r['seam_source_raw_slot'],r['seam_directed_source_corners'],r['bulk_source_raw_slot'],r['bulk_hinge_source_corner'],r['hinge_vertex']))
            for rank,candidate in enumerate(queue):
                trial,info,why=build_trial(state,candidate,base,pre,delta);record=dict(candidate=candidate,rank=rank,**info)
                if trial is not None:
                    detail,pairs,edges=evaluate(state,trial,info,base,bp,curpairs,be,curedges);record.update(detail)
                    why=detail['all_guard_failures'][0] if detail['all_guard_failures'] else 'ACCEPTED'
                    B.save_npz(dest/f'attempt_{rank:03d}.npz',**exported(trial,base))
                    C.write(dest/f'attempt_{rank:03d}_pairs.json',pairs)
                    C.csvwrite(dest/f'attempt_{rank:03d}_edges.csv',edges)
                    if why=='ACCEPTED':state,curpairs,curedges=trial,pairs,edges
                record['decision']=why;records.append(record);allrecords.append(dict(request_id=q['request_id'],split=q['split'],delta_fraction=delta,**record))
                print(json.dumps(dict(request=q['request_id'],delta=delta,rank=rank,decision=why,failures=record.get('all_guard_failures'))),flush=True)
            B.save_npz(dest/'post.npz',**exported(state,base));C.write(dest/'candidate_records.json',records);C.write(dest/'pairs.json',curpairs);C.csvwrite(dest/'valid_edges.csv',curedges)
            rec=dict(status='COMPLETE',request_id=q['request_id'],split=q['split'],delta_fraction=delta,registered_candidates=len(queue),accepted=sum(r['decision']=='ACCEPTED' for r in records),
                decisions=dict(collections.Counter(r['decision'] for r in records)),actual_faces=len(state['faces']),baseline_faces=len(base['faces']),
                seconds=time.perf_counter()-start,mesh_path=str(dest/'post.npz'),pairs_path=str(dest/'pairs.json'),edges_path=str(dest/'valid_edges.csv'),model_calls=0,
                files={str(z):C.sha(z) for z in dest.iterdir() if z.is_file()})
            C.write(dest/'complete.json',rec);summaries.append(rec)
    C.csvwrite(OUT/'per_request.csv',[{k:v for k,v in r.items() if k!='files'} for r in summaries]);C.write(OUT/'all_trials.json',allrecords)
    B.verify(sources['protected']);B.verify(sources['code']);B.verify(code);C.resources(S.OUT/'resources/after_split.json')
    result=dict(status='COMPLETE',version=VERSION,cells=len(summaries),candidate_attempts=len(allrecords),accepted=sum(r['accepted'] for r in summaries),
        decisions=dict(collections.Counter(r['decision'] for r in allrecords)),seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        CPU_only='torch' not in sys.modules,model_calls=0,records=summaries,code=code,files={str(z):C.sha(z) for z in OUT.rglob('*') if z.is_file()})
    C.write(done,result);return result


if __name__=='__main__':
    try:
        r=run();print(json.dumps({k:v for k,v in r.items() if k not in ('records','code','files')}),flush=True)
    except BaseException as exc:
        p=OUT/'execution_failure.json'
        if not p.exists():C.write(p,dict(utc=C.utc(),error=repr(exc),traceback=traceback.format_exc()))
        raise
