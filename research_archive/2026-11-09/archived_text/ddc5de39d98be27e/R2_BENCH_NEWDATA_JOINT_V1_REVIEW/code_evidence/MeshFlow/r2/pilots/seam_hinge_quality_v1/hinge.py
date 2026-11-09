"""CPU-only shared-vertex open-edge zipper with zero new CF/FF guards.

No model imports, radius reclustering, face deletion, reorientation or uncrossing.
Both free endpoint clusters move to their FP32 midpoint; C clusters never move.
All actual original GEO members, including mapped deleted-face members, obey a
cumulative displacement budget measured from the unchanged PRE_GEO positions.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, concurrent.futures, json, multiprocessing, resource, sys, time, traceback
from pathlib import Path
import numpy as np
from r2.pilots.seam_followup_v1 import binding as B
from r2.pilots.seam_pool47_couple_v1.endpoint_scoring import metric_row
from r2.pilots.owner_layout_v1.binding_adapter import validate_arrays

C,Q,PE=B.C,B.Q,B.PE
OUT=C.ROOT/'experiments/r2_seam_hinge_quality_v1/hinge'
OLD=C.ROOT/'experiments/r2_seam_quality_wave1_wave2_v1'
NEW=OUT.parent
VERSION='hinge_zip_original_POST_zero_new_CF_FF_v1'
ARMS=('PLAIN_S2000','CLEAN_EDGE_COH','SMOOTH_PRIOR')
DELTAS=(.015,.025)


def paths(arm,q):
    root=OLD if arm=='PLAIN_S2000' else NEW
    folder=root/'evaluation'/arm/q['request_id']
    score=root/'evaluation/scores'/arm/q['request_id']/'POST'
    return dict(post=folder/'post/post.npz',pre=folder/'post/pre_geo.npz',raw=folder/'raw/raw.npz',
        pairs=score/'pairs.json',score=score/'score.json')


def label(arm,delta): return arm+'_HINGE'+str(round(delta*1000)).zfill(3)


def requests():
    return C.read(OLD/'protocol.json')['evaluation']['requests']


def edge_keys(ledger):
    return {Q._edge_key(row) for row in ledger['edges'] if row['valid_edge']}


def nm_keys(faces,sf,sc):
    return {B.edge_identity(occ,sf,sc) for occ in PE.incidence(faces).values() if len(occ)>2}


def scan(tri,K):
    return dict(CF=Q.pair_scan(tri[:K],tri[K:],C.L),FF=Q.pair_scan(tri[K:],tri[K:],C.L,self_pairs=True))


def inventory(base,pre,raw,delta):
    v,f=base['vertices'],base['faces'];K=int(base['known_faces']);sf,sc=base['source_face_ids'],base['source_corner_ids']
    cv=base['corner_vertex_ids'].reshape(-1);members={i:np.flatnonzero(cv==i) for i in range(len(v))}
    if any(not len(m) for m in members.values()): raise ValueError('POST vertex without original GEO members')
    fixed=set(map(int,f[:K].reshape(-1)));roots=np.arange(len(v),dtype=np.int64)
    if not set(map(int,f[base['output_anchor_ids']>=0]))<=fixed: raise ValueError('Seam bases not actual C anchors')
    preflat=pre.reshape(-1,3).astype(float);by_hinge=collections.defaultdict(list);counts=collections.Counter()
    for edge,occ in sorted(PE.incidence(f).items()):
        if len(occ)!=1: continue
        i,a,b=occ[0]
        if i<K: continue
        u,w=map(int,f[i,[a,b]])
        by_hinge[u].append((i,a,b,u,w,1));by_hinge[w].append((i,b,a,w,u,-1))
    cand=[];cap=C.L*delta
    for h,half in sorted(by_hinge.items()):
        for x in range(len(half)):
            for y in range(x+1,len(half)):
                i,a,b,_,u,d=half[x];j,c,e,_,w,dd=half[y]
                if i==j or u==w: continue
                counts['shared_vertex_open_edge_pairs']+=1
                if d==dd: counts['same_winding']+=1;continue
                gap=float(np.linalg.norm(v[u].astype(float)-v[w]))
                rawgap=float(np.linalg.norm(raw[int(sf[i]),int(sc[i,b])].astype(float)-raw[int(sf[j]),int(sc[j,e])]))
                if gap>cap+1e-12: counts['endpoint_gap_over_delta']+=1;continue
                trial,why=B.state_merge(v,roots,members,preflat,fixed,[(u,w)])
                if why: counts[why]+=1;continue
                seam=base['raw_seam_slot_mask'][sf]
                kind='seam-seam' if seam[i] and seam[j] else 'seam-bulk' if seam[i] or seam[j] else 'bulk-bulk'
                cand.append(dict(hinge_original_vertex=h,endpoint_original_vertices=[u,w],
                    post_face_ids=[i,j],source_raw_slots=[int(sf[i]),int(sf[j])],
                    halfedge_corner_ids=[[int(sc[i,a]),int(sc[i,b])],[int(sc[j,c]),int(sc[j,e])]],
                    endpoint_current_corner_ids=[b,e],role_pair=kind,original_POST_endpoint_gap=gap,
                    original_RAW_endpoint_gap=rawgap,RAW_gap_within_delta=rawgap<=cap,
                    sort_PRE_GEO_max_displacement=trial['maxdisp']))
    cand.sort(key=lambda r:(r['sort_PRE_GEO_max_displacement'],r['source_raw_slots'],r['hinge_original_vertex'],r['endpoint_original_vertices']))
    return cand,dict(counts),members,fixed


def decode(base,pre,raw,pairs,delta):
    v=base['vertices'].copy();f0=base['faces'];K=int(base['known_faces']);sf,sc=base['source_face_ids'],base['source_corner_ids']
    roots=np.arange(len(v),dtype=np.int64);tri0=v[f0].copy();shape0=PE.shape(tri0)
    preflat=pre.reshape(-1,3).astype(float);cap=C.L*delta
    candidates,filters,members,fixed=inventory(base,pre,raw,delta)
    baseline=Q.connectivity_ledger(v,f0,K,pairs,sf,sc);current=baseline;current_pairs=pairs
    original_bad=Q.illegal_pair_sets(pairs,sf,K);old_nm=nm_keys(f0,sf,sc);old_dups=B.numeric_duplicate_pairs(v,f0)
    records=[];affected=set();accepted=0;checks=0;move_path=np.zeros(len(v),np.float64)
    for rank,cand in enumerate(candidates):
        row=dict(cand,rank=rank);why=None;f=roots[f0]
        i,j=cand['post_face_ids'];u,w=cand['endpoint_original_vertices'];h=int(roots[cand['hinge_original_vertex']]);a,b=int(roots[u]),int(roots[w])
        inc=PE.incidence(f)
        if a==b: why='ALREADY_MERGED'
        elif len(inc.get(tuple(sorted((h,a))),()))!=1 or len(inc.get(tuple(sorted((h,b))),()))!=1: why='NO_LONGER_OPEN_BOUNDARY'
        elif float(np.linalg.norm(v[a].astype(float)-v[b]))>cap+1e-12: why='CURRENT_ENDPOINT_GAP_OVER_DELTA'
        trial=None
        if why is None: trial,why=B.state_merge(v,roots,members,preflat,fixed,[(u,w)])
        if why is None:
            row['merged_clusters']=trial['operations'];row['maximum_PRE_GEO_displacement']=trial['maxdisp']
            if trial['maxdisp']>cap+1e-12: why='CLUSTER_PRE_GEO_OVER_BUDGET'
        if why is None:
            tv,tr=trial['vertices'],trial['roots'];tf=tr[f0];tt=tv[tf]
            aff=np.flatnonzero(np.any(np.isin(f0,list(trial['changed_clusters'])),axis=1));aff=aff[aff>=K]
            row['affected_source_raw_slots']=sf[aff].tolist()
            row['incremental_POST_motion']=float(np.linalg.norm(tv[list(trial['changed_clusters'])].astype(float)-v[list(trial['changed_clusters'])],axis=1).max())
            if not np.array_equal(tt[:K].view(np.uint32),tri0[:K].view(np.uint32)): raise AssertionError('C geometry changed')
            if not np.array_equal(tt[base['output_anchor_ids']>=0].view(np.uint32),tri0[base['output_anchor_ids']>=0].view(np.uint32)): raise AssertionError('Seam anchor changed')
            if any(len(set(tf[z].tolist()))!=3 for z in aff): why='COLLAPSED_VERTEX_IDS'
            ss=PE.shape(tt[aff]);prev=PE.shape(v[f[aff]])
            if why is None and np.any(ss['area']<=1e-12*C.L*C.L): why='AREA_DEGENERATE'
            # Both original and current states are guarded; damage cannot accumulate across accepted edits.
            for ref in (shape0,):
                if why is None and np.any((ss['angle']<5)&(ss['angle']<ref['angle'][aff]-1e-10)):why='ANGLE_NEW_OR_WORSE'
            if why is None and np.any((ss['angle']<5)&(ss['angle']<prev['angle']-1e-10)):why='ANGLE_CURRENT_WORSENED'
            if why is None and np.any((ss['q']>10)&(ss['q']>shape0['q'][aff]+1e-10)):why='Q_NEW_OR_WORSE'
            if why is None and np.any((ss['q']>10)&(ss['q']>prev['q']+1e-10)):why='Q_CURRENT_WORSENED'
            cos0=(ss['normal']*shape0['normal'][aff]).sum(1);cosprev=(ss['normal']*prev['normal']).sum(1)
            row['maximum_normal_rotation_deg']=float(np.degrees(np.arccos(np.clip(cos0,-1,1))).max()) if len(aff) else 0.
            if why is None and (np.any(cos0<.5-1e-12) or np.any(cosprev<.5-1e-12)):why='NORMAL_ROTATION_GT60'
            if why is None and B.numeric_duplicate_pairs(tv,tf)-old_dups:why='NEW_EXACT_DUPLICATE'
            if why is None and nm_keys(tf,sf,sc)-old_nm:why='NEW_NONMANIFOLD_EDGE'
            if why is None:
                trial_pairs=scan(tt,K);checks+=1;bad=Q.illegal_pair_sets(trial_pairs,sf,K);curbad=Q.illegal_pair_sets(current_pairs,sf,K)
                for kind in ('CF','FF'):
                    newbad=bad[kind]-(original_bad[kind]&curbad[kind])
                    if newbad:why='NEW_'+kind+'_ILLEGAL';row['new_'+kind+'_source_pairs']=sorted(newbad);break
            if why is None:
                trial_ledger=Q.connectivity_ledger(tv,tf,K,trial_pairs,sf,sc)
                lost=edge_keys(current)-edge_keys(trial_ledger)
                if lost:why='LOST_LEGAL_CF_OR_FF_EDGE';row['lost_legal_source_edges']=sorted(lost)
            if why is None:
                # The nominated hinge itself must now be a legal oppositely wound two-face edge.
                target=tuple(sorted((int(tr[h]),int(tr[a]))));candidate_edge=[r for r in trial_ledger['edges'] if (r['vertex_u'],r['vertex_v'])==target]
                if len(candidate_edge)!=1 or not candidate_edge[0]['valid_edge'] or not candidate_edge[0]['winding_opposite']:why='HINGE_NOT_LEGAL_OPPOSITE_EDGE'
            if why is None:
                move_path+=np.linalg.norm(tv.astype(float)-v,axis=1)
                v,roots=tv,tr;current,current_pairs=trial_ledger,trial_pairs;affected.update(sf[aff].tolist());accepted+=1;why='ACCEPTED'
        row['decision']=why;records.append(row)
    out={k:z.copy() for k,z in base.items()};ff=roots[f0];out.update(vertices=v,faces=ff,output=v[ff],C=v[ff[:K]],free=v[ff[K:]],
        original_POST_vertex_to_hinge_vertex=roots,index_domain=np.asarray('HINGE_ZIP_ORIGINAL_POST_C_SEAM_FIXED'))
    cv=out['corner_vertex_ids'];good=cv>=0;cv[good]=roots[cv[good]]
    changed=[i for i in range(len(v)) if roots[i]!=i or not np.array_equal(v[i].view(np.uint32),base['vertices'][i].view(np.uint32))]
    flatids=np.concatenate([members[i] for i in changed]) if changed else np.zeros(0,np.int64)
    displacement=np.linalg.norm(preflat[flatids]-v[roots[base['corner_vertex_ids'].reshape(-1)[flatids]]].astype(float),axis=1) if len(flatids) else np.asarray([0.])
    if displacement.max()>cap+1e-12:raise AssertionError('Cumulative PRE_GEO cap failed')
    if len(ff)!=len(f0):raise AssertionError('No deletion permitted')
    summary=dict(version=VERSION,delta_fraction=delta,delta_model=cap,candidates=len(candidates),accepted=accepted,
        accepted_by_role=dict(collections.Counter(r['role_pair'] for r in records if r['decision']=='ACCEPTED')),
        decisions=dict(collections.Counter(r['decision'] for r in records)),prefilter=filters,
        affected_source_raw_slots=sorted(affected),changed_original_clusters=len(changed),moved_original_corners=len(flatids),
        max_PRE_GEO_displacement=float(displacement.max()),max_accumulated_POST_path_length=float(move_path.max()),
        additional_deleted_faces=0,candidate_full_geometry_scans=checks,model_calls=0)
    return out,records,summary,baseline,current,current_pairs


def freeze(arm):
    if arm not in ARMS:raise ValueError(arm)
    path=OUT/'registrations'/(arm+'.json')
    if path.exists():r=C.read(path);B.verify(r['protected']);B.verify(r['code']);return r
    parent=NEW/'protocol.json'
    if not parent.exists() or C.read(parent)['status']!='FROZEN':raise RuntimeError('Root protocol must be frozen')
    protected={str(parent):C.sha(parent)};qq=requests()
    for q in qq:
        for p in paths(arm,q).values():protected[str(p)]=C.sha(p)
        for key in ('inference_npz','metadata_npz'):protected[q[key]]=C.sha(q[key])
        protected[q['query_descriptor']['path']]=q['query_descriptor']['sha256']
    code={str(Path(__file__)):C.sha(__file__),str(Path(B.__file__)):C.sha(B.__file__),str(Path(PE.__file__)):C.sha(PE.__file__),**Q.definition_hashes()}
    reg=dict(status='FROZEN',version=VERSION,arm=arm,requests=qq,deltas=DELTAS,cells=56,L=C.L,protected=protected,code=code,
        base='Saved source-oriented POST; not BOUND; no D0 or GEO call',
        candidates='Original POST incidence1 free-face halfedges sharing one actual vertex, opposite directed winding; seam-seam/seam-bulk/bulk-bulk; no C-face halfedge, no dynamic expansion',
        budgets='Both original/current endpoint separation<=delta; every moved original GEO member cumulative displacement from PRE_GEO<=delta; no RAW budget',
        order='Original proposed max member displacement, raw slots, original hinge ID, original endpoint IDs',
        geometry='Both free groups FP32 midpoint; C-bearing group fixed; distinct C anchors cannot merge; original whole clusters only',
        guards='Zero new CF/FF illegal pairs vs both original/current; zero lost current legal CF/FF; zero newly nonmanifold/duplicate; no degenerate/no new or worse angle<5 or q>10; normal rotation<=60 vs original/current; no deletions/flips',
        no_uncross='Undefined in attachment; not implemented or silently substituted',
        closest_prior='R05/R10 and partial-edge/BOUND variants; new hinge-qualified bulk-bulk coverage and hard zero-new-FF/zero-lost-FF guards',
        inference='No sourceY needed; RAW used only for distance diagnosis; slot IDs only within-output identity')
    C.write(path,reg);return reg


def worker(job):
    arm,q=job;p=paths(arm,q);base=C.arrays(p['post']);pre=C.arrays(p['pre'])['output'];raw=C.arrays(p['raw'])['output']
    validate_arrays(base,pre,q);bp=C.read(p['pairs']);results=[]
    for delta in DELTAS:
        name=label(arm,delta);dest=OUT/'cells'/name/q['request_id'];done=dest/'complete.json'
        if done.exists():r=C.read(done);B.verify(r['files']);results.append(r);continue
        if dest.exists():raise RuntimeError('Incomplete attempt preserved; no silent retry '+str(dest))
        tick=time.perf_counter();C.write(dest/'started.json',dict(utc=C.utc(),arm=arm,request_id=q['request_id'],delta=delta,pid=os.getpid()))
        try:
            out,records,summary,bl,nl,_=decode(base,pre,raw,bp,delta)
            B.save_npz(dest/'post.npz',**out);C.write(dest/'hinge.json',summary);C.write(dest/'candidate_records.json',records);C.csvwrite(dest/'candidates.csv',records)
            score,row,pairs,shape,edges,vectors,face_rows,roles=B.score_output(out,q,name)
            C.write(dest/'score.json',score);C.write(dest/'pairs.json',pairs);C.write(dest/'shape_per_face.json',shape)
            C.csvwrite(dest/'valid_edges.csv',edges);C.csvwrite(dest/'seam_bulk_faces.csv',face_rows);C.csvwrite(dest/'illegal_pairs_by_role.csv',roles);B.save_npz(dest/'vectors.npz',**vectors)
            nl=Q.connectivity_ledger(out['vertices'],out['faces'],int(q['K']),pairs,out['source_face_ids'],out['source_corner_ids'])
            paired=Q.compare_ledgers(bl,nl,bp,pairs,summary['affected_source_raw_slots']);C.write(dest/'paired_ledger.json',paired);C.write(dest/'connectivity.json',nl)
            for name2 in ('baseline_edge_fates','new_edges','illegal_pair_fates'):C.csvwrite(dest/(name2+'.csv'),paired[name2])
            for k in ('CF_lost_legal_edges','FF_lost_legal_edges','CF_new_illegal_pairs','FF_new_illegal_pairs','deleted_source_faces'):
                if paired['summary'][k]:raise AssertionError('Final zero-damage guard '+k)
            row.update(base_arm=arm,delta_fraction=delta,accepted=summary['accepted'],candidates=summary['candidates'],max_PRE_GEO_motion=summary['max_PRE_GEO_displacement'],**paired['summary'])
            r=dict(status='COMPLETE',version=VERSION,row=row,seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                files={str(z):C.sha(z) for z in dest.iterdir() if z.is_file()},CPU_only='torch' not in sys.modules,model_calls=0)
            if not r['CPU_only']:raise AssertionError('Unexpected torch import')
            C.write(done,r);results.append(r)
            print(json.dumps(dict(arm=arm,request=q['request_id'],delta=delta,accepted=summary['accepted'],seconds=r['seconds'])),flush=True)
        except BaseException as exc:
            C.write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    return results


def aggregate(arm):
    rows=[];registry=[];paired=[];funnels=[]
    for q in requests():
        p=paths(arm,q);base=C.arrays(p['post']);score=C.read(p['score']);pairs=C.read(p['pairs']);b,_,_=metric_row(score,base,pairs)
        for d in DELTAS:
            l=label(arm,d);dest=OUT/'cells'/l/q['request_id'];rr=C.read(dest/'complete.json');B.verify(rr['files']);row=rr['row'];rows.append(row)
            registry.append(dict(label=l,base_arm=arm,request_id=q['request_id'],stage='POST_HINGE'+str(round(d*1000)).zfill(3),
                mesh_path=str(dest/'post.npz'),score_path=str(dest/'score.json'),pairs_path=str(dest/'pairs.json'),edges_path=str(dest/'valid_edges.csv')))
            funnels.append(dict(label=l,request_id=q['request_id'],split=q['split'],**C.read(dest/'hinge.json')))
            for key,value in b.items():
                if key in B.E.META or key not in row:continue
                if isinstance(value,(int,float,bool)) and isinstance(row[key],(int,float,bool)):
                    paired.append(dict(label=l,request_id=q['request_id'],split=q['split'],parent_id=q['parent_id'],noise_index=q['noise_index'],metric=key,baseline=value,candidate=row[key],delta=float(row[key])-float(value)))
    folder=OUT/'aggregate'/arm
    for name,data in (('per_output',rows),('paired_metrics',paired),('rejection_funnel',funnels)):C.csvwrite(folder/(name+'.csv'),data)
    C.write(folder/'registry.json',dict(records=registry))
    summaries=[]
    for d in DELTAS:
        for split in ('TRAIN','DEV','ALL'):
            rr=[r for r in rows if r['delta_fraction']==d and (split=='ALL' or r['split']==split)]
            for metric in sorted({r['metric'] for r in paired}):
                pp=[p for p in paired if p['label']==label(arm,d) and p['metric']==metric and (split=='ALL' or p['split']==split)]
                summaries.append(dict(label=label(arm,d),split=split,metric=metric,n=len(pp),baseline_mean=float(np.mean([p['baseline'] for p in pp])),candidate_mean=float(np.mean([p['candidate'] for p in pp])),delta_mean=float(np.mean([p['delta'] for p in pp]))))
    C.csvwrite(folder/'summary.csv',summaries)
    return dict(cells=len(rows),files={str(p):C.sha(p) for p in folder.iterdir() if p.is_file()})


def run(arm,workers):
    reg=freeze(arm);done=OUT/(arm+'_completion.json')
    if done.exists():r=C.read(done);B.verify(r['files']);return r
    C.resources(OUT/('resources_'+arm+'_before_'+str(time.time_ns())+'.json'));tick=time.perf_counter()
    first=worker((arm,reg['requests'][0]));snap=C.resources(OUT/('resources_'+arm+'_measured_'+str(time.time_ns())+'.json'))
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (snap.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    peak=max(r['peak_RSS_KiB'] for r in first)*1024
    count=2 if workers==2 and (os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*peak) and (snap.get('loadavg') or [999])[0]<(os.cpu_count() or 1)-2 else 1
    C.write(OUT/('dispatch_'+arm+'_'+str(time.time_ns())+'.json'),dict(workers=count,threads_per_worker=1,first_authorized_peak_RSS_bytes=peak,resources=snap))
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(worker,[(arm,q) for q in reg['requests'][1:]]):pass
    result=aggregate(arm);B.verify(reg['protected']);B.verify(reg['code'])
    result.update(status='COMPLETE',arm=arm,workers=count,seconds=time.perf_counter()-tick,model_calls=0,GPU_calls=0,additional_deleted_faces=0)
    C.write(done,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','run','aggregate'));p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    r=freeze(a.arm) if a.stage=='freeze' else run(a.arm,a.workers) if a.stage=='run' else aggregate(a.arm)
    print(json.dumps({k:v for k,v in r.items() if k not in ('protected','code','requests','files')}),flush=True)
