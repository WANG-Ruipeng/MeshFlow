"""CPU-only frozen-layout evaluation; original scorers and queries are reused.

Roles can change between layouts. Fixed-slot partitions are therefore separate
from each layout's own SEAM/BULK partition. No cross-layout slot pair is called
a physically removed face pair; exact connectivity ledgers compare only the two
backends derived from the SAME pre-GEO geometry.
"""
from __future__ import annotations
import os
for _n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_n]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import collections
import concurrent.futures
import csv
import json
import multiprocessing
import resource
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from r2.pilots.seam_slot_c200_v1 import common as C, evaluate as E

OUT=C.ROOT/'experiments/r2_seam_occupancy_rollout_v1'
OLD=C.OUT
GUARD=C.ROOT/'experiments/r2_seam_geometry_guard_v1'
VERSION='occupancy_frozen_rollout_original_scores_fixed_slot_partitions_v1'
ARMS=('PREDICTED','ALL_ON','SOURCE_ORACLE')
STAGES=('RAW','PRE_GEO','POST','CAPACITY_POST')
META=('request_id','condition_id','parent_id','noise_index','N','K','split')

def manifest(root):
    m=C.read(C.resolve(root)/'eval_manifest.json')
    if tuple(m['arms'])!=ARMS[1:] or len(m['requests'])!=16 or len(m['conditions'])!=4:
        raise ValueError('Expected exactly two new arms and the frozen sixteen requests')
    old=C.read(OLD/'eval_manifest.json')
    if m['requests']!=old['requests'] or m['conditions']!=old['conditions']:
        raise ValueError('Noise/condition panel changed')
    return m

def artifact(root,arm,rid,stage):
    root=C.resolve(root)
    base=OLD if arm=='PREDICTED' else root
    label='SEAM_SLOT_C200' if arm=='PREDICTED' else arm
    if stage=='CAPACITY_POST':
        base=GUARD/'capacity' if arm=='PREDICTED' else root/'capacity'
        return base/'evaluation'/label/rid/'post/post.npz'
    part='raw/raw.npz' if stage=='RAW' else 'post/pre_geo.npz' if stage=='PRE_GEO' else 'post/post.npz'
    return base/'evaluation'/label/rid/part

def cached_score(arm,rid,stage):
    if arm!='PREDICTED':return None
    if stage=='PRE_GEO':return GUARD/'attribution/pre_geo'/rid/'score.json'
    base=GUARD/'capacity' if stage=='CAPACITY_POST' else OLD
    return base/'evaluation/scores/SEAM_SLOT_C200'/rid/('POST' if stage=='CAPACITY_POST' else stage)/'score.json'

def source_hashes():
    paths=[Path(__file__),C.ROOT/'r2/pilots/seam_geometry_guard_v1/summarize_capacity.py']
    return dict(E.source_hashes(),**{str(p):C.sha(p) for p in paths})

def prepare(root):
    root=C.resolve(root);m=manifest(root);protected={};old,*_=E.helpers()
    for q in m['requests']:
        for f in ('inference_npz','metadata_npz','noise_npz'):protected[q[f]]=C.sha(q[f])
        query=q['query_descriptor'];protected[query['path']]=query['sha256']
        inp=C.arrays(q['inference_npz']);cc=inp['C_vertices'][inp['C_faces_model']]
        if query['identity']['C_sha256']!=old.array_sha(cc):raise ValueError('Query C differs')
        old.load_queries(query)
        for s in STAGES:
            for p in (artifact(root,'PREDICTED',q['request_id'],s),cached_score('PREDICTED',q['request_id'],s)):
                protected[str(p)]=C.sha(p)
            p=cached_score('PREDICTED',q['request_id'],s).parent/'pairs.json';protected[str(p)]=C.sha(p)
    for c in m['conditions']:protected[c['labels_npz']]=C.sha(c['labels_npz'])
    E.verify(protected)
    value=dict(version=VERSION,status='FROZEN',manifest_sha256=C.sha(root/'eval_manifest.json'),
        protected=protected,source_hashes=source_hashes(),arms=ARMS,stages=STAGES,
        registered_logical_cells=192,new_full_quality_scans=128,cached_scores=64,
        L=C.L,query_points=2048,quality_definition=old.cpu_math()['control'].frozen_definition(),
        comparison='Same C, fixed noise and frozen generator. SOURCE_ORACLE is source-label-assisted diagnostic, not C-only deployment.',
        fixed_slot_partition='Common BULK is intersection of BULK RAW slot sets across all three layouts. Fixed role categories use original PREDICTED roles plus conversion-to-source-SEAM or source-Null; retained denominators recorded.',
        connectivity_ledger='Only original POST versus capacity POST of identical pre-GEO; source RAW slots and corners identify edges.',
        PRE_GEO_topology='NA_NOT_EXPORTED_INDEXED_TOPOLOGY; only continuous geometry scores, original query distances and role-pair partitions.',
        GT='NA_OPEN_ENDED_COMPLETION',adoption='PENDING_ACTUAL_AI_PROXY_REVIEW',model_calls=0,new_queries=0)
    C.write(root/'analysis/registration.json',value);return value

def registration(root):
    root=C.resolve(root);r=C.read(root/'analysis/registration.json')
    if r['manifest_sha256']!=C.sha(root/'eval_manifest.json') or r['source_hashes']!=source_hashes():
        raise ValueError('Evaluation definition changed after freeze')
    E.verify(r['protected']);return r

def score_folder(root,arm,rid,stage):return C.resolve(root)/'analysis/scores'/arm/rid/stage

def _boundary(old,A,q,inp,metadata,F):
    original=inp['C_vertices'][inp['C_faces_original']]
    ids=metadata['C_source_vertex_ids'][inp['C_faces_original']]
    bi,points,audit=A.topology(original,ids);dist,near=A.nearest(points,F)
    rows=[dict(setting=n,threshold_model=t,scale=s,tau=v,domain='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY',**A.metric(dist,t)) for n,t,s,v in A.SETTINGS]
    detail=[dict(source_vertex_id=int(i),distance_model=float(d) if np.isfinite(d) else None,
        nearest_free_face=int(j)//3 if j>=0 else None,nearest_free_corner=int(j)%3 if j>=0 else None) for i,d,j in zip(bi,dist,near)]
    return rows,detail,audit

def score_one(job):
    root,arm,q,stage,rhash=job;root=C.resolve(root);rid=q['request_id']
    dest=score_folder(root,arm,rid,stage);final=dest/'score.json';mesh=artifact(root,arm,rid,stage)
    identity=dict(version=VERSION,arm=arm,request_id=rid,stage=stage,mesh_path=str(mesh),mesh_sha256=C.sha(mesh),registration_sha256=rhash)
    if final.exists():
        r=C.read(final)
        if r['identity']!=identity:raise ValueError('Completed score identity differs')
        E.verify(r['files']);return dict(path=str(final),sha256=C.sha(final),reused=True,seconds=0,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if dest.exists():raise RuntimeError('Incomplete cell: explicit recovery required '+str(dest))
    C.write(dest/'started.json',identity);tick=time.perf_counter()
    try:
        cp=cached_score(arm,rid,stage)
        if cp:
            r=C.read(cp);pairs=C.read(cp.parent/'pairs.json')
            C.write(dest/'cached_score_reference.json',dict(path=str(cp),sha256=C.sha(cp),pairs_sha256=C.sha(cp.parent/'pairs.json')))
            # Cached immutable baseline is referenced, never rewritten or rescored.
            r=dict(r);r.update({k:q[k] for k in META});r.update(label=arm,stage=stage)
            r['cached_original_score_path']=str(cp);r['model_calls']=0;r['full_quality_scans']=0
            r.pop('files',None);r['original_identity']=r.pop('identity',None)
            C.write(dest/'pairs.json',pairs)
        else:
            old,A,G,H,Q=E.helpers();mod=old.cpu_math();a=C.arrays(mesh)
            inp=C.arrays(q['inference_npz']);metadata=C.arrays(q['metadata_npz']);k=q['K']
            cc=inp['C_vertices'][inp['C_faces_model']];original=inp['C_vertices'][inp['C_faces_original']]
            tri=a['output'].reshape(-1,3,3);F=tri[k:]
            if tri.dtype!=np.float32 or not np.isfinite(tri).all():raise ValueError('Invalid geometry')
            if not old.byte_equal(tri[:k],cc if stage=='PRE_GEO' else original):raise ValueError('C changed')
            if stage in ('RAW','PRE_GEO') and len(tri)!=q['N']:raise ValueError('Face budget changed')
            metrics,cf,ff=mod['geometry'].scan_geometry(cc,F,C.L) if len(F) else ({},*Q._empty_scans(k))
            pairs=dict(CF=cf,FF=ff);ref=old.load_queries(q['query_descriptor'])
            trusted=mod['control'].trusted_cf(cc,F,C.L,cf_scan=cf)
            im,iv=mod['control'].conflict_metrics(F,dict(C=cc,L0=C.L),ref,trusted);metrics.update(im)
            for kind,scan in pairs.items():
                illegal=[p for p in scan['contacts'] if p['classification'] in E.ILLEGAL]
                aff={int(p['pair'][1]) for p in illegal} if kind=='CF' else {int(i) for p in illegal for i in p['pair']}
                metrics.update({kind+'_illegal_pair_count':len(illegal),kind+'_illegal_affected_free_count':len(aff),kind+'_illegal_affected_free_fraction':len(aff)/len(F) if len(F) else None})
            shape={name:mod['shape'].shape_quality(x,C.L) for name,x in (('C',cc),('free',F),('whole',tri))}
            for name,s in shape.items():
                metrics.update({name+'_minimum_angle_lt5_fraction':s['min_angle_lt5_fraction'],name+'_q_gt10_fraction':s['q_actual_gt10_fraction'],name+'_near_degenerate_count':s['near_degenerate_area_count'],name+'_exact_zero_area_count':s['exact_zero_area_count'],name+'_total_surface_area':s['total_area']})
            metrics.update(mod['geometry'].topology(tri,k),actual_free_faces=len(F),deleted_free_faces=q['N']-len(tri))
            if stage=='PRE_GEO':
                graph=dict(status='NA_NOT_EXPORTED_INDEXED_TOPOLOGY');internal=dict(status='NA_NOT_EXPORTED_INDEXED_TOPOLOGY');edges=[];faces=[]
            else:
                vertices=a['vertices'];fidx=a['faces']
                if not old.byte_equal(vertices[fidx],tri):raise ValueError('Indexed geometry differs')
                graph,edges,_=G.actual_valid_graph(vertices,fidx,k,pairs)
                graph.update(H.opposite_reach(tri,k,edges),C_F_has_opposite_winding_edge=int(graph['C_F_valid_opposite_winding_edge_count']>0),domain=str(a.get('index_domain','SAVED_INDEX')))
                raw_ids=a['raw_slot_ids'];raw_seam=a['raw_seam_slot_mask']
                internal,faces=E.seam_to_bulk(tri,k,edges,raw_seam[raw_ids],raw_ids,raw_seam,True)
            boundary,bd,audit=_boundary(old,A,q,inp,metadata,F)
            C.write(dest/'pairs.json',pairs);C.write(dest/'shape_per_face.json',shape)
            C.write(dest/'topology.json',graph);C.write(dest/'seam_to_bulk.json',internal)
            C.csvwrite(dest/'valid_edges.csv',edges);C.csvwrite(dest/'seam_bulk_faces.csv',faces);C.csvwrite(dest/'boundary_vertices.csv',bd)
            E.save_npz(dest/'vectors.npz',iv)
            r=dict(**{k:q[k] for k in META},label=arm,stage=stage,status='PASS',metrics=metrics,topology=graph,seam_to_bulk=internal,boundary=boundary,boundary_audit=audit,
                C_preserved_exact=True,C_original_order_preserved=stage!='PRE_GEO',C_query_count=2048,C_query_weight_denominator=float(ref['arrays']['C_weights'].sum()),
                free_face_denominator=len(F),CF_tested_pair_denominator=k*len(F),FF_tested_pair_denominator=len(F)*(len(F)-1)//2,
                status_by_metric=dict(I=im['ConflictCov_C_status'],S='NA_NO_REQUIRED_INTERFACE',GT='NA_OPEN_ENDED_COMPLETION',semantic_use='PENDING_ACTUAL_AI_PROXY'),model_calls=0,new_queries=0,full_quality_scans=1)
        r.update(identity=identity,seconds=time.perf_counter()-tick,files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
        if 'torch' in sys.modules:raise RuntimeError('CPU scorer imported torch')
        C.write(final,r)
        return dict(path=str(final),sha256=C.sha(final),reused=False,seconds=r['seconds'],peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,quality_scans=r['full_quality_scans'])
    except BaseException as exc:
        C.write(dest/'failure.json',dict(status='STOPPED',error=repr(exc),traceback=traceback.format_exc()));raise

def run(root,workers=2):
    root=C.resolve(root);registration(root);m=manifest(root);final=root/'analysis/score_completion.json'
    if final.exists():
        r=C.read(final)
        for x in r['receipts']:E.verify({x['path']:x['sha256']});E.verify(C.read(x['path'])['files'])
        return r
    for a in ARMS:
        for q in m['requests']:
            for s in STAGES:
                if not artifact(root,a,q['request_id'],s).exists():raise FileNotFoundError(artifact(root,a,q['request_id'],s))
    rh=C.sha(root/'analysis/registration.json');jobs=[(str(root),a,q,s,rh) for a in ARMS for q in m['requests'] for s in STAGES]
    tick=time.perf_counter();first=score_one(jobs[0]);resources=C.resources();memory={x.split(':')[0]:int(x.split()[1])*1024 for x in (resources.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    count=workers if (os.cpu_count() or 1)>=4 and memory.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024) and (resources['loadavg'] or [999])[0]<(os.cpu_count() or 1)-2 else 1
    C.write(root/'analysis'/('dispatch_'+str(time.time_ns())+'.json'),dict(workers=count,threads_each=1,resources=resources,first_actual_cell=first,reason='Original CPU scorer; no synthetic probe'))
    receipts=[first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        todo=iter(jobs[1:]);active={}
        for _ in range(count):
            job=next(todo,None)
            if job:active[pool.submit(score_one,job)]=job
        while active:
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            results=[f.result() for f in done]
            for f,res in zip(done,results):
                del active[f];receipts.append(res);print(json.dumps(dict(scored=len(receipts),registered=192)),flush=True)
                job=next(todo,None)
                if job:active[pool.submit(score_one,job)]=job
    receipts.sort(key=lambda x:x['path']);value=dict(status='COMPLETE',cells=len(receipts),receipts=receipts,workers=count,seconds=time.perf_counter()-tick,model_calls=0,new_queries=0,registration_sha256=rh)
    C.write(final,value);return value

def _pairs(score_dir,a,k):
    pairs=C.read(score_dir/'pairs.json');ids=np.asarray(a.get('raw_slot_ids',np.arange(len(a['output']))),int);out={}
    for kind in ('CF','FF'):
        for p in pairs[kind]['contacts']:
            if p['classification'] not in E.ILLEGAL:continue
            i,j=map(int,p['pair']);i=int(ids[i if kind=='CF' else k+i]);j=int(ids[k+j])
            out[(kind,tuple(sorted((i,j))))]=p['classification']
    return out,ids

def _flat(s,q,arm,stage):
    r={k:q[k] for k in META};r.update(label=arm,stage=stage)
    for group in ('metrics','topology','seam_to_bulk','distance'):
        for k,v in s.get(group,{}).items():
            if not isinstance(v,(dict,list)):r[(group+'_status') if k=='status' else k]=v
    for b in s.get('boundary',[]):r['A_VMR__'+b['setting']]=b['VMR'];r['PMR__'+b['setting']]=b['perfect']
    r['free_face_denominator']=s['free_face_denominator'];r['cached_score']=arm=='PREDICTED'
    return r

def aggregate(root):
    root=C.resolve(root);registration(root);m=manifest(root);completion=C.read(root/'analysis/score_completion.json')
    if completion['cells']!=192:raise ValueError('Incomplete score panel')
    done=root/'analysis/aggregate_completion.json'
    if done.exists():r=C.read(done);E.verify(r['files']);return r
    rows=[];parts=[];details=[];fixed_rows=[];identities=[];converted=[]
    for q in m['requests']:
        rid=q['request_id'];k=q['K'];n=q['N'];raw={a:C.arrays(artifact(root,a,rid,'RAW')) for a in ARMS}
        masks={a:np.asarray(x['raw_seam_slot_mask'],bool) for a,x in raw.items()}
        for a,b in (('PREDICTED','ALL_ON'),('PREDICTED','SOURCE_ORACLE'),('ALL_ON','SOURCE_ORACLE')):
            equal_layout=bool(np.array_equal(masks[a],masks[b]))
            for st in STAGES:
                aa=C.arrays(artifact(root,a,rid,st));bb=C.arrays(artifact(root,b,rid,st))
                equal=aa['output'].dtype==bb['output'].dtype and aa['output'].shape==bb['output'].shape and aa['output'].tobytes()==bb['output'].tobytes()
                identities.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],baseline=a,candidate=b,stage=st,equal_layout=equal_layout,output_byte_identical=equal,status='PASS_SAME_LAYOUT_REPEAT' if equal_layout and equal else 'FAIL_SAME_LAYOUT_NOT_IDENTICAL' if equal_layout else 'DIFFERENT_LAYOUT_COMPARISON'))
        common_bulk=np.ones(n,bool);common_bulk[:k]=False
        for mask in masks.values():common_bulk &= ~mask
        base=masks['PREDICTED'];truth=masks['SOURCE_ORACLE']
        def fixed_role(i):
            if i<k:return 'C'
            if base[i]:return 'ORIGINAL_SEAM'
            if common_bulk[i]:return 'COMMON_BULK'
            return 'CONVERTED_TRUE_SEAM' if truth[i] else 'CONVERTED_SOURCE_NULL'
        for a in ARMS:
            for i in range(k,n):fixed_rows.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],label=a,raw_slot_id=i,fixed_role=fixed_role(i),current_role='SEAM' if masks[a][i] else 'BULK'))
            for stage in STAGES:
                sd=score_folder(root,a,rid,stage);s=C.read(sd/'score.json');arr=C.arrays(artifact(root,a,rid,stage));pp,ids=_pairs(sd,arr,k);row=_flat(s,q,a,stage)
                retained=set(map(int,ids[k:]));common=retained&set(np.flatnonzero(common_bulk).tolist());cp=[p for (kind,p) in pp if kind=='FF' and all(i in common for i in p)];aff={i for p in cp for i in p}
                row.update(common_bulk_registered_raw_faces=int(common_bulk.sum()),common_bulk_retained_faces=len(common),common_bulk_deleted_faces=int(common_bulk.sum())-len(common),
                    FF_common_bulk_illegal_pair_count=len(cp),FF_common_bulk_affected_faces=len(aff),FF_common_bulk_affected_fraction=len(aff)/len(common) if common else None,
                    FF_common_bulk_tested_pair_denominator=len(common)*(len(common)-1)//2)
                # Derived count denominator remains actual retained free, never raw N-K.
                allff=[p for (kind,p) in pp if kind=='FF'];allcf=[p for (kind,p) in pp if kind=='CF']
                row.update(FF_illegal_affected_free_count=len({i for p in allff for i in p}),CF_illegal_affected_free_count=len({p[1] for p in allcf}),
                    FF_tested_pair_denominator=len(retained)*(len(retained)-1)//2,CF_tested_pair_denominator=k*len(retained))
                for sid in range(k,n):
                    if fixed_role(sid) not in ('CONVERTED_TRUE_SEAM','CONVERTED_SOURCE_NULL'):continue
                    loc=np.flatnonzero(ids==sid);kept=bool(len(loc));area=None
                    if kept:
                        t=arr['output'][int(loc[0])].astype(float);area=float(.5*np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0])))
                    converted.append(dict(request_id=rid,parent_id=q['parent_id'],condition_id=q['condition_id'],noise_index=q['noise_index'],label=a,stage=stage,
                        raw_slot_id=sid,edge_id=int(raw[a]['raw_slot_edge_id'][sid]),fixed_role=fixed_role(sid),current_role='SEAM' if masks[a][sid] else 'BULK',
                        retained=kept,area_model2=area,illegal_CF_pair_count=sum(sid in p for p in allcf),illegal_FF_pair_count=sum(sid in p for p in allff),
                        cleanup_reason=str(arr['source_face_reason'][sid]) if 'source_face_reason' in arr else 'NOT_CLEANED',
                        semantic_role='UNKNOWN_REQUIRES_ACTUAL_IMAGE_REVIEW'))
                rows.append(row);groups=collections.defaultdict(list)
                for (kind,p),cl in pp.items():
                    current=['C' if i<k else 'SEAM' if masks[a][i] else 'BULK' for i in p];fixed=[fixed_role(i) for i in p]
                    category='-'.join(sorted(current));fcat='-'.join(sorted(fixed));groups[kind,category,fcat].append(p)
                    details.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],label=a,stage=stage,pair_kind=kind,raw_slot_i=p[0],raw_slot_j=p[1],current_role_pair=category,fixed_role_pair=fcat,classification=cl))
                for (kind,cat,fcat),pairs in sorted(groups.items()):
                    af={j for _,j in pairs} if kind=='CF' else {i for p in pairs for i in p}
                    parts.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],label=a,stage=stage,pair_kind=kind,current_role_pair=cat,fixed_role_pair=fcat,pair_count=len(pairs),unique_affected_free=len(af),actual_free_denominator=len(retained),affected_fraction=len(af)/len(retained),fractions_not_additive=True))
    out=root/'analysis';C.csvwrite(out/'per_output.csv',rows);C.csvwrite(out/'role_pair_partition.csv',parts);C.csvwrite(out/'illegal_pairs_by_fixed_slot.csv',details);C.csvwrite(out/'slot_roles.csv',fixed_rows);C.csvwrite(out/'same_layout_repeat_identity.csv',identities);C.csvwrite(out/'converted_slot_outcomes.csv',converted)
    keys=sorted(k for k in set().union(*(r.keys() for r in rows)) if k not in (*META,'label','stage','cached_score') and any(isinstance(r.get(k),(int,float,bool)) for r in rows))
    pairs=[];parents=[];summaries=[]
    look={(r['label'],r['request_id'],r['stage']):r for r in rows}
    for before,after in (('PREDICTED','ALL_ON'),('PREDICTED','SOURCE_ORACLE'),('ALL_ON','SOURCE_ORACLE')):
        for q in m['requests']:
            for st in STAGES:
                x=look[before,q['request_id'],st];y=look[after,q['request_id'],st]
                for key in keys:
                    xx,yy=x.get(key),y.get(key);valid=isinstance(xx,(int,float,bool)) and isinstance(yy,(int,float,bool))
                    pairs.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],condition_id=q['condition_id'],noise_index=q['noise_index'],baseline=before,candidate=after,stage=st,metric=key,baseline_value=xx,candidate_value=yy,delta=float(yy)-float(xx) if valid else None,status='PASS' if valid else 'NA_CACHED_OR_STAGE_NOT_APPLICABLE'))
    groups=collections.defaultdict(list)
    for r in pairs:groups[r['baseline'],r['candidate'],r['parent_id'],r['stage'],r['metric']].append(r)
    for key,rr in sorted(groups.items()):
        ok=[r for r in rr if r['status']=='PASS'];parents.append(dict(zip(('baseline','candidate','parent_id','stage','metric'),key),n_registered=4,n_valid=len(ok),**{f:float(np.mean([r[f] for r in ok])) if len(ok)==4 else None for f in ('baseline_value','candidate_value','delta')},status='PASS' if len(ok)==4 else 'NA'))
    for a in ARMS:
        for st in STAGES:
            rr=[r for r in rows if r['label']==a and r['stage']==st]
            for key in keys:
                vv=[r[key] for r in rr if isinstance(r.get(key),(int,float,bool))]
                summaries.append(dict(label=a,stage=st,metric=key,n_registered=16,n_valid=len(vv),mean=float(np.mean(vv)) if vv else None,sum=float(np.sum(vv)) if vv else None,status='PASS' if len(vv)==16 else 'NA_OR_PARTIAL'))
    C.csvwrite(out/'paired_per_noise.csv',pairs);C.csvwrite(out/'paired_per_parent.csv',parents);C.csvwrite(out/'summary.csv',summaries)
    connectivity(root,m)
    files={str(p):C.sha(p) for p in out.glob('*.csv')};r=dict(status='COMPLETE_NUMERIC_PENDING_ACTUAL_IMAGE_REVIEW',cells=192,files=files,model_calls=0,notes='Role partitions use fixed RAW slot identities; same-slot cross-layout changes are paired observations, not face-wise causal removals. PRE_GEO graph NA. Historical PRE_GEO lacks cached shape/boundary fields: NA, no hidden rescoring.')
    C.write(done,r);return r

def connectivity(root,m):
    from r2.pilots.seam_geometry_guard_v1 import summarize_capacity as S
    erows=[];prows=[];summ=[];frows=[]
    for arm in ARMS:
        for q in m['requests']:
            rid=q['request_id'];k=q['K'];data={}
            for stage in ('POST','CAPACITY_POST'):
                arr=C.arrays(artifact(root,arm,rid,stage));sd=score_folder(root,arm,rid,stage)
                edge_dir=cached_score(arm,rid,stage).parent if arm=='PREDICTED' else sd
                ed,all_ed=S.read_edges(edge_dir/'valid_edges.csv',arr);pp,_=_pairs(sd,arr,k)
                data[stage]=dict(arr=arr,edges=ed,all_edges=all_ed,pairs=pp)
            a=data['POST'];b=data['CAPACITY_POST'];local=[];lp=[]
            for key in sorted(a['edges'].keys()|b['edges'].keys()):
                ae=a['edges'].get(key);be=b['edges'].get(key);kind='CF' if any(i<k for i,_ in key) else 'FF'
                r=dict(label=arm,request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],edge_kind=kind,source_slot_corner_identity=key,status='RETAINED' if ae and be else 'LOST' if ae else 'NEW',original_length=ae['length_model'] if ae else None,capacity_length=be['length_model'] if be else None,lost_reason=S.missing_reason(key,b['arr'],b['all_edges']) if ae and not be else '')
                erows.append(r);local.append(r)
            for kind,p in sorted(a['pairs'].keys()|b['pairs'].keys()):
                ac=a['pairs'].get((kind,p));bc=b['pairs'].get((kind,p));r=dict(label=arm,request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],pair_kind=kind,source_slot_pair=p,status='RETAINED_ILLEGAL' if ac and bc else 'ELIMINATED' if ac else 'NEW_ILLEGAL',original_classification=ac,capacity_classification=bc)
                prows.append(r);lp.append(r)
            sa=set(map(int,a['arr']['source_face_ids']));sb=set(map(int,b['arr']['source_face_ids']))
            for sid in sorted(sa^sb):frows.append(dict(label=arm,request_id=rid,raw_slot_id=sid,original_kept=sid in sa,capacity_kept=sid in sb,original_reason=str(a['arr']['source_face_reason'][sid]),capacity_reason=str(b['arr']['source_face_reason'][sid])))
            r=dict(label=arm,request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],newly_deleted_faces=len(sa-sb),additionally_retained_faces=len(sb-sa))
            for kind in ('CF','FF'):
                for status in ('RETAINED','LOST','NEW'):
                    rr=[x for x in local if x['edge_kind']==kind and x['status']==status];r[kind+'_'+status+'_edge_count']=len(rr)
                    for st in ('original','capacity'):r[kind+'_'+status+'_'+st+'_length']=sum(x[st+'_length'] or 0 for x in rr)
                for status in ('RETAINED_ILLEGAL','ELIMINATED','NEW_ILLEGAL'):r[kind+'_'+status+'_pair_count']=sum(x['pair_kind']==kind and x['status']==status for x in lp)
            summ.append(r)
    out=C.resolve(root)/'analysis';C.csvwrite(out/'same_pregeo_legal_edge_ledger.csv',erows);C.csvwrite(out/'same_pregeo_illegal_pair_ledger.csv',prows);C.csvwrite(out/'same_pregeo_face_retention.csv',frows);C.csvwrite(out/'same_pregeo_connectivity_per_output.csv',summ)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(OUT));p.add_argument('--stage',choices=('prepare','score','aggregate','all'),default='prepare');p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    if a.stage=='prepare':r=prepare(a.root)
    elif a.stage=='score':r=run(a.root,a.workers)
    elif a.stage=='aggregate':r=aggregate(a.root)
    else:prepare(a.root);run(a.root,a.workers);r=aggregate(a.root)
    print(json.dumps(dict(status=r['status'],stage=a.stage)),flush=True)

if __name__=='__main__':main()
