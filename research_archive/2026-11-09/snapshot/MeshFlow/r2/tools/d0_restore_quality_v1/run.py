"""CPU-only fixed 16 cached D0 proposals; no model import or GEO calls."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import argparse,collections,concurrent.futures,json,multiprocessing,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from r2.pilots.seam_slot_c200_v1 import common as C,evaluate as E

OUT=C.ROOT/'experiments/r2_d0_restore_quality_v1'
OLD=C.OUT
GUARD=C.ROOT/'experiments/r2_seam_geometry_guard_v1/attribution/pre_geo'
VERSION='d0_original_proposal_proper_rigid_hard_restore_v1'
STAGES=('UNCONSTRAINED','RIGID','HARD_RESTORE')


def save(p,a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise ValueError('Refuse overwrite '+str(p))
    with p.open('xb') as f:np.savez_compressed(f,**a)


def resources():
    r=C.resources()
    import psutil
    r.update(ram_available_bytes=psutil.virtual_memory().available,cpu_percent=psutil.cpu_percent(interval=.1))
    return r


def paths(q):
    b=OLD/'evaluation/SEAM_SLOT_C200'/q['request_id']
    return dict(raw=b/'raw/raw.npz',proposal=b/'post/proposal.npz',hard=b/'post/pre_geo.npz',post=b/'post/post.npz',
        hard_score=GUARD/q['request_id']/'score.json',hard_pairs=GUARD/q['request_id']/'pairs.json',
        raw_score=OLD/'evaluation/scores/SEAM_SLOT_C200'/q['request_id']/'RAW/score.json',
        post_score=OLD/'evaluation/scores/SEAM_SLOT_C200'/q['request_id']/'POST/score.json')


def prepare():
    if (OUT/'registration.json').exists():return registration()
    m=C.read(OLD/'eval_manifest.json');old,*_=E.helpers();protected={str(OLD/'eval_manifest.json'):C.sha(OLD/'eval_manifest.json')};missing=[]
    for q in m['requests']:
        for p in list(paths(q).values())+[C.resolve(q['inference_npz']),C.resolve(q['metadata_npz']),C.resolve(q['query_descriptor']['path'])]:
            if not p.exists():missing.append(str(p))
            else:protected[str(p)]=C.sha(p)
    if missing:
        x=dict(status='BLOCKED_MISSING_CACHE',missing=missing,model_calls=0);C.write(OUT/'BLOCKED.json',x);return x
    if len(m['requests'])!=16:raise ValueError('Fixed16 contract')
    sources=E.source_hashes();sources[str(Path(__file__))]=C.sha(__file__)
    for p in Path(__file__).parent.glob('*.py'):sources[str(p)]=C.sha(p)
    r=dict(version=VERSION,status='FROZEN',requests=m['requests'],conditions=m['conditions'],protected=protected,source_hashes=sources,
        L=C.L,model_calls=0,new_noise=0,new_GEO=0,rigid_fits=16,new_full_internal_quality_scans=32,existing_hard_quality_caches=16,
        stages=STAGES,rigid='Weighted proper Kabsch (det R=+1): C corner copies each weight inverse source-C-ID multiplicity; one R,t applied to ALL proposal corners; no C writeback; FP64 arithmetic/output.',
        original_C_I='Original fixed C and frozen2048 queries versus method F. Diagnostic only for UNCONSTRAINED/RIGID, whose actual C differs. Not an actual-mesh control success score.',
        actual_C_I='NA_NO_FROZEN_QUERY_FOR_MOVED_C; no query redraw or remapping',
        exact_topology='Existing exact-coordinate diagnostic internally casts FP32; no tolerance welding. Pre-GEO face components are representation diagnostics, not semantic fragmentation.',
        paired_faces='Same original RAW slots for all three stages. Same bulk mask is fixed; no role conversion.',
        comparisons='UNCONSTRAINED to HARD isolates writeback C plus seam fixed corners. UNCONSTRAINED to RIGID tests global coordinate motion, not deformations. RAW/POST retained references.',
        numerical_rigid='Exact mathematics preserves distances/intersections; floating point classifier changes recorded, not silently asserted invariant.',
        concurrency='At most2 independent CPU workers, BLAS1; measure first authorized request then decide',
        stopping='Complete16 caches, audit, tables, images/review/report; no deployment/default/commit')
    C.write(OUT/'resources_before.json',resources());C.write(OUT/'registration.json',r);return r


def registration():
    r=C.read(OUT/'registration.json');E.verify(r['protected']);E.verify(r['source_hashes']);return r


def fit(raw,proposal):
    k=int(raw['known_faces']);x=proposal[:k].reshape(-1,3).astype(np.float64);y=raw['raw'][:k].reshape(-1,3).astype(np.float64)
    ids=raw['raw_anchor_ids'][:k].reshape(-1);unique,inv,counts=np.unique(ids,return_inverse=True,return_counts=True)
    w=1/counts[inv];w=w/w.sum();mx=(x*w[:,None]).sum(0);my=(y*w[:,None]).sum(0)
    u,s,vt=np.linalg.svd((x-mx).T@(w[:,None]*(y-my)));d=np.ones(3);d[-1]=np.linalg.det(u@vt);rot=u@np.diag(d)@vt
    trans=my-mx@rot;out=proposal.astype(np.float64)@rot+trans
    err0=np.linalg.norm(x-y,axis=1);err=np.linalg.norm(out[:k].reshape(-1,3)-y,axis=1)
    translated=x+(my-mx);terr=np.linalg.norm(translated-y,axis=1)
    pair0=np.linalg.norm(x[:,None]-x[None],axis=-1);pairy=np.linalg.norm(y[:,None]-y[None],axis=-1)
    copyrows=[]
    for uid in unique:
        group=x[ids==uid];diam=float(np.linalg.norm(group[:,None]-group[None],axis=-1).max())
        copyrows.append(dict(C_vertex_id=int(uid),copies=len(group),proposal_copy_diameter=diam,proposal_copy_diameter_L=diam/C.L,
            source_coordinate=y[np.flatnonzero(ids==uid)[0]].tolist()))
    flat=proposal.reshape(-1,3).astype(float);after=out.reshape(-1,3)
    # All corner-pair distances, no detector or mesh modification.
    distance_error=float(np.max(np.abs(np.linalg.norm(flat[:,None]-flat[None],axis=-1)-np.linalg.norm(after[:,None]-after[None],axis=-1))))
    summary=dict(R=rot.tolist(),t=trans.tolist(),determinant=float(np.linalg.det(rot)),singular_values=s.tolist(),
        C_weighted_RMS_before=float(np.sqrt(w@(err0**2))),C_weighted_RMS_after=float(np.sqrt(w@(err**2))),
        C_weighted_translation_RMS=float(np.sqrt(w@(terr**2))),C_max_error_before=float(err0.max()),C_max_error_after=float(err.max()),
        C_exact_restoration=False,C_max_pairwise_distance_change_from_input=float(np.max(np.abs(pair0-pairy))),
        source_vertex_groups=len(unique),duplicate_groups=sum(r['copies']>1 for r in copyrows),
        split_groups=sum(r['copies']>1 and r['proposal_copy_diameter']>0 for r in copyrows),
        max_copy_split=max(r['proposal_copy_diameter'] for r in copyrows),
        all_mesh_pairwise_distance_invariance_max_error=distance_error,
        rotation_orthogonality_error=float(np.max(np.abs(rot.T@rot-np.eye(3)))))
    return out,summary,copyrows


def pairs_by_slot(scan,kind,k):
    return {(int(p['pair'][0])+(k if kind=='FF' else 0),int(p['pair'][1])+k):p['classification']
        for p in scan['contacts'] if p['classification'] in E.ILLEGAL}


def measure(q,stage,tri,original_C,raw,mod,old,cache=None):
    k=q['K'];nf=len(tri)-k;cc=tri[:k];free=tri[k:];mask=raw['raw_seam_slot_mask']
    if cache:
        met=C.read(cache[0])['metrics'];pairs=C.read(cache[1]);fixed_I={a:b for a,b in met.items() if a.startswith('ConflictCov_C')};iv=None
    else:
        met,cf,ff=mod['geometry'].scan_geometry(cc,free,C.L);pairs=dict(CF=cf,FF=ff)
        original_cf=mod['geometry'].pair_scan(original_C,free,C.L,independent=True)
        trusted=mod['control'].trusted_cf(original_C,free,C.L,cf_scan=original_cf)
        fixed_I,iv=mod['control'].conflict_metrics(free,dict(C=original_C,L0=C.L),old.load_queries(q['query_descriptor']),trusted)
    met=dict(met);met.update(mod['geometry'].topology(tri,k))
    role_rows=[];pairsets={kind:pairs_by_slot(scan,kind,k) for kind,scan in pairs.items()}
    for kind,pp in pairsets.items():
        aff={j for i,j in pp} if kind=='CF' else {i for pair in pp for i in pair}
        met[kind+'_illegal_pair_count']=len(pp);met[kind+'_illegal_affected_free_fraction']=len(aff)/nf
        buckets=collections.defaultdict(list)
        for i,j in pp:
            typ='C-'+('SEAM' if mask[j] else 'BULK') if kind=='CF' else '-'.join(sorted(('SEAM' if mask[i] else 'BULK','SEAM' if mask[j] else 'BULK')))
            buckets[typ].append((i,j))
        for typ in (('C-SEAM','C-BULK') if kind=='CF' else ('BULK-BULK','BULK-SEAM','SEAM-SEAM')):
            rows=buckets[typ];ids={j for i,j in rows} if kind=='CF' else {x for pair in rows for x in pair}
            role_rows.append(dict(stage=stage,kind=kind,category=typ,pairs=len(rows),affected_faces=len(ids),free_denominator=nf))
    return dict(metrics=met,pairs=pairs,fixed_original_C_conflict=fixed_I,
        actual_C_I='NA_NO_QUERY_FOR_MOVED_C' if stage!='HARD_RESTORE' else fixed_I['ConflictCov_C'],
        C_faithful=np.array_equal(cc,original_C),role_rows=role_rows,pairsets=pairsets),iv


def one(q):
    rid=q['request_id'];dest=OUT/'outputs'/rid;done=dest/'completion.json'
    if done.exists():r=C.read(done);E.verify(r['files']);return dict(path=str(done),sha256=C.sha(done),reused=True,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if dest.exists():raise RuntimeError('Incomplete cell; explicit recovery required '+str(dest))
    C.write(dest/'attempt.json',dict(request=q,model_calls=0,unix=time.time()));tick=time.perf_counter()
    try:
        old,*_=E.helpers();mod=old.cpu_math();ps=paths(q);raw=C.arrays(ps['raw']);proposal=C.arrays(ps['proposal'])['official_unconstrained_proposal'];hard=C.arrays(ps['hard'])['output']
        k=q['K'];original=raw['raw'][:k];fixed=raw['raw_anchor_ids']>=0;bulk=~raw['raw_seam_slot_mask'];bulk[:k]=False
        expected=proposal.copy();expected[fixed]=raw['C_vertices'][raw['raw_anchor_ids'][fixed]]
        if not old.byte_equal(expected,hard) or not old.byte_equal(proposal[bulk],hard[bulk]):raise ValueError('Hard restoration is not exactly the expected cache operation')
        rigid,fitinfo,copies=fit(raw,proposal);save(dest/'rigid.npz',dict(output=rigid,R=np.array(fitinfo['R']),t=np.array(fitinfo['t'])))
        C.write(dest/'rigid_fit.json',fitinfo);C.csvwrite(dest/'C_copy_groups.csv',copies)
        stage_data={};summary=[]
        for stage,tri in zip(STAGES,(proposal,rigid,hard)):
            score,vec=measure(q,stage,tri,original,raw,mod,old,(ps['hard_score'],ps['hard_pairs']) if stage=='HARD_RESTORE' else None)
            stage_data[stage]=score;sd=dest/stage;sd.mkdir()
            C.write(sd/'pairs.json',score['pairs']);C.write(sd/'metrics.json',dict(metrics=score['metrics'],fixed_original_C_conflict=score['fixed_original_C_conflict'],
                actual_C_I=score['actual_C_I'],C_faithful=score['C_faithful'],original_C_I_is_control_success_score=stage=='HARD_RESTORE'))
            if vec is not None:save(sd/'fixed_original_C_conflict_vectors.npz',vec)
            bb=next(x for x in score['role_rows'] if x['category']=='BULK-BULK')
            met=score['metrics'];summary.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],stage=stage,
                CF=met['CF_illegal_affected_free_fraction'],FF=met['FF_illegal_affected_free_fraction'],CF_pairs=met['CF_illegal_pair_count'],FF_pairs=met['FF_illegal_pair_count'],
                bulk_bulk_pairs=bb['pairs'],bulk_bulk_affected_faces=bb['affected_faces'],bulk_faces=int(bulk.sum()),
                I_against_fixed_original_C=score['fixed_original_C_conflict']['ConflictCov_C'],C_faithful=score['C_faithful'],
                exact_coordinate_components=met['EXACT_COORD_face_components'],exact_coordinate_CF_edges=met['EXACT_COORD_CF_shared_edges'],
                free_small_angle=met.get('free_minimum_angle_lt5_fraction'),free_faces=len(tri)-k))
        transitions=[];details=[]
        for after in ('RIGID','HARD_RESTORE'):
            for kind in ('CF','FF'):
                b=stage_data['UNCONSTRAINED']['pairsets'][kind];a=stage_data[after]['pairsets'][kind];bs=set(b);aas=set(a)
                transitions.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],before='UNCONSTRAINED',after=after,kind=kind,
                    original_pairs=len(bs),final_pairs=len(aas),retained=len(bs&aas),resolved=len(bs-aas),new=len(aas-bs),changed_classification=sum(b[p]!=a[p] for p in bs&aas)))
                for action,pp in (('NEW',aas-bs),('RESOLVED',bs-aas)):
                    for i,j in sorted(pp):details.append(dict(request_id=rid,after=after,kind=kind,action=action,slot_i=i,slot_j=j,
                        role_i='C' if i<k else 'SEAM' if raw['raw_seam_slot_mask'][i] else 'BULK',role_j='SEAM' if raw['raw_seam_slot_mask'][j] else 'BULK'))
        C.csvwrite(dest/'stage_metrics.csv',summary);C.csvwrite(dest/'pair_transitions.csv',transitions);C.csvwrite(dest/'pair_changes.csv',details)
        C.csvwrite(dest/'role_pairs.csv',[dict(request_id=rid,**row) for s in STAGES for row in stage_data[s]['role_rows']])
        originalbb={(i,j) for i,j in stage_data['UNCONSTRAINED']['pairsets']['FF'] if bulk[i] and bulk[j]}
        hardbb={(i,j) for i,j in stage_data['HARD_RESTORE']['pairsets']['FF'] if bulk[i] and bulk[j]}
        if originalbb!=hardbb:raise AssertionError('Bulk geometry unchanged but cached pair set differs')
        if 'torch' in sys.modules:raise RuntimeError('Unexpected model framework import')
        files={str(p):C.sha(p) for p in dest.rglob('*') if p.is_file()};C.write(done,dict(status='COMPLETE',request_id=rid,seconds=time.perf_counter()-tick,
            model_calls=0,bulk_geometry_byte_equal=True,bulk_illegal_pair_sets_equal=True,files=files))
        print(json.dumps(dict(request_id=rid,status='COMPLETE',seconds=time.perf_counter()-tick)),flush=True)
        return dict(path=str(done),sha256=C.sha(done),reused=False,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException as e:
        C.write(dest/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),model_calls=0));raise


def run(workers=2):
    r=registration();done=OUT/'score_completion.json'
    if done.exists():x=C.read(done);E.verify(x['files']);return x
    tick=time.perf_counter();qs=r['requests'];first=one(qs[0]);snap=resources()
    n=2 if workers==2 and snap['ram_available_bytes']>max(4<<30,3*first['peak_RSS_KiB']*1024) and (snap['loadavg'] or [999])[0]<max(1,(os.cpu_count() or 1)-2) else 1
    C.write(OUT/('dispatch_'+str(time.time_ns())+'.json'),dict(workers=n,threads_per_worker=1,first=first,resources=snap))
    rs=[first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n,mp_context=multiprocessing.get_context('spawn')) as pool:
        todo=iter(qs[1:]);active={}
        for _ in range(n):
            q=next(todo,None)
            if q:active[pool.submit(one,q)]=q
        while active:
            finished,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            results=[f.result() for f in finished]
            for f,x in zip(finished,results):
                del active[f];rs.append(x);q=next(todo,None)
                if q:active[pool.submit(one,q)]=q
    E.verify(r['protected']);E.verify(r['source_hashes']);files={x['path']:x['sha256'] for x in rs}
    result=dict(status='COMPLETE',requests=16,model_calls=0,new_GEO=0,workers=n,wall_seconds=time.perf_counter()-tick,files=files)
    C.write(done,result);return result


def aggregate():
    import csv
    r=registration();C.read(OUT/'score_completion.json');tables=collections.defaultdict(list);fits=[];refs=[]
    for q in r['requests']:
        d=OUT/'outputs'/q['request_id'];fits.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],**C.read(d/'rigid_fit.json')))
        for n in ('stage_metrics','pair_transitions','pair_changes','role_pairs','C_copy_groups'):
            with (d/(n+'.csv')).open(encoding='utf-8-sig',newline='') as f:tables[n]+= [dict(request_id=q['request_id'],**row) for row in csv.DictReader(f)]
        for stage in ('RAW','POST'):
            s=C.read(paths(q)[stage.lower()+'_score']);met=s['metrics'];refs.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],stage=stage,
                CF=met['CF_illegal_affected_free_fraction'],FF=met['FF_illegal_affected_free_fraction'],I=met['ConflictCov_C'],source_score=str(paths(q)[stage.lower()+'_score'])))
    for n,rows in tables.items():C.csvwrite(OUT/(n+'.csv'),rows)
    C.csvwrite(OUT/'rigid_fit_per_output.csv',fits);C.csvwrite(OUT/'reference_RAW_POST.csv',refs)
    values=tables['stage_metrics'];summary=[];parents=[]
    nums=('CF','FF','CF_pairs','FF_pairs','bulk_bulk_pairs','bulk_bulk_affected_faces','I_against_fixed_original_C','exact_coordinate_components','exact_coordinate_CF_edges')
    for stage in STAGES:
        rr=[x for x in values if x['stage']==stage]
        summary.append(dict(stage=stage,outputs=len(rr),**{n:float(np.mean([float(x[n]) for x in rr])) for n in nums},
            FF_pairs_sum=sum(int(x['FF_pairs']) for x in rr),CF_pairs_sum=sum(int(x['CF_pairs']) for x in rr),bulk_bulk_pairs_sum=sum(int(x['bulk_bulk_pairs']) for x in rr)))
        for pid in sorted({x['parent_id'] for x in rr}):
            pp=[x for x in rr if x['parent_id']==pid];parents.append(dict(parent_id=pid,stage=stage,outputs=len(pp),**{n:float(np.mean([float(x[n]) for x in pp])) for n in nums}))
    C.csvwrite(OUT/'summary.csv',summary);C.csvwrite(OUT/'per_parent.csv',parents)
    transitions=[]
    for after in ('RIGID','HARD_RESTORE'):
        for kind in ('CF','FF'):
            rr=[x for x in tables['pair_transitions'] if x['after']==after and x['kind']==kind]
            transitions.append(dict(after=after,kind=kind,**{n:sum(int(x[n]) for x in rr) for n in ('original_pairs','final_pairs','retained','resolved','new','changed_classification')}))
    result=dict(status='COMPLETE',summary=summary,transitions=transitions,
        max_rigid_distance_invariance_error=max(x['all_mesh_pairwise_distance_invariance_max_error'] for x in fits),
        mean_C_RMS_before=float(np.mean([x['C_weighted_RMS_before'] for x in fits])),mean_C_RMS_after=float(np.mean([x['C_weighted_RMS_after'] for x in fits])),
        max_C_residual=max(x['C_max_error_after'] for x in fits),split_C_groups=sum(x['split_groups'] for x in fits),
        duplicate_C_groups=sum(x['duplicate_groups'] for x in fits),max_C_copy_split=max(x['max_copy_split'] for x in fits))
    C.write(OUT/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','run','aggregate','all','status'],default='status');p.add_argument('--workers',type=int,choices=[1,2],default=2);a=p.parse_args()
    if a.stage=='status':x=C.read(OUT/'score_completion.json') if (OUT/'score_completion.json').exists() else dict(status='NOT_COMPLETE')
    elif a.stage=='prepare':x=prepare()
    elif a.stage=='run':x=run(a.workers)
    elif a.stage=='aggregate':x=aggregate()
    else:prepare();run(a.workers);x=aggregate()
    print(json.dumps(C.clean(x),indent=2))

if __name__=='__main__':main()
