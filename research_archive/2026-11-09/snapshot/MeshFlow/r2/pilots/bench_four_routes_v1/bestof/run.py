"""Registered CPU/GPU stages; GPU stages are launched only by the root dispatcher."""
from __future__ import annotations
import argparse, collections, concurrent.futures, copy, inspect, json, multiprocessing, resource, sys, time, types
from pathlib import Path
from .common import *


def raw_completion(label=LABEL):
    if label != LABEL: raise ValueError(label)
    return OUT / 'evaluation' / LABEL / 'raw_complete_full.json'


def raw_receipt(q, label=LABEL):
    if label != LABEL: raise ValueError(label)
    return raw_dir(q) / 'receipt.json'


def verify_raw(label=LABEL):
    qs = new_requests(label); done = read(raw_completion(label))
    if done['status'] != 'COMPLETE' or done['outputs'] != 46 or done['total_new_generator_forwards'] != 2300:
        raise ValueError('New RAW wave incomplete')
    verify(done['files'])
    return qs, [checked_raw(q) for q in qs]


def all_raw_barrier():
    registered();verify_raw()
    result=[checked_raw(q) for q in candidates()]
    if len(result)!=92:raise ValueError('All92 RAW required before selection/D0')
    return result


def label_spec(label=LABEL):
    if label!=LABEL:raise ValueError(label)
    return dict(mode='PLAIN',arm='NATURAL_HIGH',local_step=1500,global_step=22500,subset='full',
        schema='bench_natural_n_recipe_generator_v1',generator_path=str(GENERATOR))


def freeze_sample(label=LABEL):
    reg=registered()
    if sha(GENERATOR)!=GENERATOR_SHA:raise ValueError('Narrow B checkpoint changed')
    result=dict(status='FROZEN',label=LABEL,spec=label_spec(label),outputs=46,new_rollouts=46,reused_rollouts=0,
        maximum_generator_forwards=2300,maximum_head_forwards=0,solver='Euler50',representation='NATIVE_ALL_FREE',
        source_labels_read=False,code=reg['code'],protected={**reg['inputs'],str(GENERATOR):GENERATOR_SHA})
    immutable(OUT/'evaluation/inference_registrations/B4__RAW.json',result);return result


def sampler_function():
    """Build an isolated callable; CPU tests can inspect it without invoking it."""
    from r2.pilots.bench_group_weight_match_v1 import evaluation as original
    # Independent globals dictionary: never mutate imported/frozen modules.
    ns=dict(vars(original));ns.update(globals())
    ns.update(__package__=__package__,NEW_LABELS=(LABEL,),requests=new_requests,
              reuse_spec=lambda q,label:None,load_head=lambda label,payload:(None,dict(kind='NONE_NATIVE_ALL_FREE',parameters=0,source_labels_read=False)),
              source_assets=original.source_assets)
    def clone(fn):
        result=types.FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
        result.__kwdefaults__=fn.__kwdefaults__;return result
    ns['save_layout']=clone(original.save_layout)
    return clone(original.sample_run)


def sample(estimated_peak_mib=2800):
    registered()
    if raw_completion().exists():verify_raw();return read(raw_completion())
    if list((OUT/'evaluation'/LABEL).glob('sampling_attempt_*')):raise RuntimeError('Prior sample attempt; no automatic retry')
    return sampler_function()(LABEL,estimated_peak_mib)


def cpu_modules():
    if 'torch' in sys.modules:raise RuntimeError('Run CPU scoring in a separate process')
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
    return CPU


def score_registration():
    reg=registered()
    result=dict(status='FROZEN',registration_sha256=sha(OUT/'registration.json'),source_hashes=reg['code'],
        original_metric_fields=read(MANIFEST)['metric_fields'],new_RAW_score_budget=46,reused_RAW_scores=46,
        selected_POST_only=True,maximum_new_POST_scores=46,model_calls=0)
    immutable(OUT/'evaluation/score_registration.json',result);return result


def score_raw_one(q):
    CPU=cpu_modules(); checked_raw(q)
    if q['noise_index']<2:raise ValueError('Do not rescore cached B')
    folder=score_dir(q,'RAW');done=folder/'audit_receipt.json'
    identity=dict(request_id=q['request_id'],raw_sha256=sha(raw_dir(q)/'raw.npz'),registration_sha256=sha(OUT/'registration.json'))
    if done.exists():
        r=read(done)
        if r['identity']!=identity:raise ValueError('Score identity changed')
        verify(r['files']);return r
    if folder.exists():raise RuntimeError('Partial score retained; no silent retry')
    tick=time.perf_counter()
    CPU.score_current(q,arrays(raw_dir(q)/'raw.npz'),folder,'RAW',False)
    result=dict(status='PASS',identity=identity,seconds=time.perf_counter()-tick,model_calls=0,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        files={str(p):sha(p) for p in folder.iterdir() if p.is_file()})
    immutable(done,result)
    print('RAW_SCORE',q['request_id'],flush=True);return result


def selected():
    r=read(OUT/'selection/complete.json');verify(r['files']);verify(r['inputs'])
    if r['status']!='COMPLETE':raise ValueError('Selection incomplete')
    rows=read(OUT/'selection/winners.json')
    lookup={q['request_id']:q for q in candidates()}
    return [lookup[r['request_id']] for r in rows if r['status']=='SELECTED']


def new_selected():
    return [q for q in selected() if q['noise_index']>=2]


def score_post_one(q):
    CPU=cpu_modules()
    if q['noise_index']<2:raise ValueError('Do not rerun cached post')
    rp=post_dir(q)/'receipt.json';receipt=read(rp)
    if receipt['D0_forwards']!=1 or receipt['GEO_calls']!=1 or receipt['raw_sha256']!=sha(raw_dir(q)/'raw.npz'):
        raise ValueError('New selected D0 identity')
    verify(receipt['files'])
    cache=OUT/'base_pair_caches'/LABEL/q['request_id'];done=cache/'complete.json'
    if done.exists():verify(read(done)['files'])
    else:
        if cache.exists():raise RuntimeError('Partial pair scan retained')
        immutable(cache/'started.json',dict(mesh_sha256=sha(post_dir(q)/'post.npz'),model_calls=0))
        immutable(cache/'pairs.json',CPU.H.scan(arrays(post_dir(q)/'post.npz')['output'],q['K']))
        immutable(done,dict(status='COMPLETE',files={str(p):sha(p) for p in cache.iterdir() if p.is_file()}))
    return CPU.process_one(q,LABEL,post_dir(q)/'post.npz',post_dir(q)/'pre_geo.npz',raw_dir(q)/'raw.npz',
        cache/'pairs.json',OUT/'cpu'/LABEL/q['request_id'],False,sha(OUT/'registration.json'))


def cpu_job(job):
    mode,q=job
    return score_raw_one(q) if mode=='raw' else score_post_one(q)


def run_cpu(mode,workers=3):
    score_registration();all_raw_barrier()
    done=OUT/'evaluation/completions'/(mode+'.json')
    if done.exists():r=read(done);verify(r['files']);return r
    qs=new_requests() if mode=='raw' else new_selected()
    jobs=[(mode,q) for q in qs];tick=time.perf_counter();count=0
    if jobs:
        CPU=cpu_modules()
        from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
        before=CPU.C.resources();first=cpu_job(jobs[0]);after=CPU.C.resources()
        count=worker_count(after,first,max(1,workers))
        immutable(OUT/'evaluation/dispatch'/(mode+'.json'),dict(resources_before=before,resources_after_first=after,
            workers=count,threads_per_worker=1,first_peak_RSS_KiB=first['peak_RSS_KiB'],model_calls=0))
        with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(cpu_job,jobs[1:]))
    paths=[score_dir(q,'RAW')/'audit_receipt.json' if mode=='raw' else OUT/'cpu'/LABEL/q['request_id']/'complete.json' for q in qs]
    result=dict(status='COMPLETE',mode=mode,new_requests=len(qs),workers=count,seconds=time.perf_counter()-tick,
        model_calls=0,files={str(p):sha(p) for p in paths})
    immutable(done,result);return result


def diversity(candidates_for_C):
    """Area-weighted finite-face-centroid NN distance; descriptive, not TMD."""
    import numpy as np
    from scipy.spatial import cKDTree
    rows=[];cache=[]
    for q in candidates_for_C:
        inp=arrays(q['inference_npz']);C=inp['C_vertices'][inp['C_faces_original']]
        scale=float(np.linalg.norm(np.ptp(C.reshape(-1,3).astype(np.float64),axis=0)))
        F=arrays(raw_dir(q)/'raw.npz')['output'][q['K']:].astype(np.float64)
        area=.5*np.linalg.norm(np.cross(F[:,1]-F[:,0],F[:,2]-F[:,0]),axis=1)
        mask=area>1e-12*scale*scale;cache.append((q,F[mask].mean(1),area[mask],scale))
    for i,(a,ac,aw,scale) in enumerate(cache):
        for b,bc,bw,_ in cache[i+1:]:
            value=None
            if len(ac) and len(bc):
                value=float((np.dot(cKDTree(bc).query(ac)[0],aw)/aw.sum()+np.dot(cKDTree(ac).query(bc)[0],bw)/bw.sum())/(2*scale))
            rows.append(dict(condition_id=a['condition_id'],parent_id=a['parent_id'],request_a=a['request_id'],request_b=b['request_id'],
                area_weighted_centroid_NN_over_C_diagonal=value,status='PASS' if value is not None else 'NA_NO_VALID_FREE_TRIANGLE',
                interpretation='geometric_difference_only_not_semantic_diversity_or_TMD'))
    return rows


def select():
    from .selector import measure,choose
    reg=registered();all_raw_barrier()
    done=OUT/'selection/complete.json'
    if done.exists():r=read(done);verify(r['files']);verify(r['inputs']);return r
    qs=candidates();by=collections.defaultdict(list);rows=[];inputs={};winners=[];div=[]
    for q in qs:
        row,score=checked_score(q,'RAW');mesh=arrays(raw_dir(q)/'raw.npz');inp=arrays(q['inference_npz'])
        # Only C, the generated output, and the two frozen C/F & F/F fractions
        # enter measure. Neither source target nor human/AI visual labels enter.
        detail=measure(mesh['output'],inp['C_vertices'],inp['C_faces_original'],q['N'],q['K'],
            row['CF_illegal_affected_free_fraction'],row['FF_illegal_affected_free_fraction'],q['noise_sha256'])
        detail.update(request_id=q['request_id'],condition_id=q['condition_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],
            report_group=q['report_group'],ConflictCov_C_report_only=row['ConflictCov_C'],
            I_status_report_only=score['status_by_metric']['I'],original_free_near_degenerate_fraction=row['free_near_degenerate_fraction'],
            raw_sha256=sha(raw_dir(q)/'raw.npz'),reused_candidate=q['noise_index']<2)
        rows.append(detail);by[q['condition_id']].append(detail)
        for p in (raw_dir(q)/'raw.npz',score_dir(q,'RAW')/'row.json',score_dir(q,'RAW')/'score.json',resolve(q['inference_npz'])):
            inputs[str(p)]=sha(p)
    for cid,part in sorted(by.items()):
        if len(part)!=4:raise ValueError('Four candidates required')
        winner=choose(part)
        winners.append(dict(condition_id=cid,status='SELECTED' if winner else 'FAILED_ALL_INELIGIBLE',
            request_id=winner['request_id'] if winner else None,noise_index=winner['noise_index'] if winner else None,
            rank=winner['rank'] if winner else None,eligible_candidates=sum(r['eligible'] for r in part),
            new_D0_needed=bool(winner and winner['noise_index']>=2),no_automatic_fallback=True))
        div.extend(diversity([q for q in qs if q['condition_id']==cid]))
    immutable(OUT/'selection/candidates.json',rows);csvwrite(OUT/'selection/candidates.csv',rows)
    immutable(OUT/'selection/winners.json',winners);csvwrite(OUT/'selection/winners.csv',winners)
    immutable(OUT/'selection/diversity.json',div);csvwrite(OUT/'selection/diversity.csv',div)
    result=dict(status='COMPLETE',candidates=92,conditions=23,selected=sum(w['status']=='SELECTED' for w in winners),
        failed_conditions=sum(w['status']!='SELECTED' for w in winners),new_D0_needed=sum(w['new_D0_needed'] for w in winners),
        reused_D0=sum(w['status']=='SELECTED' and not w['new_D0_needed'] for w in winners),selector_sha256=sha(OUT/'data/selector_spec.json'),
        code=reg['code'],inputs=inputs,model_calls=0,UTC=utc(),
        files={str(p):sha(p) for p in (OUT/'selection').iterdir() if p.is_file()})
    immutable(done,result);return result


def d0(estimated_peak_mib=3500):
    reg=registered();barrier=all_raw_barrier();pending=new_selected()
    dest=OUT/'evaluation'/LABEL;done=dest/'post_complete_full.json'
    if done.exists():r=read(done);verify(r['files']);return r
    if list(dest.glob('postprocess_attempt_*')):raise RuntimeError('Prior D0 attempt requires review; no retry')
    if len(pending)>23:raise ValueError('D0 budget exceeded')
    if not pending:
        result=dict(status='COMPLETE',outputs=0,new_D0_forwards=0,new_GEO_calls=0,reason='All eligible winners reused',files={})
        immutable(done,result);return result
    from r2.pilots.bench_parts_convergence_v1 import postprocess as legacy
    immutable(OUT/'evaluation/inference_registrations/B4__D0.json',dict(status='FROZEN',outputs=len(pending),
        new_D0_budget=len(pending),max_D0_budget=23,selection_sha256=sha(OUT/'selection/complete.json'),
        D0_checkpoint=str(legacy.OLD.D0),D0_checkpoint_sha256=legacy.OLD.D0_SHA,code=reg['code'],
        RAW_barrier={str(p):sha(p) for p in barrier}))
    def checked(root,labels,subset):
        if resolve(root)!=OUT or labels!=[LABEL] or subset!='full':raise ValueError('D0 scope differs')
        return pending,all_raw_barrier()
    ns=dict(vars(legacy),verify_raw_barrier=checked)
    fn=types.FunctionType(legacy.run.__code__,ns,legacy.run.__name__,legacy.run.__defaults__,legacy.run.__closure__)
    fn.__kwdefaults__=legacy.run.__kwdefaults__
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    observer=GPUObserver('bestof4_selected_single_D0',interval=1.);observer.__enter__()
    try:
        result=fn(OUT,LABEL,[LABEL],'full',legacy.OLD.D0,estimated_peak_mib)
        if result['new_D0_forwards']!=len(pending):raise ValueError('D0 ledger differs')
        verify(reg['code']);return result
    finally:
        observer.__exit__(None,None,None)
        immutable(OUT/'evaluation/D0_resource_observation.json',dict(summary=observer.summary(),samples=observer.samples))


def collect():
    registered();all_raw_barrier();done=OUT/'results/completion.json'
    if done.exists():r=read(done);verify(r['files']);return r
    from r2.pilots.bench_adaptation_strength_v1.collect_v2 import aggregate,numeric
    qs=candidates();chosen={q['condition_id']:q for q in selected()};rows=[];allraw=[];sources={}
    metrics=read(MANIFEST)['metric_fields']
    for q in qs:
        row,score=checked_score(q,'RAW')
        allraw.append(dict(row,report_group=q['report_group'],candidate_noise_index=q['noise_index'],origin=q['bestof_source']))
    for cid in sorted({q['condition_id'] for q in qs}):
        part=[q for q in qs if q['condition_id']==cid]
        for label,q in [('FIXED_N0',next(q for q in part if q['noise_index']==0)),
                        ('FIXED_N1',next(q for q in part if q['noise_index']==1)),('SELECTED',chosen.get(cid))]:
            if q is None:
                base=part[0]
                for stage in STAGES:
                    rows.append(dict({field:None for field in metrics},label=label,stage=stage,report_group=base['report_group'],
                        parent_id=base['parent_id'],condition_id=cid,request_id=None,selected_noise_index=None,
                        status='FAILED_ALL_INELIGIBLE',comparison_unit='condition',result_origin='NO_FALLBACK',
                        seam_applicability='NA_NOT_SEAM_REPRESENTATION'))
                continue
            for stage in STAGES:
                row,score=checked_score(q,stage)
                rows.append(dict(row,label=label,stage=stage,report_group=q['report_group'],selected_noise_index=q['noise_index'],
                    comparison_unit='condition',result_origin='REUSED_B' if q['noise_index']<2 else 'NEW_B',seam_applicability=score['seam_to_bulk']['status']))
                for name in ('row.json','score.json','complete.json'):
                    p=score_dir(q,stage)/name;sources[str(p)]=sha(p)
    condition=aggregate(rows,('label','stage','report_group','parent_id','condition_id'),metrics)
    parents=aggregate(condition,('label','stage','report_group','parent_id'),metrics)
    summary=aggregate(parents,('label','stage','report_group'),metrics)
    lookup={(r['label'],r['stage'],r['condition_id']):r for r in rows};paired=[]
    for row in rows:
        if row['label']!='SELECTED':continue
        for ref in ('FIXED_N0','FIXED_N1'):
            a=lookup[ref,row['stage'],row['condition_id']]
            p=dict(reference=ref,candidate='SELECTED',stage=row['stage'],condition_id=row['condition_id'],parent_id=row['parent_id'],
                report_group=row['report_group'],reference_request_id=a['request_id'],selected_request_id=row['request_id'])
            for field in metrics:
                av,bv=a[field],row[field];p[field+'__reference']=av;p[field+'__candidate']=bv
                p[field+'__delta']=bv-av if numeric(av) and numeric(bv) else None
            paired.append(p)
    tables=dict(all_RAW_candidates=allraw,per_output=rows,per_condition=condition,per_parent=parents,parent_macro_summary=summary,paired_per_condition=paired)
    for name,table in tables.items():csvwrite(OUT/'results'/(name+'.csv'),table)
    immutable(OUT/'results/metric_rows.json',rows)
    result=dict(status='COMPLETE',logical_rows=len(rows),all_RAW_candidate_rows=92,numeric_fields=len(metrics),selected_conditions=len(chosen),
        failed_conditions=23-len(chosen),reference_labels=['FIXED_N0','FIXED_N1'],weight_update_claim=False,
        generator_new_forwards=2300,new_D0=read(OUT/'evaluation'/LABEL/'post_complete_full.json')['new_D0_forwards'],
        visual_review='PENDING_ROOT_UNIFIED_FIVE_AXES',sources=sources,
        files={str(p):sha(p) for p in (OUT/'results').iterdir() if p.is_file()})
    immutable(done,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['sample','score-raw','select','d0','score-post','collect'])
    p.add_argument('--workers',type=int,default=3);p.add_argument('--estimated-peak-mib',type=float)
    a=p.parse_args()
    if a.action=='sample':r=sample(a.estimated_peak_mib or 2800)
    elif a.action=='d0':r=d0(a.estimated_peak_mib or 3500)
    elif a.action=='score-raw':r=run_cpu('raw',a.workers)
    elif a.action=='score-post':r=run_cpu('post',a.workers)
    elif a.action=='select':r=select()
    else:r=collect()
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','inputs','code','sources')},indent=2))


if __name__=='__main__':main()
