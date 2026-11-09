"""Final108 RAW/POST CPU evaluation with isolated, blinded model aliases.

The original24 BASE POST outputs are read-only cache. Exactly84 new outputs
need existing official-D0 proposals. No weights or model module are imported.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import concurrent.futures
import itertools
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from . import data as D
from . import evaluation as E
from .cpu_worker import _single_thread, _verify_files, torch_free, _post

VERSION='terminal_reward_nft_final_cpu_v1'
LABELS=('BASE','NFT','SELF_FM')
ALIASES=('A','B','C')
EV=D.OUT/'evaluation'
PRIVATE=EV/'private'


def prepare_blinding():
    """Freeze mappings from request IDs only, before final output review."""
    torch_free()
    manifest=D.read(D.OUT/'inputs/manifest.json')
    queries=sorted(manifest['evaluation_requests'],key=lambda q:q['request_id'])
    if len(queries)!=36:raise ValueError('Expected36 frozen evaluation requests')
    path=PRIVATE/'label_mapping.json'
    if path.exists():
        old=D.read(path)
        if old['manifest_sha256']!=D.sha(D.OUT/'inputs/manifest.json'):
            raise ValueError('Blind mapping manifest changed')
        return old
    permutations=list(itertools.permutations(LABELS))*6
    rng=np.random.RandomState(D.noise_seed(VERSION,'BLIND_LABEL_ASSIGNMENT'))
    order=rng.permutation(36)
    rows=[]
    for q,index in zip(queries,order):
        mapping=dict(zip(ALIASES,permutations[int(index)]))
        rows.append(dict(request_id=q['request_id'],aliases=mapping))
    result=dict(status='FROZEN',version=VERSION,manifest_sha256=D.sha(D.OUT/'inputs/manifest.json'),
        rows=rows,mapping_scope='PRIVATE; do not pass this file or real-label source paths to image reviewers',
        permutation='Each of6 permutations appears6 times; shuffled with fixed preregistered MT19937 seed from request-independent namespace',
        selection_uses='Frozen request IDs only; no metric, image, or generated outcome',
        model_calls=0,quality_scores=0)
    D.write(path,result)
    D.write(EV/'blind_design.json',dict(status='FROZEN',version=VERSION,requests=36,aliases=list(ALIASES),
        assignment='Independent fixed model permutation per request, balanced across36 requests',
        review_scope='Model labels hidden; condition/noise identity visible. Reviewers receive only anonymous gallery artifacts, not the private mapping or model-labelled source paths.',
        review_labels='A/B/C are local to one request and must never be interpreted as fixed global model identities.',
        output_selection='All36 requests and all3 models retained; no outcome selection',model_calls=0))
    return result


def _cached_post_index():
    path=D.FULL/'evaluation/indices/FULL_S6000/post_index.json'
    saved=D.read(path)
    if saved['status']!='COMPLETE' or len(saved['rows'])!=24:
        raise ValueError('Original FULL POST index is not complete24')
    return path,{r['request_id']:r for r in saved['rows']}


def _records(label,q,manifest,cached):
    if label=='BASE' and q['evaluation_noise_partition']=='OLD':
        raw=manifest['base_raw_cache'][q['request_id']]
        post=cached[q['request_id']]
        if raw['checkpoint_sha256']!=manifest['source_binding']['sha256']:
            raise ValueError('Wrong cached BASE model')
        # Historical descriptive fields can say HALF. Trust the frozen FULL q
        # and actual C arrays, never reinterpret its metadata as input geometry.
        for key in ('input_sha256','C_sha256','Gaussian_sha256','N','K'):
            if raw[key]!=q[key] or post[key]!=q[key]:
                raise ValueError('Cached geometry identity mismatch: '+key)
        if D.sha(post['post_path'])!=post['post_sha256']:
            raise ValueError('Cached BASE POST changed')
        return raw,dict(post_path=post['post_path'],post_sha256=post['post_sha256'],
                        cached_original_POST=True,original_record=post)
    root=EV/label
    raw=D.read(root/'raw'/q['request_id']/'receipt.json')
    postpath=root/'post'/q['request_id']/'receipt.json'
    return raw,D.read(postpath) if postpath.exists() else None


def _validate_geometry(q,raw,post):
    E.package()
    from .sample import verify_receipt
    verify_receipt(q,raw)
    if post is None:raise ValueError('POST not yet completed')
    if D.sha(post['post_path'])!=post['post_sha256']:
        raise ValueError('Final POST changed')
    a=D.arrays(raw['raw_path'])['raw'].reshape(q['N'],3,3)
    p=D.arrays(post['post_path'])
    source=E.source_metadata(q['condition_source'],a[:q['K']])
    if not D.same(p['output'][:q['K']],source['C_original']):
        raise ValueError('Final source-oriented C differs from exact manifest input')
    if not D.same(p['vertices'][p['faces']],p['output']):
        raise ValueError('Final indexed POST inconsistent')


def _job(job):
    label,q,raw,post,query,stages,regsha=job
    torch_free()
    tick=time.perf_counter()
    with _single_thread():
        if post is None and 'post' in stages:
            post=_post(q,EV/label,raw)
        elif post is None:
            raise ValueError('Final POST unavailable: '+label+'/'+q['request_id'])
        _validate_geometry(q,raw,post)
        result=None
        if 'score' in stages:
            result=E.score_one(q,raw['raw_path'],post['post_path'],q['condition_source'],query,
                EV/label/'scores'/q['request_id'],label=label,registration_sha256=regsha,
                stages=('RAW','POST'))
    torch_free()
    files={raw['raw_path']:raw['raw_sha256'],post['post_path']:post['post_sha256']}
    if result:files.update({p:D.sha(p) for p in result['score_paths'].values()})
    return dict(label=label,request_id=q['request_id'],condition_id=q['condition_id'],
        raw_path=raw['raw_path'],post_path=post['post_path'],condition_path=q['condition_source'],
        cached_original_POST=bool(post.get('cached_original_POST',False)),files=files,
        score_paths=result['score_paths'] if result else {},seconds=time.perf_counter()-tick,
        model_calls=0,torch_imported=False)


def collect_inputs():
    """Verify all108 RAW before allowing any final postprocessing."""
    torch_free()
    m=D.read(D.OUT/'inputs/manifest.json')
    path,cache=_cached_post_index()
    rows=[]
    for label in LABELS:
        for q in m['evaluation_requests']:
            raw,post=_records(label,q,m,cache)
            if raw['status']!='COMPLETE' or raw['input_sha256']!=q['input_sha256']:
                raise ValueError('All108 final RAW must precede final Stage2')
            rows.append((label,q,raw,post))
    if len(rows)!=108 or sum(p is not None and bool(p.get('cached_original_POST')) for _,_,_,p in rows)!=24:
        raise ValueError('Final108/BASE24 cache accounting mismatch')
    return m,rows,dict(path=str(path),sha256=D.sha(path))


def _copy_frozen(source,dest):
    source=D.resolve(source);dest=D.checked(dest)
    h=D.sha(source)
    if dest.exists():
        if D.sha(dest)!=h:raise ValueError('Anonymous geometry copy changed')
        return h
    dest.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as src,dest.open('xb') as dst:
        for block in iter(lambda:src.read(1024*1024),b''):dst.write(block)
        dst.flush();os.fsync(dst.fileno())
    if D.sha(dest)!=h:raise ValueError('Anonymous copy lost bytes')
    return h


def render_blind(*,workers=1):
    m,rows,_=collect_inputs()
    mapping=prepare_blinding()
    byid={r['request_id']:r['aliases'] for r in mapping['rows']}
    cases=[];private_provenance=[]
    with _single_thread():
        for label,q,raw,post in rows:
            _validate_geometry(q,raw,post)
            alias=next(a for a,name in byid[q['request_id']].items() if name==label)
            dest=EV/'anonymous_inputs'/q['request_id']/alias
            rawsha=_copy_frozen(raw['raw_path'],dest/'raw.npz')
            postsha=_copy_frozen(post['post_path'],dest/'post.npz')
            # Only explicit safe fields enter the public renderer. Full receipts,
            # model IDs, source model paths and score identities remain private.
            public={k:q[k] for k in ('request_id','condition_id','parent_id','candidate','split','N','K')}
            public.update(label=alias,raw_path=str(dest/'raw.npz'),post_path=str(dest/'post.npz'),
                          condition_path=q['condition_source'])
            cases.append(public)
            private_provenance.append(dict(request_id=q['request_id'],label=label,alias=alias,
                original_raw_path=raw['raw_path'],original_post_path=post['post_path'],
                anonymous_raw_path=str(dest/'raw.npz'),anonymous_post_path=str(dest/'post.npz'),
                raw_sha256=rawsha,post_sha256=postsha))
        D.write(PRIVATE/'anonymous_geometry_provenance.json',dict(status='COMPLETE',version=VERSION,
            rows=private_provenance,mapping_sha256=D.sha(PRIVATE/'label_mapping.json'),copies_are_byte_identical=True))
        # Separate renderer keeps the already frozen training renderer unchanged.
        from .render_blind import render_panel
        result=render_panel(cases,EV/'blind_gallery',label_order=list(ALIASES),expected_cases=108,workers=workers)
    torch_free()
    # Explicit check that public serialized gallery metadata cannot leak true labels.
    public_text=json.dumps(result,ensure_ascii=False)
    if 'SELF_FM' in public_text or '/evaluation/BASE/' in public_text or '/evaluation/NFT/' in public_text:
        raise RuntimeError('Real model identity leaked into anonymous gallery metadata')
    return result


def run(*,stages=('post','score','render'),workers=1):
    torch_free()
    if workers not in (1,2):raise ValueError('Measured CPU worker count must be1 or2')
    stages=tuple(s for s in ('post','score','render') if s in stages)
    if not stages:raise ValueError('No final stage selected')
    m,rows,old_index=collect_inputs()
    queries={q['condition_id']:q for q in m['evaluation_queries']}
    regsha=D.sha(D.OUT/'registration.json')
    done=EV/'cpu'/('_'.join(stages)+'_complete.json')
    if done.exists():
        old=D.read(done);_verify_files(old['files'])
        if old['registration_sha256']!=regsha:raise ValueError('Final registration changed')
        return old
    tick=time.perf_counter();results=[];gallery=None
    if any(s in stages for s in ('post','score')):
        jobs=[(label,q,raw,post,queries[q['condition_id']],stages,regsha) for label,q,raw,post in rows]
        if workers==1:results=[_job(j) for j in jobs]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                results=list(pool.map(_job,jobs))
        D.write(EV/'indices'/('_'.join(s for s in stages if s!='render')+'_index.json'),
                dict(status='COMPLETE',version=VERSION,rows=results,requests=108))
    if 'render' in stages:gallery=render_blind(workers=workers)
    torch_free()
    files={old_index['path']:old_index['sha256']}
    for r in results:files.update(r['files'])
    if gallery:files[str(EV/'blind_gallery/manifest.json')]=D.sha(EV/'blind_gallery/manifest.json')
    result=dict(status='COMPLETE',version=VERSION,stages=stages,requests=108,distinct_requests=36,
                BASE_cached_old_POST=24,required_new_POST=84,model_calls=0,torch_imported=False,
                registration_sha256=regsha,manifest_sha256=D.sha(D.OUT/'inputs/manifest.json'),
                workers=workers,CPU_threads_per_worker=1,seconds=time.perf_counter()-tick,files=files,
                gallery_manifest=str(EV/'blind_gallery/manifest.json') if gallery else None)
    D.write(done,result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='all',help='post,score,render,all, or comma combination')
    parser.add_argument('--workers',type=int,choices=(1,2),default=1)
    parser.add_argument('--prepare-blinding',action='store_true')
    args=parser.parse_args()
    if args.prepare_blinding:
        result=prepare_blinding()
        print(json.dumps(dict(status=result['status'],requests=len(result['rows']),model_calls=0)),flush=True)
        return
    stages=('post','score','render') if args.stage=='all' else tuple(args.stage.replace('+',',').split(','))
    if any(s not in ('post','score','render') for s in stages):parser.error('Unknown stage')
    r=run(stages=stages,workers=args.workers)
    print(json.dumps({k:r[k] for k in ('status','requests','required_new_POST','workers','seconds','model_calls')}),flush=True)


if __name__=='__main__':main()
