"""Recoverable CPU-only data entrypoint. Old data and results are read-only."""
from __future__ import annotations

from . import common as c
import argparse
import concurrent.futures
import math
import traceback

SIMPLIFIER_MODULE='simplifier'


def freeze():
    path=c.OUT/'protocol.json'
    if path.exists(): return c.read(path)
    source=c.read(c.SOURCE)
    rows=[]
    for row in source['train']:
        if c.sha(row['case_path']) != row['case_sha256']: raise RuntimeError('Source hash mismatch')
        n=int(row['N']); floor=max(128,int(row['K'])+1)
        rows.append(dict(parent_id=row['parent_id'],condition_id=row['condition_id'],
            case_path=row['case_path'],case_sha256=row['case_sha256'],N=n,K=int(row['K']),
            target_faces=max(floor,int(math.ceil(n*.8))),
            minimum_reduction=max(8,int(math.ceil(n*.05)))))
    p=dict(schema=c.VERSION,scope='CPU_DATA_ONLY_FIRST_STAGE',source_manifest=str(c.SOURCE),
        source_manifest_sha256=c.sha(c.SOURCE),rows=rows,parents=len(rows),
        max_new_versions_per_parent=1,max_candidate_paths_per_parent=1,
        proposal='Deterministic constrained QEM edge collapse; preserve C and its original common-edge neighbor faces.',
        reduction_target_fraction=.20,minimum_reduction_fraction=.05,minimum_reduction_faces=8,
        minimum_N=128,maximum_N=799,coordinates='Keep original model transform and source vertex indices; no normalization or C change.',
        admission=dict(exact_C=True,exact_original_CF_interface_faces=True,
            sampled_bidirectional_max_distance_L=.0025,total_area_ratio=[.98,1.02],
            per_component_area_ratio=[.95,1.05],bbox_max_shift_L=.0025,
            no_new_illegal_CF_or_FF=True,no_new_degenerate_or_duplicate_faces=True,
            components_preserved=True,actual_AI_PROXY_visual_review_required=True),
        distance_limit='Deterministic finite-surface probes, NOT a certified Hausdorff bound.',
        source_defects='Retain and disclose old defects; do not automatically weld, deduplicate, repair, or replace parents.',
        split_policy='All versions retain their parent split. No DEV/HOLDOUT parent is used to build the pool.',
        later_training_policy='Sample a parent first, then a registered variant; extra versions do not add parent exposure.',
        current_model_updates=0,current_model_forwards=0,current_generation=0,
        later_stages='Generator comparison budget and relation-head stage will be frozen separately after data admission.',
        dispatch=dict(cpu_workers=2,numeric_threads=1,gpu_workers=0),
        prior_routes=dict(natural_N='Expanded parent/N coverage, no same-parent multi-LOD mix.',
            C_EDGEFLIP='Changed C interior diagonal with fixed N and free target; not the present fixed-C free simplification.',
            JOINT_SHIFT='Changed C/Y coordinates; not the present geometry-version augmentation.'),
        evidence='Official MeshFlow paper S5 describes quadric decimation plus Hausdorff filtering. Fixed-C/interface protection is our adaptation, not an official reproduction.')
    c.write(path,p)
    c.event('PROTOCOL_FROZEN',parents=len(rows),cpu_workers=2,gpu_workers=0)
    return p


def propose(row):
    import time
    folder=c.OUT/'parents'/row['parent_id']
    folder.mkdir(parents=True,exist_ok=True)
    if row['N']-max(128,row['K']+1)<row['minimum_reduction']:
        return dict(parent_id=row['parent_id'],status='SKIP_N_FLOOR',N_original=row['N'],N_candidate=row['N'])
    proposal=folder/'proposal.npz'
    if not proposal.exists():
        if (folder/'proposal_attempt.json').exists():
            raise RuntimeError('Incomplete proposal attempt; inspect ledger before explicit engineering recovery.')
        if c.sha(row['case_path']) != row['case_sha256']: raise RuntimeError('Original changed')
        old=c.arrays(row['case_path'])
        import importlib
        simplify=importlib.import_module('r2.data_tools.bench_multilod_v1.'+SIMPLIFIER_MODULE).simplify
        c.write(folder/'proposal_attempt.json',dict(unix=time.time(),case_sha256=row['case_sha256'],
            target_faces=row['target_faces'],source_code_sha256=c.sha(c.ROOT/('r2/data_tools/bench_multilod_v1/'+SIMPLIFIER_MODULE+'.py'))))
        v,f,ids,ledger=simplify(old['model_vertices'],old['source_faces'],old['known_ids'],row['target_faces'],c.L)
        c.write(folder/'collapse_ledger.json',ledger)
        c.save(proposal,dict(model_vertices=v,source_faces=f,kept_original_face_ids=ids))
    a=c.arrays(proposal)
    return dict(parent_id=row['parent_id'],status='PROPOSED',N_original=row['N'],N_candidate=len(a['source_faces']))


def one(row):
    import time
    import numpy as np
    folder=c.OUT/'parents'/row['parent_id']
    receipt=folder/'receipt.json'
    if receipt.exists():
        old=c.read(receipt)
        for p,h in old.get('artifact_hashes',{}).items():
            if c.sha(p)!=h: raise RuntimeError('Completed data artifact changed: '+p)
        return old
    folder.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    if c.sha(row['case_path']) != row['case_sha256']: raise RuntimeError('Original changed')
    old=c.arrays(row['case_path'])
    base=dict(parent_id=row['parent_id'],condition_id=row['condition_id'],N_original=row['N'],K=row['K'],
        source_case_path=row['case_path'],source_case_sha256=row['case_sha256'],target_faces=row['target_faces'],
        minimum_reduction=row['minimum_reduction'],model_forwards=0,model_updates=0)
    if row['N']-max(128,row['K']+1)<row['minimum_reduction']:
        result=dict(**base,status='BLOCKED_N_FLOOR',N_candidate=row['N'],reduction=0,
            reasons=['Cannot make a meaningful lower-N version while keeping N>=128.'],wall_seconds=time.monotonic()-start)
        c.write(receipt,result)
        return result
    proposal=folder/'proposal.npz'
    if not proposal.exists():
        propose(row)
    a=c.arrays(proposal)
    n=len(a['source_faces'])
    result=dict(**base,N_candidate=n,reduction=row['N']-n,proposal_path=str(proposal),
        proposal_sha256=c.sha(proposal))
    if row['N']-n<row['minimum_reduction']:
        result.update(status='BLOCKED_INSUFFICIENT_REDUCTION',reasons=['Protected simplification did not yield a meaningful alternate N.'])
    else:
        quality_path=folder/'quality.json'
        if not quality_path.exists():
            from .quality import evaluate
            q=evaluate(old['model_vertices'],old['source_faces'],a['model_vertices'],a['source_faces'],
                       a['kept_original_face_ids'],old['known_ids'],c.L)
            c.write(quality_path,q)
        q=c.read(quality_path)
        result.update(quality_path=str(quality_path),quality_sha256=c.sha(quality_path))
        # Independent checker provides a documented pass flag and explicit rejection reasons.
        passed=bool(q.get('geometry_pass',False))
        if not passed:
            result.update(status='BLOCKED_QUALITY',reasons=q.get('rejection_reasons',[])+q.get('blocked_reasons',[]))
        else:
            from .pack import pack_case
            case,audit=pack_case(old,a['model_vertices'],a['source_faces'],a['kept_original_face_ids'])
            dest=folder/'case.npz'
            c.save(dest,case)
            c.write(folder/'packing_audit.json',audit)
            from r2.pilots.bench_natural_n_recipe_v1.data_prepare import verify_case
            checked=verify_case(dict(parent_id=row['parent_id'],condition_id=row['condition_id']+'__LOD80',
                case_path=str(dest),case_sha256=c.sha(dest),N=n,K=row['K']))
            c.write(folder/'native_case_check.json',checked)
            result.update(status='NUMERIC_PASS_PENDING_VISUAL',reasons=[],case_path=str(dest),case_sha256=c.sha(dest))
    result['wall_seconds']=time.monotonic()-start
    result['artifact_hashes']={str(p):c.sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p!=receipt}
    c.write(receipt,result)
    return result


def build(workers=2, proposals_only=False):
    p=freeze()
    if c.sha(c.SOURCE)!=p['source_manifest_sha256']: raise RuntimeError('Source manifest changed')
    res=c.resources()
    if res['available_ram_bytes']<4*(1<<30): raise RuntimeError('CPU RAM headroom insufficient; no dispatch')
    c.event('CPU_DISPATCH',workers=workers,proposals_only=proposals_only,resources=res)
    result=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(propose if proposals_only else one,row):row for row in p['rows']}
        for future in concurrent.futures.as_completed(futures):
            row=futures[future]
            try:
                r=future.result();result.append(r)
                c.event('PARENT_COMPLETE',parent_id=row['parent_id'],status=r['status'],N=r['N_candidate'])
                print(c.json.dumps({k:r[k] for k in ('parent_id','status','N_original','N_candidate','reduction') if k in r},ensure_ascii=False),flush=True)
            except Exception as exc:
                c.event('CPU_ENGINEERING_FAILURE',parent_id=row['parent_id'],error=str(exc),traceback=traceback.format_exc())
                raise
    result.sort(key=lambda r:r['parent_id'])
    c.write(c.OUT/('proposal_results.json' if proposals_only else 'numeric_results.json'),dict(rows=result,model_forwards=0,model_updates=0))
    c.event('CPU_DATA_BUILD_COMPLETE',parents=len(result),numeric_pass=sum(r['status']=='NUMERIC_PASS_PENDING_VISUAL' for r in result),resources=c.resources())


def status():
    p=freeze()
    from collections import Counter
    done=[c.read(c.OUT/'parents'/r['parent_id']/'receipt.json') for r in p['rows'] if (c.OUT/'parents'/r['parent_id']/'receipt.json').exists()]
    print(c.json.dumps(dict(parents=len(p['rows']),completed=len(done),statuses=dict(Counter(r['status'] for r in done))),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','propose','build','status'])
    parser.add_argument('--workers',type=int,choices=[1,2],default=2);args=parser.parse_args()
    if args.stage=='freeze': print(c.json.dumps(dict(parents=freeze()['parents'])))
    elif args.stage in ('build','propose'):build(args.workers,proposals_only=args.stage=='propose')
    else:status()
