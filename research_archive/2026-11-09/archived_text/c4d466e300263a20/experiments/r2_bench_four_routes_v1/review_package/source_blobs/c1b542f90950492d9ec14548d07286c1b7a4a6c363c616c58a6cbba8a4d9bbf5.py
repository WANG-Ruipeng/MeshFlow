"""Prepare and freeze candidates using only CPU file/array checks."""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, sys
import numpy as np
from .common import *
from .selector import SPEC


def prepare():
    done = OUT / 'audits/cache_audit.json'
    if done.exists():
        r = read(done); verify(r['files']); verify(r['protected']); return r
    source_manifest = SOURCE / 'data/eval_manifest.json'
    source = read(source_manifest)
    original = source['requests']
    if len(original) != 46 or len(source['metric_fields']) != 177:
        raise ValueError('Original panel size/metrics differ')
    groups = collections.defaultdict(list)
    for q in original:
        groups[q['condition_id']].append(q)
    if len(groups) != 23 or any(sorted(q['noise_index'] for q in qs) != [0, 1] for qs in groups.values()):
        raise ValueError('Expected 23 conditions with two original noises each')
    sidecar = read(GENERATOR.with_suffix('.json'))
    if (sidecar['sha256'] != GENERATOR_SHA or sidecar['arm'] != 'NATURAL_HIGH'
            or sidecar['global_step'] != 22500 or sidecar['local_step'] != 1500):
        raise ValueError('Narrow B identity differs')
    protected = {str(source_manifest): sha(source_manifest), str(GENERATOR.with_suffix('.json')): sha(GENERATOR.with_suffix('.json'))}
    cache = []
    for q in original:
        p = checked_raw(q); rr = read(p)
        protected[str(p)] = sha(p); protected.update(rr['files'])
        post = post_dir(q) / 'receipt.json'; pr = read(post)
        if pr['D0_forwards'] != 1 or pr['GEO_calls'] != 1 or pr['raw_sha256'] != rr['raw_sha256']:
            raise ValueError('Cached B post identity differs')
        verify(pr['files']); protected[str(post)] = sha(post); protected.update(pr['files'])
        for stage in STAGES:
            row, score = checked_score(q, stage)
            if any(m not in row for m in source['metric_fields']):
                raise ValueError('Original metric missing')
            folder = score_dir(q, stage)
            for name in ('row.json', 'score.json', 'complete.json'):
                path = folder / name; protected[str(path)] = sha(path)
        for key, hkey in (('inference_npz', 'inference_npz_sha256'), ('metadata_npz', 'metadata_sha256'),
                          ('source_npz', 'source_npz_sha256'), ('noise_npz', 'noise_sha256')):
            if sha(q[key]) != q[hkey]:
                raise ValueError('Frozen input differs: ' + key)
            protected[str(resolve(q[key]))] = q[hkey]
        qp = q['query_descriptor']
        if sha(qp['path']) != qp['sha256']:
            raise ValueError('Original query changed')
        protected[str(resolve(qp['path']))] = qp['sha256']
        cache.append(dict(request_id=q['request_id'],condition_id=q['condition_id'],report_group=q['report_group'],
            raw=str(p.parent/'raw.npz'),raw_sha256=rr['raw_sha256'],generator_sha256=rr['generator']['sha256'],
            post=str(post.parent/'post.npz'),score_stages=list(STAGES),origin=str(cached_root(q))))
    seeds = []
    used_seeds = {q['seed'] for q in original}
    for cid in sorted(groups):
        for ni in (2, 3):
            seed = int.from_bytes(hashlib.sha256((VERSION+'|EVAL|'+cid+'|'+str(ni)).encode()).digest()[:4], 'little')
            if seed in used_seeds:
                raise ValueError('Seed collision; no automatic seed substitution')
            used_seeds.add(seed)
            seeds.append(dict(condition_id=cid, noise_index=ni, seed=seed))
    immutable(OUT/'data/seed_registration.json', dict(status='FROZEN_BEFORE_NEW_NOISE',version=VERSION,
        rule='sha256(version|EVAL|condition_id|noise_index)[:4]little; NumPy RandomState normal FP32',rows=seeds))
    qs = [copy.deepcopy(q) for q in original]
    for row in seeds:
        q = copy.deepcopy(sorted(groups[row['condition_id']], key=lambda r:r['noise_index'])[0])
        ni, seed = row['noise_index'], row['seed']
        rid = q['condition_id'] + '__n' + str(ni)
        noise = OUT / 'evaluation_inputs/requests' / rid / 'noise.npz'
        z = np.random.RandomState(seed).standard_normal((q['N']-q['K'],3,3)).astype(np.float32)
        array_file(noise, dict(z=z, seed=np.asarray(seed,np.int64)))
        q.update(request_id=rid,noise_index=ni,seed=seed,noise_npz=str(noise),noise_sha256=sha(noise),noise_npz_sha256=sha(noise),
                 panel_origin='BESTOF_NEW_PREREGISTERED_NOISE',baseline_reuse=False)
        qs.append(q)
    for q in qs:
        q.update(bestof_candidate=True, bestof_source='REUSED_B' if q['noise_index']<2 else 'NEW_B',
                 generator_sha256=GENERATOR_SHA)
    immutable(OUT/'data/selector_spec.json', SPEC)
    immutable(MANIFEST, dict(version=VERSION,status='PREPARED',source_manifest=str(source_manifest),source_manifest_sha256=sha(source_manifest),
        generator=dict(path=str(GENERATOR),sha256=GENERATOR_SHA,arm='NATURAL_HIGH',local_step=1500,global_step=22500,EMA=False),
        conditions=23,parents=len({q['parent_id'] for q in qs}),candidates=qs,metric_fields=source['metric_fields'],
        budget=dict(new_RAW=46,reused_RAW=46,total_RAW=92,generator_forwards=2300,new_D0_max=23,
                    new_RAW_scores=46,reused_RAW_scores=46,post_winners=23,new_POST_scores_max=46),
        comparisons=['SELECTED_vs_fixed_n0','SELECTED_vs_fixed_n1'],
        selector_spec_sha256=sha(OUT/'data/selector_spec.json'),new_original_queries=0,new_selector_boundary_queries='8 deterministic midpoints per C source open edge',
        unique_source_reconstruction_success=False,model_weights_improved=False))
    outputs = [MANIFEST, OUT/'data/selector_spec.json', OUT/'data/seed_registration.json']
    outputs.extend(resolve(q['noise_npz']) for q in qs if q['noise_index']>=2)
    result = dict(status='PASS',cached_B_RAW=46,cached_B_POST=46,cached_B_score_rows=138,
        original_natural_B=38,wide_new_parent_B=8,conditions=23,new_noise_arrays=46,cache=cache,
        model_calls=0,scoring_calls=0,OT_calls=0,protected=protected,files={str(p):sha(p) for p in outputs})
    immutable(done,result); return result


def cpu_gate():
    from .tests import run_tests
    prepare(); tests = run_tests(); rows=[]
    qs = candidates()
    for q in qs:
        inp, z = arrays(q['inference_npz']), arrays(q['noise_npz'])['z']
        if set(inp) != {'C_vertices','C_faces_original','C_faces_model','N','K'}:
            raise ValueError('Unexpected model input fields')
        if not 128 <= q['N'] <= 799 or not 2 <= q['K'] < q['N']:
            raise ValueError('N/K contract')
        if z.shape != (q['N']-q['K'],3,3) or z.dtype != np.float32 or not np.isfinite(z).all():
            raise ValueError('Noise shape/dtype')
        query = arrays(q['query_descriptor']['path'])
        if len(query['C_points']) != 2048 or not np.isclose(query['C_weights'].sum(),1):
            raise ValueError('Original fixed-query contract')
        from .selector import boundary_samples
        points, weights, edges = boundary_samples(inp['C_vertices'],inp['C_faces_original'])
        if len(edges) != q['boundary_edges'] or bool(len(edges)==0) != bool(q['closed_C']):
            raise ValueError('C-only boundary identity differs')
        rows.append(dict(request_id=q['request_id'],N=q['N'],K=q['K'],boundary_edges=len(edges),closed_C=q['closed_C'],
            old_cache=q['noise_index']<2,noise_sha256=q['noise_sha256'],model_calls=0))
    for cid in {q['condition_id'] for q in qs}:
        part=[q for q in qs if q['condition_id']==cid]
        if sorted(q['noise_index'] for q in part)!=[0,1,2,3] or len({q['noise_sha256'] for q in part})!=4:
            raise ValueError('Candidate cardinality/noise collision')
        for field in ('N','K','inference_npz_sha256','metadata_sha256','source_npz_sha256','query_descriptor','gallery_bounds','gallery_depth_bounds'):
            if any(q[field]!=part[0][field] for q in part):raise ValueError('Within-C fixed metadata changed '+field)
    if 'torch' in sys.modules:raise RuntimeError('CPU preparation imported torch')
    result=dict(status='PASS',tests=tests,rows=rows,conditions=23,candidates=92,new_noise=46,model_calls=0,scoring_calls=0,OT_calls=0,
        files={str(p):sha(p) for p in (MANIFEST,OUT/'data/selector_spec.json',OUT/'audits/cache_audit.json')})
    immutable(OUT/'audits/cpu_gate.json',result);return result


def freeze():
    if (OUT/'registration.json').exists():
        r=read(OUT/'registration.json');verify(r['code']);verify(r['inputs']);return r
    gate=read(OUT/'audits/cpu_gate.json');verify(gate['files'])
    if gate['status']!='PASS':raise ValueError('CPU gate not passed')
    from . import run as runner
    from r2.pilots.bench_group_weight_match_v1 import evaluation as prior_eval
    fn=runner.sampler_function();ns=fn.__globals__
    if (ns is vars(prior_eval) or ns['OUT']!=OUT or len(ns['requests']('B4'))!=46
            or ns['save_layout'].__globals__ is not ns or ns['label_spec']('B4')['arm']!='NATURAL_HIGH'
            or prior_eval.LABELS!=('B','W','M') or prior_eval.OUT==OUT):
        raise ValueError('Isolated inherited sampler namespace differs')
    if 'torch' in sys.modules:raise RuntimeError('Freeze imported model framework')
    immutable(OUT/'audits/runtime_binding_gate.json',dict(status='PASS',new_sample_requests=46,
        old_modules_unmodified=True,independent_function_globals=True,relative_runtime_package=ns['__package__'],
        generator='NATURAL_HIGH',model_calls=0,scoring_calls=0))
    from r2.pilots.bench_group_weight_match_v1 import evaluation as E
    from r2.pilots.chair_back_transfer_v1.evaluate import source_hashes
    from r2.tools.cref_layer_audit_v1 import geometry
    code={str(p):sha(p) for p in CODE.glob('*.py')}
    code.update(source_hashes());code[str(Path(E.__file__))]=sha(E.__file__)
    code[str(Path(geometry.__file__))]=sha(geometry.__file__)
    for kind in ('RAW','D0'):
        inherited=read(SOURCE/'evaluation/inference_registrations'/('B__'+kind+'.json'))['code']
        verify(inherited);code.update(inherited)
    protocol=OUT.parent/'PROTOCOL.md'
    inputs={str(p):sha(p) for p in (MANIFEST,OUT/'data/selector_spec.json',OUT/'data/seed_registration.json',OUT/'audits/cpu_gate.json',OUT/'audits/runtime_binding_gate.json',protocol)}
    result=dict(status='FROZEN',version=VERSION,UTC=utc(),generator_sha256=GENERATOR_SHA,code=code,inputs=inputs,
        budget=read(MANIFEST)['budget'],selector=SPEC,all92_RAW_before_D0=True,root_dispatch_only=True,
        diversity_definition='All six within-C candidate pairs: area-weighted symmetric nearest free-face-centroid distance / C-only bbox diagonal; descriptive, not TMD or semantic diversity.',
        retries=0,parameter_selection='Frozen before all new RAW and before selection',scientific_outputs_mutable=False)
    immutable(OUT/'registration.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','cpu-gate','freeze']);a=p.parse_args()
    r={'prepare':prepare,'cpu-gate':cpu_gate,'freeze':freeze}[a.action]()
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','protected','cache','rows','code','inputs')},indent=2))


if __name__=='__main__':main()
