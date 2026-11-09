"""Fixed 46-request evaluation, isolated from the read-only EXPOSURE baseline.

Existing production implementations are rebound into a PRIVATE function-global
dictionary. No imported module globals or original result paths are modified.
Only explicit sample/d0 commands perform model forwards.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from functools import lru_cache
import inspect
import json
from pathlib import Path
import sys
import types
from . import common as C

ROOT, OUT, OLD = C.ROOT, C.OUT, C.OLD
MANIFEST = OUT/'data/eval_manifest.json'
LABELS = ('EXPOSURE', 'PLAIN1500', 'MULTILOD1500')
NEW_LABELS = C.ARMS
STAGES = ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')
VERSION = 'bench_multilod_training_fixed46_evaluation_v1'


def budget():
    return dict(requests=46,parents=23,new_RAW=92,new_D0=92,
                generator_forwards=4600,new_score_rows=276,reused_score_rows=138,total_score_rows=414)


def root_for(label,q=None):
    if label not in LABELS: raise ValueError(label)
    return OLD if label=='EXPOSURE' else OUT


def requests(label=None):
    if label is not None and label not in LABELS: raise ValueError(label)
    qs=C.read(MANIFEST)['requests']
    if len(qs)!=46 or len({q['request_id'] for q in qs})!=46 or len({q['parent_id'] for q in qs})!=23:
        raise ValueError('Frozen 23-parent/46-request panel changed')
    return qs


def new_requests(label):
    return requests(label) if label in NEW_LABELS else []


def label_spec(label):
    if label not in NEW_LABELS: raise ValueError('No baseline sampling authorized')
    return dict(mode='PLAIN',arm=label,local_step=1500,global_step=27000,subset='full',
        schema='bench_multilod_generator_v1',generator_path=str(OUT/'training'/label/'generator_step1500.pt'))


def _code_hashes(code,kind):
    # Authenticate inherited production implementations/configs, never rewrite them.
    inherited=C.read(OLD/'evaluation/inference_registrations'/('EXPOSURE__'+kind+'.json'))['code']
    C.verify(inherited)
    own={str(p):C.sha(p) for p in Path(__file__).parent.glob('*.py')}
    return {**inherited,**own,**code}


def all_raw_barrier():
    panel=C.read(OUT/'audits/reference_cache_audit.json')
    C.verify(panel['files'])
    paths=[]
    for label in NEW_LABELS:
        engine().verify_raw(label)
        paths.extend(engine().raw_receipt(q,label) for q in requests())
    if len(paths)!=92: raise ValueError('All 92 new RAWs must precede any D0')
    return paths


def _cpu_worker(job):
    """Spawn-pickleable new-module entry; cloned functions never leave process."""
    label,q,mode,regsha=job
    e=engine()
    result=e.raw_one(label,q,regsha) if mode=='raw' else e.post_one(label,q,regsha)
    if 'torch' in sys.modules:raise RuntimeError('CPU scorer imported torch')
    print('EVALUATED',mode,label,q['request_id'],flush=True)
    return result


def cpu_dispatch_identity(label):
    """Zero-work spawn/path contract probe; no scorer, mesh or model calls."""
    e=engine()
    from r2.pilots.bench_four_routes_v1.exposure import evaluation as original
    if label not in NEW_LABELS:raise ValueError(label)
    return dict(label=label,output_root=str(e.root_for(label)),original_root=str(original.OUT),
                worker_module=e.cpu_one.__module__,new_root_preserved=e.OUT==OUT,
                original_root_unchanged=original.OUT==OLD,model_calls=0,scoring_calls=0)


@lru_cache(maxsize=1)
def engine():
    """Reuse audited logic with an isolated namespace, never monkeypatch OLD."""
    from r2.pilots.bench_four_routes_v1.exposure import evaluation as original
    original_out, original_manifest, original_labels = original.OUT, original.MANIFEST, original.LABELS
    ns=dict(vars(original))
    for name, obj in vars(original).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(original):
            clone=types.FunctionType(obj.__code__,ns,obj.__name__,obj.__defaults__,obj.__closure__)
            clone.__kwdefaults__=obj.__kwdefaults__
            ns[name]=clone
    ns.update(OUT=OUT,OLD_M=OLD,MANIFEST=MANIFEST,LABELS=LABELS,NEW_LABELS=NEW_LABELS,
        ARMS={arm:arm for arm in NEW_LABELS},VERSION=VERSION,__file__=__file__,
        requests=requests,new_requests=new_requests,root_for=root_for,budget=budget,
        prepare_panel=prepare_panel,prepare_manifest=prepare_panel,
        registration_path=lambda:C.REGISTRATION,label_spec=label_spec,
        reuse_spec=lambda q,label:dict(root=str(OLD),label=label) if label=='EXPOSURE' else None,
        inference_code_hashes=_code_hashes,all_raw_barrier=all_raw_barrier,cpu_one=_cpu_worker)
    result=types.SimpleNamespace(**ns)
    if (original.OUT,original.MANIFEST,original.LABELS)!=(original_out,original_manifest,original_labels):
        raise AssertionError('Original globals changed')
    return result


def prepare_panel():
    """Freeze original inputs and authenticate cached baseline; no rescore/model."""
    done=OUT/'audits/reference_cache_audit.json'
    if done.exists():
        r=C.read(done);C.verify(r['files']);C.verify(r['protected']);return r
    src=OLD/'data/eval_manifest.json'
    raw=C.resolve(src).read_bytes()
    if MANIFEST.exists():
        if MANIFEST.read_bytes()!=raw:raise ValueError('Fixed evaluation manifest changed')
    else:
        MANIFEST.parent.mkdir(parents=True,exist_ok=True)
        with MANIFEST.open('xb') as f:f.write(raw)
    manifest=C.read(MANIFEST);qs=requests();fields=manifest['metric_fields']
    if len(fields)!=177 or len(set(fields))!=177:raise ValueError('Original 177 metrics changed')
    C.verify(manifest['protected'])
    protected={**manifest['protected'],str(src):C.sha(src),str(OLD/'results/completion.json'):C.sha(OLD/'results/completion.json')}
    original_rows={(r['stage'],r['request_id']):r for r in C.read(OLD/'results/metric_rows.json') if r['label']=='EXPOSURE'}
    if len(original_rows)!=138:raise ValueError('Baseline cached rows incomplete')
    protected[str(OLD/'results/metric_rows.json')]=C.sha(OLD/'results/metric_rows.json')
    e=engine();cells=[]
    for q in qs:
        if 'gallery_bounds' not in q or 'gallery_depth_bounds' not in q:raise ValueError('Original display contract missing')
        for key,hkey in (('inference_npz','inference_npz_sha256'),('metadata_npz','metadata_sha256'),('source_npz','source_npz_sha256'),('noise_npz','noise_sha256')):
            p=C.resolve(q[key]);h=C.sha(p)
            if h!=q[hkey]:raise ValueError('Fixed input hash changed '+key)
            protected[str(p)]=h
        query=C.resolve(q['query_descriptor']['path'])
        if C.sha(query)!=q['query_descriptor']['sha256']:raise ValueError('Fixed query changed')
        protected[str(query)]=C.sha(query)
        for kind in ('raw','post'):
            folder=e.producer('EXPOSURE',q,kind)
            r=C.read(folder/'receipt.json');protected.update(r['files'])
            protected[str(folder/'receipt.json')]=C.sha(folder/'receipt.json')
        for stage in STAGES:
            row,score,folder=e.authenticate_score('EXPOSURE',q,stage)
            if any(row[m]!=original_rows[stage,q['request_id']][m] for m in fields):
                raise ValueError('Cached baseline numeric value changed')
            cells.append(dict(label='EXPOSURE',request_id=q['request_id'],stage=stage,source=str(folder)))
            for name in ('row.json','score.json','complete.json'):
                p=folder/name;protected[str(p)]=C.sha(p)
        p=OLD/'cpu/EXPOSURE'/q['request_id']/'complete.json'
        rr=C.read(p);C.verify(rr['files']);protected.update(rr['files']);protected[str(p)]=C.sha(p)
    # These registrations already contain the complete original scorer/runtime
    # sources and configs. Read/hash checks perform no scoring or model work.
    oldscore=C.read(OLD/'evaluation/score_registration.json')
    C.verify(oldscore['source_hashes']);protected.update(oldscore['source_hashes'])
    for kind in ('RAW','D0'):
        p=OLD/'evaluation/inference_registrations'/('EXPOSURE__'+kind+'.json')
        r=C.read(p);C.verify(r['code']);protected.update(r['code']);protected[str(p)]=C.sha(p)
    plan=dict(status='FROZEN',schema=VERSION,labels=list(LABELS),new_labels=list(NEW_LABELS),
        stages=list(STAGES),budget=budget(),fixed_endpoint='global27000/local1500; no checkpoint selection',
        baseline='EXPOSURE cached RAW/BASE/FINAL, read-only, no scoring/model rerun',
        primary_contrast='MULTILOD1500-PLAIN1500',secondary_contrasts=['PLAIN1500-EXPOSURE','MULTILOD1500-EXPOSURE'],
        original_manifest=str(src),original_manifest_sha256=C.sha(src),original_manifest_bytes_preserved=True,
        new_noise_arrays=0,new_queries=0,new_camera_or_sections=0,
        raw_barrier='All 92 new RAW receipts before the first D0 forward',
        chain='Single official Bench D0 -> GEO_C/source winding -> HINGE_OPP015 -> ALPHA -> HOLE',
        gallery='138 pages: each request/stage, 3 model rows x (2 oblique views + original front/side sections)',
        rating='Actual AI_PROXY viewing recorded separately; no synthetic labels')
    C.write(OUT/'data/evaluation_plan.json',plan)
    result=dict(status='PASS',schema=VERSION,requests=46,parents=23,score_rows_reused=138,numeric_fields=177,
        source_score_rows=cells,protected=protected,model_calls=0,scoring_calls=0,new_noise=0,new_queries=0,
        code_sha256=C.sha(__file__),files={str(p):C.sha(p) for p in (MANIFEST,OUT/'data/evaluation_plan.json')})
    C.write(done,result)
    return result


def sample(label,estimated_peak_mib=3500):
    if label not in NEW_LABELS:raise ValueError(label)
    prepare_panel()
    return engine().sample(label,estimated_peak_mib)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','prepare-scores','cpu-contract','sample'))
    p.add_argument('--label',choices=NEW_LABELS)
    p.add_argument('--estimated-peak-mib',type=float,default=3500)
    a=p.parse_args()
    if a.action=='prepare':r=prepare_panel()
    elif a.action=='prepare-scores':r=engine().prepare_scores()
    elif a.action=='cpu-contract':r=engine().cpu_contract()
    else:
        if not a.label:p.error('--label required')
        r=sample(a.label,a.estimated_peak_mib)
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','protected','source_score_rows','source_hashes','rows','metric_fields')},ensure_ascii=False))


if __name__=='__main__':main()
