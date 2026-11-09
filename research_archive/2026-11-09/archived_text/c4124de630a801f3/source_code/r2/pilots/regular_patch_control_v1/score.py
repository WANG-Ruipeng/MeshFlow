"""CPU-only regular-patch scoring, with unchanged inherited numerical definitions.

64 new stage scores are computed; 32 historical scores require exact input,
query, geometry, model and definition provenance. No model framework is imported.
The old affine-query prose is corrected only in this independent adapter.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, concurrent.futures, csv, functools, json, multiprocessing
from pathlib import Path
import sys, time
import numpy as np
from . import common as C

VERSION='r2_regular_patch_original_metrics_adapter_v1'
LABELS=('SMALL_C200','CREF_C200')
EV=C.OUT/'evaluation'
PRIOR=C.ROOT/'experiments/r2_c_reference_frame_v1/evaluation'
WIDTH=C.ROOT/'r2/pilots/condition_width_response_v1/evaluate.py'
META=('request_id','parent_id','condition_id','panel','candidate','region_type','tier','N','K','ratio',
      'base_request_id','base_condition_id','variant','alpha','pair_id','split','condition_type')
QUERY_MEASURE='REGISTERED_CONDITION_ORIGINAL_IID_AREA_SAMPLES; frozen within each C; no affine transform or importance reweighting'
QUALITY_C_ORIENTATION='Registered canonical C for scalar geometry and frozen queries; same unordered geometry as final source-restored C'

def verify(files):
    for p,h in files.items():
        if C.sha(p)!=h:raise ValueError('Frozen file changed: '+str(p))

class _Binding:
    """Redirect output only, replacing two inherited textual descriptions."""
    def __getattr__(self,name):return getattr(C,name)
    def write(self,path,value,replace=False):
        if Path(path).name=='score.json' and isinstance(value,dict) and 'query_measure' in value:
            value=dict(value)
            value['inherited_adapter_metadata_original']={k:value[k] for k in ('query_measure','quality_C_orientation')}
            value['query_measure']=QUERY_MEASURE
            value['quality_C_orientation']=QUALITY_C_ORIENTATION
            value['metadata_correction']='Text only: this panel uses original per-C area queries, not affine variants. All inherited numerical functions unchanged.'
        return C.write(path,value,replace=replace)

@functools.lru_cache(None)
def cpu_scorer():
    C.package()
    W=C.module('r2.pilots.condition_width_response_v1._regular_patch_frozen_metrics',WIDTH)
    original=W.definition_hashes()
    W.E=_Binding();W.VERSION=VERSION;W.META=META
    W.definition_hashes=lambda:{**original,str(Path(__file__)):C.sha(__file__),str(Path(C.__file__)):C.sha(C.__file__)}
    if 'torch' in sys.modules:raise RuntimeError('CPU adapter imported model framework')
    return W

def definition_hashes():return cpu_scorer().definition_hashes()

def base_registration():
    r=C.load_registration()
    if r['status']!='FROZEN' or tuple(r['labels'])!=LABELS:raise ValueError('Wrong frozen experiment')
    qs=r['requests']
    if len(qs)!=24 or len({q['request_id'] for q in qs})!=24 or len({q['condition_id'] for q in qs})!=6:raise ValueError('Expected 24 requests / six C')
    if sum(q['action']=='NEW' for q in qs)!=16 or sum(q['action']=='CACHE' for q in qs)!=8:raise ValueError('NEW/CACHE budget changed')
    if len(r['queries'])!=6 or {x['condition_id'] for x in r['queries']}!={q['condition_id'] for q in qs}:raise ValueError('Query set changed')
    for q in qs:
        if not 2<=int(q['K'])<int(q['N']) or not 128<=int(q['N'])<=256:raise ValueError('C/N contract')
        for k in META:
            if k not in q:raise ValueError('Missing frozen metadata '+k)
        verify({q['input_path']:q['input_sha256'],q['condition_source']:q['condition_source_sha256']})
    verify(r['code_hashes'])
    for query in r['queries']:verify({query['path']:query['sha256']})
    if 'manifest_sha256' in r:verify({C.OUT/'inputs/manifest.json':r['manifest_sha256']})
    if 'scoring_definition_hashes' in r and r['scoring_definition_hashes']!=definition_hashes():raise ValueError('Scoring definitions changed')
    return r


def freeze():
    r=base_registration();p=EV/'scoring_registration.json'
    old=PRIOR/'score_completion.json'
    value=dict(status='FROZEN',version=VERSION,experiment_registration_sha256=C.sha(C.OUT/'registration.json'),
        definition_hashes=definition_hashes(),original_score_completion_path=str(old),original_score_completion_sha256=C.sha(old),
        labels=LABELS,new_stages=64,historical_stages=32,total_stages=96,query_measure=QUERY_MEASURE,
        no_model_framework=True,no_new_queries=True,original_scores_readonly=True)
    C.write(p,value);return dict(status='FROZEN',path=str(p),sha256=C.sha(p))

def registration():
    r=base_registration();p=C.read(EV/'scoring_registration.json')
    if p['status']!='FROZEN' or p['definition_hashes']!=definition_hashes() or p['experiment_registration_sha256']!=C.sha(C.OUT/'registration.json'):
        raise ValueError('Scoring registration changed')
    verify({p['original_score_completion_path']:p['original_score_completion_sha256']})
    return r


def indices(r):
    expected={(arm,q['request_id']) for arm in LABELS for q in r['requests']};idx={}
    qs={q['request_id']:q for q in r['requests']}
    for stage in ('raw','post'):
        x=C.read(EV/(stage+'_index.json'));rows=x['rows']
        if x['status']!='COMPLETE' or len(rows)!=48 or {(v['label'],v['request_id']) for v in rows}!=expected:raise ValueError('All 48 RAW and POST must complete before scoring')
        idx[stage]={}
        for v in rows:
            q=qs[v['request_id']];verify({v[stage+'_path']:v[stage+'_sha256']})
            if v['input_sha256']!=q['input_sha256']:raise ValueError('Indexed input identity differs')
            if stage=='raw' and v['checkpoint_sha256']!=r['models'][v['label']]['sha256']:raise ValueError('Indexed model differs')
            if stage=='post':verify(v['files'])
            idx[stage][v['label'],v['request_id']]=v
    return idx

def score_one(job):
    q,raw,post,query,stage,rh=job
    query=dict(query,transform_audit=dict(method='REGISTERED_FULL_C_ORIGINAL_QUERY_NO_AFFINE',new_queries=0,
        query_identity_sha256=query['sha256'],retained_sampling_definition='original IID finite-triangle area sampling'))
    value=cpu_scorer().score_one((q,raw,post,query,stage,rh))
    if 'torch' in sys.modules:raise RuntimeError('No model framework permitted in CPU score')
    value.update(label=raw['label'],request_id=q['request_id'],stage=stage,cache_action='NEW_REGISTERED_SCORE',
                 new_quality_scans_this_call=0 if value['reused'] else 1)
    return value

def cached_scores(r,idx):
    completion_path=PRIOR/'score_completion.json';completion=C.read(completion_path)
    if completion['status']!='COMPLETE' or len(completion['receipts'])!=384:raise ValueError('Original CREF full score set unavailable')
    original={}
    # The completion receipt is frozen by the new protocol; every reused original
    # score is then individually checked against this immutable parent receipt.
    for ref in completion['receipts']:
        s=C.read(ref['path']);key=(s['label'],s['request_id'],s['stage'])
        if key in original:raise ValueError('Duplicate old score identity')
        original[key]=(ref,s)
    queries={x['condition_id']:x for x in r['queries']};receipts=[]
    for arm in LABELS:
        for q in r['requests']:
            if q['action']!='CACHE':continue
            cache=q['cache'][arm]
            verify({cache[s+'_path']:cache[s+'_sha256'] for s in ('raw','post')})
            oldraw=C.read(cache['raw_path']);oldpost=C.read(cache['post_path']);rid=oldraw['request_id']
            if oldpost['request_id']!=rid or oldraw['label']!=arm or oldpost['label']!=arm:raise ValueError('Old RAW/POST/model identity')
            for k in ('input_sha256','C_sha256','Gaussian_sha256','condition_source_sha256','N','K','noise_seed'):
                if oldraw[k]!=q[k]:raise ValueError('Cached input differs: '+k)
            if oldraw['checkpoint_sha256']!=r['models'][arm]['sha256']:raise ValueError('Wrong cached weights')
            query=queries[q['condition_id']]
            for stage in ('RAW','POST'):
                ref,s=original[arm,rid,stage];verify({ref['path']:ref['sha256']});verify(s['files']);identity=s['identity']
                if s['status']!='PASS' or s['stage']!=stage or s['label']!=arm:raise ValueError('Old score not PASS')
                verify(identity['definition_hashes'])
                for k in ('input_sha256','C_sha256','Gaussian_sha256','condition_source_sha256'):
                    if identity[k]!=q[k]:raise ValueError('Cached score identity differs: '+k)
                if identity['query_sha256']!=query['sha256'] or query['identity']['C_sha256']!=q['C_sha256']:raise ValueError('Cached query differs')
                current=idx[stage.lower()][arm,q['request_id']];path=current[stage.lower()+'_path'];digest=current[stage.lower()+'_sha256']
                if identity['mesh_sha256']!=digest or C.resolve(identity['mesh_path'])!=C.resolve(path):raise ValueError('Cached score mesh differs')
                if current['historical_receipt_sha256']!=cache[stage.lower()+'_sha256']:raise ValueError('New cache receipt lineage differs')
                if int(s['N'])!=int(q['N']) or int(s['K'])!=int(q['K']) or s['C_query_count']!=2048:raise ValueError('Old denominators differ')
                # Shared numerical implementation files must be identical. Adapter
                # metadata/version differences alone are intentionally retained.
                common=set(identity['definition_hashes']) & set(definition_hashes())
                if any(identity['definition_hashes'][p]!=definition_hashes()[p] for p in common):raise ValueError('Metric implementation changed')
                receipt_path=EV/'score_reuse'/arm/q['request_id']/stage/'receipt.json'
                receipt=dict(status='VERIFIED_REUSE',version=VERSION,label=arm,request_id=q['request_id'],historical_request_id=rid,stage=stage,
                    registration_sha256=C.sha(C.OUT/'registration.json'),old_completion_path=str(completion_path),old_completion_sha256=C.sha(completion_path),
                    score_path=ref['path'],score_sha256=ref['sha256'],original_score_identity=identity,
                    new_geometry_receipt_path=str(EV/stage.lower()/arm/q['request_id']/'receipt.json'),
                    new_geometry_receipt_sha256=C.sha(EV/stage.lower()/arm/q['request_id']/'receipt.json'),
                    original_query_measure=s['query_measure'],query_measure=QUERY_MEASURE,quality_C_orientation=QUALITY_C_ORIENTATION,
                    metadata_note='Corrected explanatory prose only; original cached score and numerical values remain unchanged.',
                    full_quality_scans=0,new_queries=0,model_calls=0,verified_common_definition_files=sorted(common))
                C.write(receipt_path,receipt)
                receipts.append(dict(path=ref['path'],sha256=ref['sha256'],label=arm,request_id=q['request_id'],historical_request_id=rid,
                    stage=stage,reused=True,cache_action='VERIFIED_HISTORICAL_SCORE_REUSE',reuse_receipt_path=str(receipt_path),
                    reuse_receipt_sha256=C.sha(receipt_path),seconds=0,new_quality_scans_this_call=0))
    if len(receipts)!=32:raise ValueError('Expected exactly 32 historical stage scores')
    return receipts

def validate_completion(p):
    if p['status']!='COMPLETE' or p['stages']!=96 or p['definition_hashes']!=definition_hashes():raise ValueError('Completion contract differs')
    for x in p['receipts']:
        verify({x['path']:x['sha256']});s=C.read(x['path']);verify(s['files'])
        if 'reuse_receipt_path' in x:verify({x['reuse_receipt_path']:x['reuse_receipt_sha256']})

def score(workers=2):
    r=registration();idx=indices(r);done=EV/'score_completion.json'
    if done.exists():p=C.read(done);validate_completion(p);return p
    tick=time.perf_counter();C.resources(EV/('score_resources_before_'+str(time.time_ns())+'.json'))
    cached=cached_scores(r,idx)
    query={x['condition_id']:x for x in r['queries']};rh=C.sha(C.OUT/'registration.json')
    jobs=[(q,idx['raw'][arm,q['request_id']],idx['post'][arm,q['request_id']],query[q['condition_id']],stage,rh)
          for arm in LABELS for q in r['requests'] if q['action']=='NEW' for stage in ('RAW','POST')]
    if len(jobs)!=64:raise ValueError('New score budget changed')
    # Completed new-stage caches are checked first. One real pending score then
    # supplies the per-process RSS measurement; there is no extra benchmark.
    completed={};pending=[]
    for i,j in enumerate(jobs):
        p=EV/'scores'/j[1]['label']/j[0]['request_id']/j[4]/'score.json'
        if p.exists():completed[i]=score_one(j)
        else:pending.append((i,j))
    actual=0;first=None
    if pending:
        i,j=pending.pop(0);first=score_one(j);completed[i]=first;snapshot=C.resources()
        mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (snapshot.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
        parallel=(os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024) and (snapshot.get('loadavg') or [999])[0]<(os.cpu_count() or 1)-2
        actual=2 if workers==2 and parallel else 1
        C.write(EV/('score_dispatch_'+str(time.time_ns())+'.json'),dict(workers=actual,threads_per_worker=2,first_authorized_score=first,resources=snapshot,reason='First actual pending task RSS, CPU and RAM guards; no model or benchmark'))
        def progress():C.write(EV/'score_progress.json',dict(status='RUNNING',new_registered_complete=len(completed),new_registered=64,cached=32,workers=actual),replace=True)
        progress()
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(score_one,j):i for i,j in pending}
            try:
                for f in concurrent.futures.as_completed(futures):
                    completed[futures[f]]=f.result();progress()
                    print(json.dumps(dict(stage='SCORE',new_registered_complete=len(completed),new_registered=64,cached=32)),flush=True)
            except BaseException:
                for f in futures:f.cancel()
                raise
    receipts=[completed[i] for i in sorted(completed)]
    for x in receipts:x['sha256']=C.sha(x['path'])
    receipts+=cached
    if len(receipts)!=96 or len({(x['label'],x['request_id'],x['stage']) for x in receipts})!=96:raise ValueError('Logical score set differs')
    result=dict(status='COMPLETE',version=VERSION,stages=96,new_stages=64,reused_historical_stages=32,
        new_quality_scans_this_invocation=sum(x['new_quality_scans_this_call'] for x in receipts),receipts=receipts,
        workers=actual,seconds=time.perf_counter()-tick,definition_hashes=definition_hashes(),query_measure=QUERY_MEASURE,
        scientific_definition='Original width diagnostic score_one and all numerical scientific functions unchanged; independent metadata adapter only.',
        new_queries=0,model_calls=0,training=0)
    C.write(done,result);C.write(EV/'score_progress.json',dict(status='COMPLETE',new_registered_complete=64,new_registered=64,cached=32,workers=actual),replace=True)
    if 'torch' in sys.modules:raise RuntimeError('CPU score imported model framework')
    return result

def aggregate():
    r=registration();done=C.read(EV/'score_completion.json');validate_completion(done)
    qs={q['request_id']:q for q in r['requests']};outputs=[];boundary=[]
    for ref in done['receipts']:
        s=C.read(ref['path']);q=qs[ref['request_id']];meta={k:q[k] for k in META};meta.update(label=ref['label'],stage=ref['stage'],input_action=q['action'])
        vals={**s['metrics'],**{k:v for k,v in s['topology'].items() if isinstance(v,(int,float,bool)) or v is None},**s['distance']}
        for b in s['boundary']:
            boundary.append(dict(meta,**b));vals['A_VMR__'+b['setting']]=b['VMR'];vals['PMR__'+b['setting']]=b['perfect']
        vals.update({'DESC__'+k:v for k,v in s['descriptive_boundary_distance']['metrics'].items()})
        vals.update(C_preserved_exact=s['C_preserved_exact'],I_status=s['status_by_metric']['I'],topology_status=s['topology'].get('status'),
            free_face_denominator=s['free_face_denominator'],CF_tested_pair_denominator=s['CF_tested_pair_denominator'],FF_tested_pair_denominator=s['FF_tested_pair_denominator'],
            C_query_weight_denominator=s['C_query_weight_denominator'],boundary_vertex_denominator=s['boundary'][0]['boundary_vertex_count'],
            source_C_connectivity_status=s['source_C_connectivity_status'],score_path=ref['path'],score_sha256=ref['sha256'],
            historical_request_id=ref.get('historical_request_id'),score_origin=ref['cache_action'],
            reuse_receipt_path=ref.get('reuse_receipt_path'),reuse_receipt_sha256=ref.get('reuse_receipt_sha256'),
            query_measure=QUERY_MEASURE,original_score_query_measure=s['query_measure'])
        outputs.append(dict(meta,**vals))
    if len(outputs)!=96:raise ValueError('Exactly 96 output rows required')
    W=cpu_scorer();lookup={(x['label'],x['request_id'],x['stage']):x for x in outputs};paired=[]
    metrics=sorted(set(W.ORIGINAL_SCALARS)|{k for x in outputs for k in x if k.startswith('DESC__')})
    for q in r['requests']:
        for stage in ('RAW','POST'):
            a=lookup[LABELS[0],q['request_id'],stage];b=lookup[LABELS[1],q['request_id'],stage]
            for metric in metrics:
                av,bv=a.get(metric),b.get(metric);valid=isinstance(av,(int,float,bool)) and isinstance(bv,(int,float,bool)) and np.isfinite(av) and np.isfinite(bv)
                paired.append(dict(comparison='CREF_C200-minus-SMALL_C200',panel=q['panel'],input_action=q['action'],request_id=q['request_id'],parent_id=q['parent_id'],
                    condition_id=q['condition_id'],condition_type=q['condition_type'],candidate=q['candidate'],stage=stage,metric=metric,baseline_value=av,candidate_value=bv,
                    baseline_free_face_denominator=a['free_face_denominator'],candidate_free_face_denominator=b['free_face_denominator'],
                    boundary_vertex_denominator=a['boundary_vertex_denominator'],delta=float(bv)-float(av) if valid else None,status='PASS' if valid else 'NA'))
    parents=W.average(paired,['comparison','panel','input_action','parent_id','condition_id','condition_type','stage','metric'])
    summary=W.average(parents,['comparison','panel','input_action','stage','metric'])
    C.csvwrite(EV/'per_output.csv',outputs);C.csvwrite(EV/'AVMR_PMR_all8.csv',boundary)
    C.csvwrite(EV/'paired_per_output.csv',paired);C.csvwrite(EV/'paired_per_parent.csv',parents);C.csvwrite(EV/'paired_summary.csv',summary)
    result=dict(status='COMPLETE',outputs=96,new_quality_scans=64,historical_scores_reused=32,parents=6,
        query_measure=QUERY_MEASURE,full_condition_and_reference_panels_separate=True,model_calls=0,NA_policy='All inherited NA/status and denominators retained; no GT reconstruction ranking')
    C.write(EV/'aggregation.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('freeze','score','aggregate','check','definitions'));p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    if a.stage=='score':v=score(a.workers)
    elif a.stage=='aggregate':v=aggregate()
    elif a.stage=='definitions':v=definition_hashes()
    elif a.stage=='freeze':v=freeze()
    else:v=dict(status='PASS',requests=len(registration()['requests']),model_calls=0)
    print(json.dumps(C.clean({k:x for k,x in v.items() if k!='receipts'}),allow_nan=False),flush=True)

if __name__=='__main__':main()
