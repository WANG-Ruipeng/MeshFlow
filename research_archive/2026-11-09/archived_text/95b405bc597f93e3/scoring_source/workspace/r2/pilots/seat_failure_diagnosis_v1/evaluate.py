"""Frozen EXP67 seat diagnostic: inherit original CPU scores, isolate new outputs."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,collections,concurrent.futures,csv,json,multiprocessing,time
from fractions import Fraction
from pathlib import Path
import numpy as np
from . import common as E
from r2.pilots.condition_width_response_v1 import evaluate as W

VERSION='r2_seat_diagnosis_original_metrics_adapter_v1'
LABEL='EXP67_FROZEN'
META=(*W.META,'pair_id','split','condition_type')
INHERITED_HASHES=W.definition_hashes
W.E=E
W.VERSION=VERSION
W.META=META
LOG=E.OUT/'scoringlogs'
STATS=E.OUT/'statistics'

def hashes():
    return {**INHERITED_HASHES(),str(Path(__file__)):E.sha(__file__),
            str(E.CODE/'prepare.py'):E.sha(E.CODE/'prepare.py')}
W.definition_hashes=hashes

def registration():
    r=E.load_registration()
    assert len(r['requests'])==40 and len({q['request_id'] for q in r['requests']})==40
    assert len(r['queries'])==20 and len({q['condition_id'] for q in r['queries']})==20
    assert {q['variant'] for q in r['requests']}=={'base','sub4'}
    for q in r['requests']:
        for k in META:assert k in q,k
    sub=[q for q in r['requests'] if q['variant']=='sub4'];assert len(sub)==8
    base={q['base_request_id']:q for q in r['requests'] if q['variant']=='base'}
    for q in sub:
        b=base[q['base_request_id']]
        assert q['free_Gaussian_sha256']==b['free_Gaussian_sha256']
        assert q['N']-q['K']==b['N']-b['K'] and q['K']==4*b['K'] and q['N']==b['N']+3*b['K']
    return r

def freeze():
    r=registration();definition={
        'version':VERSION,'status':'FROZEN','registration_sha256':E.sha(E.OUT/'registration.json'),
        'score_hashes':hashes(),'score_one_inherited_unchanged_from':str(E.ROOT/'r2/pilots/condition_width_response_v1/evaluate.py'),
        'logical_scores':80,'raw':40,'post':40,'max_quality_scans':80,'GPU':0,'new_queries':0,
        'interpretation':'TRAIN minus DEV is an input-matched observational comparison; different parents/N/K/free counts; same seed free-prefix only, not point correspondence.',
        'subdivision':'sub4 minus base fixes C physical surface and complete free Gaussian; changes C tessellation, K and total N. Not isolated token multiplicity causality.',
        'AVMR':'All eight inherited thresholds and actual boundary denominators retained. Subdivision adds boundary vertices so VMR/PMR and uniform-vertex distances are not tessellation invariant.',
        'queries':'Physical C2048 points and weights are byte-identical for base/sub4; point ownership, normals and borders adapted by frozen preparation.',
        'metrics':'All 96 inherited scalar fields plus four existing finite-edge/surface descriptive distances. S/E/unique GT remain NA.',
        'aggregation':'Two noises averaged per condition before condition/pair averages. Separate TRAIN/DEV, type and RAW/POST. No score determines which images are reviewed.'}
    E.write(LOG/'definition.json',definition)
    return definition

def indices(r):
    expected={q['request_id'] for q in r['requests']};out=[]
    for name,field in [('raw_index.json','raw'),('post_index.json','post')]:
        p=E.OUT/'evaluation'/name
        if not p.exists():raise RuntimeError('BLOCKED: all RAW and POST must finish before scoring: '+str(p))
        z=E.read(p);rr=z['rows']
        assert z['status']=='COMPLETE' and len(rr)==40 and {q['request_id'] for q in rr}==expected
        for q in rr:
            assert q['label']==LABEL;W.verify({q[field+'_path']:q[field+'_sha256']})
        out.append({q['request_id']:q for q in rr})
    return out

def score_one(job):
    q,raw,post,query,stage,rh=job
    query=dict(query,transform_audit={'method':'FROZEN_SEAT_CONDITION_QUERY','variant':q['variant'],
        'physical_points_and_weights':'base/sub4 preserved bytewise by preparation','preparation_audit':str(E.OUT/'subdivision_audit.json')})
    result=W.score_one((q,raw,post,query,stage,rh))
    p=Path(result['path']);s=E.read(p)
    if not result['reused']:
        s['query_measure']='FROZEN_BASE_AREA_QUERIES; sub4 same physical points and weights, child-face ownership remapped'
        s['quality_C_orientation']='Canonical input C and registered queries; same unordered final source-oriented C surface'
        s['representation_caveat']='sub4 changes N/K and source boundary vertex denominator; TRAIN/DEV different parents and free counts'
        s['free_Gaussian_sha256']=q['free_Gaussian_sha256']
        s['noise_provenance']=q['noise_provenance']
        E.write(p,s,replace=True)
    else:
        assert s['free_Gaussian_sha256']==q['free_Gaussian_sha256']
    return result

def run(max_workers=2):
    r=registration();definition=freeze();raw,post=indices(r);rh=E.sha(E.OUT/'registration.json')
    final=LOG/'score_completion.json'
    if final.exists():
        z=E.read(final);assert z['status']=='COMPLETE' and z['stages']==80 and z['definition_hashes']==hashes()
        for a in z['receipts']:W.verify({a['path']:a['sha256']});W.verify(E.read(a['path'])['files'])
        return z
    queries={q['condition_id']:q for q in r['queries']}
    jobs=[(q,raw[q['request_id']],post[q['request_id']],queries[q['condition_id']],stage,rh) for q in r['requests'] for stage in ['RAW','POST']]
    assert len(jobs)==80
    resources=E.resources();E.write(LOG/('resources_before_'+str(time.time_ns())+'.json'),resources)
    start=time.perf_counter();first=score_one(jobs[0]);resources=E.resources()
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (resources.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    parallel=(os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024)
    parallel=parallel and (resources.get('loadavg') or [999])[0]<(os.cpu_count() or 1)-2
    workers=2 if max_workers==2 and parallel else 1
    E.write(LOG/('dispatch_'+str(time.time_ns())+'.json'),{'workers':workers,'threads_per_worker':2,'first_registered_task':first,'resources':resources,'reason':'Real first score RSS, RAM and CPU guard; no unbudgeted probe'})
    receipts=[first];results={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futs={pool.submit(score_one,j):i for i,j in enumerate(jobs[1:],1)}
        try:
            for fut in concurrent.futures.as_completed(futs):
                results[futs[fut]]=fut.result();receipts.append(results[futs[fut]])
                E.write(LOG/'progress.json',{'status':'RUNNING','completed':len(receipts),'expected':80},replace=True)
                print(json.dumps({'stage':'SCORE','completed':len(receipts),'expected':80}),flush=True)
        except BaseException:
            for f in futs:f.cancel()
            raise
    receipts=[first,*[results[i] for i in sorted(results)]]
    for a in receipts:a['sha256']=E.sha(a['path'])
    z={'status':'COMPLETE','stages':80,'receipts':receipts,'seconds':time.perf_counter()-start,'workers':workers,
       'definition_hashes':hashes(),'new_queries':0,'model_calls':0,'scientific_definition':'Inherited score_one and helper computations unchanged; metadata adapted only.'}
    E.write(final,z);E.write(LOG/'progress.json',{'status':'COMPLETE','completed':80,'expected':80},replace=True)
    return z

def avg(rows,keys):
    return W.average(rows,keys)

def aggregate():
    r=registration();done=E.read(LOG/'score_completion.json');assert done['status']=='COMPLETE' and done['stages']==80 and done['definition_hashes']==hashes()
    final=STATS/'completion.json'
    if final.exists():
        z=E.read(final);W.verify(z['files']);return z
    outputs=[];quality=[];topology=[];boundary=[];vertices=[];distances=[];names=set();exact={};qs={q['request_id']:q for q in r['requests']}
    for rec in done['receipts']:
        W.verify({rec['path']:rec['sha256']});s=E.read(rec['path']);W.verify(s['files'])
        meta={k:s[k] for k in (*META,'label','stage')};key=(s['request_id'],s['stage'])
        vals={**s['metrics'],**{k:v for k,v in s['topology'].items() if W.numeric(v) or v is None},**s['distance']}
        quality.append(dict(meta,**s['metrics'],free_face_denominator=s['free_face_denominator'],C_query_weight_denominator=s['C_query_weight_denominator'],I_status=s['status_by_metric']['I']))
        topology.append(dict(meta,**{k:v for k,v in s['topology'].items() if not isinstance(v,(dict,list))}))
        for b in s['boundary']:
            boundary.append(dict(meta,**b));vm='A_VMR__'+b['setting'];pm='PMR__'+b['setting'];vals[vm]=b['VMR'];vals[pm]=b['perfect']
            if b['status']=='PASS' and b['boundary_vertex_count']:
                exact[key,vm]=Fraction(b['matched_count'],b['boundary_vertex_count']);exact[key,pm]=Fraction(b['perfect'])
        names.update(k for k,v in vals.items() if W.numeric(v) or v is None);names.add('C_preserved_exact')
        desc=s['descriptive_boundary_distance'];distances.append(dict(meta,**desc['metrics'],status=desc['status'],boundary_vertex_denominator=desc['boundary_vertex_denominator']))
        vals.update({'DESC__'+k:v for k,v in desc['metrics'].items()})
        vals.update(C_preserved_exact=s['C_preserved_exact'],source_C_connectivity_status=s['source_C_connectivity_status'],I_status=s['status_by_metric']['I'],
                    free_face_denominator=s['free_face_denominator'],C_query_weight_denominator=s['C_query_weight_denominator'],C_query_count=s['C_query_count'],
                    boundary_vertex_denominator=s['boundary'][0]['boundary_vertex_count'],CF_tested_pair_denominator=s['CF_tested_pair_denominator'],FF_tested_pair_denominator=s['FF_tested_pair_denominator'],
                    free_Gaussian_sha256=s['free_Gaussian_sha256'],noise_provenance=s['noise_provenance'],score_path=rec['path'],score_sha256=rec['sha256'])
        outputs.append(dict(meta,**vals));vertices.extend(dict(meta,**x) for x in W.read_csv(Path(rec['path']).parent/'boundary_vertices.csv'))
    assert len(outputs)==80 and len({(x['request_id'],x['stage']) for x in outputs})==80
    assert names==set(W.ORIGINAL_SCALARS),{'missing':sorted(set(W.ORIGINAL_SCALARS)-names),'extra':sorted(names-set(W.ORIGINAL_SCALARS))}
    descnames=sorted(k for k in outputs[0] if k.startswith('DESC__'));allnames=sorted(names|set(descnames))
    def compare(base,cand,kind):
        out=[]
        for k in allnames:
            a,b=base.get(k),cand.get(k);valid=W.numeric(a) and W.numeric(b)
            keya=((base['request_id'],base['stage']),k);keyb=((cand['request_id'],cand['stage']),k)
            delta=exact[keyb]-exact[keya] if keya in exact and keyb in exact else None
            out.append({'comparison':kind,'stage':cand['stage'],'pair_id':cand['pair_id'],'candidate':cand['candidate'],'condition_type':cand['condition_type'],'split':cand['split'],
                        'baseline_request_id':base['request_id'],'candidate_request_id':cand['request_id'],'baseline_parent':base['parent_id'],'candidate_parent':cand['parent_id'],
                        'baseline_N':base['N'],'candidate_N':cand['N'],'baseline_K':base['K'],'candidate_K':cand['K'],
                        'baseline_boundary_vertices':base['boundary_vertex_denominator'],'candidate_boundary_vertices':cand['boundary_vertex_denominator'],
                        'metric':k,'metric_origin':'INHERITED_DESCRIPTIVE_DISTANCE' if k.startswith('DESC__') else 'INHERITED_ORIGINAL',
                        'baseline_value':a,'candidate_value':b,'delta':b-a if valid else None,'status':'PASS' if valid else 'NA_OR_MISSING',
                        'delta_fraction':str(delta) if delta is not None else None,'delta_sign_exact':int(delta>0)-int(delta<0) if delta is not None else None})
        return out
    baseby={(x['split'],x['pair_id'],x['candidate'],x['stage']):x for x in outputs if x['variant']=='base'}
    td=[];sub=[]
    for x in outputs:
        if x['variant']=='base' and x['split']=='TRAIN':td+=compare(baseby['DEV',x['pair_id'],x['candidate'],x['stage']],x,'TRAIN-minus-DEV')
        elif x['variant']=='sub4':
            b=baseby[x['split'],x['pair_id'],x['candidate'],x['stage']]
            assert b['free_Gaussian_sha256']==x['free_Gaussian_sha256'];sub+=compare(b,x,'sub4-minus-base')
    tp=avg(td,('comparison','stage','pair_id','condition_type','metric','metric_origin'))
    sp=avg(sub,('comparison','stage','pair_id','condition_type','split','metric','metric_origin'))
    tsum=avg(tp,('comparison','stage','condition_type','metric','metric_origin'))
    ssum=avg(sp,('comparison','stage','condition_type','split','metric','metric_origin'))
    absolute=[]
    for x in outputs:
        for k in allnames:
            val=x.get(k);absolute.append({'split':x['split'],'variant':x['variant'],'stage':x['stage'],'pair_id':x['pair_id'],'parent_id':x['parent_id'],'condition_type':x['condition_type'],'metric':k,
                'baseline_value':val,'candidate_value':val,'delta':0 if W.numeric(val) else None,'status':'PASS' if W.numeric(val) else 'NA_OR_MISSING'})
    ap=avg(absolute,('split','variant','stage','pair_id','parent_id','condition_type','metric'))
    asum=avg(ap,('split','variant','stage','condition_type','metric'))
    tables={'per_output':outputs,'quality_per_output':quality,'topology_per_output':topology,'AVMR_PMR_all8':boundary,'boundary_vertices':vertices,'descriptive_distances':distances,
            'TRAIN_DEV_per_pair_noise':td,'TRAIN_DEV_per_pair':tp,'TRAIN_DEV_by_type':tsum,'sub4_base_per_noise':sub,'sub4_base_per_condition':sp,'sub4_base_by_type_split':ssum,
            'absolute_per_condition':ap,'absolute_by_type_split':asum}
    files={}
    for name,rr in tables.items():
        p=STATS/(name+'.csv');E.csvwrite(p,rr);files[str(p)]=E.sha(p)
    result={'status':'COMPLETE','stage_rows':80,'inherited_metrics':len(names),'descriptive_metrics':len(descnames),'TRAIN_DEV_noise_metric_rows':len(td),'sub4_base_noise_metric_rows':len(sub),
            'metric_names':allnames,'definition_hashes':hashes(),'files':files,'model_calls':0,'geometry_rescored':0,'visual_review':'Separate actual image review; no automated semantic inference',
            'caveats':['TRAIN/DEV is observational across distinct parents; same noise prefix does not create geometry correspondence.','sub4 changes N/K and source boundary vertex denominator; surface queries stay fixed.','Raw numeric deltas are not all higher-is-better; retain contradictory costs.']}
    E.write(final,result);return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['freeze','score','aggregate'],required=True);a.add_argument('--max-workers',type=int,choices=[1,2],default=2);args=a.parse_args()
    result=freeze() if args.stage=='freeze' else run(args.max_workers) if args.stage=='score' else aggregate()
    print(json.dumps({k:v for k,v in result.items() if k not in ('receipts','definition_hashes','score_hashes','files','metric_names')},ensure_ascii=False),flush=True)