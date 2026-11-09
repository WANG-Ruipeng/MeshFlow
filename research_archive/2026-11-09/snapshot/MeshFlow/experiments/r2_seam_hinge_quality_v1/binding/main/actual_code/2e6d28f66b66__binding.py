"""Unchanged corrected actual-cluster decoder, only 1.5%L per new output.

Whole PRE_GEO member displacement, immutable C, geometric vetoes and source
ledgers are inherited. Tip point aliases never count as a joined surface.
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
from r2.pilots.owner_layout_v1 import binding_adapter as A
from . import scoring as S
from .evaluation import C, ROOT, OUT, WAVES, private_functions, requests, dispatch_cpu, source_paths

LEGACY, E, Q, PE = A.B, A.E, A.Q, A.PE
VERSION = 'seam_hinge_quality_corrected_actual_cluster_bound015_v1'
_cores = {}


def folder(wave):
    if wave not in WAVES:
        raise ValueError('Unknown wave')
    return OUT/'binding'/wave


def core(wave):
    if wave not in _cores:
        if C.sha(LEGACY.__file__) != A.LEGACY_SHA256:
            raise ValueError('Frozen binding source changed')
        ns = private_functions(LEGACY, BASE=OUT, OUT=folder(wave), ARMS=WAVES[wave],
                               DELTAS=(.015,), VERSION=VERSION)
        ns['state_merge'] = A.state_merge
        # score_output uses explicit seam applicability=True and the original
        # seam_to_bulk function; no label membership test is lost in this path.
        _cores[wave] = ns
    return _cores[wave]


def freeze(wave):
    path = folder(wave)/'registration.json'; api=core(wave)
    if path.exists():
        reg=C.read(path); api['verify'](reg['protected']); api['verify'](reg['code']); return reg
    complete=C.read(S.files(wave)['summary_completion'])
    if complete['rows'] != len(WAVES[wave])*56 or complete['numeric_field_count'] != 124:
        raise ValueError('Binding requires all original wave RAW/POST scores')
    protected={str(p):C.sha(p) for p in (*S.files(wave).values(),
               OUT/'evaluation'/(wave+'_per_output.csv'))}
    qq=requests()
    for q in qq:
        for key in ('inference_npz','metadata_npz'):
            protected[q[key]]=C.sha(q[key])
        desc=q['query_descriptor']
        if C.sha(desc['path']) != desc['sha256']:
            raise ValueError('Fixed query identity changed')
        protected[desc['path']]=desc['sha256']
        for arm in WAVES[wave]:
            base=OUT/'evaluation'/arm/q['request_id']
            for rel in ('post/post.npz','post/pre_geo.npz','post/receipt.json'):
                p=base/rel; protected[str(p)]=C.sha(p)
            A.validate_arrays(C.arrays(base/'post/post.npz'),C.arrays(base/'post/pre_geo.npz')['output'],q)
            for name in ('score.json','pairs.json'):
                p=OUT/'evaluation/scores'/arm/q['request_id']/'POST'/name; protected[str(p)]=C.sha(p)
    code={**E.source_hashes(), **Q.definition_hashes()}
    for p in (*source_paths(),Path(__file__),Path(A.__file__),Path(LEGACY.__file__)):
        code[str(p)]=C.sha(p)
    for name in ('geo_c','winding_solver'):
        p=Path(PE.cpu_module(name).__file__); code[str(p)]=C.sha(p)
    result=dict(status='FROZEN',wave=wave,version=VERSION,arms=list(WAVES[wave]),deltas=[.015],
        cells=28*len(WAVES[wave]),requests_manifest=qq,L=C.L,protected=protected,code=code,
        algorithm='Original TIP_CLUSTER + LATERAL_EDGE deterministic queue; corrected current-root C guard; no new algorithm',
        displacement='Every affected original actual GEO_C cluster member measured from PRE_GEO <=1.5%L',
        guards='Exact C, identity, collapse, shape/normal worsening, new illegal CF, old legal CF, newly nonmanifold and orientation guards inherited',
        FF_policy='New/resolved illegal FF and lost legal FF measured, not vetoed',
        model_calls=0,new_queries=0,new_noise=0)
    C.write(path,result)
    for p,h in code.items():
        dest=folder(wave)/'actual_code'/(h[:12]+'__'+Path(p).name)
        dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(C.resolve(p).read_bytes())
    return result


def worker(job):
    wave,arm,q=job
    if arm not in WAVES[wave]:
        raise ValueError('Unregistered binding arm')
    return core(wave)['worker']((arm,q))


def aggregate(wave):
    reg=C.read(folder(wave)/'registration.json')
    numeric=C.read(S.files(wave)['summary_completion'])['numeric_fields']
    rows,paired,receipts,ledger,funnels=[],[],[],[],[]
    for arm in WAVES[wave]:
        label=core(wave)['kind_label'](arm,.015)
        for q in reg['requests_manifest']:
            dest=folder(wave)/'cells'/label/q['request_id']; path=dest/'complete.json'
            r=C.read(path); E.verify(r['files']); row=r['row']; rows.append(row)
            receipts.append(dict(path=str(path),sha256=C.sha(path)))
            original=OUT/'evaluation/scores'/arm/q['request_id']/'POST'
            score=C.read(original/'score.json')
            baseline,_,_=S.metric_row(score,C.arrays(score['identity']['mesh_path']),C.read(original/'pairs.json'))
            for metric in numeric:
                x,y=baseline.get(metric),row.get(metric)
                good=isinstance(x,(int,float,bool)) and isinstance(y,(int,float,bool))
                paired.append(dict(label=label,base_arm=arm,request_id=q['request_id'],parent_id=q['parent_id'],
                    split=q['split'],noise_index=q['noise_index'],metric=metric,baseline=x,candidate=y,
                    delta=float(y)-float(x) if good else None,status='PASS' if good else 'NA'))
            summary=C.read(dest/'paired_ledger.json')['summary']
            ledger.append(dict(label=arm,request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],
                noise_index=q['noise_index'],accepted_tip=row['accepted_tip'],accepted_edge=row['accepted_edge'],
                max_PRE_GEO_motion=row['max_PRE_GEO_motion'],**summary))
        for split in ('TRAIN','DEV','ALL'):
            rr=[r for r in rows if r['base_arm']==arm and (split=='ALL' or r['split']==split)]
            counts=collections.Counter()
            for r in rr:
                info=C.read(folder(wave)/'cells'/label/r['request_id']/'binding.json')
                counts.update(info['rejections']); counts.update({'prefilter_'+k:v for k,v in info['prefilter'].items()})
            funnels.append(dict(label=label,split=split,n=len(rr),**counts,
                **{'sum_'+k:sum(r[k] for r in rr) for k in ('accepted','accepted_tip','accepted_edge',
                'CF_lost_legal_edges','FF_lost_legal_edges','FF_new_illegal_pairs','FF_illegal_pairs_removed','deleted_source_faces')}))
    files={}
    for name,rr in (('per_output',rows),('paired_metrics',paired),('binding_ledger',ledger),('rejection_funnel',funnels)):
        p=folder(wave)/(name+'.csv'); C.csvwrite(p,rr); files[str(p)]=C.sha(p)
    return dict(rows=rows,receipts=receipts,numeric_metric_count=len(numeric),files=files)


def run(wave,workers=2):
    reg=freeze(wave); path=folder(wave)/'completion.json'
    if path.exists():
        done=C.read(path); E.verify(done['files'])
        for r in done['receipts']:
            E.verify({r['path']:r['sha256']}); E.verify(C.read(r['path'])['files'])
        return done
    jobs=[(wave,arm,q) for q in reg['requests_manifest'] for arm in WAVES[wave]]
    _,count,seconds=dispatch_cpu(jobs,worker,folder(wave),workers)
    result=aggregate(wave)
    if len(result['rows'])!=reg['cells'] or result['numeric_metric_count']!=124:
        raise ValueError('Binding matrix/metric count incomplete')
    E.verify(reg['protected']); E.verify(reg['code'])
    done=dict(status='COMPLETE',wave=wave,cells=reg['cells'],model_calls=0,GPU_calls=0,new_queries=0,
        workers=count,seconds=seconds,numeric_metric_count=124,receipts=result['receipts'],files=result['files'],
        visual_review='PENDING_ACTUAL_VIEW')
    C.write(path,done); return done


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--wave',choices=tuple(WAVES),required=True)
    parser.add_argument('--workers',type=int,choices=(1,2),default=2); args=parser.parse_args()
    result=freeze(args.wave) if args.stage=='freeze' else run(args.wave,args.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('protected','code','requests_manifest','receipts','files')}),flush=True)


if __name__=='__main__':
    main()
