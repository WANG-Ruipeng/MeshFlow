"""CPU-only original 124-field scoring and fixed part/parent aggregation.

The historical finite-face detectors are unchanged. Only saved-artifact routing,
checkpoint labels, and closed-C seam applicability are adapted privately.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import collections
import concurrent.futures
import inspect
import multiprocessing
from pathlib import Path
import shutil
import sys
import time
import types
import numpy as np
from . import panel as P
from r2.pilots.seam_abc_v1 import evaluate as E, eval_common as C
from r2.pilots.seam_pool47_couple_v1.endpoint_scoring import metric_row

VERSION = 'bench_parts_original_124_fields_closed_C_applicability_v1'


def requested(root, subset='full'):
    m = P.read(P.resolve(root)/'evaluation_inputs/manifest.json')
    allowed = set(m['process_request_ids'])
    return [q for q in m['requests'] if subset == 'full' or q['request_id'] in allowed]


def artifact(root, label, request_id, stage):
    path = P.resolve(root)/'evaluation'/label/request_id/stage.lower()/(stage.lower()+'.npz')
    receipt = path.parent/'receipt.json'; r = P.read(receipt)
    if r.get('status') not in ('COMPLETE','PASS','VERIFIED_CACHE') or r.get('label',r.get('arm')) != label or r['request_id'] != request_id:
        raise ValueError('Saved producer identity differs: '+str(receipt))
    key = 'raw_sha256' if stage == 'RAW' else 'post_sha256'
    if key in r:
        if P.sha(path) != r[key]: raise ValueError('Saved mesh hash differs')
    elif str(path) not in r.get('files',{}) or P.sha(path) != r['files'][str(path)]:
        raise ValueError('Producer does not bind mesh SHA')
    if stage == 'RAW' and r.get('generator_forwards') != 50:
        raise ValueError('Not the fixed Euler50 protocol')
    if stage == 'POST' and (r.get('D0_forwards') != 1 or r.get('GEO_calls') != 1):
        raise ValueError('Not the single-D0/GEO_C protocol')
    if r.get('files'): E.verify(r['files'])
    return path, receipt


def private_score(q, label, seam_model):
    common = types.SimpleNamespace(**vars(C)); common.ARMS = (label,) if seam_model else ()
    namespace = dict(vars(E)); namespace['C'] = common
    for name, obj in vars(E).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(E):
            fn = types.FunctionType(obj.__code__, namespace, obj.__name__, obj.__defaults__, obj.__closure__)
            fn.__kwdefaults__ = obj.__kwdefaults__; namespace[name] = fn
    previous = namespace['seam_to_bulk']
    def scoped_graph(*args, **kwargs):
        if int(q['boundary_edges']) == 0:
            return dict(status='NA_CLOSED_C_NO_SEAM_INTERFACE', **{k:None for k in E.INTERNAL_FIELDS}), []
        return previous(*args, **kwargs)
    namespace.update(artifact=artifact, VERSION=VERSION, seam_to_bulk=scoped_graph)
    return namespace


def source_hashes():
    return {**E.source_hashes(), str(Path(metric_row.__code__.co_filename)):P.sha(metric_row.__code__.co_filename),
            str(Path(__file__)):P.sha(__file__)}


def prepare(root, label, seam_model=True):
    root = P.resolve(root); dest = root/'evaluation/score_registrations'/(label+'.json')
    if dest.exists():
        r=P.read(dest); E.verify(r['protected']); E.verify(r['source_hashes'])
        if r['seam_model'] != seam_model: raise ValueError('Checkpoint representation changed')
        return r
    m=P.read(root/'evaluation_inputs/manifest.json'); old,A,_,_,_=E.helpers()
    descriptors={}; qq=[]; protected={str(root/'evaluation_inputs/manifest.json'):P.sha(root/'evaluation_inputs/manifest.json')}
    for initial in m['requests']:
        q=dict(initial); inp=C.arrays(q['inference_npz']); meta=C.arrays(q['metadata_npz'])
        cc=inp['C_vertices'][inp['C_faces_model']]; original=inp['C_vertices'][inp['C_faces_original']]
        ids=meta['C_source_vertex_ids'][inp['C_faces_original']]
        _,_,topology=A.topology(original,ids)
        if int(topology['boundary_edge_count']) != int(q['boundary_edges']):
            raise ValueError('Frozen C boundary count changed')
        cid=q['condition_id']
        if cid not in descriptors:
            bank,seed=old.query_bank(cc,cid); path=root/'evaluation/queries'/(cid+'.npz'); P.array_file(path,bank)
            descriptors[cid]=dict(condition_id=cid,parent_id=q['parent_id'],path=str(path),sha256=P.sha(path),
                array_sha256={k:old.array_sha(v) for k,v in bank.items()},
                identity=dict(condition_id=cid,C_seed=seed,C_sha256=old.array_sha(cc),L0=C.L),
                query_count=2048,weight_denominator=float(bank['C_weights'].sum()),source_boundary_audit=topology)
        q['query_descriptor']=descriptors[cid]; qq.append(q)
        for key in ('inference_npz','metadata_npz','noise_npz','source_npz'):
            protected[q[key]]=P.sha(q[key])
        protected[descriptors[cid]['path']]=descriptors[cid]['sha256']
    hashes=source_hashes()
    for path,digest in hashes.items():
        p=P.resolve(path)
        try: relative=p.relative_to(P.ROOT)
        except ValueError: relative=Path('external')/p.name
        target=root/'scoring_source_snapshot'/relative; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and P.sha(target)!=digest: raise ValueError('Scorer snapshot changed')
        if not target.exists(): shutil.copyfile(p,target)
    r=dict(status='FROZEN',version=VERSION,label=label,seam_model=seam_model,requests=qq,
        stages=['RAW','POST'],source_hashes=hashes,protected=protected,L=C.L,C_query_count=2048,
        source_geometry_use='C-only source identity; no unique-parent reconstruction ranking',
        closed_C_policy='AVMR/PMR NA_EMPTY_BOUNDARY; seam graph NA_CLOSED_C_NO_SEAM_INTERFACE; CF/FF/I/quality remain applicable',
        open_C_all_null_policy='Retain zero graph reach; do not convert prediction failures into NA',
        model_calls=0,quality_definition=old.cpu_math()['control'].frozen_definition())
    P.immutable(dest,r)
    if 'torch' in sys.modules: raise RuntimeError('CPU scoring imported model framework')
    return r


def worker(job):
    root,label,q,stage,regsha,seam_model=job
    return private_score(q,label,seam_model)['score_one']((root,label,q,stage,regsha))


def run(root,label,seam_model=True,subset='full',workers=2):
    root=P.resolve(root); reg=prepare(root,label,seam_model); ids={q['request_id'] for q in requested(root,subset)}
    qq=[q for q in reg['requests'] if q['request_id'] in ids]
    completion=root/'evaluation/completions'/(label+'_'+subset+'.json')
    if completion.exists():
        done=P.read(completion)
        for receipt in done['receipts']: E.verify({receipt['path']:receipt['sha256']}); E.verify(P.read(receipt['path'])['files'])
        return done
    for q in qq:
        for stage in reg['stages']: artifact(root,label,q['request_id'],stage)
    regsha=P.sha(root/'evaluation/score_registrations'/(label+'.json'))
    jobs=[(str(root),label,q,s,regsha,seam_model) for q in qq for s in reg['stages']]
    started=time.perf_counter(); first=worker(jobs[0]); snap=C.resources()
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in (snap.get('memory') or '').splitlines()
            if len(line.split())>1 and line.split()[1].isdigit()}
    count=2 if workers==2 and (os.cpu_count() or 1)>=4 and memory.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024) and (snap.get('loadavg') or [999])[0]<(os.cpu_count() or 1)-2 else 1
    dispatch=root/'evaluation/dispatch'/(label+'_'+subset+'.json')
    P.immutable(dispatch,dict(workers=count,threads_per_worker=1,resources=snap,first_actual_cell=first))
    receipts=[first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        for r in pool.map(worker,jobs[1:]):
            receipts.append(r); print('SCORE',label,len(receipts),'/',len(jobs),flush=True)
    E.verify(reg['protected']);E.verify(reg['source_hashes']);receipts.sort(key=lambda r:r['path'])
    done=dict(status='COMPLETE',label=label,subset=subset,cells=len(receipts),receipts=receipts,
        workers=count,seconds=time.perf_counter()-started,model_calls=0)
    P.immutable(completion,done);return done


def number(x):
    return isinstance(x,(int,float,bool)) and np.isfinite(float(x))


def aggregate(root,labels,subset='full'):
    root=P.resolve(root); qq=requested(root,subset); lookup={q['request_id']:q for q in qq}; rows=[];roles=[];deleted=[]
    for label in labels:
        done=P.read(root/'evaluation/completions'/(label+'_'+subset+'.json'))
        if done['cells'] != 2*len(qq): raise ValueError('Incomplete scoring matrix')
        for q in qq:
            for stage in ('RAW','POST'):
                folder=root/'evaluation/scores'/label/q['request_id']/stage
                s=P.read(folder/'score.json');E.verify(s['files'])
                row,role,drop=metric_row(s,C.arrays(s['identity']['mesh_path']),P.read(folder/'pairs.json'))
                row.update(part_type=q['part_type'],part_family=q['part_family'],closed_C=q['closed_C'],official_split=q['official_split'])
                rows.append(row);roles.extend(role);deleted.extend(drop)
    numeric=P.read(P.ROOT/'experiments/r2_bench_control_transfer_v1/evaluation/summary_completion.json')['numeric_fields']
    if len(numeric)!=124 or any(k not in row for row in rows for k in numeric): raise ValueError('Original 124-field schema missing')
    summaries=[];conditions=[];parents=[]
    for label in labels:
        for stage in ('RAW','POST'):
            for split in ('TRAIN','DEV','ALL'):
                for part in ('ALL',*sorted({q['part_type'] for q in qq})):
                    rr=[r for r in rows if r['label']==label and r['stage']==stage and (split=='ALL' or r['split']==split) and (part=='ALL' or r['part_type']==part)]
                    if not rr:continue
                    for metric in numeric:
                        cc=collections.defaultdict(list)
                        for r in rr:cc[(r['parent_id'],r['condition_id'])].append(r)
                        pp=collections.defaultdict(list)
                        for (parent,cid),rs in cc.items():
                            values=[float(r[metric]) for r in rs if number(r[metric])]
                            v=float(np.mean(values)) if values else None
                            pp[parent].append(v)
                            if split!='ALL' and part=='ALL':conditions.append(dict(label=label,stage=stage,split=split,parent_id=parent,condition_id=cid,metric=metric,value=v,n_registered=len(rs),n_valid=len(values)))
                        pv=[]
                        for parent,vs in pp.items():
                            valid=[v for v in vs if v is not None];v=float(np.mean(valid)) if valid else None
                            if v is not None:pv.append(v)
                            if split!='ALL' and part=='ALL':parents.append(dict(label=label,stage=stage,split=split,parent_id=parent,metric=metric,value=v,n_conditions=len(vs),n_valid_conditions=len(valid)))
                        vals=[float(r[metric]) for r in rr if number(r[metric])]
                        summaries.append(dict(label=label,stage=stage,split=split,part_type=part,metric=metric,
                            mean_parent_equal=float(np.mean(pv)) if pv else None,mean_output_descriptive=float(np.mean(vals)) if vals else None,
                            median_output=float(np.median(vals)) if vals else None,min=min(vals) if vals else None,max=max(vals) if vals else None,
                            n_outputs=len(rr),n_valid=len(vals),n_parents=len(pp),n_valid_parents=len(pv),
                            status='PASS' if len(vals)==len(rr) else 'NA_OR_PARTIAL'))
    destination=root/'evaluation/tables'/('__'.join(labels)+'_'+subset)
    files={}
    for name,values in [('per_output',rows),('summary_parent_equal',summaries),('per_condition',conditions),('per_parent',parents),('illegal_pairs_by_role',roles),('deleted_faces',deleted)]:
        path=destination/(name+'.csv');C.csvwrite(path,values);files[str(path)]=P.sha(path)
    pairs=[]
    if len(labels)>1:
        baseline=labels[0];by={(r['label'],r['stage'],r['request_id']):r for r in rows}
        for label in labels[1:]:
            for q in qq:
                for stage in ('RAW','POST'):
                    a,b=by[baseline,stage,q['request_id']],by[label,stage,q['request_id']]
                    for metric in numeric:
                        valid=number(a[metric]) and number(b[metric]);pairs.append(dict(baseline=baseline,candidate=label,stage=stage,
                            request_id=q['request_id'],parent_id=q['parent_id'],condition_id=q['condition_id'],split=q['split'],part_type=q['part_type'],metric=metric,
                            baseline_value=a[metric],candidate_value=b[metric],delta=float(b[metric])-float(a[metric]) if valid else None,status='PASS' if valid else 'NA_NOT_COMMON'))
        path=destination/'paired_per_output.csv';C.csvwrite(path,pairs);files[str(path)]=P.sha(path)
    result=dict(status='COMPLETE',labels=labels,subset=subset,rows=len(rows),numeric_fields=numeric,files=files,
        aggregation='Noise mean -> condition mean -> parent equal mean; part strata separate; NA denominators disclosed',model_calls=0)
    P.immutable(destination/'completion.json',result);return result


def gallery(root,labels,subset='full',kind=None):
    from r2.pilots.seam_followup_v1 import gallery as G
    root=P.resolve(root);qs=requested(root,subset);kind=kind or '__'.join(labels)+'_'+subset
    pages=[]
    for q in qs:
        cells=[]
        for label in labels:
            for stage in ('RAW','POST'):
                path,_=artifact(root,label,q['request_id'],stage);cells.append(dict(label=label+'_'+stage,path=str(path)))
        pages.append(dict(**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','K','N','split')},
            page_id=q['request_id'],stage='RAW / POST',kind='Bench '+q['part_type']+' | '+q['extent_description'],cells=cells))
    manifest=dict(pages=pages,model_calls=0,metric_based_selection=False,
        section_policy='Two fixed opposing surface views and finite front/side sections at C bbox center; common per-C scale',
        review_policy='AI_PROXY actual viewing, no metric-to-label conversion; source style not displayed')
    P.immutable(root/'gallery_manifests'/(kind+'.json'),manifest);G.OUT=root;G.run(kind)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','score','aggregate','gallery'))
    p.add_argument('--root',default=str(P.OUT));p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--subset','--panel',choices=('full','process'),default='full');p.add_argument('--plain',action='store_true');p.add_argument('--workers',type=int,choices=(1,2),default=2)
    a=p.parse_args()
    if a.stage=='prepare':
        for label in a.labels:prepare(a.root,label,not a.plain)
    elif a.stage=='score':
        for label in a.labels:run(a.root,label,not a.plain,a.subset,a.workers)
    elif a.stage=='aggregate':aggregate(a.root,a.labels,a.subset)
    else:gallery(a.root,a.labels,a.subset)
