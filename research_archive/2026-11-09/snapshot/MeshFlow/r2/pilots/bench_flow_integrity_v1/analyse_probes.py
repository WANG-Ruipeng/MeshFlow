"""Reduce the frozen 200-forward probe artifacts on CPU; no model calls."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import time
import numpy as np
from .common import OUT, ROOT, read, resolve, sha, write_new
from . import probe_metrics as M

DEST = OUT / 'probe_analysis_v1'
NATIVE = ('COMMON_S12000', 'NATIVE_S6000', 'COVERAGE_S1500')
MODELS = NATIVE + ('SEAM_S3000',)
METRICS = ('velocity_mse', 'velocity_mse_L2', 'E_shared_pair_mean',
           'endpoint_mse', 'endpoint_rms_L', 'endpoint_gap_L_mean', 'endpoint_gap_L_p95')


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return {key:z[key].copy() for key in z.files}


def csv_write(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    value=stream.getvalue()
    if path.exists():
        with path.open('r',encoding='utf-8-sig',newline='') as f: existing=f.read()
        if existing != value:
            raise RuntimeError('Existing diagnostic CSV differs: '+str(path))
        return
    with path.open('x',encoding='utf-8-sig',newline='') as f:f.write(value)


def balanced(rows,keys,stage):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key, group in sorted(groups.items()):
        row=dict(zip(keys,key)); row['aggregation']=stage; row['input_rows']=len(group)
        row['available_rows']=sum(g['status']=='AVAILABLE' for g in group)
        row['status']='AVAILABLE' if row['available_rows'] else 'NA_NO_APPLICABLE_SOURCE_ELEMENTS'
        for metric in METRICS:
            values=[float(g[metric]) for g in group if g.get(metric) is not None]
            row[metric]=float(np.mean(values)) if values else None
        for name in ('count','scalar_count'):
            values=[g[name] for g in group if name in g]
            if values: row[name+'_min'],row[name+'_max']=min(values),max(values)
        result.append(row)
    return result


def _normalize_manifest(package):
    source=package.get('rows', package.get('traces'))
    if not isinstance(source,list): raise ValueError('No trace rows')
    result=[]
    for raw in source:
        row=dict(raw)
        row['representation']=row.get('representation',row.get('arm'))
        if row['representation'] in ('PLAIN','NATIVE'): row['representation']='NATIVE'
        if row['representation'] not in ('NATIVE','SEAM'): raise ValueError('Unknown representation')
        row['trace_path']=row.get('trace_path',row.get('sample_path'))
        row['trace_sha256']=row.get('trace_sha256',row.get('sample_sha256'))
        if not row['trace_sha256'] and row.get('audit_json'):
            audit=read(row['audit_json']);row['trace_sha256']=audit['sample_sha256']
        if not all(row.get(k) for k in ('trace_path','trace_sha256','case_path','request_id','parent_id','condition_id')):
            raise ValueError('Missing trace metadata')
        result.append(row)
    return result


def analyze(manifest_path):
    manifest_path=resolve(manifest_path)
    if (DEST/'COMPLETE.json').exists():
        result=read(DEST/'COMPLETE.json')
        if sha(manifest_path)!=result['manifest_sha256']:raise ValueError('Manifest changed')
        for path,digest in result['output_hashes'].items():
            if sha(path)!=digest:raise ValueError('Diagnostic output changed')
        return result
    package=read(manifest_path);traces=_normalize_manifest(package)
    missing=[]
    for model in MODELS:
        completion=OUT/'probes'/model/'completion.json'
        if not completion.exists():missing.append(str(completion))
    if missing:return dict(status='PENDING',missing=missing,model_calls=0)
    start=time.perf_counter();protected={str(manifest_path):sha(manifest_path)}
    cases={};sources={};scalar=[];pair_rows=[];identities=[];forward_count=0
    for model in MODELS:
        completion_path=OUT/'probes'/model/'completion.json';complete=read(completion_path)
        protected[str(completion_path)]=sha(completion_path);forward_count+=complete['forwards']
        if complete['backwards'] or complete['updates']:raise ValueError('Probe unexpectedly updated model')
        representation='SEAM' if model=='SEAM_S3000' else 'NATIVE'
        for row in (r for r in traces if r['representation']==representation):
            trace_path=resolve(row['trace_path']);case_path=resolve(row['case_path'])
            if sha(trace_path)!=row['trace_sha256']:raise ValueError('Trace changed')
            if row.get('case_sha256') and sha(case_path)!=row['case_sha256']:raise ValueError('Source case changed')
            protected[str(trace_path)]=row['trace_sha256'];protected[str(case_path)]=sha(case_path)
            sample=arrays(trace_path)
            if str(case_path) not in cases:cases[str(case_path)]=arrays(case_path)
            case=cases[str(case_path)]
            if row.get('pair_path'):
                pair_path=resolve(row['pair_path'])
                if row.get('pair_sha256') and sha(pair_path)!=row['pair_sha256']:raise ValueError('Pair map changed')
                pairs=read(pair_path);protected[str(pair_path)]=sha(pair_path)
            else:
                if str(case_path) not in sources:sources[str(case_path)]=M.source_pairs(case)
                pairs=M.prepare_pairs(sample,case,source=sources[str(case_path)])
            prediction_path=OUT/'probes'/model/(row['request_id']+'.npz')
            expected=complete['files'].get(str(prediction_path))
            if expected is None:
                expected=next((h for p,h in complete['files'].items() if resolve(p)==prediction_path),None)
            if not expected or sha(prediction_path)!=expected:raise ValueError('Unregistered/changed probe prediction')
            protected[str(prediction_path)]=expected;prediction=arrays(prediction_path)
            times=np.asarray(sample['times']);values=prediction['pred']
            if len(times)!=5 or len(values)!=5 or not np.array_equal(times,prediction['times']):raise ValueError('Probe time mismatch')
            for ti,t in enumerate(times):
                current=dict(sample,t=np.asarray(t,np.float32),xt=sample['x_by_time'][ti].reshape(int(sample['N']),3,3))
                scores,details=M.compute_metrics(current,values[ti],pairs=pairs)
                meta=dict(model=model,representation=representation,request_id=row['request_id'],
                    parent_id=row['parent_id'],condition_id=row['condition_id'],noise_index=row.get('noise_index'),
                    split=row.get('split',row.get('panel','UNSPECIFIED')),time_index=ti,t=float(t),
                    L=M.L_DEFAULT,metric_version=M.VERSION,interpretation='SOURCE_BRIDGE_DIAGNOSTIC')
                scalar.extend(dict(meta,**s) for s in scores)
                pair_rows.extend(dict(meta,**p) for p in details['pairs'])
                identities.append(dict(meta,pair_count=details['pair_count'],
                    bridge_identity_max_abs=details['bridge_identity_max_abs'],
                    shared_identity_max_abs=details['shared_identity_max_abs'],
                    identity_status=details['identity_status'],
                    raw_fixed_prediction_squared_sum=details['raw_fixed_prediction_squared_sum']))
    if forward_count!=200 or len(identities)!=200:raise ValueError('Frozen 200-forward scope differs')
    common=('model','representation','kind','stratum')
    request=balanced(scalar,common+('parent_id','condition_id','request_id'),'equal_five_fixed_times')
    condition=balanced(request,common+('parent_id','condition_id'),'equal_existing_noises')
    parent=balanced(condition,common+('parent_id',),'equal_conditions_within_parent')
    summary=balanced(parent,common,'equal_parents')
    native=[r for r in scalar if r['representation']=='NATIVE']
    index={(r['model'],r['request_id'],r['time_index'],r['kind'],r['stratum']):r for r in native}
    paired=[]
    for first,second in ((NATIVE[0],NATIVE[1]),(NATIVE[1],NATIVE[2]),(NATIVE[0],NATIVE[2])):
        for r in (r for r in native if r['model']==first):
            key=(second,r['request_id'],r['time_index'],r['kind'],r['stratum']);other=index[key]
            if r['count']!=other['count']:raise ValueError('Same-state Native denominator differs')
            out={k:r[k] for k in ('request_id','parent_id','condition_id','noise_index','time_index','t','kind','stratum','count','status')}
            out.update(first_model=first,second_model=second)
            for metric in METRICS:
                a,b=r.get(metric),other.get(metric)
                out[metric+'_before']=a;out[metric+'_after']=b
                out[metric+'_delta']=b-a if a is not None and b is not None else None
                out[metric+'_ratio']=b/a if a is not None and b is not None and a>0 else None
            paired.append(out)
    DEST.mkdir(parents=True,exist_ok=True)
    tables={'per_time_stratum':scalar,'per_source_pair':pair_rows,'identity_checks':identities,
            'per_request_time_mean':request,'per_condition_noise_mean':condition,
            'per_parent_summary':parent,'parent_equal_summary':summary,'native_paired_per_time':paired}
    for name,rows in tables.items():csv_write(DEST/(name+'.csv'),rows)
    # Requested explicit request/parent paired tables use the same balanced means.
    for table_name,base,ids in [('native_paired_per_request',request,('parent_id','condition_id','request_id')),
                              ('native_paired_per_parent',parent,('parent_id',))]:
        lookup={(r['model'],r['kind'],r['stratum'])+tuple(r[k] for k in ids):r for r in base}
        output=[]
        for first,second in ((NATIVE[0],NATIVE[1]),(NATIVE[1],NATIVE[2]),(NATIVE[0],NATIVE[2])):
            for r in (x for x in base if x['model']==first):
                other=lookup[(second,r['kind'],r['stratum'])+tuple(r[k] for k in ids)]
                out={k:r[k] for k in ids+('kind','stratum','status')};out.update(first_model=first,second_model=second)
                for metric in METRICS:
                    a,b=r.get(metric),other.get(metric)
                    out[metric+'_before']=a;out[metric+'_after']=b
                    out[metric+'_delta']=b-a if a is not None and b is not None else None
                output.append(out)
        csv_write(DEST/(table_name+'.csv'),output)
    plot(scalar)
    lines=['# Frozen source-bridge probe readout','',
        '200 frozen forwards; zero updates/backward. These measurements use source-target interpolation states. They do not measure autonomous completion, certify generated topology, or require a unique source style.',
        '', 'Native models share exactly the same 60 input states. SEAM has a distinct source-owner layout and is descriptive only; its errors are not a same-input architecture comparison.', '',
        '| Model | Free velocity MSE | Both-free shared-pair difference MSE | One-fixed shared-pair difference MSE |',
        '|---|---:|---:|---:|']
    def val(model,kind,stratum):
        rs=[r for r in summary if r['model']==model and r['kind']==kind and r['stratum']==stratum]
        x=rs[0]['velocity_mse'] if rs else None
        return 'NA' if x is None else f'{x:.8g}'
    for model in MODELS:
        lines.append('| '+model+' | '+val(model,'corner','all_free')+' | '+val(model,'shared_source_pair','all_both_free')+' | '+val(model,'shared_source_pair','all_one_fixed')+' |')
    lines.extend(['','Means: equal five fixed times, equal noises, equal conditions within parent, then equal parents. MSE divides by three scalar coordinates per corner/pair. No valence-squared pooling. Nonapplicable strata stay NA.',
        '', 'The primary pair mask excludes duplicate/degenerate source faces, incidence other than two, nonmanifold endpoint links, and unverified/non-legal adjacent contacts. Legality is pair-local, not a whole-source clean certificate. Slender-triangle proxy means minimum altitude / longest edge <0.1 on either incident triangle; it is not a semantic support label.',
        '', 'Fixed network predictions are projected to zero for the exact production-motion identity and separately logged. Source face and corner metadata reconstruct x1 exactly. The endpoint-gap identity is checked per pair, with FP32 bridge rounding explicitly tolerated.',
        '', 'Do not infer a broken coupling, numerical integration error, or improved free-generation quality from a change in these MSEs alone. Use the separately frozen rollout and step-sensitivity results for those questions.'])
    readout='\n'.join(lines)+'\n'
    target=DEST/'READOUT.md'
    if target.exists() and target.read_text(encoding='utf-8')!=readout:raise ValueError('Existing readout differs')
    if not target.exists():target.write_text(readout,encoding='utf-8')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected input mutated: '+path)
    result=dict(status='COMPLETE',model_calls_here=0,registered_forward_artifacts=200,
        scalar_rows=len(scalar),source_pair_rows=len(pair_rows),identity_rows=len(identities),
        max_bridge_residual=max(r['bridge_identity_max_abs'] for r in identities),
        max_shared_identity_residual=max(r['shared_identity_max_abs'] for r in identities),
        seconds=time.perf_counter()-start,manifest_sha256=sha(manifest_path),
        source_hashes={str(Path(__file__)):sha(__file__),str(Path(M.__file__)):sha(M.__file__)},
        protected_input_hashes=protected,
        output_hashes={str(p):sha(p) for p in sorted(DEST.iterdir()) if p.is_file() and p.name!='COMPLETE.json'})
    write_new(DEST/'COMPLETE.json',result)
    return result


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    wanted=(('corner','all_free','Native all-free FM error'),
            ('shared_source_pair','all_both_free','Native adjacent free-corner difference'),
            ('shared_source_pair','all_one_fixed','Native source C/free difference'),
            ('shared_source_pair','bulk--seam_tip','SEAM source tip/bulk difference (descriptive)'))
    for ax,(kind,stratum,title) in zip(axes.ravel(),wanted):
        models=('SEAM_S3000',) if stratum=='bulk--seam_tip' else NATIVE
        any_data=False
        for model in models:
            selected=[r for r in rows if r['model']==model and r['kind']==kind and r['stratum']==stratum]
            bycondition=balanced(selected,('time_index','t','parent_id','condition_id'),'equal_noise')
            byparent=balanced(bycondition,('time_index','t','parent_id'),'equal_condition')
            means=balanced(byparent,('time_index','t'),'equal_parent')
            points=[r for r in means if r.get('velocity_mse') is not None]
            if points:
                ax.plot([r['t'] for r in points],[r['velocity_mse'] for r in points],marker='o',label=model)
                any_data=True
        ax.set_title(title);ax.set_xlabel('Supervised bridge time t');ax.set_ylabel('MSE per scalar coordinate');ax.grid(alpha=.2)
        if any_data:ax.legend(fontsize=7)
        else:ax.text(.5,.5,'NA: no applicable legal source pairs',ha='center',transform=ax.transAxes)
    path=DEST/'source_bridge_probe_curves.png'
    if path.exists():raise RuntimeError('Refuse overwrite existing diagnostic chart')
    fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.manifest),ensure_ascii=False))
