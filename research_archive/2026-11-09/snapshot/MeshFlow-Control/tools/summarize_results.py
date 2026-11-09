#!/usr/bin/env python3
"""Read-only descriptive summary of completed evaluation tables.

No model/scorer imports. Does not rank models or change evaluation gates.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,statistics,tempfile
from collections import Counter,defaultdict
from pathlib import Path

VERSION='direct_context_readout_descriptive_summary_v1'
LABELS={'P1':('START','A500','B500','C500','A1000','B1000','C1000'),
        'P2':('START','A1000','B1000','C1000')}
COHORTS={'P1':{'OLD':(0,1),'NEW':(4,5),'ALL':(0,1,4,5)},'P2':{'OPEN':(0,1,2,3)}}
FOCUS=('CF_proper_affected_free_fraction','CF_coplanar_pair_count','CF_unresolved_or_degenerate_pair_count',
       'FF_illegal_affected_free_fraction','FF_illegal_pair_count','FF_unresolved_or_degenerate_pair_count',
       'ConflictCov_C','ConflictCov_C_cross','ConflictCov_C_dup','ConflictCov_C_cross_and_dup',
       'free_near_degenerate_fraction','free_exact_zero_area_fraction','free_minimum_angle_lt5_fraction',
       'free_exact_duplicate_excess_fraction','EXACT_COORD_face_components','EXACT_COORD_CF_shared_edges',
       'EXACT_COORD_boundary_edges','EXACT_COORD_nonmanifold_edges',
       'SeamCov','ExposedRetain','ExposedRetain_global_lower','ExposedRetain_global_upper',
       'GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','D_sym2048','fixed_query_MSE','fixed_query_p95_L','coverage_fraction',
       'D_free_mean')
BOUNDS={'SeamCov':('SeamCov','SeamCov_upper'),'ConflictCov_C':('ConflictCov_C','ConflictCov_C_upper'),
        'ExposedRetain':('ExposedRetain','ExposedRetain_upper'),
        'ExposedRetain_global':('ExposedRetain_global_lower','ExposedRetain_global_upper')}

def path(value):
    s=str(value).replace('\\','/')
    if os.name!='nt' and len(s)>2 and s[1]==':':s='/mnt/'+s[0].lower()+s[2:]
    if os.name=='nt' and s.startswith('/mnt/'):s=s[5].upper()+':/'+s[7:]
    return Path(s)
def read(p):return json.loads(path(p).read_text(encoding='utf-8-sig'))
def rows(value):
    d=read(value) if isinstance(value,(str,Path)) else value
    return d['rows'] if isinstance(d,dict) else d
def sha(p):return hashlib.sha256(path(p).read_bytes()).hexdigest()
def dump(p,value):
    p=path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rr):
    p=path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rr for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
def numeric(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def number(v):
    if v in (None,'','NA','None'):return None
    try:x=float(v)
    except (ValueError,TypeError):return None
    return x if math.isfinite(x) else None
def mean(v):return statistics.fmean(v) if v else None
def quantile(v,q):
    if not v:return None
    a=sorted(v);i=(len(a)-1)*q;lo=int(i);hi=min(lo+1,len(a)-1)
    return a[lo]+(a[hi]-a[lo])*(i-lo)
def stats(values,expected):
    finite=[float(x) for x in values if numeric(x)];complete=len(finite)==expected and expected>0
    return dict(expected_units=expected,finite_units=len(finite),missing_units=expected-len(finite),
        mean=mean(finite) if complete else None,median=statistics.median(finite) if complete else None,
        observed_mean=mean(finite),observed_median=statistics.median(finite) if finite else None,
        minimum=min(finite) if finite else None,p05=quantile(finite,.05),p95=quantile(finite,.95),maximum=max(finite) if finite else None,
        positive_count=sum(x>0 for x in finite),zero_count=sum(x==0 for x in finite),negative_count=sum(x<0 for x in finite),
        status='COMPLETE' if complete else 'NA_OR_INCOMPLETE',
        observed_distribution_warning=None if complete else 'Observed subset only; not a full-cohort result')
def interval(m,key):
    fields=BOUNDS.get(key)
    if not fields:return None,None
    a,b=m.get(fields[0]),m.get(fields[1])
    return (float(a),float(b)) if numeric(a) and numeric(b) else (None,None)
def reference_only(key):
    return key.startswith(('SeamCov','ExposedRetain','Gamma','GT_','F_to_GT','D_sym','fixed_query','coverage','free_to_R12','R12_','remainder_','boundary_','Precision','Recall','Fscore','reference_','ConflictCov_C_GT')) or 'minus_GT' in key or key=='free_area_over_GT_free_area'

def constraints(m,key):
    v=m.get(key)
    if numeric(v):return float(v),'PASS'
    return None,'NA_OR_MISSING'

def compute(evaluation,manifest,out,training_logs=None):
    evaluation,out=path(evaluation),path(out);base=rows(manifest)
    source=evaluation/'per_output.json';observed=rows(source)
    index={};duplicates=[]
    for r in observed:
        key=(r['label'],r['request_id'])
        if key in index:duplicates.append(key)
        index[key]=r
    if duplicates:raise ValueError('Duplicate evaluated cells; refusing silent selection: '+str(duplicates[:3]))
    bypanel={p:[r for r in base if r['panel']==p] for p in LABELS}
    metrics=sorted(set(FOCUS)-{'D_free_mean'} | {k for r in observed for k,v in r.get('metrics',{}).items() if numeric(v) or v is None})
    unit_rows=[];profile_rows=[];noise_rows=[];pair_rows=[];contrast_rows=[];contracts=[]
    keyed_units={};noise_index=defaultdict(list)
    for panel,labels in LABELS.items():
        parent_ids=sorted({r['parent_id'] for r in bypanel[panel]})
        for label in labels:
            for cohort,noises in COHORTS[panel].items():
                requests=[r for r in bypanel[panel] if r['candidate'] in noises]
                for pid in parent_ids:
                    pp=[r for r in requests if r['parent_id']==pid]
                    expected=len(pp)
                    for metric in metrics:
                        vals=[];low=[];high=[];global_low=[];global_high=[];states=Counter()
                        for req in pp:
                            rr=index.get((label,req['request_id']))
                            if rr is None:states['MISSING_OUTPUT']+=1;continue
                            if rr.get('status')!='PASS':states[rr.get('status','UNKNOWN')]+=1;continue
                            val=rr.get('metrics',{}).get(metric)
                            state=rr.get('metric_status',{}).get(metric,'PASS' if numeric(val) else 'NA_OR_MISSING')
                            states[state]+=1
                            if numeric(val):vals.append(float(val))
                            lo,hi=interval(rr.get('metrics',{}),metric)
                            if lo is not None:low.append(lo);high.append(hi)
                            if metric=='ExposedRetain':
                                gl,gu=interval(rr.get('metrics',{}),'ExposedRetain_global')
                                if gl is not None:global_low.append(gl);global_high.append(gu)
                        value=mean(vals) if expected and len(vals)==expected else None
                        item=dict(label=label,panel=panel,cohort=cohort,parent_id=pid,metric=metric,value=value,
                            expected_noises=expected,finite_noises=len(vals),lower=mean(low) if expected and len(low)==expected else None,
                            upper=mean(high) if expected and len(high)==expected else None,
                            global_lower=mean(global_low) if expected and len(global_low)==expected else None,
                            global_upper=mean(global_high) if expected and len(global_high)==expected else None,status_counts=dict(states),
                            unit='parent' if panel=='P1' else 'condition; seat2/seat32 same geometric surface',
                            status='COMPLETE' if value is not None else 'NA_OR_INCOMPLETE')
                        if panel=='P2' and reference_only(metric):item.update(status='NA_NOT_APPLICABLE',status_counts={'NA_NO_SOURCE_GT':expected})
                        unit_rows.append(item);keyed_units[(label,panel,cohort,pid,metric)]=item
                for metric in metrics:
                    vals=[keyed_units[(label,panel,cohort,pid,metric)]['value'] for pid in parent_ids]
                    profile_rows.append(dict(label=label,panel=panel,cohort=cohort,metric=metric,
                        endpoint_role='baseline' if label=='START' else 'primary1000' if label.endswith('1000') else 'secondary500',
                        **stats(vals,len(parent_ids))))
            lr=[index.get((label,r['request_id'])) for r in bypanel[panel]]
            keys=sorted({k for r in lr if r for k in r.get('contracts',{})})
            for k in keys:
                counts=Counter('MISSING_OUTPUT' if r is None else 'TRUE' if r.get('contracts',{}).get(k) is True else
                    'FALSE' if r.get('contracts',{}).get(k) is False else 'UNKNOWN' for r in lr)
                contracts.append(dict(label=label,panel=panel,contract=k,registered=len(lr),**dict(counts)))
        contrasts=[]
        for step in ((500,1000) if panel=='P1' else (1000,)):
            contrasts += [(f'B{step}',f'A{step}'),(f'C{step}',f'A{step}'),(f'C{step}',f'B{step}')]
            contrasts += [(f'{arm}{step}','START') for arm in 'ABC']
        for candidate,reference in contrasts:
            role='primary1000' if candidate.endswith('1000') else 'secondary500'
            contrast=candidate+'-'+reference
            for req in bypanel[panel]:
                a=index.get((candidate,req['request_id']));b=index.get((reference,req['request_id']))
                for metric in metrics:
                    av=a.get('metrics',{}).get(metric) if a and a.get('status')=='PASS' else None
                    bv=b.get('metrics',{}).get(metric) if b and b.get('status')=='PASS' else None
                    alo,ahi=interval(a.get('metrics',{}),metric) if a else (None,None)
                    blo,bhi=interval(b.get('metrics',{}),metric) if b else (None,None)
                    noise_rows.append(dict(contrast=contrast,endpoint_role=role,panel=panel,cohort=req['cohort'],
                        parent_id=req['parent_id'],condition_id=req['condition_id'],request_id=req['request_id'],noise=req['candidate'],
                        metric=metric,candidate_value=av if numeric(av) else None,reference_value=bv if numeric(bv) else None,
                        delta=float(av-bv) if numeric(av) and numeric(bv) else None,
                        delta_lower=alo-bhi if alo is not None and bhi is not None else None,
                        delta_upper=ahi-blo if ahi is not None and blo is not None else None))
                    noise_index[(contrast,panel,req['parent_id'],metric)].append(noise_rows[-1])
            for cohort,noises in COHORTS[panel].items():
                for metric in metrics:
                    values=[];pitems=[]
                    for pid in parent_ids:
                        a=keyed_units[(candidate,panel,cohort,pid,metric)];b=keyed_units[(reference,panel,cohort,pid,metric)]
                        av,bv=a['value'],b['value'];delta=av-bv if numeric(av) and numeric(bv) else None
                        values.append(delta)
                        if delta is not None and numeric(bv) and bv!=0:pct=100*delta/abs(bv)
                        else:pct=None
                        parent_noise=[r for r in noise_index[(contrast,panel,pid,metric)] if r['noise'] in noises]
                        ds=[r['delta'] for r in parent_noise if numeric(r['delta'])]
                        item=dict(contrast=contrast,endpoint_role=role,panel=panel,cohort=cohort,parent_id=pid,metric=metric,
                            candidate_value=av,reference_value=bv,delta=delta,relative_delta_percent=pct,
                            delta_lower=a['lower']-b['upper'] if numeric(a['lower']) and numeric(b['upper']) else None,
                            delta_upper=a['upper']-b['lower'] if numeric(a['upper']) and numeric(b['lower']) else None,
                            delta_global_lower=a['global_lower']-b['global_upper'] if numeric(a['global_lower']) and numeric(b['global_upper']) else None,
                            delta_global_upper=a['global_upper']-b['global_lower'] if numeric(a['global_upper']) and numeric(b['global_lower']) else None,
                            expected_noises=a['expected_noises'],finite_paired_noises=len(ds),
                            positive_noises=sum(v>0 for v in ds),negative_noises=sum(v<0 for v in ds),
                            zero_noises=sum(v==0 for v in ds),noise_sign_disagreement=any(v>0 for v in ds) and any(v<0 for v in ds))
                        pair_rows.append(item);pitems.append(item)
                    value=stats(values,len(parent_ids))
                    complete=value['status']=='COMPLETE'
                    biggest=max(pitems,key=lambda x:abs(x['delta']) if numeric(x['delta']) else -1) if pitems else None
                    leave=[dict(removed_parent_id=p['parent_id'],mean=mean([q['delta'] for q in pitems if q['parent_id']!=p['parent_id']])) for p in pitems] if complete and len(pitems)>1 else []
                    av=[p['candidate_value'] for p in pitems];bv=[p['reference_value'] for p in pitems]
                    cmean=mean(av) if all(numeric(v) for v in av) and av else None
                    rmean=mean(bv) if all(numeric(v) for v in bv) and bv else None
                    contrast_rows.append(dict(contrast=contrast,endpoint_role=role,panel=panel,cohort=cohort,metric=metric,**value,
                        candidate_parent_mean=cmean,reference_parent_mean=rmean,
                        mean_relative_delta_percent=100*(cmean-rmean)/abs(rmean) if numeric(cmean) and numeric(rmean) and rmean else None,
                        noise_disagreement_parents=sum(p['noise_sign_disagreement'] for p in pitems),
                        mean_delta_lower=mean([p['delta_lower'] for p in pitems]) if pitems and all(numeric(p['delta_lower']) for p in pitems) else None,
                        mean_delta_upper=mean([p['delta_upper'] for p in pitems]) if pitems and all(numeric(p['delta_upper']) for p in pitems) else None,
                        mean_delta_global_lower=mean([p['delta_global_lower'] for p in pitems]) if pitems and all(numeric(p['delta_global_lower']) for p in pitems) else None,
                        mean_delta_global_upper=mean([p['delta_global_upper'] for p in pitems]) if pitems and all(numeric(p['delta_global_upper']) for p in pitems) else None,
                        largest_absolute_parent_id=biggest['parent_id'] if biggest and numeric(biggest['delta']) else None,
                        largest_absolute_parent_delta=biggest['delta'] if biggest else None,
                        leave_one_parent_out=leave,interpretation='Descriptive only; no removal, ranking or promotion gate'))
    div_pairs=[];div_units=[]
    dp=evaluation/'diversity_pairs.csv'
    if dp.exists():
        with dp.open(encoding='utf-8-sig',newline='') as f:div_pairs=list(csv.DictReader(f))
        for panel,labels in LABELS.items():
            cids=sorted({r['condition_id'] for r in bypanel[panel]})
            for label in labels:
                for cohort,ns in COHORTS[panel].items():
                    expected=len(ns)*(len(ns)-1)//2
                    for cid in cids:
                        rr=[r for r in div_pairs if r['label']==label and r['panel']==panel and r['condition_id']==cid and int(r['candidate_a']) in ns and int(r['candidate_b']) in ns]
                        vals=[number(r['D_free_pair']) for r in rr];finite=[v for v in vals if v is not None]
                        div_units.append(dict(label=label,panel=panel,cohort=cohort,condition_id=cid,
                            expected_pairs=expected,finite_pairs=len(finite),D_free_mean=mean(finite) if len(rr)==expected and len(finite)==expected else None,
                            status='COMPLETE' if len(rr)==expected and len(finite)==expected else 'INCOMPLETE',
                            interpretation='Pairwise variation, not quality or original TMD; no candidate selection'))
    # Diversity contrasts use condition means, never individual pair counts as independent units.
    div_paired=[];div_contrasts=[];di={(r['label'],r['panel'],r['cohort'],r['condition_id']):r for r in div_units}
    for panel in LABELS:
        cids=sorted({r['condition_id'] for r in bypanel[panel]})
        for step in ((500,1000) if panel=='P1' else (1000,)):
            comparisons=[(f'B{step}',f'A{step}'),(f'C{step}',f'A{step}'),(f'C{step}',f'B{step}')]+[(f'{a}{step}','START') for a in 'ABC']
            for candidate,reference in comparisons:
                for cohort in COHORTS[panel]:
                    vals=[]
                    for cid in cids:
                        a=di.get((candidate,panel,cohort,cid),{}).get('D_free_mean')
                        b=di.get((reference,panel,cohort,cid),{}).get('D_free_mean')
                        delta=a-b if numeric(a) and numeric(b) else None;vals.append(delta)
                        div_paired.append(dict(contrast=candidate+'-'+reference,panel=panel,cohort=cohort,condition_id=cid,candidate_value=a,reference_value=b,delta=delta))
                    div_contrasts.append(dict(contrast=candidate+'-'+reference,panel=panel,cohort=cohort,metric='D_free_mean',
                        endpoint_role='primary1000' if step==1000 else 'secondary500',**stats(vals,len(cids)),
                        interpretation='Variation difference only; positive is not automatically better'))
    for collection in (profile_rows,contrast_rows):
        for r in collection:
            if r['panel']=='P2' and reference_only(r['metric']):
                r.update(status='NA_NOT_APPLICABLE',not_applicable_units=r['expected_units'],applicable_units=0,missing_units=0,observed_distribution_warning='No unique source GT; not a measurement failure')
    for r in pair_rows:
        if r['panel']=='P2' and reference_only(r['metric']):r['status']='NA_NOT_APPLICABLE'
    table(out/'diversity_paired.csv',div_paired);table(out/'diversity_contrasts.csv',div_contrasts)
    table(out/'per_unit_metrics.csv',unit_rows);table(out/'checkpoint_distributions.csv',profile_rows)
    table(out/'paired_noise.csv',noise_rows);table(out/'paired_parent.csv',pair_rows);table(out/'contrast_distributions.csv',contrast_rows)
    table(out/'contract_counts.csv',contracts);table(out/'diversity_by_cohort.csv',div_units)
    training=training_summary(training_logs,out) if training_logs else dict(status='NOT_REQUESTED')
    source_hashes={str(source):sha(source),str(path(manifest)):sha(manifest)}
    if dp.exists():source_hashes[str(dp)]=sha(dp)
    expected_keys={(label,r['request_id']) for panel in LABELS for label in LABELS[panel] for r in bypanel[panel]}
    expected_cells=len(expected_keys)
    result=dict(schema=VERSION,status='COMPLETE_COHORT' if set(index)==expected_keys and all(r.get('status')=='PASS' for r in observed) else 'PARTIAL_COHORT',
        registered_requests=len(base),expected_cells=expected_cells,observed_cells=len(index),missing_cells=len(expected_keys-set(index)),unexpected_cells=len(set(index)-expected_keys),source_hashes=source_hashes,
        primary_endpoint=1000,secondary_endpoint=500,focus_contrasts=[r for r in contrast_rows if r['metric'] in FOCUS],
        contracts=contracts,diversity=div_units,diversity_contrasts=div_contrasts,training=training,
        interpretation=dict(parent_weighting='noise average first; equal parents for P1; missing parent never silently removed',
            P1='Observed historical DEV parents; OLD and new noises separated. Source reconstruction is diagnostic, not a rejection gate.',
            P2='No unique GT. Four conditions are not four independent source parents; seat2 and seat32 are one surface.',
            sign='positive/negative counts are mathematical signs; do not automatically mean improvement/degradation',
            leave_one_out='descriptive sensitivity only; never removes parent from primary analysis',
            mechanism='Norms/entropy/gradient reachability are implementation evidence, not control-quality improvements',
            human_usability='NOT_RUN',automatic_ranking='NOT_DEFINED',promotion='NOT_DECIDED'))
    dump(out/'summary.json',result)
    files=[p for p in out.iterdir() if p.is_file() and p.name!='output_hashes.json']
    dump(out/'output_hashes.json',{p.name:sha(p) for p in files})
    return result

def training_summary(folder,out):
    folder=path(folder);cost=[];mechanism=[];sources={}
    for p in sorted(folder.rglob('training_log.csv')):
        sources[str(p)]=sha(p)
        with p.open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
        grouped=defaultdict(list)
        for r in rr:grouped[r.get('arm',p.parent.name)].append(r)
        for arm,ar in grouped.items():
            steps=[int(r['step']) for r in ar if r.get('step','').isdigit()]
            base=dict(arm=arm,source=str(p),logged_rows=len(ar),unique_logged_updates=len(set(steps)),duplicate_step_rows=len(steps)-len(set(steps)),
                minimum_step=min(steps) if steps else None,maximum_step=max(steps) if steps else None,
                caveat='Logged update rows only; authoritative attempted/failed call counts remain in budget ledger')
            for key in ('FM','old_gradient_norm','new_gradient_norm','new_actual_update_norm','input_seconds','forward_seconds',
                        'backward_seconds','optimizer_seconds','update_seconds','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes'):
                vals=[number(r.get(key)) for r in ar];vals=[v for v in vals if v is not None]
                base[key+'_n']=len(vals);base[key+'_mean']=mean(vals);base[key+'_max']=max(vals) if vals else None
                if key.endswith('_seconds'):base[key+'_sum']=sum(vals) if vals else None
            base['memory_scope']='PyTorch allocated/reserved are distinct, non-additive; not process or device peak'
            cost.append(base)
    for p in sorted(folder.rglob('diagnostics.jsonl')):
        sources[str(p)]=sha(p)
        for line_no,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
            if not line.strip():continue
            try:r=json.loads(line)
            except json.JSONDecodeError:
                mechanism.append(dict(source=str(p),line=line_no,status='INCOMPLETE_JSONL_LINE'));continue
            if r.get('event')!='TRAIN_DIAGNOSTICS':continue
            for entry in r.get('readout_observations',[]):
                mechanism.append(dict(arm=r['arm'],step=r['step'],kind='readout_observation',source=str(p),**{k:v for k,v in entry.items() if numeric(v) or isinstance(v,str)}))
            for kind in ('readout_parameter_gradients','readout_parameter_updates'):
                for name,val in r.get(kind,{}).items():
                    mechanism.append(dict(arm=r['arm'],step=r['step'],kind=kind,parameter=name,value=val,source=str(p)))
    table(out/'training_cost_summary.csv',cost);table(out/'mechanism_observations.csv',mechanism)
    return dict(status='AVAILABLE_LOGS' if cost else 'NOT_AVAILABLE',cost=cost,mechanism_records=len(mechanism),sources=sources,
        limitation='No sum of overlapped stages is reported as wall time; readout usage does not imply quality benefit.')

def self_test():
    assert stats([1.,None],2)['mean'] is None
    assert stats([1.,3.],2)['mean']==2.
    a=interval({'SeamCov':.2,'SeamCov_upper':.5},'SeamCov');b=interval({'SeamCov':.3,'SeamCov_upper':.4},'SeamCov')
    assert math.isclose(a[0]-b[1],-.2) and math.isclose(a[1]-b[0],.2)
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);req=[];actual=[]
        for panel,pids,ns in [('P1',['p0','p1'],[0,1,4,5]),('P2',['seat2','seat32'],[0,1,2,3])]:
            for pid in pids:
                for n in ns:req.append(dict(request_id=f'{panel}_{pid}_{n}',panel=panel,cohort='OLD' if panel=='P1' and n<2 else 'NEW' if panel=='P1' else 'OPEN',parent_id=pid,condition_id=pid,candidate=n))
        for r in req:
            for label in LABELS[r['panel']]:
                m=dict(SeamCov=.1 if r['panel']=='P1' else None,ConflictCov_C=.2)
                if label=='B1000' and r['parent_id']=='p0':m['SeamCov']=.3
                if label=='B1000' and r['parent_id']=='p1':m['SeamCov']=.1
                actual.append(dict(r,label=label,status='PASS',metrics=m,contracts=dict(C_output_bitwise=True)))
        dump(p/'manifest.json',dict(rows=req));dump(p/'eval/per_output.json',dict(rows=actual))
        result=compute(p/'eval',p/'manifest.json',p/'out')
        row=next(r for r in result['focus_contrasts'] if r['contrast']=='B1000-A1000' and r['panel']=='P1' and r['cohort']=='ALL' and r['metric']=='SeamCov')
        assert math.isclose(row['mean'],.1) and row['positive_count']==1 and row['zero_count']==1
        assert row['expected_units']==2 and result['status']=='COMPLETE_COHORT'
    return dict(status='PASS',model_calls=0,geometry_queries=0,synthetic_only=True,checks=5)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluation');ap.add_argument('--manifest');ap.add_argument('--out');ap.add_argument('--training-logs');ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    if not args.evaluation or not args.manifest or not args.out:ap.error('--evaluation, --manifest and --out are required')
    result=compute(args.evaluation,args.manifest,args.out,args.training_logs)
    print(json.dumps({k:result[k] for k in ('status','expected_cells','observed_cells')}))
if __name__=='__main__':main()
