"""Parent-equal descriptive summaries; no quality veto or automatic promotion."""
from collections import defaultdict
import numpy as np
from .io import write_json,write_csv,resolve
KEYS=('SeamCov','SeamCov_upper','ConflictCov_C','ConflictCov_C_upper','ConflictCov_C_cross','ConflictCov_C_dup',
      'ExposedRetain','ExposedRetain_upper','ExposedRetain_global_lower','ExposedRetain_global_upper',
      'GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','D_sym2048','fixed_query_MSE','fixed_query_p95_L','coverage_fraction',
      'CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction','CF_proper_pair_count','CF_coplanar_pair_count',
      'FF_illegal_pair_count','free_minimum_angle_lt5_fraction','free_near_degenerate_fraction','free_exact_duplicate_excess_fraction',
      'EXACT_COORD_CF_shared_edges','EXACT_COORD_face_components')
def number(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x)
def delta_interval(a,b,key):
    upper={'SeamCov':'SeamCov_upper','ConflictCov_C':'ConflictCov_C_upper','ExposedRetain':'ExposedRetain_upper'}.get(key)
    if upper and all(number(v) for v in (a.get(key),a.get(upper),b.get(key),b.get(upper))):
        return float(a[key]-b[upper]),float(a[upper]-b[key])
    return None,None
def scopes(r):return [r['cohort'],'ALL'] if r['panel']=='P1' else ['OPEN']
def aggregate(rows,out):
    valid=[r for r in rows if 'metrics' in r]; groups=defaultdict(list);parents=[];summaries=[]
    for r in valid:
        for scope in scopes(r):groups[(r['label'],r['panel'],scope,r['parent_id'])].append(r)
    for (label,panel,scope,pid),rr in sorted(groups.items()):
        for k in KEYS:
            values=[r['metrics'].get(k) for r in rr];good=[float(v) for v in values if number(v)]
            parents.append(dict(label=label,panel=panel,cohort=scope,parent_id=pid,metric=k,requested=len(rr),finite=len(good),
                value=float(np.mean(good)) if len(good)==len(rr) and good else None,
                status='PASS' if len(good)==len(rr) and good else 'NA_OR_INCOMPLETE'))
    for key in sorted({(r['label'],r['panel'],r['cohort'],r['metric']) for r in parents}):
        rr=[r for r in parents if (r['label'],r['panel'],r['cohort'],r['metric'])==key]
        vals=[r['value'] for r in rr if number(r['value'])]
        summaries.append(dict(zip(('label','panel','cohort','metric'),key),value=float(np.mean(vals)) if len(vals)==len(rr) and vals else None,
            parent_count=len(rr),finite_parents=len(vals),status='PASS' if len(vals)==len(rr) and vals else 'NA_OR_INCOMPLETE',
            aggregation='noise mean within parent then equal parent weights; missing values not silently dropped'))
    lookup={(r['label'],r['request_id']):r for r in valid}
    contrasts=[(f'{x}{step}',f'{y}{step}') for step in (500,1000) for x,y in (('B','A'),('C','A'),('C','B'))]
    contrasts += [(f'{x}{step}','START') for step in (500,1000) for x in 'ABC']
    noise=[];paired=[]
    for candidate,reference in contrasts:
        matched=[(r,lookup[(reference,r['request_id'])]) for r in valid if r['label']==candidate and (reference,r['request_id']) in lookup]
        for a,b in matched:
            for k in KEYS:
                va,vb=a['metrics'].get(k),b['metrics'].get(k)
                dlo,dhi=delta_interval(a['metrics'],b['metrics'],k)
                noise.append(dict(contrast=candidate+'-'+reference,panel=a['panel'],cohort=a['cohort'],parent_id=a['parent_id'],
                    condition_id=a['condition_id'],request_id=a['request_id'],candidate=a['candidate'],metric=k,
                    candidate_value=va,reference_value=vb,delta=float(va-vb) if number(va) and number(vb) else None,delta_lower=dlo,delta_upper=dhi,
                    interpretation='descriptive paired difference; reference reconstruction is not control-success gate'))
        for panel in sorted({r['panel'] for r,_ in matched}):
            for scope in (('OLD','NEW','ALL') if panel=='P1' else ('OPEN',)):
                rr=[r for r in noise if r['contrast']==candidate+'-'+reference and r['panel']==panel and (scope=='ALL' or r['cohort']==scope)]
                for k in KEYS:
                    kk=[r for r in rr if r['metric']==k];pvals=[]
                    for pid in sorted({r['parent_id'] for r in kk}):
                        pr=[r for r in kk if r['parent_id']==pid];d=[r['delta'] for r in pr]
                        if d and all(number(v) for v in d):pvals.append(float(np.mean(d)))
                        paired.append(dict(contrast=candidate+'-'+reference,panel=panel,cohort=scope,parent_id=pid,metric=k,
                            delta=float(np.mean(d)) if d and all(number(v) for v in d) else None,noise_count=len(d),
                            delta_lower=float(np.mean([r['delta_lower'] for r in pr])) if pr and all(number(r['delta_lower']) for r in pr) else None,
                            delta_upper=float(np.mean([r['delta_upper'] for r in pr])) if pr and all(number(r['delta_upper']) for r in pr) else None))
    write_csv(resolve(out)/'per_parent_metrics.csv',parents);write_csv(resolve(out)/'checkpoint_profiles.csv',summaries)
    write_csv(resolve(out)/'paired_noise_metrics.csv',noise);write_csv(resolve(out)/'paired_parent_metrics.csv',paired)
    result=dict(status='DESCRIPTIVE_AVAILABLE_ROWS',profiles=summaries,quality_gate='NOT_DEFINED_NO_SOURCE_RECONSTRUCTION_VETO',
        human_usability='NOT_RUN',source_scope='P1 observed DEV parents; P2 four frozen C, seat2/seat32 same geometric surface')
    write_json(resolve(out)/'aggregate.json',result);return dict(parent_rows=len(parents),paired_noise_rows=len(noise),paired_parent_rows=len(paired))
