"""Extract bounded report evidence from completed tables; never score or view."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'experiments/r2_bench_group_weight_match_v1'
METRICS = {
    'CF': 'CF_illegal_affected_free_fraction',
    'FF': 'FF_illegal_affected_free_fraction',
    'I_PASS': 'ConflictCov_C',
    'angle_lt5': 'free_minimum_angle_lt5_fraction',
}
STAGES = ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return statistics.fmean(values) if values else None


def main():
    paths = [OUT/'results/metric_rows.json', OUT/'results/completion.json',
             OUT/'results/analysis_v1/paired_primary_parent_macro.csv',
             OUT/'results/analysis_v1/deletion_area_per_output.csv']
    data, complete = read(paths[0]), read(paths[1])
    assert complete['status'] == 'COMPLETE' and len(data) == 414 and len(complete['metric_fields']) == 177
    by = {(r['label'], r['stage'], r['request_id']):r for r in data}
    assert len(by) == len(data)
    paired = csvrows(paths[2]); areas = csvrows(paths[3])
    area_by = {(r['label'], r['stage'], r['request_id']):r for r in areas}
    groups = sorted({r['report_group'] for r in data})
    attribution = []
    for stage in ('RAW', 'CLEAN_FINAL_POST'):
        refs = [r for r in data if r['label']=='W' and r['stage']==stage and r['report_group']=='OLD']
        parents = {r['parent_id'] for r in refs}
        assert len(parents)==8 and len(refs)==16
        requests, by_parent = [], defaultdict(list)
        for w in refs:
            m = by['M', stage, w['request_id']]
            assert w['ConflictCov_C_status']==m['ConflictCov_C_status']=='PASS'
            delta = 100*(m['ConflictCov_C']-w['ConflictCov_C'])
            row = dict(request_id=w['request_id'], parent_id=w['parent_id'], noise_index=w['noise_index'],
                       W_I_pct=100*w['ConflictCov_C'], M_I_pct=100*m['ConflictCov_C'], delta_pp=delta,
                       contribution_to_OLD_parent_macro_delta_pp=delta/16)
            requests.append(row); by_parent[w['parent_id']].append(row)
        parent_rows = []
        for parent, rows in by_parent.items():
            assert len(rows)==2
            delta = mean([r['delta_pp'] for r in rows])
            parent_rows.append(dict(parent_id=parent, W_I_pct=mean([r['W_I_pct'] for r in rows]),
                M_I_pct=mean([r['M_I_pct'] for r in rows]), delta_pp=delta,
                contribution_to_OLD_parent_macro_delta_pp=delta/8))
        parent_rows.sort(key=lambda r:r['contribution_to_OLD_parent_macro_delta_pp'], reverse=True)
        requests.sort(key=lambda r:r['contribution_to_OLD_parent_macro_delta_pp'], reverse=True)
        net=sum(r['contribution_to_OLD_parent_macro_delta_pp'] for r in parent_rows)
        increase=sum(max(0,r['contribution_to_OLD_parent_macro_delta_pp']) for r in parent_rows)
        attribution.append(dict(stage=stage, OLD_net_delta_pp=net,
            top_parent_share_of_net_increase=parent_rows[0]['contribution_to_OLD_parent_macro_delta_pp']/net,
            top_parent_share_of_positive_contributions=parent_rows[0]['contribution_to_OLD_parent_macro_delta_pp']/increase,
            parents=parent_rows, requests=requests))
    new_parents = sorted({r['parent_id'] for r in data if r['report_group']=='NEW_WIDE_TRAIN'})
    new_rows = []
    for parent in new_parents:
        for stage in STAGES:
            refs=sorted([r for r in data if r['label']=='W' and r['stage']==stage and r['parent_id']==parent], key=lambda r:r['noise_index'])
            assert len(refs)==2
            noises=[]
            for w in refs:
                m=by['M',stage,w['request_id']]
                row=dict(request_id=w['request_id'],noise_index=w['noise_index'],
                    W_FF_pct=100*w[METRICS['FF']],M_FF_pct=100*m[METRICS['FF']],
                    FF_delta_pp=100*(m[METRICS['FF']]-w[METRICS['FF']]))
                for label, score in (('W',w),('M',m)):
                    raw=by[label,'RAW',w['request_id']]
                    row[label+'_actual_free_faces']=score['actual_free_faces']
                    row[label+'_actual_free_area']=score['free_total_surface_area']
                    row[label+'_I_status']=score['ConflictCov_C_status']
                    if stage=='RAW':
                        row[label+'_retained_original_slots']=score['N']-score['K']
                        row[label+'_deleted_original_slots']=0
                        row[label+'_deleted_RAW_reference_area']=0.
                        row[label+'_deleted_RAW_reference_area_fraction']=0.
                        row[label+'_final_over_RAW_area']=1.
                        row[label+'_added_bridge_faces']=0
                    else:
                        area=area_by[label,stage,w['request_id']]
                        for source,target in (('retained_original_free_faces','retained_original_slots'),
                                              ('deleted_original_free_faces','deleted_original_slots'),
                                              ('deleted_raw_reference_area','deleted_RAW_reference_area'),
                                              ('deleted_raw_area_fraction','deleted_RAW_reference_area_fraction'),
                                              ('added_bridge_faces','added_bridge_faces')):
                            row[label+'_'+target]=float(area[source])
                        row[label+'_final_over_RAW_area']=float(area['actual_final_free_area'])/float(area['raw_free_area'])
                noises.append(row)
            averaged={k:mean([float(r[k]) for r in noises]) for k,v in noises[0].items()
                      if isinstance(v,(float,int)) and k!='noise_index'}
            new_rows.append(dict(parent_id=parent,N=refs[0]['N'],K=refs[0]['K'],closed_C=refs[0]['closed_C'],
                stage=stage,averages_over_two_registered_noises=averaged,noises=noises))
    count_rows, parent_vectors = [], {}
    for row in paired:
        if (row['reference'],row['candidate'])!=('W','M') or row['metric'] not in METRICS.values():
            continue
        short=next(k for k,v in METRICS.items() if v==row['metric'])
        deltas=json.loads(row['parent_deltas'])
        group=row['report_group'];stage=row['stage']
        count_rows.append(dict(group=group,stage=stage,metric=short,
            improved=sum(v<0 for v in deltas.values()),equal=sum(v==0 for v in deltas.values()),
            worsened=sum(v>0 for v in deltas.values()),valid_parents=int(row['valid_parents']),
            expected_parents=int(row['expected_parents']),NA_parents=int(row['expected_parents'])-int(row['valid_parents']),
            parent_delta_pp={p:100*v for p,v in deltas.items()}))
        for parent,delta in deltas.items():
            parent_vectors.setdefault((group,stage,parent),{})[short]=delta
    vector_rows=[]
    for (group,stage,parent),ds in sorted(parent_vectors.items()):
        down,up=sum(v<0 for v in ds.values()),sum(v>0 for v in ds.values())
        vector_rows.append(dict(group=group,stage=stage,parent_id=parent,
            available_metrics=list(ds),missing_metrics=sorted(set(METRICS)-set(ds)),
            improved_fields=down,worsened_fields=up,equal_fields=len(ds)-down-up,
            pattern='MIXED' if down and up else 'NONWORSE_WITH_DECREASE' if down else 'NONBETTER_WITH_INCREASE' if up else 'ALL_EQUAL',
            all_four_metrics_available=len(ds)==4))
    stage_totals=[]
    for stage in STAGES:
        for metric in METRICS:
            rr=[r for r in count_rows if r['stage']==stage and r['metric']==metric]
            stage_totals.append(dict(stage=stage,metric=metric,**{k:sum(r[k] for r in rr)
                for k in ('improved','equal','worsened','valid_parents','expected_parents','NA_parents')}))
    scientific_NA=dict(
        I_display='ConflictCov_C main display requires PASS. LOW_SUPPORT is retained but excluded; not replaced with0.',
        I_LOW_SUPPORT_parent_ids=sorted({r['parent_id'] for r in data if r['ConflictCov_C_status']=='LOW_SUPPORT'}),
        Native_seam='All414 rows NA_NOT_SEAM_REPRESENTATION; no seam-to-bulk success criterion.',
        closed_C='Closed C does not imply a required open interface. C/F graph diagnostics are descriptive; do not force connection success or use empty boundary as success.',
        source_GT='No unique-source-style success requirement; RAW-slot deletion area is not source-GT surface loss.',
        visual='No new visual review or visual inference in this extraction. Use separately frozen W/M V2 observations.',
        gradient='Group gradient shares are not logged and remain UNKNOWN; scalar coefficients/loss shares are not gradient shares.')
    result=dict(schema='BENCH_GROUP_WEIGHT_MATCH_REPORT_NUMERIC_FOCUS_V1',status='PASS',
        primary='M-W',secondary_reference='B only in existing complete tables',
        old_I_attribution=attribution,new_wide_train_by_parent=new_rows,
        parent_metric_counts=count_rows,all_group_parent_metric_counts=stage_totals,
        parent_available_metric_patterns=vector_rows,
        count_definition='Sign of exact existing common-support parent mean delta for four frozen lower-is-better metrics. No statistical significance threshold; these are descriptive geometry counts, not joint-success or C-adoption labels. NA excludes a metric, never zero-fills it.',
        area_definition='Actual area is model^2; original generated RAW slot retained/deleted provenance. FINAL/RAW ratios include displacement and added bridges, not complete-source retention. Each displayed parent has exactly two frozen noises; request details retained.',
        scientific_NA=scientific_NA,
        evidence={str(p.relative_to(ROOT)):digest(p) for p in paths},code_sha256=digest(Path(__file__)),
        model_calls=0,scoring_calls=0,visual_calls=0)
    target=OUT/'audits/report_numeric_focus.json'
    raw=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if target.exists():
        assert target.read_text(encoding='utf-8')==raw,'Refusing changed report evidence'
    else:
        with target.open('x',encoding='utf-8') as f:f.write(raw)
    print(json.dumps(dict(status='PASS',path=str(target),old_I_top_parents=[dict(stage=r['stage'],net_delta_pp=r['OLD_net_delta_pp'],top_parent=r['parents'][0]) for r in attribution],
                          all_group_parent_metric_counts=stage_totals),indent=2))


if __name__=='__main__':main()
