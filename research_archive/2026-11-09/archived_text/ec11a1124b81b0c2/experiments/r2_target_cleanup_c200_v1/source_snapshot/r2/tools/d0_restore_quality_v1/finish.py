"""Post-score table recovery: no scorer/model/geometry rerun.

Original frozen run.py aggregate had duplicate request_id kwargs. Preserve it
and its completed scientific cells; this independent aggregation fixes merging.
"""
import csv,collections,json
from pathlib import Path
import numpy as np
from . import run as R
from r2.pilots.seam_slot_c200_v1 import common as C


def main():
    r=R.registration();C.read(R.OUT/'score_completion.json');tables=collections.defaultdict(list);fits=[];refs=[]
    C.write(R.OUT/'aggregation_recovery.json',dict(status='RECOVERED_TABLE_ONLY',error='TypeError duplicate request_id in dict(request_id=..., **row)',
        stage='Aggregate after all16 scientific cells and score_completion saved',frozen_code_preserved=True,new_scoring=0,model_calls=0,
        correction='Use mapping merge with explicit existing-row identity check',source_sha256=C.sha(__file__)))
    for q in r['requests']:
        d=R.OUT/'outputs'/q['request_id'];fits.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],**C.read(d/'rigid_fit.json')))
        for n in ('stage_metrics','pair_transitions','pair_changes','role_pairs','C_copy_groups'):
            with (d/(n+'.csv')).open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):
                    if row.get('request_id',q['request_id'])!=q['request_id']:raise ValueError('Wrong row identity')
                    tables[n].append({'request_id':q['request_id'],**row})
        for stage in ('RAW','POST'):
            p=R.paths(q)[stage.lower()+'_score'];met=C.read(p)['metrics'];refs.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],stage=stage,
                CF=met['CF_illegal_affected_free_fraction'],FF=met['FF_illegal_affected_free_fraction'],I=met['ConflictCov_C'],source_score=str(p)))
    for n,rows in tables.items():C.csvwrite(R.OUT/(n+'.csv'),rows)
    C.csvwrite(R.OUT/'rigid_fit_per_output.csv',fits);C.csvwrite(R.OUT/'reference_RAW_POST.csv',refs)
    values=tables['stage_metrics'];summary=[];parents=[]
    nums=('CF','FF','CF_pairs','FF_pairs','bulk_bulk_pairs','bulk_bulk_affected_faces','I_against_fixed_original_C','exact_coordinate_components','exact_coordinate_CF_edges')
    for stage in R.STAGES:
        rr=[x for x in values if x['stage']==stage]
        summary.append(dict(stage=stage,outputs=len(rr),**{n:float(np.mean([float(x[n]) for x in rr])) for n in nums},
            FF_pairs_sum=sum(int(x['FF_pairs']) for x in rr),CF_pairs_sum=sum(int(x['CF_pairs']) for x in rr),bulk_bulk_pairs_sum=sum(int(x['bulk_bulk_pairs']) for x in rr)))
        for pid in sorted({x['parent_id'] for x in rr}):
            pp=[x for x in rr if x['parent_id']==pid];parents.append(dict(parent_id=pid,stage=stage,outputs=len(pp),**{n:float(np.mean([float(x[n]) for x in pp])) for n in nums}))
    C.csvwrite(R.OUT/'summary.csv',summary);C.csvwrite(R.OUT/'per_parent.csv',parents)
    transitions=[]
    for after in ('RIGID','HARD_RESTORE'):
        for kind in ('CF','FF'):
            rr=[x for x in tables['pair_transitions'] if x['after']==after and x['kind']==kind]
            transitions.append(dict(after=after,kind=kind,**{n:sum(int(x[n]) for x in rr) for n in ('original_pairs','final_pairs','retained','resolved','new','changed_classification')}))
    result=dict(status='COMPLETE',summary=summary,transitions=transitions,
        max_rigid_distance_invariance_error=max(x['all_mesh_pairwise_distance_invariance_max_error'] for x in fits),
        mean_C_RMS_before=float(np.mean([x['C_weighted_RMS_before'] for x in fits])),mean_C_RMS_after=float(np.mean([x['C_weighted_RMS_after'] for x in fits])),
        max_C_residual=max(x['C_max_error_after'] for x in fits),split_C_groups=sum(x['split_groups'] for x in fits),
        duplicate_C_groups=sum(x['duplicate_groups'] for x in fits),max_C_copy_split=max(x['max_copy_split'] for x in fits))
    C.write(R.OUT/'summary.json',result);print(json.dumps(result,indent=2));return result

if __name__=='__main__':main()
