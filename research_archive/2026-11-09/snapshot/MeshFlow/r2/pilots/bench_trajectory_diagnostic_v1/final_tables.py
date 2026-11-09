"""CPU-only final aggregations; no new detector, query, or model call."""
from .common import *
from collections import defaultdict
import numpy as np

def rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))

def grouped(data, keys):
    groups=defaultdict(list)
    for r in data: groups[tuple(r[k] for k in keys)].append(r)
    return groups

def main():
    d=OUT/'analysis_full/summaries'
    continuous=rows(OUT/'stage_audit/continuous_stage_pairs.csv')
    result=[]
    for key,rr in sorted(grouped(continuous,['label','split','stage']).items()):
        result.append(dict(zip(['label','split','stage'],key))|dict(outputs=len(rr),**{
            k:float(np.mean([float(r[k]) for r in rr])) for k in (
                'free_area','C_corner_RMS_displacement','CF_illegal_pairs','FF_illegal_pairs',
                'CF_affected_free_fraction','FF_affected_free_fraction')}))
    csvwrite(d/'continuous_stage_summary.csv',result)
    result=[]
    for key,rr in sorted(grouped(rows(OUT/'stage_audit/pair_transitions.csv'),['label','split','transition','kind']).items()):
        result.append(dict(zip(['label','split','transition','kind'],key))|dict(outputs=len(rr),**{
            k:sum(int(r[k]) for r in rr) for k in ('old','new','retained','removed','added')}))
    csvwrite(d/'pair_transition_summary.csv',result)
    survival=rows(OUT/'stage_audit/GEO_face_survival.csv')
    result=[]
    for key,rr in sorted(grouped(survival,['label','split','reason']).items()):
        result.append(dict(zip(['label','split','reason'],key))|dict(faces=len(rr),**{
            k:sum(float(r[k]) for r in rr) for k in ('RAW_area','D0_area','PRE_GEO_area')}))
    csvwrite(d/'GEO_reason_summary.csv',result)
    result=[]
    for key,rr in sorted(grouped(survival,['label','request_id','split','part']).items()):
        kept=[r for r in rr if r['kept']=='True'];deleted=[r for r in rr if r['kept']=='False']
        raw=sum(float(r['RAW_area']) for r in rr);d0=sum(float(r['D0_area']) for r in rr);pre=sum(float(r['PRE_GEO_area']) for r in rr)
        result.append(dict(zip(['label','request_id','split','part'],key))|dict(
            free_faces=len(rr),kept=len(kept),deleted=len(deleted),RAW_area=raw,D0_area=d0,PRE_GEO_area=pre,
            D0_over_RAW_area=d0/raw,deleted_pre_area=sum(float(r['PRE_GEO_area']) for r in deleted),
            deleted_pre_area_fraction=sum(float(r['PRE_GEO_area']) for r in deleted)/pre,
            deleted_D0_area=sum(float(r['D0_area']) for r in deleted),deleted_RAW_area=sum(float(r['RAW_area']) for r in deleted)))
    csvwrite(d/'GEO_survival_per_request.csv',result)
    aggregate=[]
    for key,rr in sorted(grouped(result,['label','split']).items()):
        aggregate.append(dict(zip(['label','split'],key))|dict(outputs=len(rr),**{
            k:float(np.mean([r[k] for r in rr])) for k in ('free_faces','kept','deleted','RAW_area','D0_area','PRE_GEO_area','D0_over_RAW_area','deleted_pre_area','deleted_pre_area_fraction','deleted_RAW_area')}))
    csvwrite(d/'GEO_survival_summary.csv',aggregate)
    allmetrics=rows(OUT/'analysis_full/per_output_all_metrics.csv')
    keys=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
        'free_minimum_angle_lt5_fraction','nonmanifold_edges','hinge_diag_isolated_1face_edge_components',
        'C_F_valid_edge_count','free_only_reached_bulk_area_fraction','actual_free_faces','deleted_free_faces']
    parents=[]
    for key,rr in sorted(grouped(allmetrics,['label','split','parent_id','part','stage']).items()):
        parents.append(dict(zip(['label','split','parent_id','part','stage'],key))|dict(outputs=len(rr),**{
            k:float(np.mean([float(r[k]) for r in rr])) for k in keys}))
    csvwrite(d/'per_parent_stage.csv',parents)
    spectra=rows(OUT/'weights/exact_selected_spectra.csv')
    csvwrite(d/'weight_spectra_compact.csv',[{k:v for k,v in r.items() if k!='singular_values_json'} for r in spectra])
    print(json.dumps(dict(continuous=rows(d/'continuous_stage_summary.csv'),pairs=rows(d/'pair_transition_summary.csv'),
        survival=aggregate,reasons=rows(d/'GEO_reason_summary.csv')),ensure_ascii=False))

if __name__=='__main__':main()
