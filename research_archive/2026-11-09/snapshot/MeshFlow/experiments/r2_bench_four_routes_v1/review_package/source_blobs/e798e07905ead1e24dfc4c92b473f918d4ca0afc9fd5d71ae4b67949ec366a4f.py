"""Small derived report values from frozen completed CSVs; no geometry/scorer."""
import csv,json,math,statistics
from collections import Counter
from . import common as C

def rows(name,where='results'):
    with (C.OUT/where/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def number(value):
    try:return float(value)
    except (ValueError,TypeError):return None
def mean(rs,key):
    values=[number(r.get(key)) for r in rs];values=[x for x in values if x is not None and math.isfinite(x)]
    return dict(mean=statistics.mean(values) if values else None,n=len(values))
def run():
    pr=rows('per_parent.csv');out=rows('per_output.csv');sel=rows('selection_per_output.csv');ar=rows('area_deletion_per_output.csv');led=rows('connectivity_per_output.csv');tim=rows('phase_runtime.csv')
    keys=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','CF_illegal_pair_count','FF_illegal_pair_count','C_F_valid_edge_count','C_F_valid_edge_total_length','C_F_has_valid_edge','actual_free_faces','deleted_free_faces','free_minimum_angle_lt5_fraction','free_q_gt10_fraction','nonmanifold_edges','hinge_diag_isolated_1face_edge_components','hinge_diag_legal_opposed_edge_C_reached_free_area_fraction']
    result={'scope':'Derived arithmetic only; frozen metric/visual rows unchanged','by_stage':{},'selection':{},'connectivity':{},'runtime':{}}
    for stage in ('BASE_POST','CLEAN_FINAL_POST'):
        result['by_stage'][stage]={}
        for label in ('M','HYBRID'):
            rr=[r for r in pr if r['stage']==stage and r['label']==label];oo=[r for r in out if r['stage']==stage and r['label']==label]
            aa=[r for r in ar if r['stage']==stage and r['label']==label]
            result['by_stage'][stage][label]={'parent_count':len(rr),'means':{k:mean(rr,k) for k in keys},
                'I_status':dict(Counter(r['I_status'] for r in oo)),
                'C_preserved_exact':dict(Counter(r['C_preserved_exact'] for r in oo)),
                'sum_CF_pairs':sum(float(r['CF_illegal_pair_count']) for r in oo),'sum_FF_pairs':sum(float(r['FF_illegal_pair_count']) for r in oo),
                'requests_with_interface':sum(float(r['C_F_valid_edge_count'])>0 for r in oo),
                'area_means':{k:mean(aa,k) for k in ('deleted_RAW_area_fraction','actual_area_over_RAW','deleted_original_free_faces','retained_original_free_faces','added_bridge_faces')}}
        by={(r['label'],r['request_id']):r for r in out if r['stage']==stage}
        matched=[rid for label,rid in by if label=='M' and by['M',rid]['I_status']=='PASS' and by['HYBRID',rid]['I_status']=='PASS']
        ip={}
        for group in ('ALL',*sorted({by['M',rid]['report_group'] for rid in matched})):
            ids=[rid for rid in matched if group=='ALL' or by['M',rid]['report_group']==group]
            parents=sorted({by['M',rid]['parent_id'] for rid in ids});vals={}
            for label in ('M','HYBRID'):
                pm=[statistics.mean(float(by[label,rid]['ConflictCov_C']) for rid in ids if by[label,rid]['parent_id']==parent) for parent in parents]
                vals[label]=statistics.mean(pm) if pm else None
            ip[group]={'requests':len(ids),'parents':len(parents),'parent_ids':parents,**vals}
        result['by_stage'][stage]['I_matched_PASS']=ip
        rr=[r for r in led if r['stage']==stage]
        result['connectivity'][stage]={k:sum(float(r[k]) for r in rr) for k in rr[0] if k.startswith(('CF_','FF_','original_faces_'))}
    result['selection']={'all_free_RAW_requests':sum(r['all_free_selected_RAW']=='True' for r in sel),
        **{k:mean(sel,k) for k in ('seed_free_face_fraction','selected_RAW_free_face_fraction','selected_RAW_source_area_fraction')},
        'RAW_D0_boundary_clusters':sum(int(r['RAW_D0_boundary_clusters']) for r in sel),
        'RAW_D0_noncoincident_clusters':sum(int(r['RAW_D0_noncoincident_clusters']) for r in sel),
        'C_bytes_all_preserved':all(r['C_bytes_preserved']=='True' for r in sel)}
    rt={r['request_id']:float(r['request_total_seconds']) for r in tim}
    adjusted={r['request_id']:float(r['request_total_seconds'])-float(r['request_explicit_pause_lower_seconds']) for r in tim}
    result['runtime']={'geometry_collect':C.read(C.OUT/'completions/geometry_scores.json')['seconds'],
        'original_pool_workers':1,'recovery_active_geometry_peak':2,'request_wall_median':statistics.median(rt.values()),
        'request_wall_max':max(rt.values()),'request_wall_excluding_explicit_pause_median':statistics.median(adjusted.values()),
        'request_wall_excluding_explicit_pause_max':max(adjusted.values()),
        'slow_phases':sorted([{k:r[k] for k in ('request_id','stage','stage_seconds','stage_wall_excluding_explicit_pause_upper','requested_N','stage_output_total_faces','candidates','accepted','candidate_full_geometry_scans','new_face_pair_scans','location_attempts')} for r in tim],key=lambda r:float(r['stage_seconds']),reverse=True)[:9]}
    visual=rows('counts.csv','visual_review');pairs=rows('paired_labels.csv','visual_review')
    result['visual_all']=[r for r in visual if r['report_group']=='ALL']
    result['joint_pair_fates']=dict(Counter(r['joint_fate'] for r in pairs if r['reference']=='M_D0'))
    C.write(C.OUT/'audits/report_values_v2.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
if __name__=='__main__':run()
