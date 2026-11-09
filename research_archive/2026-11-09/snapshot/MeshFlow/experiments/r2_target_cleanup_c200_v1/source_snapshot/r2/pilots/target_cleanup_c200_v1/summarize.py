"""CPU-only full paired summaries; never selects checkpoints or hides costs."""
from collections import defaultdict
import json
import numpy as np
from . import common as C

METRICS={
 'CF_pct':'CF_illegal_affected_free_fraction',
 'FF_pct':'FF_illegal_affected_free_fraction',
 'I_pct':'ConflictCov_C',
 'free_lt5_pct':'free_minimum_angle_lt5_fraction',
 'free_q_gt10_pct':'free_q_gt10_fraction',
 'oriented_reached_free_area_pct':'opposite_winding_reached_free_area_fraction',
 'AVMR_005_pct':'A_VMR__L_tau_0.005',
 'valid_CF_edge_count':'C_F_valid_edge_count',
 'valid_CF_edge_length':'C_F_valid_edge_total_length',
 'actual_free_faces':'actual_free_faces',
 'deleted_free_faces':'deleted_free_faces',
 'exact_face_components':'EXACT_COORD_face_components',
 'nonmanifold_edges':'nonmanifold_edges',
 'finite_boundary_to_surface_pctL':'DESC__C_boundary_to_F_surface_mean_model',
 'finite_boundary_to_edges_pctL':'DESC__C_boundary_to_F_all_finite_edges_mean_model',
}

def scale(name):
    return 100/C.L if name.endswith('pctL') else 100 if name.endswith('pct') else 1

def main():
    rows=C.csvread(C.OUT/'evaluation/tables/per_output.csv');assert len(rows)==144
    summaries=[];parents=[];paired=[]
    for label in ('FULL_S6000',)+C.ARMS:
        for stage in ('RAW','POST'):
            for split in ('ALL','TRAIN','DEV'):
                rs=[r for r in rows if r['label']==label and r['stage']==stage and (split=='ALL' or r['split']==split)]
                d=dict(label=label,stage=stage,split=split,requests=len(rs),parents=len({r['parent_id'] for r in rs}),
                    has_valid_CF=sum(float(r['C_F_has_valid_edge'])>0 for r in rs),
                    oriented_reach_ge30=sum(float(r['opposite_winding_reached_free_area_fraction'])>=.3 for r in rs),
                    C_exact=sum(r['C_preserved_exact']=='True' for r in rs))
                for name,key in METRICS.items():
                    values=[float(r[key])*scale(name) for r in rs]
                    d[name]=float(np.mean(values));d[name+'__median']=float(np.median(values));d[name+'__max']=float(max(values))
                summaries.append(d)
            for parent in sorted({r['parent_id'] for r in rows}):
                rs=[r for r in rows if (r['label'],r['stage'],r['parent_id'])==(label,stage,parent)]
                d=dict(label=label,stage=stage,parent_id=parent,split=rs[0]['split'],requests=len(rs),
                    has_valid_CF=sum(float(r['C_F_has_valid_edge'])>0 for r in rs),continued_in_this_pilot=False)
                d.update({name:float(np.mean([float(r[key])*scale(name) for r in rs])) for name,key in METRICS.items()})
                parents.append(d)
    lookup={(r['label'],r['request_id'],r['stage']):r for r in rows}
    for stage in ('RAW','POST'):
        for split in ('ALL','TRAIN','DEV'):
            rs=[r for r in rows if r['label']=='DEDUP_C200' and r['stage']==stage and (split=='ALL' or r['split']==split)]
            for ref in ('ORIGINAL_C200','FULL_S6000'):
                for name,key in METRICS.items():
                    ds=np.array([(float(r[key])-float(lookup[ref,r['request_id'],stage][key]))*scale(name) for r in rs])
                    paired.append(dict(stage=stage,split=split,candidate='DEDUP_C200',reference=ref,metric=name,requests=len(rs),
                        mean_delta=float(ds.mean()),median_delta=float(np.median(ds)),minimum_delta=float(ds.min()),maximum_delta=float(ds.max()),
                        lower=int(np.count_nonzero(ds < -1e-12)),equal=int(np.count_nonzero(abs(ds)<=1e-12)),higher=int(np.count_nonzero(ds>1e-12))))
    C.csvwrite(C.OUT/'analysis/summary.csv',summaries)
    C.csvwrite(C.OUT/'analysis/per_parent.csv',parents)
    C.csvwrite(C.OUT/'analysis/paired_metric_deltas.csv',paired)
    C.write(C.OUT/'analysis/metric_notes.json',dict(rows=144,metric_source='unchanged frozen evaluator',
        percentage='fractions multiplied by100, no changed denominator',finite_distance='divide by fixedL then100',
        direction='CF/FF/I and small angles lower; interface/reach larger is descriptive, not mandatory closure',
        source_GT_ranking=False,exact_components='exact-index connected components; not identical to visually detached fragments',
        exposure='three historicalTRAIN, three historicalDEV; all six absent from current16-parent continuation pool'))
    print(json.dumps([r for r in summaries if r['stage']=='POST' and r['split']=='ALL'],indent=2))

if __name__=='__main__':main()
