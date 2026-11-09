"""Reduce sealed tables/arrays only; no models, queries or geometric re-scoring."""
from __future__ import annotations
import csv,json
from collections import defaultdict
import numpy as np
from .common import *
from r2.pilots.seam_abc_v1.eval_common import L

FIELDS=['FF_illegal_affected_free_fraction','CF_illegal_affected_free_fraction','ConflictCov_C',
 'free_minimum_angle_lt5_fraction','free_q_gt10_fraction','free_face_denominator','deleted_free_faces',
 'free_total_surface_area','hinge_diag_isolated_1face_unfiltered_edge_components',
 'hinge_diag_isolated_1face_edge_components','C_F_valid_edge_count','opposite_winding_reached_free_area_fraction']

def load_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    dest=OUT/'candidate_evaluation/analysis'
    met=read(dest/'metrics_complete.json');verify(met['files']);verify(met['protected'])
    rows=load_csv(dest/'native_full177_per_output.csv');assert len(rows)==60
    summaries=[]
    for rep in ('ORIGINAL','EDGE_FLIP'):
      for stage in ('RAW','CLEAN_FINAL_POST'):
       for label in ('EXPOSURE','PLAIN200','FLIP200'):
        selected=[r for r in rows if r['representation']==rep and r['stage']==stage and r['label']==label]
        by=defaultdict(list)
        for r in selected:by[r['parent_id']].append(r)
        result=dict(representation=rep,stage=stage,label=label,outputs=len(selected),parents=len(by),aggregation='within_parent_mean_then_equal_parent_mean')
        for field in FIELDS:
            result[field]=float(np.mean([np.mean([float(r[field]) for r in rr]) for rr in by.values()]))
        result['outputs_with_legal_CF']=sum(float(r['C_F_valid_edge_count'])>0 for r in selected)
        summaries.append(result)
    cells=read(OUT/'candidate_evaluation/tasks.json')+read(OUT/'candidate_evaluation/baseline.json')
    by={(t['label'],t['representation'],t['q']['request_id']):t for t in cells};responses=[]
    for label,rep,rid in sorted(by):
      if rep!='EDGE_FLIP':continue
      aa=by[label,'ORIGINAL',rid];bb=by[label,rep,rid];k=int(aa['q']['K'])
      a=arrays(aa['raw_path'])['raw'][k:].astype(np.float64);b=arrays(bb['raw_path'])['raw'][k:].astype(np.float64)
      dd=np.linalg.norm(b-a,axis=-1);assert a.shape==b.shape
      responses.append(dict(label=label,request_id=rid,parent=aa['q']['parent_id'],noise=aa['q']['noise_index'],
        N=aa['q']['N'],K=k,mean_corner_displacement_L=float(dd.mean()/L),RMS_corner_displacement_L=float(np.sqrt((dd**2).mean())/L),
        max_corner_displacement_L=float(dd.max()/L),interpretation='Same saved slots/noise, response magnitude only, not a quality or unique-GT reconstruction metric'))
    csvwrite(dest/'report_parent_balanced_main_tables.csv',summaries)
    csvwrite(dest/'same_physical_C_response.csv',responses)
    write(dest/'report_summary.json',dict(status='COMPLETE',primary=summaries,response=responses,
        source_metrics_sha256=sha(dest/'native_full177_per_output.csv'),new_model_calls=0,new_scores=0))
    print(json.dumps(dict(primary=[r for r in summaries if r['stage']=='CLEAN_FINAL_POST'],response=responses)))

if __name__=='__main__':main()
