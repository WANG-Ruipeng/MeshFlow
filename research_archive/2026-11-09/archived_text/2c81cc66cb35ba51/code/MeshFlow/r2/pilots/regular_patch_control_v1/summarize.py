"""Deterministic aggregation of numerical results and actual saved AI_PROXY reviews."""
import csv,collections,statistics
from . import common as C
EV=C.OUT/'evaluation'
def rows(p):return list(csv.DictReader(C.resolve(p).open(encoding='utf-8-sig')))
def main(version='v1'):
    reg=C.read(C.OUT/'registration.json');numeric=rows(EV/'per_output.csv');suffix='_v2' if version=='v2' else '';result_root=C.OUT/('results'+suffix)
    review=rows(EV/('gallery/review_train'+suffix+'.csv'))+rows(EV/('gallery/review_dev'+suffix+'.csv'))
    key=lambda x:(x['request_id'],x['label'],x['stage'])
    if len(review)!=96 or len(set(map(key,review)))!=96:raise ValueError('96 actual unique reviews required')
    for x in review:
        if x['reviewed_image_sha256']!=C.sha(x['image_path']) or x['C_adoption'] not in ['YES','NO','UNCERTAIN']:raise ValueError('Unreviewed image')
    C.csvwrite(EV/('gallery/AI_PROXY_reviews'+suffix+'.csv'),review)
    nv={key(x):x for x in numeric};rv={key(x):x for x in review};old=rows(C.OUT/'historical_context/historical_actual_reviews.csv')
    metrics=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','ConflictCov_C_dup','ConflictCov_C_cross','free_minimum_angle_lt5_fraction','opposite_winding_reached_free_area_fraction','A_VMR__L_tau_0.001','PMR__L_tau_0.001']
    summary=[];parent=[]
    for arm in reg['labels']:
      for stage in ['RAW','POST']:
        for uid in sorted(set(x['parent_id'] for x in numeric))+['ALL','NEW_ONLY','CACHE_ONLY','RECTANGLE_LIKE']:
            a=[x for x in numeric if x['label']==arm and x['stage']==stage and (uid=='ALL' or uid=='NEW_ONLY' and x['input_action']=='NEW' or uid=='CACHE_ONLY' and x['input_action']=='CACHE' or uid=='RECTANGLE_LIKE' and not x['parent_id'].startswith('3f') or x['parent_id']==uid)]
            rs=[rv[key(x)] for x in a];d=dict(group=uid,label=arm,stage=stage,outputs=len(a),valid_interface_outputs=sum(int(x['C_F_valid_edge_count'])>0 for x in a),valid_interface_edges=sum(int(x['C_F_valid_edge_count']) for x in a),opposite_interface_edges=sum(int(x['C_F_valid_opposite_winding_edge_count']) for x in a))
            for name in ['C_adoption','whole_body_complete','visible_join']:
                for v in ['YES','NO','UNCERTAIN']:d[name+'_'+v]=sum(x[name]==v for x in rs)
            for m in metrics:d[m+'_mean']=statistics.mean(float(x[m]) for x in a)
            summary.append(d)
    C.csvwrite(result_root/'summary.csv',summary)
    pairs=[]
    for q in reg['requests']:
      for stage in ['RAW','POST']:
        a,b=[rv[q['request_id'],arm,stage] for arm in reg['labels']]
        pairs.append(dict(parent_id=q['parent_id'],candidate=q['candidate'],stage=stage,request_id=q['request_id'],input_action=q['action'],K=q['K'],SMALL_adoption=a['C_adoption'],CREF_adoption=b['C_adoption'],SMALL_whole=a['whole_body_complete'],CREF_whole=b['whole_body_complete'],SMALL_notes=a['notes'],CREF_notes=b['notes']))
    C.csvwrite(result_root/'paired_visual.csv',pairs)
    C.write(result_root/'summary.json',dict(status='COMPLETE',numerical_outputs=96,actual_review_rows=96,pairs=48,rows=summary,review_policy='Nonblind AI_PROXY; no unique GT style requirement; visible contact does not certify indexed topology.'))
    print({'status':'COMPLETE','actual_reviews':96,'summary_rows':len(summary)})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--version',choices=['v1','v2'],default='v1');main(p.parse_args().version)