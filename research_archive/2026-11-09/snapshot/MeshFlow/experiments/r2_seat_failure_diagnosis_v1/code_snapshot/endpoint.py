"""Secondary descriptive endpoint sensitivity; no inferred success threshold."""
import json
import numpy as np
from . import common as E

def run():
    r=E.load_registration();idx=E.read(E.OUT/'evaluation/raw_index.json');byid={x['request_id']:x for x in idx['rows']}
    bases={(q['split'],q['base_condition_id'],q['candidate']):q for q in r['requests'] if q['variant']=='base'}
    definition=dict(status='SECONDARY_DESCRIPTIVE_ANALYSIS',registered_after_generation=True,used_for_selection=False,
        quantity='RMS of corresponding free corner coordinates, RAW sub4 minus RAW base, divided by inherited L',
        same_Gaussian_required=True,free_slots_correspond=True,POST_excluded='GEO_C may delete/reorder free faces',success_threshold=None,
        inference_limit='Changing K also changes N and face boundaries; not an isolated estimate of attention dilution')
    E.write(E.OUT/'statistics/endpoint_definition.json',definition);out=[]
    for q in r['requests']:
        if q['variant']!='sub4':continue
        b=bases[q['split'],q['base_condition_id'],q['candidate']]
        aa=E.arrays(byid[q['request_id']]['raw_path']);bb=E.arrays(byid[b['request_id']]['raw_path'])
        a=np.asarray(aa['raw']).reshape(q['N'],3,3)[q['K']:];z=np.asarray(bb['raw']).reshape(b['N'],3,3)[b['K']:]
        assert a.shape==z.shape and q['free_Gaussian_sha256']==b['free_Gaussian_sha256']
        delta=a.astype(float)-z.astype(float);dist=np.linalg.norm(delta,axis=-1)
        out.append(dict(pair_id=q['pair_id'],split=q['split'],condition_type=q['condition_type'],candidate=q['candidate'],base_request_id=b['request_id'],sub4_request_id=q['request_id'],base_N=b['N'],sub4_N=q['N'],base_K=b['K'],sub4_K=q['K'],free_faces=len(a),RMS_coordinate_delta_over_L=float(np.sqrt(np.mean(delta**2))/E.L),mean_corner_displacement_over_L=float(dist.mean()/E.L),p95_corner_displacement_over_L=float(np.quantile(dist,.95)/E.L),max_corner_displacement_over_L=float(dist.max()/E.L),exact_equal=E.same(a,z),base_raw_sha256=byid[b['request_id']]['raw_sha256'],sub4_raw_sha256=byid[q['request_id']]['raw_sha256']))
    E.csvwrite(E.OUT/'statistics/sub4_RAW_endpoint_sensitivity.csv',out)
    summary=dict(status='COMPLETE',pairs=len(out),nonzero_pairs=sum(not q['exact_equal'] for q in out),RMS_coordinate_delta_over_L_mean=float(np.mean([q['RMS_coordinate_delta_over_L'] for q in out])),RMS_coordinate_delta_over_L_min=float(min(q['RMS_coordinate_delta_over_L'] for q in out)),RMS_coordinate_delta_over_L_max=float(max(q['RMS_coordinate_delta_over_L'] for q in out)),model_calls=0)
    E.write(E.OUT/'statistics/endpoint_summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':run()