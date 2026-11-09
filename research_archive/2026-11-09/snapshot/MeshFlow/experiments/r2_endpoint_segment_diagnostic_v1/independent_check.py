"""Independent NumPy read-only endpoint/segment check; no decoder/scorer imports."""
from pathlib import Path
from collections import defaultdict
import json, numpy as np
P=Path(__file__).resolve().parent
M=json.loads((P.parent/'r2_partial_edge_binding_v1/input_manifest.json').read_text())['rows']
L=np.sqrt(3)*2/.3762

def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def incidence(faces,start=0):
    out=defaultdict(list)
    for f in range(start,len(faces)):
        for i,j in ((0,1),(1,2),(2,0)):
            out[tuple(sorted(map(int,faces[f,[i,j]])))].append((f,i,j))
    return out

def ps(points,c):
    d=c[1]-c[0]; t=np.clip(((points-c[0])*d).sum(-1)/(d@d),0,1)
    return np.linalg.norm(points-(c[0]+t[:,None]*d),axis=-1)

result=[]
for r in M:
    if not r['short_id'].startswith(('608','610')):continue
    a=npz(r['canonical_path']);pre=npz(r['pre_geo_path'])['output'];v=a['vertices'].astype(float);f=a['faces'];k=r['K'];sf=a['source_face_ids'];
    cedges=[e for e,o in incidence(f[:k]).items() if len(o)==1]
    fedges=[(e,o[0]) for e,o in incidence(f,k).items() if len(o)==1 and e[0]!=e[1]]
    pairs=[]
    for e,(fi,i,j) in fedges:
        x=pre[int(sf[fi]),[i,j]].astype(float);z=v[f[fi,[i,j]]];za=z[1]-z[0]
        for ce in cedges:
            c=v[list(ce)];ca=c[1]-c[0];axis=abs(float(za@ca)/(np.linalg.norm(za)*np.linalg.norm(ca)))
            costs=[float(np.max(np.linalg.norm(x-c,axis=1))),float(np.max(np.linalg.norm(x-c[::-1],axis=1)))]
            pair=dict(source_face_id=int(sf[fi]),local_edge=[i,j],C_edge=list(ce),axis_abs_cos=axis,
                endpoint_max_model=min(costs),segment_max_model=float(ps(x,c).max()),segment_min_model=float(ps(x,c).min()),
                segment_midpoint_model=float(ps(x.mean(0)[None],c)[0]),free_pre_length=float(np.linalg.norm(x[1]-x[0])),
                C_length=float(np.linalg.norm(ca)),pre_endpoints=x.tolist(),C_endpoints=c.tolist())
            pairs.append(pair)
    qualified=[p for p in pairs if p['axis_abs_cos']>np.cos(np.pi/6)]
    witness=min(qualified,key=lambda p:p['endpoint_max_model'])
    segw=min(qualified,key=lambda p:p['segment_max_model'])
    result.append(dict(short_id=r['short_id'],pair_count=len(pairs),axis_qualified_pairs=len(qualified),
        min_endpoint_all=min(p['endpoint_max_model'] for p in pairs),min_endpoint_axis=min(p['endpoint_max_model'] for p in qualified),
        min_segment_max_all=min(p['segment_max_model'] for p in pairs),min_segment_max_axis=min(p['segment_max_model'] for p in qualified),
        axis_and_endpoint_035_count=int(sum(p['endpoint_max_model']<=.035*L for p in qualified)),
        closest_endpoint_witness=witness,closest_segment_witness=segw))
assert len(result)==8 and all(r['axis_and_endpoint_035_count']==0 for r in result)
(P/'independent_check.json').write_text(json.dumps(dict(status='PASS_NUMPY_RECOMPUTATION',L=L,requests=8,rows=result,
    no_models=True,no_binding=True,no_rescoring=True,definition='POST axis; pre_geo selected free-edge endpoints; finite C segment with clamped projection; source face/corner identities'),indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ('short_id','min_endpoint_axis','min_segment_max_axis','axis_and_endpoint_035_count')} for r in result],indent=2))# Compare saved CSVs directly, with new independent point minima from originals.
import csv
with (P/'summary.csv').open(encoding='utf-8-sig',newline='') as f: summary={r['short_id']:r for r in csv.DictReader(f)}
with (P/'edge_pairs.csv').open(encoding='utf-8-sig',newline='') as f: saved_edges=list(csv.DictReader(f))
checks=[]
for r in result:
    meta=next(x for x in M if x['short_id']==r['short_id']);a=npz(meta['canonical_path']);pr=npz(meta['pre_geo_path'])['output'];k=meta['K']
    cedges=[e for e,o in incidence(a['faces'][:k]).items() if len(o)==1]
    xyz=pr[k:].reshape(-1,3).astype(float)
    md=float(min(ps(xyz,a['vertices'][list(e)].astype(float)).min() for e in cedges))
    current=summary[r['short_id']]
    comparisons={'min_endpoint_match_any_axis_pctL':r['min_endpoint_all']/L*100,
        'min_endpoint_match_axis_pass_pctL':r['min_endpoint_axis']/L*100,
        'all_pre_corners_min_segment_pctL':md/L*100}
    for key,expected in comparisons.items():assert abs(float(current[key])-expected)<1e-11,(r['short_id'],key)
    matching=[x for x in saved_edges if x['short_id']==r['short_id'] and x['axis_pass']=='True']
    assert abs(min(float(x['segment_max_pctL']) for x in matching)-r['min_segment_max_axis']/L*100)<1e-11
    checks.append(dict(short_id=r['short_id'],all_pre_corners_min_segment_pctL=md/L*100,
        min_aligned_edge_endpoint_pctL=r['min_endpoint_axis']/L*100,
        min_aligned_edge_both_endpoint_to_segment_max_pctL=r['min_segment_max_axis']/L*100,status='CSV_MATCH_PASS'))
payload=json.loads((P/'independent_check.json').read_text());payload.update(CSV_checks=checks,CSV_checks_status='PASS',
    interpretation='Nearest corner/edge-endpoint quantities only; does not prove absence of nearby free edge or triangle interiors.',
    engineering_note='First JSON dump hit NumPy-int64 serialization; cast count to built-in int and reran read-only arithmetic. No scoring/model/binding calls.')
(P/'independent_check.json').write_text(json.dumps(payload,indent=2)+'\n')
print('INDEPENDENT_CSV_CHECK_PASS',len(checks))
