"""Derive pair fates only from existing full FF scans and source-face mappings.
No new geometry tests, query samples, or model calls.
"""
import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/e/MeshFlow')
s=importlib.util.spec_from_file_location('multiscore',ROOT/'r2/tools/multistep_denoise_v1/score.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows=m.index_rows();idx={(r['request_id'],r['stage']):r for r in rows}
illegal={'proper_transverse_penetration','coplanar_positive_area_overlap'}
def existing(r):
    K=int(r['K'])
    with np.load(r['post_path'],allow_pickle=False) as a:ids=a['source_face_ids'][K:]
    prefix=(m.OLD/'scores'/m.LABEL if r['stage']=='D1' else m.OUT/'scores'/r['stage'])/r['request_id']/'quality'
    scan=m.read(prefix/'pairs.json')['FF']
    if scan['expected_full_pairs']!=len(ids)*(len(ids)-1)//2:raise ValueError('Not a full FF scan')
    contacts={tuple(sorted((int(ids[p['pair'][0]]),int(ids[p['pair'][1]])))):p['classification'] for p in scan['contacts']}
    bad={key for key,v in contacts.items() if v in illegal}
    return set(map(int,ids)),bad,contacts
out=[];detail=[]
for rr in m.registration()['records']:
    rid=rr['request_id'];base,basebad,_=existing(idx[(rid,'D1')])
    for stage in ('D3','D5'):
        kept,bad,contacts=existing(idx[(rid,stage)])
        counts=dict(D1_illegal_pairs=len(basebad),endpoint_illegal_pairs=len(bad),D1_illegal_both_faces_retained_still_illegal=0,D1_illegal_both_faces_retained_now_nonillegal=0,D1_illegal_one_or_both_faces_deleted=0)
        for pair in sorted(basebad):
            if not set(pair)<=kept:category='one_or_both_faces_deleted';classification='NA_DELETED'
            elif pair in bad:category='both_faces_retained_still_illegal';classification=contacts[pair]
            else:category='both_faces_retained_now_nonillegal';classification=contacts.get(pair,'not_in_contact_list_full_scan_disjoint')
            counts['D1_illegal_'+category]+=1
            detail.append(dict(request_id=rid,parent_id=rr['parent_id'],region_type=rr['region_type'],stage=stage,source_face_a=pair[0],source_face_b=pair[1],fate=category,endpoint_classification=classification))
        assert sum(counts['D1_illegal_'+k] for k in ('both_faces_retained_still_illegal','both_faces_retained_now_nonillegal','one_or_both_faces_deleted'))==len(basebad)
        counts.update(endpoint_illegal_pairs_not_illegal_at_D1=len(bad-basebad),D1_retained_faces_later_deleted=len(base-kept),D1_deleted_faces_reappeared=len(kept-base))
        out.append(dict(request_id=rid,parent_id=rr['parent_id'],region_type=rr['region_type'],stage=stage,**counts))
m.csv_write(m.OUT/'FF_pair_fates_D1_to_endpoints.csv',out)
m.csv_write(m.OUT/'FF_pair_fates_detail.csv',detail)
summary=[]
for stage in ('D3','D5'):
    for group in ('ALL','seat','back'):
        sub=[r for r in out if r['stage']==stage and (group=='ALL' or r['region_type']==group)]
        summary.append(dict(stage=stage,group=group,outputs=len(sub),**{k:sum(r[k] for r in sub) for k in out[0] if k not in ('request_id','parent_id','region_type','stage')}))
m.csv_write(m.OUT/'FF_pair_fates_summary.csv',summary)
print(summary)