"""Post-hoc coordinate accounting from stored arrays only; no score rerun."""
from __future__ import annotations
import collections,json
import numpy as np
from . import run as R
C=R.C

def run():
    reg=C.read(R.ROOT/'registration.json');rows=[];faces=[];hashes={}
    for job in reg['jobs']:
        q=job['q'];rawpath=C.resolve(job['base_dir'])/'raw/raw.npz';prepath=C.resolve(job['base_dir'])/'post/pre_geo.npz'
        raw=C.arrays(rawpath);pre=C.arrays(prepath)['output'];hashes[str(rawpath)]=C.sha(rawpath);hashes[str(prepath)]=C.sha(prepath)
        rt=raw['raw'].astype(float)
        raw_area=.5*np.linalg.norm(np.cross(rt[:,1]-rt[:,0],rt[:,2]-rt[:,0]),axis=1)
        for stage,path in [('OLD_BASE',job['old_base']),('OLD_FINAL',job['old_final']),('GUARD_GEO',R.cell(job)/'GUARD_GEO/post.npz'),('FINAL',R.cell(job)/'FINAL/post.npz')]:
            path=C.resolve(path);a=C.arrays(path);hashes[str(path)]=C.sha(path)
            sf=a.get('source_face_ids',a.get('base_source_face_ids'));sc=a.get('source_corner_ids',a.get('base_source_corner_ids'))
            tri=a['output'][:len(sf)];original=raw['raw'][sf[:,None],sc];proposal=pre[sf[:,None],sc]
            isfree=sf>=q['K'];ids=raw['raw_anchor_ids'][sf[:,None],sc];fixed=ids>=0
            if tri[fixed].tobytes()!=raw['C_vertices'][ids[fixed]].tobytes():raise AssertionError('Retained anchor moved '+str(path))
            delta_raw=np.linalg.norm(tri.astype(float)-original,axis=2);delta_d0=np.linalg.norm(tri.astype(float)-proposal,axis=2)
            row=dict(panel=job['panel'],request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],stage=stage,
                retained_original_free_faces=int(isfree.sum()),extra_bridge_faces=len(a['output'])-len(sf),
                original_free_corner_denominator=int(isfree.sum())*3,retained_hard_anchor_corners=int(fixed.sum()),hard_anchors_bitwise_exact=True,
                reference='Source model corner permutation restored; BRIDGE excluded from absent RAW lineage')
            den=float(raw_area[q['K']:].sum());kept=float(raw_area[sf[isfree]].sum())
            row.update(RAW_free_area_denominator=den,retained_RAW_free_area=kept,
                deleted_RAW_free_area_fraction=(den-kept)/den if den else None)
            for name,d in (('RAW',delta_raw),('D0_RESTORED',delta_d0)):
                v=d[isfree].reshape(-1)
                row.update({f'max_distance_from_{name}_model':float(v.max()) if len(v) else None,
                    f'mean_distance_from_{name}_model':float(v.mean()) if len(v) else None,
                    f'p95_distance_from_{name}_model':float(np.quantile(v,.95)) if len(v) else None})
            rows.append(row)
            for i in np.flatnonzero(isfree):faces.append(dict(panel=job['panel'],request_id=q['request_id'],stage=stage,raw_slot=int(sf[i]),
                max_from_RAW_model=float(delta_raw[i].max()),max_from_D0_restored_model=float(delta_d0[i].max())))
    out=R.ROOT/'displacement';C.csvwrite(out/'per_output.csv',rows);C.csvwrite(out/'per_original_face.csv',faces)
    r=dict(status='COMPLETE',rows=len(rows),faces=len(faces),source_hashes=hashes,
        model_calls=0,scoring_calls=0,decoder_calls=0,code_sha256=C.sha(__file__),
        budget_note='HINGE/ALPHA 1.5%L is measured from new guarded PRE_GEO, whereas total displacement from original D0 and RAW is separately recorded here',
        files={str(p):C.sha(p) for p in (out/'per_output.csv',out/'per_original_face.csv')})
    C.write(out/'completion.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ('source_hashes','files')}));return r
if __name__=='__main__':run()
