"""Explain retained C vertex-link failures without modifying geometry/IDs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,itertools
import numpy as np
from r2.data_tools.bench_seat_coverage_v1.inspect import OUT
from r2.data_tools.bench_user_v2 import build as b
def main():
    out=[]
    for row in b.read(OUT/'candidate_manifest.json')['conditions']:
        dest=OUT/'conditions'/row['condition_id'];q=b.read(dest/'quality.json');links=b.read(dest/'vertex_links.json');case=b.load(row['case_path']);v=case['model_vertices'];f=case['source_faces'];records=[]
        for link in links:
            contacts=[]
            for fi,fj in itertools.combinations(link['target_face_ids'],2):
                for ei in range(3):
                    for ej in range(3):
                        va=f[fi][[ei,(ei+1)%3]];vb=f[fj][[ej,(ej+1)%3]];p0,p1=v[va].astype(float);q0,q1=v[vb].astype(float);d=p1-p0;L=np.linalg.norm(d)
                        if L<=1e-12:continue
                        axis=d/L
                        if max(np.linalg.norm(np.cross(q0-p0,axis)),np.linalg.norm(np.cross(q1-p0,axis)))>1e-6:continue
                        lo=max(0,min(np.dot(q0-p0,axis),np.dot(q1-p0,axis)));hi=min(L,max(np.dot(q0-p0,axis),np.dot(q1-p0,axis)))
                        if hi-lo<=1e-6 or set(va)==set(vb):continue
                        contacts.append(dict(source_faces=[fi,fj],source_edge_vertex_ids=[va.tolist(),vb.tolist()],geometric_collinear_overlap_length=float(hi-lo),same_index_edge=False,overlap_endpoints=[(p0+lo*axis).tolist(),(p0+hi*axis).tolist()]))
            records.append(dict(source_vertex_id=link['source_vertex_id'],position=link['position'],link_components=link['old_detector_item']['link_components'],source_face_ids=link['target_face_ids'],partial_edge_contacts=contacts,interpretation='Nonconforming T-junction / partial collinear boundary contact: same finite upper surface, shared vertex but unmatched endpoint segmentation. Not certified conforming manifold C. No coordinates or IDs changed.'))
        r=dict(condition_id=row['condition_id'],C_illegal_finite_pairs=len(q['illegal_C_pairs']),C_unresolved_pairs=len(q['unresolved_C_pairs']),C_degenerate=len(q['degenerate_C_face_ids']),C_exact_duplicate_groups=len(q['duplicate_C_ID_face_groups']),C_nonmanifold_edge_count=len(q['topology']['nonmanifold_edges']),C_vertex_link_failures=len(links),records=records,old_gate=q['geometry_status'],old_reasons=q['reasons'],new_admission='ROOT_DECISION_PENDING')
        b.save(dest/'link_interpretation.json',r);out.append(r)
    b.save(OUT/'vertex_link_audit.json',out)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
