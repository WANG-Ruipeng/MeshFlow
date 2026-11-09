"""Score accepted child outputs; byte-identical endpoints inherit frozen scores."""
from __future__ import annotations
import collections
import json
from pathlib import Path
import time
import numpy as np
from . import scoring as S
C,B,T=S.C,S.B,S.T
OUT=C.ROOT/'experiments/r2_hinge_subsegment_v1'
BASE=C.ROOT/'experiments/r2_hinge_opposite_v1'


def inherited_row(old,base,edges):
    """Identity-only reductions of old exact counts; no geometry rescoring."""
    row=dict(old);P=len(base['faces'])-int(base['known_faces'])
    assert row['actual_free_faces']==P
    row['fixed_parent_free_face_denominator']=P
    for kind in ('CF','FF'):
        fraction=float(old[kind+'_illegal_affected_free_fraction']);count=round(fraction*P)
        assert np.isclose(count/P,fraction,rtol=0,atol=1e-12)
        row.update({f'fixed_parent_{kind}_illegal_affected_count':count,
                    f'fixed_parent_{kind}_illegal_affected_fraction':fraction,
                    f'fixed_parent_{kind}_illegal_unique_pair_count':old[kind+'_illegal_pair_count']})
    for name,oldfield in (('minimum_angle_lt5','free_minimum_angle_lt5_fraction'),('q_gt10','free_q_gt10_fraction'),
                           ('near_degenerate','free_near_degenerate_fraction'),('exact_zero_area','free_exact_zero_area_fraction')):
        fraction=float(old[oldfield]);count=round(fraction*P)
        assert np.isclose(count/P,fraction,rtol=0,atol=1e-12)
        row.update({f'fixed_parent_{name}_count':count,f'fixed_parent_{name}_fraction':fraction})
    legal=[e for e in edges if e['valid_edge']]
    row.update(illegal_sibling_child_pair_count=0,original_POST_free_faces=P,added_child_faces=0,
               deleted_original_POST_faces=0,subdivision_internal_valid_edge_count=0,
               subdivision_internal_valid_edge_length=0,external_valid_edge_count=len(legal),
               external_valid_edge_length=sum(float(e['length_model']) for e in legal),
               scoring_definition_version=S.VERSION)
    return row


def equal_geometry(base,mesh,q):
    S.validate(mesh,q)
    for key in ('vertices','faces','output'):
        a,b=base[key],mesh[key]
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),key
    n=len(base['faces'])
    assert np.array_equal(mesh['child_face_id'],np.arange(n))
    assert np.array_equal(mesh['parent_post_face_id'],np.arange(n))
    assert np.array_equal(mesh['parent_source_raw_slot'],base['source_face_ids'])
    assert np.array_equal(mesh['child_seam_mask'],base['raw_seam_slot_mask'][base['source_face_ids']])
    assert np.array_equal(mesh['corner_parent_barycentric'],np.broadcast_to(np.eye(3),(n,3,3)))


def run():
    dest=OUT/'scores';done=dest/'completion.json'
    if done.exists():
        result=C.read(done);B.verify(result['files']);return result
    trials=C.read(OUT/'trials/completion.json');assert trials['status']=='COMPLETE' and trials['cells']==56
    B.verify(trials['code']);B.verify(trials['files'])
    equivalence=C.read(OUT/'scoring_equivalence/608b46b0_n0/comparison.json')
    assert equivalence['status']=='PASS' and equivalence['fields']==135
    qs=C.read(BASE/'protocol.json')['requests'];assert len(qs)==28
    tick=time.perf_counter();rows=[];receipts=[];counts=collections.Counter()
    C.resources(dest/('resources_before_'+str(time.time_ns())+'.json'))
    for q in qs:
        rid=q['request_id'];folder=BASE/'cells/HINGE_OPP015'/rid
        original=C.read(folder/'complete.json');B.verify(original['files'])
        base=C.arrays(folder/'post.npz');edges=T.edges_file(folder/'valid_edges.csv')
        baseline=inherited_row(original['row'],base,edges)
        baseline.update(label='PLAIN_S2000',stage='HINGE_OPP015',score_origin='INHERITED_IDENTITY',
                        baseline_mesh_path=str(folder/'post.npz'),baseline_mesh_sha256=C.sha(folder/'post.npz'),
                        mesh_path=str(folder/'post.npz'),accepted_splits=0)
        rows.append(baseline);counts['BASELINE_INHERITED']+=1
        for stage in ('SPLIT015','SPLIT025'):
            cell=OUT/'trials'/stage/rid;producer=C.read(cell/'complete.json');B.verify(producer['files'])
            mesh=C.arrays(cell/'post.npz');p=dest/stage/rid;final=p/'complete.json'
            if producer['accepted']==0:
                equal_geometry(base,mesh,q)
                row=dict(baseline,stage=stage,mesh_path=str(cell/'post.npz'),score_origin='INHERITED_IDENTITY')
                receipt=dict(status='PASS',score_origin='INHERITED_IDENTITY',geometry_byte_equal=True,
                             inheritance='All 135 fields from identical HINGE_OPP015 mesh; fixed-parent reductions use exact unchanged counts',
                             mesh_sha256=C.sha(cell/'post.npz'),baseline_mesh_sha256=C.sha(folder/'post.npz'),
                             baseline_complete_sha256=C.sha(folder/'complete.json'),row=row,model_calls=0,new_queries=0)
                if final.exists():assert C.read(final)==C.clean(receipt)
                else:C.write(final,receipt)
                counts['CANDIDATE_INHERITED_IDENTITY']+=1
            else:
                if final.exists():
                    receipt=C.read(final);B.verify(receipt['files']);row=receipt['row']
                else:
                    payload=S.score(q,mesh,p,label='PLAIN_S2000');row=payload['row'];receipt=C.read(final)
                row=dict(row,label='PLAIN_S2000',stage=stage,score_origin='RECOMPUTED',mesh_path=str(cell/'post.npz'),
                         baseline_mesh_path=str(folder/'post.npz'),baseline_mesh_sha256=C.sha(folder/'post.npz'),
                         accepted_splits=producer['accepted'])
                counts['CANDIDATE_RECOMPUTED']+=1
            rows.append(row);receipts.append(dict(path=str(final),sha256=C.sha(final)))
    assert len(rows)==84
    order={'HINGE_OPP015':0,'SPLIT015':1,'SPLIT025':2}
    rows.sort(key=lambda r:(order[r['stage']],r['request_id']))
    C.csvwrite(dest/'per_output.csv',rows)
    result=dict(status='COMPLETE',version=S.VERSION,rows=84,original_metric_fields=135,counts=dict(counts),
                baseline='HINGE_OPP015',new_model_calls=0,new_queries=0,seconds=time.perf_counter()-tick,
                score_receipts=receipts,files={str(dest/'per_output.csv'):C.sha(dest/'per_output.csv')},
                code={str(Path(__file__)):C.sha(__file__),str(Path(S.__file__)):C.sha(S.__file__)})
    C.write(done,result);print(json.dumps({k:v for k,v in result.items() if k not in ('files','score_receipts','code')},indent=2))
    return result


if __name__=='__main__':run()
