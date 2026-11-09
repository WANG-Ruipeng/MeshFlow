"""Saved new-endpoint HINGE audit; no extra detector scans or generation."""
import collections,json,time
import numpy as np
from . import hinge as H
from . import hinge_audit as A


def main():
    start=time.perf_counter();rows=[];replays=[];identities=[];receipts={};source_hashes={};verified=0
    for arm in ('CLEAN_EDGE_COH','SMOOTH_PRIOR'):
        regpath=H.OUT/'registrations'/(arm+'.json');reg=H.C.read(regpath);H.B.verify(reg['protected']);H.B.verify(reg['code'])
        source_hashes.update(reg['code']);source_hashes[str(regpath)]=H.C.sha(regpath)
        complete=H.C.read(H.OUT/(arm+'_completion.json'));assert complete['cells']==56 and complete['status']=='COMPLETE'
        for q in H.requests():
            paths=H.paths(arm,q);base=H.C.arrays(paths['post']);pre=H.C.arrays(paths['pre'])['output'];pairs=H.C.read(paths['pairs']);score0=H.C.read(paths['score']);sf=base['source_face_ids'];K=q['K']
            original_edges=A.legal(base,paths['score'].parent/'valid_edges.csv');g0=A.geometry(base['output']);free=np.arange(len(sf))>=K;bulk=free&~base['raw_seam_slot_mask'][sf]
            original_faces=A.table(paths['score'].parent/'seam_bulk_faces.csv');reached0={int(r['raw_slot_id']) for r in original_faces if r['role']=='BULK' and r['reached_free_only']=='True'}
            for delta in H.DELTAS:
                label=H.label(arm,delta);folder=H.OUT/'cells'/label/q['request_id'];receipt=H.C.read(folder/'complete.json');H.B.verify(receipt['files']);verified+=len(receipt['files']);receipts[str(folder/'complete.json')]=H.C.sha(folder/'complete.json')
                mesh=H.C.arrays(folder/'post.npz');p1=H.C.read(folder/'pairs.json');score1=H.C.read(folder/'score.json');now_edges=A.legal(mesh,folder/'valid_edges.csv');g1=A.geometry(mesh['output'])
                assert not set(original_edges)-set(now_edges) and not A.nm(mesh)-A.nm(base)
                r=A.replay(base,pre,mesh,H.C.read(folder/'candidate_records.json'),delta);replays.append(dict(label=label,request_id=q['request_id'],**r))
                detail=dict(label=label,request_id=q['request_id'],retained_legal_source_edges=sorted(set(original_edges)&set(now_edges)),added_legal_source_edges=sorted(set(now_edges)-set(original_edges)));counts={}
                for kind in ('CF','FF'):
                    old=A.pairs_by_id(pairs,sf,K,kind,lambda c:c['classification'] in H.Q.ILLEGAL);new=A.pairs_by_id(p1,sf,K,kind,lambda c:c['classification'] in H.Q.ILLEGAL)
                    u0=A.pairs_by_id(pairs,sf,K,kind,A.unknown);u1=A.pairs_by_id(p1,sf,K,kind,A.unknown);assert not new-old
                    assert p1[kind]['independent_verification'] and p1[kind]['tested_pairs']==p1[kind]['expected_full_pairs'] and p1[kind]['affected_subset'] is None
                    detail.update({kind+'_retained_illegal_source_pairs':sorted(old&new),kind+'_removed_illegal_source_pairs':sorted(old-new),kind+'_new_illegal_source_pairs':sorted(new-old),kind+'_new_unknown_source_pairs':sorted(u1-u0)})
                    counts.update({kind+'_new_illegal':len(new-old),kind+'_removed_illegal':len(old-new),kind+'_unknown_before':len(u0),kind+'_unknown_after':len(u1),kind+'_new_unknown':len(u1-u0)})
                identities.append(detail)
                new_faces=A.table(folder/'seam_bulk_faces.csv');reached1={int(x['raw_slot_id']) for x in new_faces if x['role']=='BULK' and x['reached_free_only']=='True'};assert not reached0-reached1
                sb0,sb1=score0['seam_to_bulk'],score1['seam_to_bulk'];bulk_area0=float(g0['area'][bulk].sum());bulk_area1=float(g1['area'][bulk].sum())
                assert abs(bulk_area1-sb1['retained_bulk_area'])<1e-9
                ratio=g1['area'][free]/g0['area'][free]
                rows.append(dict(label=label,arm=arm,request_id=q['request_id'],split=q['split'],delta=delta,accepted=r['accepted'],
                    zero_face_deletion=len(base['faces'])==len(mesh['faces']),lost_legal_edges=0,new_legal_edges=len(set(now_edges)-set(original_edges)),
                    free_face_denominator_before=len(base['faces'])-K,free_face_denominator_after=len(mesh['faces'])-K,
                    free_area_before=float(g0['area'][free].sum()),free_area_after=float(g1['area'][free].sum()),
                    free_area_relative_change=float(g1['area'][free].sum()/g0['area'][free].sum()-1),
                    minimum_per_face_area_ratio=float(ratio.min()),maximum_per_face_area_ratio=float(ratio.max()),
                    small_angles_before=int(np.sum(g0['angle'][free]<5)),small_angles_after=int(np.sum(g1['angle'][free]<5)),
                    bulk_area_before=bulk_area0,bulk_area_after=bulk_area1,
                    bulk30_before=int(sb0['bulk_reach_ge_0_30_descriptive']),bulk30_after=int(sb1['bulk_reach_ge_0_30_descriptive']),
                    bulk30_lost=int(sb0['bulk_reach_ge_0_30_descriptive'] and not sb1['bulk_reach_ge_0_30_descriptive']),
                    reached_bulk_faces_lost=0,reached_bulk_faces_added=len(reached1-reached0),
                    reach_fraction_before=sb0['free_only_reached_bulk_area_fraction'],reach_fraction_after=sb1['free_only_reached_bulk_area_fraction'],
                    reach_after_fixed_denominator=sb1['free_only_reached_bulk_area']/bulk_area0,
                    cell_seconds=receipt['seconds'],peak_RSS_KiB=receipt['peak_RSS_KiB'],**counts))
        H.B.verify(reg['protected']);H.B.verify(reg['code'])
    assert len(rows)==112
    summary=[]
    for arm in ('CLEAN_EDGE_COH','SMOOTH_PRIOR'):
        for d in H.DELTAS:
            for split in ('TRAIN','DEV','ALL'):
                rr=[r for r in rows if r['arm']==arm and r['delta']==d and (split=='ALL' or r['split']==split)]
                summary.append(dict(label=H.label(arm,d),split=split,n=len(rr),
                    **{key:sum(r[key] for r in rr) for key in ('accepted','lost_legal_edges','new_legal_edges','CF_new_illegal','FF_new_illegal','CF_removed_illegal','FF_removed_illegal','CF_new_unknown','FF_new_unknown','bulk30_before','bulk30_after','bulk30_lost','reached_bulk_faces_added','small_angles_before','small_angles_after')},
                    free_area_mean_relative_change=float(np.mean([r['free_area_relative_change'] for r in rr])),
                    minimum_per_face_area_ratio=min(r['minimum_per_face_area_ratio'] for r in rr),maximum_per_face_area_ratio=max(r['maximum_per_face_area_ratio'] for r in rr),
                    mean_reach_before=float(np.mean([r['reach_fraction_before'] for r in rr])),mean_reach_after=float(np.mean([r['reach_fraction_after'] for r in rr])),
                    mean_reach_after_fixed_denominator=float(np.mean([r['reach_after_fixed_denominator'] for r in rr]))))
    folder=H.OUT/'endpoint_audit'
    H.C.csvwrite(folder/'per_output.csv',rows);H.C.csvwrite(folder/'summary.csv',summary)
    H.C.write(folder/'accepted_operation_replay.json',replays);H.C.write(folder/'identity_sets.json',identities)
    r=dict(status='PASS_READ_ONLY_AUDIT',cells=112,accepted_merges=sum(r['accepted'] for r in rows),new_detector_scans=0,new_model_calls=0,new_samples=0,
        summary=summary,seconds=time.perf_counter()-start,source_hashes=source_hashes,receipt_hashes=receipts,verified_artifact_files=verified,
        audit_code_hashes={str(H.C.resolve(__file__)):H.C.sha(__file__),str(H.C.resolve(A.__file__)):H.C.sha(A.__file__)},
        files={str(p):H.C.sha(p) for p in folder.iterdir() if p.is_file()},
        limits=['Final full CF/FF pair scans independently verified by original detector; no new scan in this audit.',
            'Intermediate detector pair files were not retained; accepted operation geometry/identity replay independently checked, not intermediate intersection re-scans.',
            'No continuous collision safety or semantic/visual guarantee. Small-angle/normal guards permit changes to finite area and shape.',
            'Bulk reach fraction uses current bulk area; original denominator alternatives retained. I can increase without illegal CF pair increase.'])
    H.C.write(H.OUT/'endpoint_audit.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ('source_hashes','receipt_hashes','audit_code_hashes','files')}),flush=True)


if __name__=='__main__':main()
