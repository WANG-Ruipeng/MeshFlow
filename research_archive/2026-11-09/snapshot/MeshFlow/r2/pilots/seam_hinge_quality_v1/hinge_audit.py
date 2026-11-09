"""Independent saved-output audit: no detector calls, model calls or new cases.

Reconstructs accepted edits from saved operations and checks endpoint identities.
Intermediate detector scans were executed by hinge.py but not persisted; this
audit does NOT claim independent re-verification of those intermediate scans.
"""
from __future__ import annotations
import collections,csv,json,time
import numpy as np
from . import hinge as H

C=H.C


def table(p):
    with C.resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def edges(f):
    result=collections.defaultdict(list)
    for i,face in enumerate(f):
        for a,b in ((0,1),(1,2),(2,0)):
            result[tuple(sorted((int(face[a]),int(face[b]))))].append((i,a,b))
    return result


def edge_key(occ,sf,sc):
    return tuple(sorted((int(sf[i]),*sorted((int(sc[i,a]),int(sc[i,b])))) for i,a,b in occ))


def legal(mesh,path):
    inc=edges(mesh['faces']);sf,sc=mesh['source_face_ids'],mesh['source_corner_ids'];result={}
    for row in table(path):
        if row['valid_edge']!='True':continue
        key=(int(row['vertex_u']),int(row['vertex_v']));occ=inc[key]
        assert len(occ)==2 and sorted(i for i,a,b in occ)==sorted(json.loads(row['incident_face_ids']))
        result[edge_key(occ,sf,sc)]=row
    return result


def pairs_by_id(scan,sf,K,kind,predicate):
    out=set()
    for r in scan[kind]['contacts']:
        if not predicate(r):continue
        i,j=map(int,r['pair']);i=int(sf[i if kind=='CF' else K+i]);j=int(sf[K+j])
        out.add((i,j) if kind=='CF' else tuple(sorted((i,j))))
    return out


def unknown(row):
    s=row['classification'].lower()
    return any(k in s for k in ('unknown','unresolved','degenerate')) or row.get('detector',{}).get('resolved') is False


def geometry(tri):
    t=np.asarray(tri,np.float64);cr=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);twice=np.linalg.norm(cr,axis=1)
    angles=[]
    for a,b,c in ((0,1,2),(1,2,0),(2,0,1)):
        x=t[:,b]-t[:,a];y=t[:,c]-t[:,a];den=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
        angles.append(np.degrees(np.arccos(np.clip(np.divide((x*y).sum(1),den,out=np.ones_like(den),where=den>0),-1,1))))
    lengths=np.linalg.norm(t-np.roll(t,-1,axis=1),axis=2);area=twice/2
    return dict(area=area,angle=np.min(angles,axis=0),normal=np.divide(cr,twice[:,None],out=np.zeros_like(cr),where=twice[:,None]>0),
        q=np.divide((lengths**2).sum(1),4*np.sqrt(3)*area,out=np.full_like(area,np.inf),where=area>0))


def nm(mesh):
    return {edge_key(occ,mesh['source_face_ids'],mesh['source_corner_ids']) for occ in edges(mesh['faces']).values() if len(occ)>2}


def replay(base,pre,candidate,records,delta):
    v=base['vertices'].copy();f=base['faces'];sf,sc=base['source_face_ids'],base['source_corner_ids'];K=int(base['known_faces'])
    roots=np.arange(len(v));fixed=set(map(int,f[:K].reshape(-1)));cv=base['corner_vertex_ids'].reshape(-1);original=geometry(v[f])
    nm0=nm(base);old_dups=H.B.numeric_duplicate_pairs(v,f);accepted=0;peak_motion=peak_normal=0.;rows=[]
    for rec in records:
        if rec['decision']!='ACCEPTED':continue
        previous=geometry(v[roots[f]]);uv=rec['endpoint_original_vertices'];aa,bb=[int(roots[u]) for u in uv]
        assert aa!=bb and np.linalg.norm(v[aa].astype(float)-v[bb])<=delta*C.L+1e-12
        expected_group=set(np.flatnonzero(np.isin(roots,[aa,bb])).tolist());ops=rec['merged_clusters'];assert len(ops)==1
        op=ops[0];group=set(op['original_cluster_ids']);assert group==expected_group
        pinned=group&fixed;assert len(pinned)<=1
        target=np.asarray(op['target'],np.float32);old_a,old_b=v[aa].copy(),v[bb].copy()
        expected=v[next(iter(pinned))] if pinned else ((old_a.astype(float)+old_b.astype(float))/2).astype(np.float32)
        assert np.array_equal(target.view(np.uint32),expected.view(np.uint32))
        all_corners=np.flatnonzero(np.isin(cv,list(group)));disp=np.linalg.norm(pre.reshape(-1,3)[all_corners].astype(float)-target,axis=1)
        assert disp.max()<=delta*C.L+1e-12 and len(all_corners)==op['original_corner_count']
        assert abs(disp.max()-op['maximum_PRE_GEO_displacement'])<1e-12
        v[list(group)]=target;roots[list(group)]=op['new_root'];ff=roots[f];tri=v[ff]
        assert np.array_equal(tri[:K].view(np.uint32),base['output'][:K].view(np.uint32))
        anchor=base['output_anchor_ids']>=0;assert np.array_equal(tri[anchor].view(np.uint32),base['output'][anchor].view(np.uint32))
        affected=np.flatnonzero(np.any(np.isin(f,list(group)),axis=1));affected=affected[affected>=K]
        assert sorted(sf[affected].tolist())==sorted(rec['affected_source_raw_slots'])
        assert all(len(set(ff[i].tolist()))==3 for i in affected)
        now=geometry(tri);area=now['area'][affected];assert np.all(area>1e-12*C.L*C.L)
        for ref in (original,previous):
            assert not np.any((now['angle'][affected]<5)&(now['angle'][affected]<ref['angle'][affected]-1e-10))
            assert not np.any((now['q'][affected]>10)&(now['q'][affected]>ref['q'][affected]+1e-10))
            cos=(now['normal'][affected]*ref['normal'][affected]).sum(1);assert np.all(cos>=.5-1e-12)
        normal=float(np.degrees(np.arccos(np.clip((now['normal'][affected]*original['normal'][affected]).sum(1),-1,1))).max())
        temp=dict(base,vertices=v,faces=ff);assert not nm(temp)-nm0
        assert not H.B.numeric_duplicate_pairs(v,ff)-old_dups
        peak_motion=max(peak_motion,float(disp.max()));peak_normal=max(peak_normal,normal);accepted+=1
        rows.append(dict(rank=rec['rank'],whole_original_clusters=sorted(group),original_members=len(all_corners),
            affected_raw_slots=sf[affected].tolist(),maximum_PRE_GEO_motion=float(disp.max()),maximum_original_normal_angle=normal))
    assert np.array_equal(v.view(np.uint32),candidate['vertices'].view(np.uint32))
    assert np.array_equal(roots[f],candidate['faces']) and np.array_equal(roots,candidate['original_POST_vertex_to_hinge_vertex'])
    assert np.array_equal(v[roots[f]].view(np.uint32),candidate['output'].view(np.uint32))
    for key in ('source_face_ids','source_corner_ids','raw_slot_ids','source_face_kept','source_face_reason','duplicate_of_source_face','raw_anchor_ids','output_anchor_ids'):
        assert np.array_equal(base[key],candidate[key]),key
    mapped=base['corner_vertex_ids'].copy();good=mapped>=0;mapped[good]=roots[mapped[good]]
    assert np.array_equal(mapped,candidate['corner_vertex_ids'])
    return dict(status='PASS_EXACT_SAVED_OPERATION_REPLAY',accepted=accepted,maximum_PRE_GEO_motion=peak_motion,maximum_normal_angle=peak_normal,accepted_steps=rows)


def slot77_prefilter():
    q=next(q for q in H.requests() if q['request_id']=='608b46b0_n3');p=H.paths('PLAIN_S2000',q);mesh=C.arrays(p['post']);v,f=mesh['vertices'],mesh['faces'];sf=mesh['source_face_ids'];sc=mesh['source_corner_ids'];K=q['K']
    raw=C.arrays(p['raw'])['output'];ii=int(np.flatnonzero(sf==77)[0]);inc=edges(f);own=[];all_open=[]
    for edge,occ in sorted(inc.items()):
        if len(occ)==1 and occ[0][0]>=K:
            all_open.append((edge,occ[0]))
            if occ[0][0]==ii:own.append((edge,occ[0]))
    records=[]
    for edge,(i,a,b) in own:
        for edge2,(j,c,d) in all_open:
            if j==i:continue
            common=set(edge)&set(edge2)
            if len(common)!=1:continue
            h=next(iter(common));u=next(x for x in edge if x!=h);w=next(x for x in edge2 if x!=h)
            di=1 if int(f[i,a])==h else -1;dj=1 if int(f[j,c])==h else -1
            gap=float(np.linalg.norm(v[u].astype(float)-v[w]));ui=int(np.flatnonzero(f[i]==u)[0]);wj=int(np.flatnonzero(f[j]==w)[0])
            rgap=float(np.linalg.norm(raw[int(sf[i]),int(sc[i,ui])].astype(float)-raw[int(sf[j]),int(sc[j,wj])]))
            records.append(dict(partner_raw_slot=int(sf[j]),original_hinge_vertex=h,own_other_vertex=u,partner_other_vertex=w,
                opposite_winding=di!=dj,POST_endpoint_gap=gap,POST_endpoint_gap_L=gap/C.L,RAW_endpoint_gap=rgap,RAW_endpoint_gap_L=rgap/C.L,
                prefilter015='SAME_WINDING' if di==dj else 'ENDPOINT_GAP_OVER_DELTA' if gap>.015*C.L+1e-12 else 'PASSES_FIRST_TWO_FILTERS',
                prefilter025='SAME_WINDING' if di==dj else 'ENDPOINT_GAP_OVER_DELTA' if gap>.025*C.L+1e-12 else 'PASSES_FIRST_TWO_FILTERS'))
    return dict(request_id=q['request_id'],slot=77,source_post_sha=C.sha(p['post']),source_raw_sha=C.sha(p['raw']),
        original_open_halfedges=len(own),shared_vertex_free_boundary_pairs=len(records),records=records,
        interpretation='Read-only replay of same frozen eligibility rules. No new binding, detector scan, threshold, mesh or sample.')


def main():
    start=time.perf_counter();reg=C.read(H.OUT/'registrations/PLAIN_S2000.json');H.B.verify(reg['protected']);H.B.verify(reg['code'])
    source_hashes={str(H.OUT/'registrations/PLAIN_S2000.json'):C.sha(H.OUT/'registrations/PLAIN_S2000.json'),str(H.OUT/'PLAIN_S2000_completion.json'):C.sha(H.OUT/'PLAIN_S2000_completion.json')}
    cells=[];replay_details=[];identity_details=[];guard_failures=[];file_count=0
    for q in H.requests():
        path=H.paths('PLAIN_S2000',q);base=C.arrays(path['post']);pre=C.arrays(path['pre'])['output'];bp=C.read(path['pairs']);bs=C.read(path['score']);K=q['K'];sf=base['source_face_ids']
        bl=legal(base,path['score'].parent/'valid_edges.csv');bg=geometry(base['output']);bseam=base['raw_seam_slot_mask'][sf];bbulk=(np.arange(len(sf))>=K)&~bseam
        for delta in H.DELTAS:
            name=H.label('PLAIN_S2000',delta);dest=H.OUT/'cells'/name/q['request_id'];receipt=C.read(dest/'complete.json');H.B.verify(receipt['files']);file_count+=len(receipt['files'])
            source_hashes[str(dest/'complete.json')]=C.sha(dest/'complete.json');out=C.arrays(dest/'post.npz');npairs=C.read(dest/'pairs.json');score=C.read(dest/'score.json')
            nl=legal(out,dest/'valid_edges.csv');ng=geometry(out['output']);saved_ledger=C.read(dest/'paired_ledger.json')
            lost=set(bl)-set(nl);new=set(nl)-set(bl);assert not lost
            assert not nm(out)-nm(base)
            for key in set(bl)&set(nl):assert bl[key]['winding_opposite']==nl[key]['winding_opposite']
            detail=dict(label=name,request_id=q['request_id'],retained_legal_source_edge_ids=sorted(set(bl)&set(nl)),new_legal_source_edge_ids=sorted(new),lost_legal_source_edge_ids=sorted(lost))
            info={}
            for kind in ('CF','FF'):
                old=pairs_by_id(bp,sf,K,kind,lambda r:r['classification'] in H.Q.ILLEGAL);newbad=pairs_by_id(npairs,sf,K,kind,lambda r:r['classification'] in H.Q.ILLEGAL)
                old_unknown=pairs_by_id(bp,sf,K,kind,unknown);new_unknown=pairs_by_id(npairs,sf,K,kind,unknown)
                assert not newbad-old
                assert saved_ledger['summary'][kind+'_new_illegal_pairs']==len(newbad-old)
                assert saved_ledger['summary'][kind+'_illegal_pairs_removed']==len(old-newbad)
                info.update({kind+'_illegal_before':len(old),kind+'_illegal_after':len(newbad),kind+'_new_illegal':len(newbad-old),kind+'_resolved_illegal':len(old-newbad),
                    kind+'_unknown_before':len(old_unknown),kind+'_unknown_after':len(new_unknown),kind+'_new_unknown':len(new_unknown-old_unknown)})
                detail.update({kind+'_retained_illegal_source_pairs':sorted(old&newbad),kind+'_removed_illegal_source_pairs':sorted(old-newbad),
                    kind+'_new_illegal_source_pairs':sorted(newbad-old),kind+'_new_unknown_source_pairs':sorted(new_unknown-old_unknown)})
                assert npairs[kind]['tested_pairs']==npairs[kind]['expected_full_pairs'] and npairs[kind]['affected_subset'] is None
                assert npairs[kind]['independent_verification'] is True
            replay_result=replay(base,pre,out,C.read(dest/'candidate_records.json'),delta);replay_details.append(dict(label=name,request_id=q['request_id'],**replay_result));identity_details.append(detail)
            ar=ng['area'][K:]/bg['area'][K:];assert np.isfinite(ar).all()
            before_bulk=bs['seam_to_bulk'];after_bulk=score['seam_to_bulk']
            assert abs(after_bulk['retained_bulk_area']-ng['area'][bbulk].sum())<1e-9
            old_face=table(path['score'].parent/'seam_bulk_faces.csv');new_face=table(dest/'seam_bulk_faces.csv')
            before_reached={int(r['raw_slot_id']) for r in old_face if r['role']=='BULK' and r['reached_free_only']=='True'}
            after_reached={int(r['raw_slot_id']) for r in new_face if r['role']=='BULK' and r['reached_free_only']=='True'}
            assert not before_reached-after_reached
            old30=bool(before_bulk['bulk_reach_ge_0_30_descriptive']);new30=bool(after_bulk['bulk_reach_ge_0_30_descriptive'])
            freeold=float(bg['area'][K:].sum());freenew=float(ng['area'][K:].sum());bulkold=float(bg['area'][bbulk].sum());bulknew=float(ng['area'][bbulk].sum())
            cells.append(dict(label=name,request_id=q['request_id'],split=q['split'],delta=delta,accepted=replay_result['accepted'],
                legal_edges_retained=len(bl),legal_edges_new=len(new),legal_edges_lost=len(lost),free_faces_before=len(sf)-K,free_faces_after=len(out['source_face_ids'])-K,
                free_area_before=freeold,free_area_after=freenew,free_area_relative_change=freenew/freeold-1,
                bulk_area_before=bulkold,bulk_area_after=bulknew,bulk_area_relative_change=bulknew/bulkold-1,
                min_free_face_area_ratio=float(ar.min()),max_free_face_area_ratio=float(ar.max()),
                free_faces_area_down10=int(np.sum(ar<.9)),free_faces_area_up10=int(np.sum(ar>1.1)),
                small_angle_before=int(np.sum(bg['angle'][K:]<5)),small_angle_after=int(np.sum(ng['angle'][K:]<5)),
                min_angle_delta_min=float((ng['angle'][K:]-bg['angle'][K:]).min()),
                reach_fraction_before=before_bulk['free_only_reached_bulk_area_fraction'],reach_fraction_after=after_bulk['free_only_reached_bulk_area_fraction'],
                reached_bulk_area_before=before_bulk['free_only_reached_bulk_area'],reached_bulk_area_after=after_bulk['free_only_reached_bulk_area'],
                reach_after_with_baseline_bulk_denominator=after_bulk['free_only_reached_bulk_area']/bulkold,
                bulk30_before=old30,bulk30_after=new30,bulk30_lost=old30 and not new30,
                reached_bulk_faces_lost=len(before_reached-after_reached),reached_bulk_faces_added=len(after_reached-before_reached),
                receipt_seconds=receipt['seconds'],peak_RSS_KiB=receipt['peak_RSS_KiB'],**info))
    summary=[]
    for delta in H.DELTAS:
        for split in ('TRAIN','DEV','ALL'):
            rr=[r for r in cells if r['delta']==delta and (split=='ALL' or r['split']==split)]
            summary.append(dict(label=H.label('PLAIN_S2000',delta),split=split,n=len(rr),
                **{key:sum(r[key] for r in rr) for key in ('accepted','legal_edges_lost','CF_new_illegal','FF_new_illegal','CF_new_unknown','FF_new_unknown','bulk30_before','bulk30_after','bulk30_lost','reached_bulk_faces_lost','reached_bulk_faces_added','small_angle_before','small_angle_after','free_faces_area_down10','free_faces_area_up10')},
                mean_free_area_relative_change=float(np.mean([r['free_area_relative_change'] for r in rr])),
                min_free_area_relative_change=min(r['free_area_relative_change'] for r in rr),max_free_area_relative_change=max(r['free_area_relative_change'] for r in rr),
                minimum_per_face_area_ratio=min(r['min_free_face_area_ratio'] for r in rr),maximum_per_face_area_ratio=max(r['max_free_face_area_ratio'] for r in rr),
                mean_bulk_area_relative_change=float(np.mean([r['bulk_area_relative_change'] for r in rr])),
                mean_reach_fraction_before=float(np.mean([r['reach_fraction_before'] for r in rr])),mean_reach_fraction_after=float(np.mean([r['reach_fraction_after'] for r in rr])),
                mean_reach_after_fixed_denominator=float(np.mean([r['reach_after_with_baseline_bulk_denominator'] for r in rr]))))
    prefilter=slot77_prefilter();H.B.verify(reg['protected']);H.B.verify(reg['code'])
    dest=H.OUT/'baseline_audit';C.csvwrite(dest/'per_output.csv',cells);C.csvwrite(dest/'summary.csv',summary)
    C.write(dest/'accepted_operation_replay.json',replay_details);C.write(dest/'identity_sets.json',identity_details);C.write(dest/'slot77_prefilter.json',prefilter)
    original=C.read(H.OUT/'PLAIN_S2000_completion.json')
    result=dict(status='PASS_READ_ONLY_AUDIT',cells=56,accepted_merges=sum(r['accepted'] for r in cells),
        checks=['All original/final illegal CF/FF source-face identities, not only counts','All original/final legal CF/FF source-face+corner edge identities',
            'Exact saved accepted-operation replay, whole cluster membership, PRE_GEO cumulative budgets, fixed C/seam anchors',
            'Every accepted edit triangle shape/normal/new nonmanifold and duplicate guards','No face deletion/reordering and original corner/slot identity preserved',
            'Complete independently-verified original detector final scans and explicit UNKNOWN accounting','Bulk reach numerator/denominator and request identities'],
        limits=['No continuous collision detection: an edit trajectory could transiently cross another face.',
            'Intermediate CF/FF detector outputs were not persisted. Runtime code checked them; this audit independently checks final full scans, not unrecorded intermediate pair scans.',
            'I is a separate near-surface coverage metric and can increase without a new illegal CF pair.',
            'No face deletion does not imply identical surface area or unchanged shape. Normal-angle and small-angle thresholds permit bounded shape changes.',
            'Source RAW slots are within-output identities; not cross-model semantic correspondences.',
            'Legal graph is stricter than prior unfiltered section connectivity; connectivity alone does not certify suitability or whole-mesh safety.'],
        summary=summary,slot77_prefilter=prelim(prefilter),
        runtime=dict(original_CPU_seconds=original['seconds'],original_CPU_workers=original['workers'],original_GPU_calls=0,
            summed_cell_seconds=sum(r['receipt_seconds'] for r in cells),max_process_RSS_KiB=max(r['peak_RSS_KiB'] for r in cells),audit_seconds=time.perf_counter()-start),
        identity=dict(protected_original_file_count=len(reg['protected']),frozen_source_file_count=len(reg['code']),verified_cell_artifact_file_count=file_count,
            original_protected_hashes=reg['protected'],frozen_code_hashes=reg['code'],audit_source_sha256=C.sha(__file__),receipt_hashes=source_hashes),
        files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()},new_detector_scans=0,model_calls=0,new_samples=0)
    C.write(H.OUT/'baseline_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('identity','files','checks','limits')}),flush=True)


def prelim(data):
    rr=data['records'];op=[r for r in rr if r['opposite_winding']]
    return dict(open_halfedges=data['original_open_halfedges'],shared_hinge_boundary_pairs=len(rr),opposite_winding_pairs=len(op),
        rejection015=dict(collections.Counter(r['prefilter015'] for r in rr)),rejection025=dict(collections.Counter(r['prefilter025'] for r in rr)),
        nearest_opposite_POST_endpoint_gap_L=min((r['POST_endpoint_gap_L'] for r in op),default=None),
        direct_partner_slot113=any(r['partner_raw_slot']==113 for r in rr))


if __name__=='__main__':main()
