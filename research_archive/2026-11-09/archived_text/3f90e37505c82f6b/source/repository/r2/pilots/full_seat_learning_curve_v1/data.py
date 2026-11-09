"""Freeze existing coherent TRAIN67 SEAT_FULL inputs and source-interface labels.

Only read existing semantic reviews and exhaustive source-pair audits. No model,
new quality scan, source repair, retriangulation, noise draw or OT operation.
The corner labels are detached diagnostics, never model input or extra loss.
"""
from collections import Counter, defaultdict
from pathlib import Path
import html, json, os, time
import numpy as np
from . import common as C

DATA=C.SCALING/'data/train_manifest.json'
DEV_PREFIXES=('2d08a6','610b16','608b46')
VERSION='full_seat_coherent_existing_source_interface_v1'


def _components(face_ids,edges):
    faces=set(map(int,face_ids));adj={i:set() for i in faces}
    for owners in edges.values():
        keep=set(owners)&faces
        for i in keep:adj[i].update(keep-{i})
    groups=[];unseen=set(faces)
    while unseen:
        stack=[min(unseen)];component=[]
        while stack:
            i=stack.pop()
            if i not in unseen:continue
            unseen.remove(i);component.append(i);stack.extend(sorted(adj[i]&unseen,reverse=True))
        groups.append(sorted(component))
    return groups


def _source_edges(faces):
    result=defaultdict(list)
    for fi,face in enumerate(np.asarray(faces,np.int64)):
        if len(set(face.tolist()))!=3:raise ValueError('Repeated source vertex in face')
        for j in range(3):result[tuple(sorted((int(face[j]),int(face[(j+1)%3]))))].append(fi)
    return dict(result)


def transport_interface_mask(source_mask,sample):
    """Transport source canonical [N,3] labels after every OT/permutation.

    The maintained stream proves source_face_ids/corner_permutations reconstruct
    final x1/context, so using exactly those two arrays includes cyclic, free
    face, nested-OT corner and final all-face permutations once each.
    """
    source_mask=np.asarray(source_mask,dtype=bool)
    ids=np.asarray(sample['source_face_ids'],np.int64)
    corner=np.asarray(sample['corner_permutations'],np.int64)
    known=np.asarray(sample['known_mask'],bool)
    n=len(ids)
    if source_mask.shape!=(n,3) or corner.shape!=(n,3) or known.shape!=(n,):raise ValueError('Interface transport shape mismatch')
    if not np.array_equal(np.sort(ids),np.arange(n)):raise ValueError('Face transport is not bijective')
    if not np.array_equal(np.sort(corner,axis=1),np.tile(np.arange(3),(n,1))):raise ValueError('Corner transport is not bijective')
    mask=source_mask[ids[:,None],corner]
    if mask[known].any():raise ValueError('Known C entered free-interface diagnostic')
    if int(mask.sum())!=int(source_mask.sum()):raise ValueError('Interface corner count changed during OT transport')
    return mask


def build_interface(t,parent,quality):
    source=C.arrays(t['npz']);a=C.arrays(parent['npz']);n=int(t['N']);k=int(t['K'])
    full=a['full_target'].reshape(n,3,3);canonical_ids=a['source_face_vertex_ids'];faces=a['source_faces'];v=a['model_vertices']
    known=np.asarray(t['source_face_ids'],np.int64);free=np.asarray(t['free_source_ids'],np.int64);Cset=set(known.tolist())
    assert np.array_equal(np.sort(np.r_[known,free]),np.arange(n))
    assert C.same(v[canonical_ids],full) and C.same(source['C'],full[known])
    assert C.same(source['C_original'],v[faces[known]])
    assert np.array_equal(source['source_face_ids'],known) and np.array_equal(source['free_source_ids'],free)
    assert np.array_equal(canonical_ids,np.take_along_axis(faces,a['source_corner_permutations'],axis=1))
    assert set(known)==set(t['region_source_face_ids'])
    edgebook=_source_edges(faces);groups=_components(known,edgebook)
    assert len(groups)==int(t['C_components'])
    C_nonmanifold=sum(sum(fi in Cset for fi in owners)>2 for owners in edgebook.values())
    pairbook={tuple(row['pair']):row for row in quality['pair_contacts']}
    assert len(pairbook)==len(quality['pair_contacts'])
    assert quality['status']=='COMPLETE' and quality['independent_verification']
    assert quality['receipt']['new_unique_pair_tests']==n*(n-1)//2
    assert quality['input_npz_sha256']==parent['sha256']
    assert quality['full_target_sha256']==__import__('hashlib').sha256(full.tobytes()).hexdigest()
    mask=np.zeros((n,3),bool);legal=[];candidates=[]
    original_faces=a.get('retained_to_original_face',np.arange(n))
    knownlookup={int(fi):i for i,fi in enumerate(known)};freelookup={int(fi):i for i,fi in enumerate(free)}
    for (u,w),raw in sorted(edgebook.items()):
        co=[fi for fi in raw if fi in Cset];fo=[fi for fi in raw if fi not in Cset]
        if not co or not fo:continue
        reason=[]
        if len(raw)!=2 or len(set(raw))!=2:reason.append('NOT_EXACTLY_TWO_DISTINCT_SOURCE_FACES')
        if len(co)!=1 or len(fo)!=1:reason.append('NOT_SINGLE_C_SINGLE_FREE_INCIDENCE')
        pair=tuple(sorted([co[0],fo[0]]));contact=pairbook.get(pair)
        if contact is None:reason.append('NO_SAVED_SOURCE_CONTACT')
        elif contact.get('category')!='legal_contact' or contact.get('classification')!='shared_edge_contact':reason.append('NOT_SAVED_LEGAL_SHARED_EDGE')
        row=dict(source_vertex_u=u,source_vertex_v=w,source_incident_faces=raw,
            C_source_face_rows=co,free_source_face_rows=fo,source_pair=list(pair),
            saved_pair_classification=contact.get('classification') if contact else 'NA_MISSING',
            saved_pair_category=contact.get('category') if contact else 'NA_MISSING',
            accepted=not reason,reasons=reason,length_model=float(np.linalg.norm(v[u].astype(float)-v[w].astype(float))))
        if not reason:
            ci,fi=co[0],fo[0];corners=np.flatnonzero(np.isin(canonical_ids[fi],[u,w])).astype(np.int64)
            assert len(corners)==2
            mask[fi,corners]=True
            def directed(face):
                for j in range(3):
                    p,q=int(face[j]),int(face[(j+1)%3])
                    if {p,q}=={u,w}:return (p,q)
                raise ValueError('Shared edge absent')
            dc,df=directed(faces[ci]),directed(faces[fi])
            row.update(C_local_face_row=knownlookup[ci],free_local_face_row=freelookup[fi],
                C_original_face_id=int(original_faces[ci]),free_original_face_id=int(original_faces[fi]),
                free_canonical_corner_ids=corners.tolist(),original_source_winding_opposite=dc==df[::-1])
            legal.append(row)
        candidates.append(row)
    assert not mask[known].any()
    # A deliberately permuted label fixture, no Gaussian draw or OT solver.
    ids=np.arange(n)[::-1];corners=np.tile([2,0,1],(n,1));known_flag=np.isin(ids,known)
    transported=transport_interface_mask(mask,dict(source_face_ids=ids,corner_permutations=corners,known_mask=known_flag))
    assert transported.sum()==mask.sum()
    return source,a,mask,dict(source_C_components=len(groups),source_C_component_face_ids=groups,
        source_C_nonmanifold_edges=C_nonmanifold,legal_edges=legal,candidate_edges=candidates,
        source_interface_edges=len(legal),source_interface_free_faces=int(mask.any(1).sum()),
        source_interface_corners=int(mask.sum()),source_interface_coordinates=int(mask.sum())*3,
        diagnostic_status='AVAILABLE' if mask.any() else 'NA_NO_LEGAL_SOURCE_INTERFACE',
        corner_transport_fixture='PASS',free_source_ids=free.tolist(),C_source_ids=known.tolist())


def prepare():
    final=C.OUT/'inputs/training_data.json'
    if final.exists():
        done=C.read(final)
        for path,h in done['protected_hashes'].items():assert C.sha(path)==h,path
        for task in done['tasks'].values():
            for p,h in [(task['mask_path'],task['mask_sha256']),(task['edge_proof_path'],task['edge_proof_sha256'])]:assert C.sha(p)==h,p
        return done
    started=time.perf_counter();m=C.read(DATA);assert len(m['pools']['USER_EXPANDED_67'])==67
    all_tasks=sorted((t for t in m['tasks'].values() if t['condition_type']=='SEAT_FULL'),key=lambda t:t['task_id'])
    assert len(all_tasks)==50
    protected={}
    def protect(path,expected=None):
        path=str(C.resolve(path));h=protected.get(path)
        if h is None:h=C.sha(path);protected[path]=h
        if expected is not None:assert h==expected,path
        return h
    protect(DATA)
    cp=C.read(C.SOURCE.with_suffix('.json'));protect(C.SOURCE.with_suffix('.json'))
    initpath=C.QUICK/'training/SMALL_C200/initialization.json';init=C.read(initpath);protect(initpath)
    assert cp['sha256']==init['source_file_sha256'] and cp['generator_state_sha256']==init['loaded_generator_sha256']
    assert cp['global_step']==init['restored_global_step']==6000
    lineage=dict(checkpoint_path=str(C.SOURCE),checkpoint_bytes=C.SOURCE.stat().st_size,
        checkpoint_expected_sha256=cp['sha256'],checkpoint_full_file_hash_here='NOT_REHASHED; training loader must verify expected digest before use',
        generator_state_sha256=cp['generator_state_sha256'],optimizer_state_sha256=init['loaded_optimizer_sha256'],
        rng_sha256=init['loaded_rng_sha256'],optimizer_hyperparameters=init['hyperparameters'],global_step=6000,
        selected_state='generator_state; matched non-EMA AdamW continuation',ema_state_present=False,
        initialization_ancestor='Official generator EMA at original adaptation step0, then6000 conditional updates with EMA disabled',
        cpu_checkpoint_observation='Task subagent actually torch.load(weights_only=True,map_location=cpu,mmap=True) once before data preparation; keys/model_config/protocol/lineage inspected; all AdamW state steps equal6000',
        source_protocol_EMA=False,source_protocol_scheduler=False,optimizer_tensor_step_values=[6000],
        source_official_EMA_state_sha256='b32197096ecf8d662bfee18bbc4abf16cc4b4690cd28e5c47d14850729d2c222',
        source_official_file_sha256='bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d')
    C.write(C.OUT/'inputs/checkpoint_lineage.json',lineage)
    rows=[];tasks={};allproof=[]
    for t in all_tasks:
        uid=t['uid'];parent=m['parents'][uid]
        assert uid in m['pools']['USER_EXPANDED_67']
        for path,h in [(t['npz'],t['sha256']),(parent['npz'],parent['sha256']),
            (t['quality_path'],t['quality_sha256']),(t['annotation_source'],t['annotation_sha256']),(t['image'],t['image_sha256'])]:protect(path,h)
        quality=C.read(t['quality_path']);source,a,mask,proof=build_interface(t,parent,quality)
        reasons=[]
        if any(uid.startswith(prefix) for prefix in DEV_PREFIXES):reasons.append('DEV_PARENT_EXCLUDED')
        if t['K']<2:reasons.append('K_LT_2')
        if proof['source_C_components']!=1:reasons.append('DISCONNECTED_C_SOURCE_EDGE_COMPONENTS')
        if proof['source_C_nonmanifold_edges']:reasons.append('NONMANIFOLD_C_SOURCE_EDGE')
        if set(t['source_face_ids'])!=set(t['region_source_face_ids']) or t['area_C_over_region']!=1.:reasons.append('NOT_FULL_REVIEWED_REGION')
        for scope in ('CC','CF'):
            if t['CC_CF_FF'][scope]['illegal'] or t['CC_CF_FF'][scope]['unknown']:reasons.append('INHERITED_'+scope+'_ILLEGAL_OR_UNKNOWN')
        row=dict(task_id=t['task_id'],parent_id=uid,N=t['N'],K=t['K'],status='EXCLUDED' if reasons else 'ELIGIBLE',
            exclusion_reasons=reasons,C_components=proof['source_C_components'],C_nonmanifold_edges=proof['source_C_nonmanifold_edges'],
            source_interface_edges=proof['source_interface_edges'],source_interface_free_faces=proof['source_interface_free_faces'],
            source_interface_corners=proof['source_interface_corners'],source_interface_coordinates=proof['source_interface_coordinates'],
            diagnostic_status=proof['diagnostic_status'],CC_CF_FF=t['CC_CF_FF'],source_quality=t['source_quality'],
            review_kind=t['review_kind'],review_reason=t['review_reason'],region_declared=t['region_declared'],structure_tags=t['structure_tags'],
            condition_source=t['npz'],condition_source_sha256=t['sha256'],parent_npz=parent['npz'],parent_npz_sha256=parent['sha256'],
            source_quality_path=t['quality_path'],source_quality_sha256=t['quality_sha256'],
            annotation_source=t['annotation_source'],annotation_sha256=t['annotation_sha256'],
            inherited_review_image=t['image'],inherited_review_image_sha256=t['image_sha256'])
        rows.append(row)
        if reasons:continue
        path=C.OUT/'inputs/interface_masks'/(t['task_id']+'.npz');proofpath=path.with_suffix('.json')
        C.save_npz(path,source_interface_corner_mask=mask,source_interface_face_mask=mask.any(1),
            source_face_vertex_ids=a['source_face_vertex_ids'],source_C_face_rows=np.asarray(t['source_face_ids'],np.int64),
            source_free_face_rows=np.asarray(t['free_source_ids'],np.int64),task_id=np.asarray(t['task_id']),N=np.asarray(t['N']),K=np.asarray(t['K']))
        proof.update(task_id=t['task_id'],parent_id=uid,condition_source_sha256=t['sha256'],parent_sha256=parent['sha256'],quality_sha256=t['quality_sha256'],
            definition='Only the two endpoint corners of a free face incident to exactly one C face through an actual two-incidence source-ID edge, whose inherited pair classification is legal shared_edge_contact. Coordinates are parent canonical full_target row/corner order.',
            application='Detached diagnostics only. Uniform x-width augmentation preserves identities; transport via sample.source_face_ids and sample.corner_permutations after OT and all shuffles. Known C remains excluded.',
            new_quality_scans=0,model_calls=0)
        C.write(proofpath,proof)
        tasks[t['task_id']]=dict(row,mask_path=str(path),mask_sha256=C.sha(path),mask_key='source_interface_corner_mask',mask_count=int(mask.sum()),
            edge_proof_path=str(proofpath),edge_proof_sha256=C.sha(proofpath),source_task=t)
        allproof.extend(dict(task_id=t['task_id'],parent_id=uid,**edge) for edge in proof['candidate_edges'])
    eligible=[row for row in rows if row['status']=='ELIGIBLE'];excluded=[row for row in rows if row['status']=='EXCLUDED']
    assert len(eligible)==47 and len(excluded)==3
    assert len({row['parent_id'] for row in eligible})==47
    assert all(not any(row['parent_id'].startswith(p) for p in DEV_PREFIXES) for row in eligible)
    C.csvwrite(C.OUT/'inputs/all50_eligibility.csv',rows);C.csvwrite(C.OUT/'inputs/source_interface_edges.csv',allproof)
    C.write(C.OUT/'inputs/exclusions.json',excluded)
    htmlrows=[]
    for row in rows:
        image=os.path.relpath(C.resolve(row['inherited_review_image']),C.OUT/'inputs').replace('\\','/')
        htmlrows.append('<tr><td>'+html.escape(row['parent_id'])+'<br>'+html.escape(row['status'])+'</td><td>K='+str(row['K'])+'; N='+str(row['N'])+'; components='+str(row['C_components'])+'; legal interface edges='+str(row['source_interface_edges'])+'</td><td>'+html.escape(row['review_reason'])+'<br>Inherited FF risk: '+html.escape(str(row['CC_CF_FF']['FF']))+'</td><td><a href="'+html.escape(image)+'"><img loading="lazy" width="450" src="'+html.escape(image)+'"></a></td></tr>')
    content='<!doctype html><meta charset="utf-8"><title>Existing SEAT_FULL input qualification</title><style>body{font:15px sans-serif}table{border-collapse:collapse}td{border:1px solid #bbb;padding:8px;vertical-align:top}</style><h1>Existing TRAIN67 SEAT_FULL:47 eligible,3 disconnected exclusions</h1><p>These are inherited AI_PROXY input-review images and reasons. No new semantic relabeling or outcome-based filtering. FF-only risks are retained. Gallery sheets may show multiple nearby parents; use parent IDs and original annotations.</p><table>'+''.join(htmlrows)+'</table>'
    C._write_bytes(C.OUT/'inputs/accepted_input_review.html',content.encode('utf-8'))
    manifest=dict(schema=VERSION,status='FROZEN',source_training_data=str(DATA),source_training_data_sha256=protected[str(DATA)],
        train_task_ids=sorted(tasks),parent_ids=sorted(row['parent_id'] for row in eligible),tasks=tasks,eligible=eligible,exclusions=excluded,
        all_seat_full_candidates=50,eligible_parent_count=47,condition_type='SEAT_FULL',K_min=min(t['K'] for t in tasks.values()),K_max=max(t['K'] for t in tasks.values()),
        K_histogram=dict(sorted(Counter(t['K'] for t in tasks.values()).items())),
        source_interface_edges_total=sum(t['source_interface_edges'] for t in tasks.values()),
        source_interface_corners_total=sum(t['source_interface_corners'] for t in tasks.values()),
        interface_empty_tasks=[tid for tid,t in tasks.items() if not t['mask_count']],
        selection='All inherited TRAIN67 SEAT_FULL with K>=2, one source-edge-connected component, full reviewed region, inherited CC/CF no illegal or unknown. Three disconnected exclusions only; no filtering by generated outcomes or Y-only risk.',
        training_reader='Existing r2.pilots.user_data_scaling_32_67_v1.training_data.Dataset on source_training_data; restrict records to train_task_ids.',
        interface_mask_contract='bool[N,3] in source canonical full_target rows/corners; C rows false. Free endpoint corners on inherited legal two-face shared C/Y source-ID edges; diagnostics only; no loss/input change.',
        interface_transport='source_mask[sample.source_face_ids[:,None],sample.corner_permutations]; assert no known slots and preserved count',
        source_checkpoint_lineage_path=str(C.OUT/'inputs/checkpoint_lineage.json'),source_checkpoint_lineage_sha256=C.sha(C.OUT/'inputs/checkpoint_lineage.json'),
        protected_hashes=protected,new_quality_scans=0,new_OT_calls=0,new_noise_draws=0,model_calls=0,
        inherited_reviews_only=True,entry_source_sha256=C.sha(__file__))
    C.write(final,manifest)
    for path,h in protected.items():assert C.sha(path)==h,path
    C.write(C.OUT/'inputs/data_completion.json',dict(status='PASS',manifest_sha256=C.sha(final),
        examined_conditions=50,eligible_conditions=47,excluded_conditions=3,protected_files=len(protected),
        interface_edges=manifest['source_interface_edges_total'],interface_corners=manifest['source_interface_corners_total'],
        checkpoint_CPU_metadata_reads_this_task=1,new_quality_scans=0,model_forwards=0,OT_calls=0,seconds=time.perf_counter()-started))
    C.event('TRAIN_DATA_AND_INTERFACE_LABELS_FROZEN',eligible=47,excluded=3,interface_edges=manifest['source_interface_edges_total'],source_hashes_unchanged=len(protected))
    return manifest


if __name__=='__main__':
    m=prepare();print(json.dumps(dict(status=m['status'],eligible=len(m['tasks']),excluded=len(m['exclusions']),K_histogram=m['K_histogram'],interface_edges=m['source_interface_edges_total'],interface_corners=m['source_interface_corners_total'],manifest=str(C.OUT/'inputs/training_data.json')),sort_keys=True))