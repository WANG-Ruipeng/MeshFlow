"""Frozen CPU-only registration for the finite zero-training usability pilot."""
from __future__ import annotations
from collections import Counter, deque
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'native_t1/runs/chair_control_usability_quick_v1'
CHAIR=ROOT/'native_t1/runs/chair_domain_control_v1'
HYBRID=ROOT/'native_t1/runs/chair_hybrid_coupling_v1'
SOURCE=ROOT/'native_t1/runs/control_mvp48_v1'
V2=ROOT/'native_t1/runs/chair_moment_channel_v2'
EXPERIMENT='CHAIR_CONTROL_USABILITY_QUICK_PILOT_V1'
ARMS=('OT_BASE','OT_HYBRID')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


def bits(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def freeze_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read_json(path)!=obj:raise ValueError('Frozen JSON differs: '+str(path))
    else:
        with path.open('x',encoding='utf-8') as f:
            json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return digest(path)


def freeze_npz(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(arrays) or any(not bits(z[k],np.asarray(v)) for k,v in arrays.items()):
                raise ValueError('Frozen array archive differs: '+str(path))
    else:
        with path.open('xb') as f:
            np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    return digest(path)


def resolve(path):
    p=Path(path);return p if p.is_absolute() else CHAIR/p


def seed(uid,condition_identity,candidate):
    text=json.dumps([EXPERIMENT,'candidate_Gaussian_v1',uid,condition_identity,int(candidate)],separators=(',',':'))
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:4],'little')


def area(tri):
    tri=np.asarray(tri,dtype=np.float64)
    return float((.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)).sum())


def components(ids,adjacency):
    unseen=set(int(x) for x in ids);sizes=[]
    while unseen:
        queue=[min(unseen)];reached=set()
        while queue:
            i=queue.pop()
            if i in reached:continue
            reached.add(i);queue.extend((adjacency[i]&unseen)-reached)
        unseen-=reached;sizes.append(len(reached))
    return sizes


def nested_additions(base,initial,targets,adjacency):
    """One predetermined source-only BFS construction; never inspects predictions."""
    centers=base['full_target'].reshape(-1,3,3).astype(np.float64).mean(1)
    initial=list(map(int,initial));c1=set(initial)
    distance=np.linalg.norm(centers[:,None]-centers[initial][None],axis=-1).min(1)
    c1_neighbors=set().union(*(adjacency[i] for i in c1))-c1
    current=list(initial);selected=set(current);result=[]
    for target in targets:
        detail=dict(target_K=int(target),seed_faces=[],adjacent_to_C1_fallback_used=False)
        if not len(current)<target<len(centers):raise ValueError('Invalid strictly nested target face count')
        while len(current)<target:
            remaining=set(range(len(centers)))-selected
            detached=remaining-c1_neighbors
            allowed=detached if detached else remaining
            if not detached:detail['adjacent_to_C1_fallback_used']=True
            if not allowed:raise ValueError('No remaining source faces')
            start=min(allowed,key=lambda i:(-float(distance[i]),i))
            detail['seed_faces'].append(dict(source_face_id=start,min_distance_to_C1_centroids=float(distance[start])))
            queue=deque([start]);seen={start}
            while queue and len(current)<target:
                i=queue.popleft()
                if i not in selected:current.append(i);selected.add(i)
                for j in sorted(adjacency[i]):
                    if j in allowed and j not in seen:seen.add(j);queue.append(j)
        result.append((list(current),detail))
    return result


def true_interface(base,ids,topology):
    incident,original,adjacency,_=topology;known=set(map(int,ids))
    interface={k:[] for k in ('edge_source_vertex_ids','known_source_face_ids','free_source_face_ids',
        'known_original_source_face_ids','free_original_source_face_ids','known_corner_ids','free_corner_ids')}
    ambiguous=[]
    for edge,faces in sorted(incident.items()):
        inside=[i for i in faces if i in known];outside=[i for i in faces if i not in known]
        if not inside or not outside:continue
        expected={int(base['retained_to_original_face'][i]) for i in faces}
        if len(faces)!=2 or len(inside)!=1 or len(outside)!=1 or len(original[edge])!=2 or set(original[edge])!=expected:
            ambiguous.append(dict(edge=list(edge),retained_faces=faces,original_faces=original[edge]));continue
        interface['edge_source_vertex_ids'].append(list(edge))
        for role,face in (('known',inside[0]),('free',outside[0])):
            interface[role+'_source_face_ids'].append(face)
            interface[role+'_original_source_face_ids'].append(int(base['retained_to_original_face'][face]))
            interface[role+'_corner_ids'].append([int(np.flatnonzero(base['source_face_vertex_ids'][face]==v)[0]) for v in edge])
    status='PASS' if not ambiguous and interface['edge_source_vertex_ids'] else 'NOT_APPLICABLE'
    edges=base['model_vertices'][np.asarray(interface['edge_source_vertex_ids'],np.int64).reshape(-1,2)]
    return interface,edges,status,ambiguous


def prepare(out=OUT):
    out=Path(out).resolve();reg=out/'registration'
    if out==CHAIR.resolve() or out==HYBRID.resolve():raise ValueError('Output must be independent')
    reg.mkdir(parents=True,exist_ok=True)
    complete=reg/'registration_summary.json'
    if complete.exists():
        result=read_json(complete)
        for rel,sha in result['output_hashes'].items():
            if digest(out/rel)!=sha:raise ValueError('Completed registration output changed: '+rel)
        return result
    source_manifest={}
    def source(path,role):
        p=Path(path).resolve();key=str(p)
        if key not in source_manifest:
            source_manifest[key]=dict(path=key,sha256=digest(p),bytes=p.stat().st_size,role=role)
        return source_manifest[key]['sha256']
    def read_source(path,role):
        source(path,role);return read_json(path)
    panels=read_source(CHAIR/'panels.json','original fixed A/B/C registration')
    records=[r for r in panels['A'] if r['split']=='validation' and r['ratio']==.2 and r['alpha']==1.]
    parents=sorted({r['uid'] for r in records})
    if len(parents)!=8 or len(records)!=16:raise ValueError('Original A20 validation panel differs')
    indexed={(r['uid'],r['noise_index']):r for r in records}
    protocol=dict(schema='quick_usability_condition_construction_v1',experiment=EXPERIMENT,
        parents=parents,nested_parents=parents[:2],main_alpha=1.,main_candidates=[0,1,2,3],nested_candidates=[0,1],
        original_noise='Candidates 0/1 of C1 copy original archived Gaussian bytes; no redraw',
        new_seed='first4 SHA256 bytes little-endian of compact JSON [EXPERIMENT,candidate_Gaussian_v1,uid,condition_identity,candidate]; excludes arm',
        new_noise='numpy.random.RandomState(seed).randn(N-K,3,3).astype(float32); C-first z with known slots zero',
        nested_K='floor(ratio*N+0.5), ratios .3 and .4; original N unchanged',
        nested_rule='Preserve C1 source-face order; rank remaining faces by descending minimum FP64 centroid distance to C1 centroid set, tie ascending retained source ID. Sorted-neighbor FIFO BFS. Prefer remaining subgraph excluding faces adjacent to C1; if a component exhausts choose next farthest seed. Only after all nonadjacent faces exhaust use C1-adjacent remainder. Build C2 then extend C2 to C3. No quality-dependent search/retry.',
        adjacency='Original repair_v2 complete-source-edge incidence, same _topology and sorted-neighbor FIFO as chair_domain_control_v1',
        Gamma='All trusted degree2 known/free crossings whose retained/original incidence agrees; if any ambiguous crossing or no trusted edge, Gamma NOT_APPLICABLE. Does not change condition eligibility.',
        patch_label='multi-piece only when actual known shared-edge connected component count > 1',
        geometry='Original archived model coordinates and source rows; no vertex edit, repair, renormalization, GT OT or semantic noise slots',
        normalization='L0 is original full GT bbox diagonal, scoring/display only; generator gets C,z,N,K,known/valid masks only',
        source_registration_sha256=source_manifest[str((CHAIR/'panels.json').resolve())]['sha256'])
    freeze_json(reg/'construction_protocol.json',protocol)

    # Exact retained endpoints; never load tensors or restore the optimizer.
    manifest=read_source(HYBRID/'checkpoint_manifest.json','retained endpoint identities')
    read_source(HYBRID/'protocol.json','original training and sampler contract')
    endpoints=[]
    for arm in ARMS:
        item=dict(next(r for r in manifest['endpoints'] if r['metadata']['arm']==arm))
        meta=item['metadata']
        if (meta['pilot_step'],meta['conditional_total_step'],meta['lambda_variance'],meta['loss_recipe'])!=(1000,5000,0. if arm=='OT_BASE' else .25,'fm'):
            raise ValueError('Wrong registered endpoint')
        try:
            actual=source(item['path'],'fixed model checkpoint; file hashing only')
            if actual!=item['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
            if arm=='OT_HYBRID' and actual!='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80':raise ValueError('Wrong START')
            item.update(status='PASS',actual_sha256=actual)
        except (FileNotFoundError,ValueError) as exc:item.update(status='BLOCKED',reason=str(exc))
        endpoints.append(item)
    freeze_json(reg/'checkpoints.json',endpoints)
    model_lookup={r['metadata']['arm']:r for r in endpoints}
    implementation=read_source(HYBRID/'registration/implementation_hashes.json','original executed sampler hash')
    old_sources=read_source(HYBRID/'registration/source_hashes_before.json','original executed model dependencies')
    critical=['experiments/mf_h1_local0/chair_hybrid_coupling_v1/sampling.py',
        'experiments/mf_h1_local0/control_mvp48/model_mvp.py','experiments/mf_h1_local0/control_mvp48/eval_runtime.py',
        'native_t1/model.py','native_t1/context_geometry.py','native_t1/runtime.py','models/equidit.py','models/utils.py']
    code_checks=[]
    for rel in critical:
        expected=implementation.get(rel,old_sources.get(rel))
        try:actual=source(ROOT/rel,'executed sampler/model source dependency')
        except FileNotFoundError:actual=None
        code_checks.append(dict(path=rel,expected_sha256=expected,actual_sha256=actual,status='PASS' if expected and actual==expected else 'BLOCKED'))
    freeze_json(reg/'cache_execution_code_audit.json',code_checks)
    code_ok=all(r['status']=='PASS' for r in code_checks)
    legacy_rows={arm:{r['case_id']:r for r in read_source(HYBRID/f'evaluation/{arm}/generated_rows.json','original raw index')['rows']} for arm in ARMS}
    score_rows={arm:{r['case_id']:r for r in read_source(HYBRID/f'evaluation/{arm}/scoring/per_output.json','original score identity index')} for arm in ARMS}
    v2=read_source(V2/'evaluation/START/generated_rows.json','MOMENT START cache alias identity')
    if v2['source_checkpoint_sha256']!=model_lookup['OT_HYBRID']['checkpoint_sha256']:raise ValueError('V2 START is not HYBRID endpoint')
    v2rows={r['case_id']:r for r in v2['rows']}
    task_manifest=read_source(CHAIR/'task_manifest.json','frozen source-face task identities')
    mesh_manifest=read_source(SOURCE/'data/manifest.json','repair_v2 prepared mesh identity')
    mesh_lookup={r['uid']:r for r in mesh_manifest['objects']}
    source(SOURCE/'data/patches.json','repair_v2 original condition and face provenance')
    source(SOURCE/'data/load_sample.py','strict original prepared-data loader')
    from experiments.mf_h1_local0.chair_domain_control_v1.prepare import load_task_sample,_topology
    from experiments.mf_h1_local0.evaluate import sample_mesh_surface
    from experiments.mf_h1_local0.control_mvp48.eval_scoring import prepare_reference_from_sample
    for rel in ('experiments/mf_h1_local0/chair_domain_control_v1/prepare.py','experiments/mf_h1_local0/control_mvp48/eval_scoring.py','experiments/mf_h1_local0/evaluate.py'):
        source(ROOT/rel,'condition/reference construction code')
    conditions=[];arrays_by_condition={};condition_lookup={};construction=[]
    for uid in parents:
        r=indexed[(uid,0)];mesh=mesh_lookup[uid]
        if source(SOURCE/mesh['prepared_path'],'selected repair_v2 mesh')!=mesh['prepared_sha256']:raise ValueError('Prepared mesh changed')
        base=load_task_sample(SOURCE,task_manifest,r['task_id'],1.)
        topology=_topology(base);initial=list(map(int,base['source_face_ids']))
        variants=[(initial,dict(source='original A20 validation',target_K=len(initial)))]
        nested_error=None
        if uid in parents[:2]:
            try:variants+=nested_additions(base,initial,[int(np.floor(.3*base['N']+.5)),int(np.floor(.4*base['N']+.5))],topology[2])
            except Exception as exc:nested_error=repr(exc)
        common_ids=np.asarray(sorted(set(range(base['N']))-set(variants[-1][0])),np.int64) if len(variants)==3 else np.empty(0,np.int64)
        for level,(ids,detail) in enumerate(variants,1):
            ids=np.asarray(ids,np.int64);free=np.asarray(sorted(set(range(base['N']))-set(ids.tolist())),np.int64)
            full=base['full_target'].reshape(-1,3,3);C=full[ids].copy();Y=full[free].copy()
            interface,gamma,gamma_status,ambiguous=true_interface(base,ids,topology)
            condition_id=f'{uid}_C{level}';N,K=len(full),len(ids);L0=float(base['bbox_L0'])
            identity=hashlib.sha256(json.dumps(dict(uid=uid,C=array_hash(C),source_ids=array_hash(ids),N=N),sort_keys=True).encode()).hexdigest()
            ref_npz=resolve(r['ref_npz']) if level==1 else None
            ref_json=resolve(r['ref_json']) if level==1 else None
            if level==1:
                source(ref_npz,'original fixed512 reference arrays');audit=read_source(ref_json,'original reference identity')
                with np.load(ref_npz,allow_pickle=False) as z:ref={k:z[k].copy() for k in z.files}
                oldinput=resolve(r['input_path'])
                if source(oldinput,'original candidate0 input')!=r['input_sha256']:raise ValueError('Original input hash changed')
                with np.load(oldinput,allow_pickle=False) as z:
                    if not bits(z['C'],C) or not bits(z['source_face_ids'],ids) or not bits(z['gt_free'].reshape(-1,3,3),Y):raise ValueError('C1 source/raw input geometry identity mismatch')
            elif gamma_status=='PASS':
                modified=dict(base,source_face_ids=ids,free_source_ids=free,constraints=C,free_target=Y.reshape(-1,9),interface=interface)
                prepared=prepare_reference_from_sample(modified);ref=prepared['arrays'];audit=prepared['audit']
            else:
                # Preserve original 512 coverage/pairwise RNG rules even when Gamma is NA.
                rng=np.random.RandomState(9911);sample_mesh_surface(Y,512,rng)
                ref=dict(coverage_points=sample_mesh_surface(Y,512,np.random.RandomState(9973)),
                    pairwise_target_points=sample_mesh_surface(Y,512,rng),gamma_points=np.empty((0,3),np.float64),gamma_weights=np.empty(0,np.float64))
                audit=dict(schema='original512_rules_gamma_NA',surface_seed=9911,coverage_seed=9973,gamma_status=gamma_status)
            counts=components(ids,topology[2])
            cpath=out/'inputs/conditions'/f'{condition_id}.npz'
            payload=dict(C=C,GT_full=full.copy(),GT_free=Y,source_face_ids=ids,free_source_ids=free,
                known_source_face_ids=ids,original_known_source_face_ids=base['retained_to_original_face'][ids],
                original_free_source_face_ids=base['retained_to_original_face'][free],
                gamma_edges=gamma if gamma_status=='PASS' else np.empty((0,2,3),np.float32),
                gamma_edge_source_vertex_ids=np.asarray(interface['edge_source_vertex_ids'],np.int64).reshape(-1,2),
                L0=np.asarray(L0),N=np.asarray(N,np.int64),K=np.asarray(K,np.int64),
                legacy_query_points=np.asarray(ref['coverage_points']),legacy_pairwise_points=np.asarray(ref['pairwise_target_points']),
                legacy_gamma_points=np.asarray(ref['gamma_points']),legacy_gamma_weights=np.asarray(ref['gamma_weights']),
                common_source_face_ids=common_ids,Y_common=full[common_ids].copy(),
                model_vertices=base['model_vertices'],source_face_vertex_ids=base['source_face_vertex_ids'])
            if not len(common_ids):
                payload.pop('Y_common');payload.pop('common_source_face_ids')
            ch=freeze_npz(cpath,payload);arrays_by_condition[condition_id]=payload
            meta=dict(condition_id=condition_id,uid=uid,context_level=level,status='READY',condition_npz=str(cpath),condition_sha256=ch,
                condition_identity=identity,C_sha256=array_hash(C),known_source_face_ids_sha256=array_hash(ids),N=N,K=K,L0=L0,
                source_task_id=r['task_id'] if level==1 else None,known_face_ratio=K/N,known_area_ratio=area(C)/area(full),
                known_components=len(counts),known_component_sizes=counts,known_span=np.ptp(C.reshape(-1,3),axis=0).tolist(),
                condition_description='multi-piece condition' if len(counts)>1 else 'increased context',
                free_budget=N-K,common_area=area(full[common_ids]) if len(common_ids) else None,
                gamma_status=gamma_status,gamma_edge_count=len(gamma) if gamma_status=='PASS' else 0,
                trusted_edge_count_before_ambiguity_check=len(gamma),gamma_ambiguities=ambiguous,
                legacy_ref_npz=str(ref_npz) if ref_npz else None,legacy_ref_json=str(ref_json) if ref_json else None,
                reference_audit=audit,construction=detail)
            conditions.append(meta);condition_lookup[condition_id]=meta;construction.append(dict(uid=uid,level=level,known_ids=ids.tolist(),**detail))
        if nested_error:
            for level in (2,3):
                cid=f'{uid}_C{level}';meta=dict(condition_id=cid,uid=uid,context_level=level,status='NOT_APPLICABLE',reason=nested_error)
                conditions.append(meta);condition_lookup[cid]=meta
    freeze_json(reg/'conditions.json',conditions);freeze_json(reg/'construction_trace.json',construction)
    requests=[];cache_audit=[];input_records=[]
    for condition in conditions:
        uid=condition['uid'];level=condition['context_level'];cid=condition['condition_id']
        candidates=range(4) if level==1 else range(2)
        for candidate in candidates:
            common=dict(uid=uid,condition_id=cid,context_level=level,candidate=candidate,
                panel='main' if level==1 else 'multi',panels=(['main','multi'] if uid in parents[:2] and candidate<2 else ['main']) if level==1 else ['multi'],
                condition_npz=condition.get('condition_npz'),condition_sha256=condition.get('condition_sha256'),
                sensitivity_check=level==1 and uid in parents[:2] and candidate==0,
                identity_replay=level==1 and uid==parents[0] and candidate==0)
            if condition['status']!='READY':
                for arm in ARMS:requests.append(dict(common,arm=arm,request_id=f'{cid}_n{candidate}_{arm}',status='NA',reason=condition['reason'],input_npz=None,legacy_case_json=None,legacy_raw=None))
                continue
            a=arrays_by_condition[cid];N,K=condition['N'],condition['K'];C=a['C']
            legacy_record=indexed[(uid,candidate)] if level==1 and candidate<2 else None
            if legacy_record:
                oldinput=resolve(legacy_record['input_path'])
                if source(oldinput,'original fixed candidate input')!=legacy_record['input_sha256']:raise ValueError('Historical input changed')
                with np.load(oldinput,allow_pickle=False) as z:bank={k:z[k].copy() for k in ('z','C','known_mask','valid_mask','N','K')}
                if not bits(bank['C'],C) or int(bank['N'])!=N or int(bank['K'])!=K:raise ValueError('Historical candidate C/N/K differs')
                gaussian_seed=legacy_record['gaussian_seed'];legacy_input=str(oldinput)
            else:
                gaussian_seed=seed(uid,condition['condition_identity'],candidate)
                free_noise=np.random.RandomState(gaussian_seed).randn(N-K,3,3).astype(np.float32)
                z=np.zeros((1,N,9),np.float32);z[:,K:]=free_noise.reshape(1,N-K,9)
                bank=dict(z=z,C=C.copy(),known_mask=np.arange(N)<K,valid_mask=np.ones(N,bool),N=np.asarray(N),K=np.asarray(K));legacy_input=None
            input_path=out/'inputs/gaussians'/f'{cid}_n{candidate}.npz';ih=freeze_npz(input_path,bank)
            common.update(input_npz=str(input_path),input_sha256=ih,condition_identity=condition['condition_identity'],
                gaussian_seed=gaussian_seed,gaussian_sha256=array_hash(bank['z']),C_array_sha256=array_hash(C),
                known_mask_sha256=array_hash(bank['known_mask']),N=N,K=K,L0=condition['L0'],legacy_input_npz=legacy_input)
            input_records.append(dict(condition_id=cid,candidate=candidate,gaussian_seed=gaussian_seed,input_npz=str(input_path),input_sha256=ih,gaussian_sha256=common['gaussian_sha256'],legacy_input_npz=legacy_input))
            for arm in ARMS:
                req=dict(common,arm=arm,request_id=f'{cid}_n{candidate}_{arm}',checkpoint_sha256=model_lookup[arm]['checkpoint_sha256'],
                    status='READY',legacy_case_json=None,legacy_raw=None,legacy_score_path=None,cache_status='NOT_REQUESTED')
                if model_lookup[arm]['status']!='PASS':req.update(status='NA',reason='MODEL_ASSET_BLOCKED')
                if legacy_record:
                    old=legacy_rows[arm][legacy_record['case_id']];case=HYBRID/f'evaluation/{arm}/cases/{legacy_record["case_id"]}.json'
                    req.update(legacy_case_json=str(case),legacy_case_id=legacy_record['case_id'],legacy_raw=old.get('raw_path'),legacy_raw_sha256=old.get('raw_sha256'),
                        legacy_metadata=old.get('metadata',{}),cache_status='UNVERIFIED')
                    checks={}
                    try:
                        checks['execution_source_hashes']=code_ok
                        case_document=read_source(case,'selected historical case metadata')
                        checks['case_metadata_matches_index']=case_document==old
                        checks['checkpoint']=old['checkpoint_sha256']==req['checkpoint_sha256'] and model_lookup[arm]['status']=='PASS'
                        checks['request_identity']=all(old[k]==legacy_record[k] for k in ('uid','task_id','N','K','alpha','noise_index','input_sha256'))
                        checks['raw_file_hash']=source(old['raw_path'],'selected historical raw')==old['raw_sha256']
                        with np.load(old['raw_path'],allow_pickle=False) as z:
                            raw=z['raw'].copy();checks['raw_array_hash']=array_hash(raw)==old['raw_array_sha256']
                            checks['C_noise_mask_bytes']=bits(z['C'],bank['C']) and bits(z['noise'],bank['z']) and bits(z['known_mask'],bank['known_mask'])
                            checks['raw_geometry_contract']=raw.dtype==np.float32 and raw.shape==(N,9) and np.isfinite(raw).all().item() and bits(raw[:K],C.reshape(K,9))
                        meta=old['metadata']
                        checks['sampler_contract']=all(meta.get(k)==v for k,v in dict(solver='original_Euler50',nfe=50,accepted_states=51,forward_dtype='bfloat16',integration_dtype='float32',known_all51_bitwise=True,known_state_checks=51,actual_Geo_recomputations=50,GT_model_inputs=False,geometry_repairs=0).items())
                        checks['metadata_actual_arrays']=meta['noise_sha256']==array_hash(bank['z']) and meta['condition_sha256']==array_hash(C.reshape(K,9)) and meta['N']==N and meta['K']==K
                        if arm=='OT_HYBRID':checks['V2_START_same_raw']=v2rows[old['case_id']]['raw_sha256']==old['raw_sha256'] and v2rows[old['case_id']]['raw_path']==old['raw_path']
                        score=score_rows[arm].get(old['case_id'])
                        if score and score.get('raw_sha256')==old['raw_sha256'] and Path(score['score_path']).exists():
                            if source(score['score_path'],'selected historical score')==score['score_sha256']:
                                req.update(legacy_score_path=score['score_path'],legacy_score_sha256=score['score_sha256'],legacy_score_row=score)
                        if not all(checks.values()):raise ValueError('Historical cache identity checks failed')
                        req.update(status='CACHED',cache_status='PASS')
                    except Exception as exc:
                        req.update(cache_status='REJECTED',cache_reason=repr(exc),rejected_legacy_raw=req.get('legacy_raw'),rejected_legacy_case_json=req.get('legacy_case_json'),legacy_raw=None,legacy_case_json=None)
                        if model_lookup[arm]['status']!='PASS':req['status']='NA'
                    cache_audit.append(dict(request_id=req['request_id'],status=req['cache_status'],checks=checks,reason=req.get('cache_reason')))
                requests.append(req)
    if len(requests)!=80 or len({r['request_id'] for r in requests})!=80:raise ValueError('Unique request budget differs')
    if sum(r['panel']=='main' for r in requests)!=64 or sum('multi' in r['panels'] for r in requests)!=24:raise ValueError('Panel membership budget differs')
    freeze_json(out/'requests.json',requests);freeze_json(reg/'candidate_inputs.json',input_records);freeze_json(reg/'cache_audit.json',cache_audit)
    freeze_json(reg/'source_files.json',list(source_manifest.values()))
    headers=['request_id','uid','condition_id','context_level','candidate','arm','panel','panels','status','input_npz','input_sha256','condition_npz','gaussian_seed','gaussian_sha256','legacy_case_json','legacy_raw','legacy_score_path','cache_status','reason']
    csv_path=out/'requests.csv'
    if not csv_path.exists():
        with csv_path.open('x',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=headers);w.writeheader()
            for r in requests:w.writerow({k:json.dumps(r.get(k)) if isinstance(r.get(k),list) else r.get(k) for k in headers})
    generated=[out/'requests.json',out/'requests.csv']+sorted(reg.glob('*.json'))+sorted((out/'inputs').rglob('*.npz'))
    result=dict(status='PASS',parents=parents,nested_parents=parents[:2],condition_count=len(conditions),unique_requests=len(requests),
        main_requests=sum(r['panel']=='main' for r in requests),multi_requests_including_C1=sum('multi' in r['panels'] for r in requests),
        request_status_counts=dict(Counter(r['status'] for r in requests)),checkpoint_status={r['metadata']['arm']:r['status'] for r in endpoints},
        cache_attempts=len(cache_audit),cache_hits=sum(r['status']=='PASS' for r in cache_audit),new_input_banks=len(input_records),
        model_forwards=0,backwards=0,optimizer_updates=0,source_file_count=len(source_manifest),
        condition_summary=[{k:c.get(k) for k in ('condition_id','N','K','known_components','known_area_ratio','gamma_status','status')} for c in conditions],
        output_hashes={str(p.relative_to(out)):digest(p) for p in generated if p!=complete},implementation_sha256=digest(__file__))
    freeze_json(complete,result)
    return result


if __name__=='__main__':
    print(json.dumps(prepare(),indent=2))
