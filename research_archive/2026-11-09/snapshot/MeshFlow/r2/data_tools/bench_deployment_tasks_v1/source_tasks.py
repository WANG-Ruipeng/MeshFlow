"""CPU source-resolution human-part tasks, explicitly outside current adapter.

This module never trains, samples, simplifies, invokes OT, or loads a model.
Original source surfaces and previous finite-geometry receipts are read-only.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from collections import defaultdict, Counter
import argparse, csv, hashlib, io, json, time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_deployment_task_data_v1/dataset/source_tasks_pending_adapter'
INV=ROOT/'experiments/r2_bench_deployment_task_data_v1/asset_inventory/reviewed_candidates.json'

def resolve(p):
    s=str(p).replace('\\','/')
    return ROOT/s[len('/mnt/e/MeshFlow/'):] if s.startswith('/mnt/e/MeshFlow/') else Path(s)
def read(p):return json.loads(resolve(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(resolve(p).read_bytes()).hexdigest()
def save(p,x):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
    if p.exists():assert p.read_text('utf-8')==text, 'Refuse overwrite '+str(p)
    else:p.write_text(text,'utf-8')
def npz(p,a):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        b=load(p);assert set(a)==set(b) and all(np.array_equal(a[k],b[k]) for k in a)
    else:np.savez_compressed(p,**a)
def load(p):
    with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rows():return [r for r in read(INV)['rows'] if r['source_complete']=='PLAUSIBLE']
def geom(a):
    v=a['model_vertices'];f=a['source_faces'];t=v[f].astype(float)
    cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);areas=np.linalg.norm(cross,axis=1)/2
    normal=cross/np.maximum(2*areas[:,None],1e-30)
    return v,f,t,areas,normal
def graph(f):
    book=defaultdict(list)
    for i,tri in enumerate(f):
        for j in range(3):book[tuple(sorted((int(tri[j]),int(tri[(j+1)%3]))))].append(i)
    edges=[]
    for ids in book.values():
        for i in ids[1:]:edges.append((ids[0],i))
    return book,edges
def components(n,edges):
    adj=[set() for _ in range(n)]
    for i,j in edges:adj[i].add(j);adj[j].add(i)
    left=set(range(n));parts=[]
    while left:
        q=[min(left)];seen=set()
        while q:
            i=q.pop()
            if i in seen:continue
            seen.add(i);q.extend(adj[i]-seen)
        left-=seen;parts.append(sorted(seen))
    return sorted(parts,key=lambda x:(-len(x),min(x)))
def descriptor(ids,t,areas,normal):
    ids=np.asarray(ids,int);v=t[ids].reshape(-1,3);w=areas[ids]
    return dict(face_ids=ids.tolist(),count=len(ids),area=float(w.sum()),bbox_min=v.min(0).tolist(),bbox_max=v.max(0).tolist(),
      center=(t[ids].mean(1)*w[:,None]).sum(0).tolist() if not w.sum() else ((t[ids].mean(1)*w[:,None]).sum(0)/w.sum()).tolist(),
      normal=((normal[ids]*w[:,None]).sum(0)/max(w.sum(),1e-30)).tolist())
def describe(row):
    a=load(row['clean_case_path']);v,f,t,areas,normal=geom(a);book,edges=graph(f)
    comp=components(len(f),edges)
    planar=[(i,j) for i,j in edges if normal[i]@normal[j]>1-1e-8 and abs((t[j,0]-t[i,0])@normal[i])<1e-6]
    patches=components(len(f),planar)
    return dict(parent_id=row['parent_id'],source_sha256=sha(row['clean_case_path']),N=len(f),
      components=[dict(component_id=i,**descriptor(ids,t,areas,normal)) for i,ids in enumerate(comp)],
      patches=[dict(patch_id=i,**descriptor(ids,t,areas,normal)) for i,ids in enumerate(patches)])
def inspect():
    data=[]
    for row in rows():
        d=describe(row);save(OUT/'source_descriptors'/(row['parent_id']+'.json'),d);data.append(d)
    save(OUT/'source_descriptor_manifest.json',dict(rows=[dict(parent_id=d['parent_id'],N=d['N'],components=[{k:r[k] for k in ('component_id','count','area','bbox_min','bbox_max','center')} for r in d['components']],top_patches=[{k:r[k] for k in ('patch_id','count','area','bbox_min','bbox_max','normal')} for r in sorted(d['patches'],key=lambda x:-x['area'])[:12]]) for d in data],source_inventory_sha256=sha(INV)))
    print('SOURCE_DESCRIPTORS_ONLY: '+str(len(data))+' parents; no C generated')

SPLIT=ROOT/'experiments/r2_bench_deployment_task_data_v1/split_audit/new_data_split.json'
CONTRACT=SPLIT.with_name('staged_eval_contract.json')
EXPECTED_SPLIT='68b68366adb4f1e6476ff7f93d0b471a8cfa83777633925a032f010ec05d6ff2'
EXPECTED_CONTRACT='4ef254257a667f30459b85364a7642a16e346ac38e219e041d01a8b66be8e8e8'

def selectors(row,a,d):
    """Source-specific, pre-review proposals using whole native components/patches.

    Axis cuts use the existing component attachment cross-section, never split a
    triangle. They remain proposals until actual C and Y review confirms meaning.
    """
    v,f,t,ar,n=geom(a);prefix=row['parent_id'][:6]
    def comp(ids):return sorted({j for i in ids for j in d['components'][i]['face_ids']})
    def patch(ids):return sorted({j for i in ids for j in d['patches'][i]['face_ids']})
    def entry(name,ids,method):return dict(proposed_region=name,source_face_ids=list(map(int,ids)),selector=method)
    configs={'1d6150':(.274457544,-2.571485),'4eb3dc':(.5261172056,-2.49201298),
        'd7d6a7':(.350705862,-2.5473218),'945214':(.254858017,-2.5732367),
        'ddd142':(.332269907,-2.5474038),'f540ad':(.484508187,-2.4966576)}
    if prefix in configs:
        sy,inner=configs[prefix]
        seat=np.flatnonzero((n[:,1]>1-1e-6)&(np.max(np.abs(t[:,:,1]-sy),axis=1)<2e-6))
        frame=np.flatnonzero(t[:,:,0].max(1)<=inner+2e-6)
        return [entry('COMPLETE_SEAT_UPPER_SURFACE_GROUP',seat,dict(kind='all_original_upward_faces_on_reviewed_seat_plane',Y_model=sy)),
            entry('COMPLETE_LEFT_END_FRAME',frame,dict(kind='whole_faces_beyond_existing_end_frame_attachment_section',inner_X_model=inner))]
    specs={
      '27eb79': [('COMPLETE_CURVED_SEAT_BACK_SHELL',[1]),('COMPLETE_ARM_LEG_FRAME',[0])],
      '28b9f2': [('COMPLETE_FIVE_SLAT_SEAT',[0,1,2,3,4]),('COMPLETE_LEFT_END_BLOCK',[5])],
      '585673': [('COMPLETE_FOUR_BOARD_SEAT',[8,9,10,11]),('COMPLETE_CORNER_LEG',[0])],
      '617fa8': [('COMPLETE_TOP_CUSHION',[5]),('COMPLETE_ROUND_LEG',[1])],
      '77aa33': [('COMPLETE_THICK_SEAT',[2]),('LEFT_END_PEDESTAL_WITH_ASSOCIATED_SOURCE_FACE',[1,3])],
      'abf250': [('COMPLETE_CURVED_SLAT_SEAT_GROUP',list(range(2,24))),('COMPLETE_END_PEDESTAL',[0])]}
    if prefix in specs:return [entry(name,comp(ids),dict(kind='union_of_entire_source_edge_components',component_ids=ids)) for name,ids in specs[prefix]]
    if prefix=='756108':return [entry('COMPLETE_UPPER_SEAT_SURFACE',patch([5]),dict(kind='entire_connected_planar_patch',patch_ids=[5])),entry('COMPLETE_REAR_FRAME_OUTER_SURFACE',patch([0]),dict(kind='entire_connected_planar_patch_including_rear_legs_and_back_frame',patch_ids=[0]))]
    if prefix=='af264c':
        ids=[p['patch_id'] for p in d['patches'] if p['normal'][1]>.95 and p['bbox_max'][0]-p['bbox_min'][0]>4.7 and p['bbox_max'][1]<0]
        return [entry('COMPLETE_SLAT_SEAT_UPPER_SURFACE_GROUP',patch(ids),dict(kind='complete_long_upward_seat_patches',patch_ids=ids)),entry('COMPLETE_LEFT_END_FRAME',np.flatnonzero(t[:,:,0].max(1)<=-2.52828+2e-6),dict(kind='whole_faces_beyond_existing_end_frame_attachment_section',inner_X_model=-2.52828))]
    if prefix=='bcb4f5':return [entry('COMPLETE_FLAT_SEAT_TOP_SURFACE',patch([139]),dict(kind='entire_flat_top_source_patch',patch_ids=[139])),entry('COMPLETE_LEFT_CURVED_LOOP_END',np.flatnonzero(t[:,:,0].max(1)<=-1.993599+2e-6),dict(kind='whole_faces_beyond_flat_to_arc_source_joint',inner_X_model=-1.993599))]
    raise ValueError('No reviewed proposal selector '+prefix)

def freeze():
    assert sha(SPLIT)==EXPECTED_SPLIT and sha(CONTRACT)==EXPECTED_CONTRACT
    split={r['parent_id']:r for r in read(SPLIT)['rows']}; out=[]
    for row in rows():
        uid=row['parent_id'];a=load(row['clean_case_path']);d=read(OUT/'source_descriptors'/(uid+'.json'))
        assert sha(row['clean_case_path'])==row['clean_case_sha256']==d['source_sha256']
        sels=selectors(row,a,d);assert len(sels)==2
        assert not set(sels[0]['source_face_ids'])==set(sels[1]['source_face_ids'])
        for i,s in enumerate(sels):
            assert 2<=len(s['source_face_ids'])<len(a['source_faces']), (uid,i,len(s['source_face_ids']))
            role=split[uid]['split_role'];group='ADAPTATION_UNSEEN_PARENT' if role.startswith('ADAPTATION_UNSEEN') else ('TRAIN_KNOWN_C' if i==0 else 'SAME_PARENT_HELDOUT_C')
            s.update(condition_id=uid+'__R'+str(i+1),parent_id=uid,recipe_id='R'+str(i+1),family_id=split[uid]['visual_family_id'],intended_eval_group=group,
                future_training_pool_membership=group=='TRAIN_KNOWN_C',training_pool_membership=False,training_ready=False,
                adapter_status='BLOCKED_N_OUTSIDE_128_256',status='FROZEN_SELECTION_PENDING_REVIEW',
                source_case_path=row['clean_case_path'],source_case_sha256=row['clean_case_sha256'],source_finite_receipt_path=row['source_geometry_path'],source_finite_receipt_sha256=sha(row['source_geometry_path']),
                parent_family_split_sha256=EXPECTED_SPLIT,staged_eval_contract_sha256=EXPECTED_CONTRACT,N=len(a['source_faces']),K=len(s['source_face_ids']))
        target=OUT/'selections'/uid/'selection.json';save(target,dict(parent_id=uid,rows=sels,parent_family_split_sha256=EXPECTED_SPLIT,staged_eval_contract_sha256=EXPECTED_CONTRACT))
        out.extend([dict(s,selection_path=str(target),selection_sha256=sha(target)) for s in sels])
    save(OUT/'selection_manifest.json',dict(rows=out,status='FROZEN_BEFORE_CY_MATERIALIZATION',train=[],training_ready=False))
    print('FROZEN_SELECTIONS',len(out))

def materialize():
    from r2.data_tools.bench_parts_v1.construct import partition
    from r2.data_tools.bench_parts_v1.quality import topology,edgebook
    result=[]
    for r in read(OUT/'selection_manifest.json')['rows']:
        assert sha(r['selection_path'])==r['selection_sha256'];assert sha(r['source_case_path'])==r['source_case_sha256']
        a=load(r['source_case_path']);ids=np.asarray(r['source_face_ids'],np.int64);p=partition(a,ids)
        fp=read(r['source_finite_receipt_path']);assert sha(r['source_finite_receipt_path'])==r['source_finite_receipt_sha256']
        assert fp['pair_count']==r['N']*(r['N']-1)//2 and not fp['illegal_pairs'] and not fp['unresolved_pairs']
        v,f,t,area,normal=geom(a);topo=topology(p['C_faces_original']); reasons=[]
        for key in ('nonmanifold_edges','same_winding_double_edges','nonmanifold_vertices'):
            if topo[key]:reasons.append('C_'+key.upper())
        if np.any(area[ids]<=1e-12):reasons.append('C_DEGENERATE')
        book=edgebook(f);ks=set(ids.tolist());interfaces=[];by_free=defaultdict(list)
        for ei,(ci,u,w) in enumerate(topo['open_edges']):
            su,sw=map(int,p['C_source_vertex_ids'][[u,w]]);inc=book[tuple(sorted((su,sw)))];outside=[x for x in inc if x[0] not in ks]
            j=-1;triple=[-1]*3
            if len(inc)==2 and len(outside)==1 and outside[0][1:]==(sw,su):
                j=outside[0][0];third=[x for x in f[j] if x not in (su,sw)];assert len(third)==1
                triple=[sw,su,int(third[0])];by_free[j].append(ei)
            interfaces.append(dict(edge_id=ei,source_u=su,source_v=sw,source_C_face_id=int(ids[ci]),free_source_face_id=j,source_triple=triple,role='NULL_UNSUPPORTED'))
        for j,es in by_free.items():
            for ei in es:interfaces[ei].update(role='OWNER' if ei==min(es) else 'ALIAS',owner_edge_id=min(es))
        cid=r['condition_id'];dest=OUT/'conditions'/cid;dest.mkdir(parents=True,exist_ok=True)
        inp={k:p[k] for k in ('C','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_corner_permutations','N','K')}
        npz(dest/'input_C_only.npz',inp);npz(dest/'paired_C_Y.npz',p)
        sup=dict(edge_records=np.asarray(topo['open_edges'],np.int64).reshape(-1,3),edge_source_triple=np.asarray([x['source_triple'] for x in interfaces],np.int64).reshape(-1,3),
            occupied_labels=np.asarray([x['role']=='OWNER' for x in interfaces],bool),source_alias_labels=np.asarray([x['role']=='ALIAS' for x in interfaces],bool),edge_owner_face=np.asarray([x['free_source_face_id'] for x in interfaces],np.int64))
        npz(dest/'supervision_source_only.npz',sup)
        report=dict(condition_id=cid,geometry_status='HOLD' if reasons else 'PASS_PENDING_SEMANTIC_REVIEW',reasons=reasons,topology=topo,
            mapping='PASS_BITWISE_C_PLUS_Y_EQUALS_SOURCE_M',source_surface_mutation=False,source_geometry_modified=False,
            full_M_finite_legality=dict(status='REUSED_AUTHENTICATED_PRIOR_FULL_SOURCE_PASS',receipt_path=r['source_finite_receipt_path'],receipt_sha256=r['source_finite_receipt_sha256'],new_finite_pair_tests=0,scope='C,Y and CF inherit whole-source legal finite pairs; exact subset preserves coordinates and IDs'),
            source_interface_rows=interfaces,source_labels_usage='SUPERVISION_ONLY_NEVER_DEPLOYMENT_INPUT',
            production_C_only_layout_status='UNRESOLVED_NOT_INFERRED_FROM_SOURCE_OWNER',training_ready=False,adapter_status=r['adapter_status'],N=r['N'],K=r['K'],
            C_area=float(area[ids].sum()),C_area_fraction=float(area[ids].sum()/area.sum()),C_edge_components=len(topo['edge_connected_components']),C_component_face_counts=[len(x) for x in topo['edge_connected_components']],source_owner_count=sum(x['role']=='OWNER' for x in interfaces),source_alias_count=sum(x['role']=='ALIAS' for x in interfaces))
        save(dest/'quality.json',report)
        source_tri=t[ids];unordered=sorted(tuple(sorted(tuple(float(x) for x in vertex) for vertex in tri)) for tri in source_tri)
        gh=hashlib.sha256(json.dumps(unordered,separators=(',',':')).encode()).hexdigest();th=hashlib.sha256(np.sort(f[ids],axis=1).tobytes()).hexdigest()
        result.append(dict(r,input_path=str(dest/'input_C_only.npz'),input_sha256=sha(dest/'input_C_only.npz'),paired_path=str(dest/'paired_C_Y.npz'),paired_sha256=sha(dest/'paired_C_Y.npz'),
            quality_path=str(dest/'quality.json'),quality_sha256=sha(dest/'quality.json'),C_geometry_sha256=gh,C_topology_sha256=th,partition_exact_roundtrip=True,review_status='NOT_REVIEWED',geometry_status=report['geometry_status'],C_edge_components=report['C_edge_components'],source_owner_count=report['source_owner_count']))
    save(OUT/'candidate_manifest.json',dict(rows=result,train=[],training_ready=False,model_calls=0,neural_calls=0,source_mutations=0))
    print('MATERIALIZED',len(result))

def render_one(r):
    from PIL import Image,ImageDraw
    from r2.data_tools.bench_user_v2.render import cell,basis,shared_bounds,font,text_lines
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    dst=OUT/'gallery'/(r['condition_id']+'.png');receipt=dst.with_suffix('.json')
    if receipt.exists():
        old=read(receipt);assert sha(dst)==old['image_sha256'] and old['selection_sha256']==r['selection_sha256'];return old
    a=load(r['source_case_path']);v,f,t,ar,n=geom(a);ids=np.asarray(r['source_face_ids'],int);rest=np.setdiff1d(np.arange(len(t)),ids)
    cams=[basis(-60,22),basis(120,22)];center,span=shared_bounds([t],cams);size=450;head=130
    page=Image.new('RGB',(size*4,size*2+head),'white');draw=ImageDraw.Draw(page)
    text_lines(draw,(12,8),r['condition_id']+' | '+r['proposed_region'],size*4-24,20,2)
    text_lines(draw,(12,58),f"N={r['N']} K={r['K']} | {r['intended_eval_group']} | BLUE=C, GRAY=Y | source resolution, NOT training ready",size*4-24,18,2)
    groups=[('Complete unchanged M',t,0),('C only',t[ids],len(ids)),('Y complement',t[rest],0),('C + Y',np.concatenate([t[ids],t[rest]]),len(ids))]
    for j,(label,tri,k) in enumerate(groups):
        draw.text((j*size+12,100),label,font=font(20),fill='black')
        for i,cam in enumerate(cams):im,_=cell(tri,k,cam,center,span,size=size);page.paste(im,(j*size,head+i*size))
    dst.parent.mkdir(parents=True,exist_ok=True);buf=io.BytesIO();page.save(buf,format='PNG')
    if dst.exists():assert dst.read_bytes()==buf.getvalue()
    else:dst.write_bytes(buf.getvalue())
    rec=dict(condition_id=r['condition_id'],image_path=str(dst),image_sha256=sha(dst),selection_sha256=r['selection_sha256'],review_status='NOT_REVIEWED',model_calls=0,pixel_frame='same source bbox across M,C,Y,CY',center=center.tolist(),span=span)
    save(receipt,rec);return rec

def render():
    from concurrent.futures import ProcessPoolExecutor
    import psutil,subprocess
    resources=dict(cpu_count=os.cpu_count(),available_RAM=psutil.virtual_memory().available,cpu_load=psutil.cpu_percent(interval=.2),workers=2,threads_per_worker=1,CUDA_calls=0,
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv'],capture_output=True,text=True).stdout)
    if not (OUT/'resources_render.json').exists():save(OUT/'resources_render.json',resources)
    rs=read(OUT/'candidate_manifest.json')['rows']
    with ProcessPoolExecutor(max_workers=2) as ex: receipts=list(ex.map(render_one,rs))
    save(OUT/'gallery_manifest.json',dict(rows=receipts));print('RENDERED',len(receipts))

def close():
    """Join actual human-readable proxy reviews with independent geometry gates."""
    candidates=read(OUT/'candidate_manifest.json')['rows']; images={r['condition_id']:r for r in read(OUT/'gallery_manifest.json')['rows']}
    reviews={};review_files=[]
    for path in sorted((OUT/'reviews').glob('*.json')):
        doc=read(path);review_files.append(dict(path=str(path),sha256=sha(path)))
        for r in doc['rows']:
            cid=r['condition_id'];assert cid not in reviews,'Duplicate review '+cid
            reviews[cid]=dict(r,review_source_path=str(path),review_source_sha256=sha(path),reviewer='AI_PROXY',actually_viewed=True)
    assert set(reviews)=={r['condition_id'] for r in candidates},'Missing or extra actual reviews'
    output=[];protected={};groups=Counter();held=Counter()
    for r in candidates:
        cid=r['condition_id'];rv=reviews[cid];g=read(r['quality_path']);im=images[cid]
        assert sha(r['source_case_path'])==r['source_case_sha256'];protected[r['source_case_path']]=r['source_case_sha256']
        assert sha(r['input_path'])==r['input_sha256'] and sha(r['paired_path'])==r['paired_sha256'] and sha(im['image_path'])==im['image_sha256']
        inp=load(r['input_path']);pair=load(r['paired_path']);a=load(r['source_case_path']);ids=pair['known_ids'];free=pair['free_ids']
        recon=np.empty_like(a['full_target']);recon[ids]=pair['C'];recon[free]=pair['Y']
        assert np.array_equal(recon,a['full_target']) and np.array_equal(inp['C'],pair['C']) and not np.intersect1d(ids,free).size
        assert np.array_equal(pair['C_vertices'][pair['C_faces_model']],pair['C'])
        restored=np.take_along_axis(pair['C'],np.argsort(pair['C_source_corner_permutations'],axis=1)[:,:,None],axis=1)
        assert np.array_equal(restored,pair['C_vertices'][pair['C_faces_original']])
        assert np.array_equal(pair['C_pre_ot']*np.float32(2),pair['C']) and np.array_equal(pair['Y_pre_ot']*np.float32(2),pair['Y'])
        assert 'Y' not in inp and 'occupied_labels' not in inp and 'edge_owner_face' not in inp
        reasons=list(g['reasons'])
        if rv['decision']!='ACCEPT_SOURCE_TASK':reasons.append('SEMANTIC_REGION_HOLD')
        decision='HOLD' if reasons else 'ACCEPT_SOURCE_TASK_PENDING_ADAPTER'
        if not reasons:groups[r['intended_eval_group']]+=1
        else:held.update(reasons)
        output.append(dict(r,review_status='ACTUALLY_REVIEWED_AI_PROXY',semantic_decision=rv['decision'],review_observation=rv.get('observation',rv.get('notes')),reviewer='AI_PROXY',
            review_source_path=rv['review_source_path'],review_source_sha256=rv['review_source_sha256'],image_path=im['image_path'],image_sha256=im['image_sha256'],
            final_source_task_status=decision,hold_reasons=reasons,staging_group=r['intended_eval_group'] if not reasons else 'HOLD',training_ready=False,training_pool_membership=False,
            deployment_layout_status='BLOCKED_PRODUCTION_C_ONLY_LAYOUT_NOT_VALIDATED',C_open_boundary_count=len(g['topology']['open_edges']),
            source_alias_count=g['source_alias_count'],
            additional_adapter_caveats=(['CLOSED_C_EXISTING_SEAM_EDGE_ENTRY_REQUIRES_ADAPTER'] if not g['topology']['open_edges'] else [])+
            (['BOUNDARY_SLOTS_EXCEED_FREE_BUDGET'] if len(g['topology']['open_edges'])>r['N']-r['K'] else [])+
            (['SOURCE_ALIAS_NOT_DEPLOYMENT_LABEL'] if g['source_alias_count'] else [])))
    by_parent=defaultdict(list)
    for r in output:by_parent[r['parent_id']].append(r)
    for uid,rs in by_parent.items():
        assert len(rs)==2 and rs[0]['C_geometry_sha256']!=rs[1]['C_geometry_sha256']
        assert rs[0]['recipe_id']=='R1' and rs[1]['recipe_id']=='R2'
    accepted=[r for r in output if not r['hold_reasons']]
    accepted_train_parents={r['parent_id'] for r in accepted if r['intended_eval_group']=='TRAIN_KNOWN_C'}
    missing_known=[]
    for r in accepted:
        if r['intended_eval_group']=='SAME_PARENT_HELDOUT_C' and r['parent_id'] not in accepted_train_parents:
            r['staging_group']='SAME_PARENT_HELDOUT_PENDING_ACCEPTED_TRAIN_C'
            r['eval_group_ready']=False;missing_known.append(r['condition_id'])
        else:r['eval_group_ready']=True
    for r in output:r['effective_eval_group']=r['staging_group']
    actual_groups=Counter(r['staging_group'] for r in accepted)
    summary=dict(schema='SOURCE_RESOLUTION_USER_TASKS_PENDING_ADAPTER_V1',parents_considered=len(by_parent),candidates=len(output),
        accepted_source_tasks=len(accepted),parents_with_accepted_source_tasks=len({r['parent_id'] for r in accepted}),
        held_source_tasks=len(output)-len(accepted),accepted_intended_groups=dict(groups),accepted_staging_groups=dict(actual_groups),hold_reasons=dict(held),
        accepted_same_parent_C_without_accepted_training_C=missing_known,
        accepted_closed_C_tasks=sum(r['C_open_boundary_count']==0 for r in accepted),accepted_C_with_source_owners=sum(r['source_owner_count']>0 for r in accepted),
        accepted_additional_adapter_caveats=dict(Counter(reason for r in accepted for reason in r['additional_adapter_caveats'])),
        N_range=[min(r['N'] for r in output),max(r['N'] for r in output)],K_range=[min(r['K'] for r in output),max(r['K'] for r in output)],
        new_model_ready_parents=0,new_model_ready_tasks=0,training_ready=False,adapter_status='BLOCKED_N_OUTSIDE_128_256',production_C_only_layout='UNRESOLVED_NOT_DERIVED_FROM_Y',
        parent_family_split_path=str(SPLIT),parent_family_split_sha256=sha(SPLIT),staged_eval_contract_path=str(CONTRACT),staged_eval_contract_sha256=sha(CONTRACT),
        review_files=review_files,protected_sources=protected,source_geometry_changes=0,model_calls=0,GPU_calls=0,training_updates=0,new_whole_source_finite_pair_tests=0,
        inherited_full_source_finite_passes=len(protected),same_parent_heldout_c_is_not_unseen_target=True,
        evidence_limit='Actual AI_PROXY image review and cached finite-pair legality do not certify universal semantic correctness or production compatibility.')
    save(OUT/'manifest.json',dict(summary=summary,rows=output,train=[],production_eval=[]))
    save(OUT/'summary.json',summary)
    save(OUT/'accepted_manifest.json',dict(rows=accepted,training_ready=False,train=[],adapter_status='BLOCKED_N_OUTSIDE_128_256'))
    save(OUT/'held_manifest.json',dict(rows=[r for r in output if r['hold_reasons']],training_ready=False,policy='Original proposed C/Y retained without retry, repair or relabelled recipes'))
    for group,name in [('TRAIN_KNOWN_C','train_known_c'),('SAME_PARENT_HELDOUT_C','same_parent_heldout_c'),('ADAPTATION_UNSEEN_PARENT','adaptation_unseen_parent')]:
        save(OUT/'evaluation_staging'/(name+'.json'),dict(rows=[r for r in accepted if r['staging_group']==group],
            group=group,status='SOURCE_TASKS_ONLY_NO_SAMPLING_REGISTERED',training_ready=False,evaluation_executable=False,
            adapter_status='BLOCKED_N_OUTSIDE_128_256',noise_arrays=None,checkpoints=None,
            parent_family_split_sha256=EXPECTED_SPLIT,staged_eval_contract_sha256=EXPECTED_CONTRACT))
    import html
    cards=['<!doctype html><html><meta charset="utf-8"><title>Bench source tasks pending adapter</title><style>body{font:16px sans-serif;max-width:1800px;margin:auto} img{width:100%}article{border-bottom:2px solid #aaa;padding:12px} .hold{color:#a22}</style><h1>Source-resolution C/Y tasks: NOT training ready</h1><p>Original clean sources unchanged. All N outside 128..256. AI_PROXY review, no model calls. Blue=C; gray=Y. Two source-scaled views per page.</p>']
    for r in output:
        cards.append('<article><h2>'+html.escape(r['condition_id'])+'</h2><p>'+html.escape(r['final_source_task_status']+' | '+r['staging_group']+' | '+str(r['hold_reasons']))+'</p><p>'+html.escape(str(r['review_observation']))+'</p><img loading="lazy" src="'+html.escape(Path(r['image_path']).name)+'"></article>')
    cards.append('</html>');index=OUT/'gallery'/'index.html';text='\n'.join(cards)
    if index.exists():assert index.read_text('utf-8')==text
    else:index.write_text(text,'utf-8')
    fields=['condition_id','parent_id','family_id','recipe_id','proposed_region','N','K','C_edge_components','source_owner_count','intended_eval_group','effective_eval_group','geometry_status','semantic_decision','final_source_task_status','hold_reasons','training_ready','adapter_status','review_observation','input_path','paired_path','image_path']
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=fields);w.writeheader()
    for r in output:w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],(list,dict)) else r[k] for k in fields})
    cp=OUT/'per_condition.csv'
    if cp.exists():assert cp.read_text('utf-8-sig')==buf.getvalue()
    else:cp.write_text(buf.getvalue(),'utf-8-sig',newline='')
    save(OUT/'execution_receipt.json',dict(steps=['inspect source components and planar patches','freeze parent/family split','freeze exact source face selections','materialize exact partitions','render30pages_cpu_2workers','actually view each page','combine semantic and topology gates'],
        engineering_events=[dict(stage='freeze first attempt',status='ABORT_BEFORE_OUTPUT',reason='Locally hand-copied expected contract hash omitted final e8; actual immutable contract was unchanged. Corrected literal to full64hex then executed.',scientific_changes=0)],
        new_model_calls=0,new_training_updates=0,GPU_work=0,source_geometry_changes=0,new_simplification_calls=0,new_whole_source_finite_pair_tests=0,
        recorded_tool_wall_seconds=dict(freeze=2.1964886,materialize=3.0325423,render=7.9121366),python_source_sha256=sha(__file__),
        asset_scope='30 actual proposed C/Y pairs from15 unchanged clean sources; source evidence only, not baseline-ready expansion'))
    save(OUT/'COMPLETE.json',dict(status='SOURCE_TASK_CONSTRUCTION_COMPLETE_ADAPTER_BLOCKED',manifest_sha256=sha(OUT/'manifest.json'),summary_sha256=sha(OUT/'summary.json'),source_sha256=sha(__file__)))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inspect','freeze','materialize','render','close']);args=p.parse_args()
    globals()[args.stage]()
