"""Frozen CPU preparation for TRAIN/DEV and same-surface subdivision diagnosis."""
import csv, json
import numpy as np
from .common import *


def verify(path,h):
    if sha(path)!=h: raise ValueError('Frozen asset mismatch: '+str(path))


def subdivide(src):
    package()
    from meshflow_control.source_winding import validate_source, capture_source
    v=validate_source(src,src['C']); C=v['C']; ids=v['C_source_vertex_ids']
    nextid=int(ids.max())+1; mid={}; verts=[]; vi=[]; parents=[]; bary=[]
    B=np.eye(3); templates=((0,3,5),(3,1,4),(5,4,2),(3,4,5))
    for f,(t,ii) in enumerate(zip(C,ids)):
        xyz=[*t]; jj=[*map(int,ii)]; bb=[*B]
        for a,b in ((0,1),(1,2),(2,0)):
            key=tuple(sorted((int(ii[a]),int(ii[b]))))
            if key not in mid:
                # float64 arithmetic rounded once; edge order independent.
                mid[key]=((t[a].astype(float)+t[b].astype(float)).astype(np.float64).__mul__(.5).astype(np.float32),nextid)
                nextid+=1
            xyz.append(mid[key][0]);jj.append(mid[key][1]);bb.append((B[a]+B[b])*.5)
        p=v['P'][f];odd=sum(p[a]>p[b] for a in range(3) for b in range(a+1,3))%2
        for inds in templates:
            inds=list(inds)
            if odd:inds[1],inds[2]=inds[2],inds[1]
            verts.append(np.asarray(xyz)[inds]);vi.append(np.asarray(jj)[inds]);parents.append(f);bary.append(np.asarray(bb)[inds])
    original=np.asarray(verts,np.float32); vid=np.asarray(vi,np.int64); parent=np.asarray(parents,np.int64)
    perm=np.asarray([np.lexsort((t[:,2],t[:,1],t[:,0])) for t in original],np.int64)
    result=capture_source(original,vid,perm,np.arange(len(original),dtype=np.int64))
    bary=np.take_along_axis(np.asarray(bary),perm[...,None],axis=1)
    result.update(subdivision_parent_C_row=parent,subdivision_parent_original_face_id=np.repeat(v['C_original_source_face_ids'],4),child_vertices_in_parent_canonical_barycentric=bary)
    rebuilt=np.einsum('nij,njk->nik',bary,C.astype(float)[parent])
    tol=float(np.max(np.abs(rebuilt-result['C'])))
    area=lambda x: np.linalg.norm(np.cross(x[:,1].astype(float)-x[:,0],x[:,2].astype(float)-x[:,0]),axis=1)/2
    err=float(abs(area(C).sum()-area(result['C']).sum())/area(C).sum())
    if tol>1e-6*L or err>1e-5:raise ValueError('Subdivision changes surface beyond frozen rounding tolerance')
    return result,dict(method='each triangle into four via shared source-ID edge midpoints',max_corner_rounding_model=tol,relative_area_rounding=err,surface_identity='within FP32 midpoint rounding; not bitwise identical tessellation',same_source_surface=True,K_before=len(C),K_after=len(result['C']))


def child_queries(oldq,src,legacy):
    q={k:v.copy() for k,v in oldq.items()};faces=[];bs=[]
    for f,b in zip(q['C_point_faces'],q['C_barycentric']):
        j=next((i for i in range(3) if b[i]>=.5),3);child=int(f)*4+j
        mat=src['child_vertices_in_parent_canonical_barycentric'][child]
        x=np.linalg.solve(mat.T,b);faces.append(child);bs.append(x)
    q['C_point_faces']=np.asarray(faces,np.int64);q['C_barycentric']=np.asarray(bs)
    ctl=legacy.cpu_math()['control'];norm,valid,_=ctl.normals(src['C'],L)
    q['C_point_normals']=norm[q['C_point_faces']];q['C_point_normal_valid']=valid[q['C_point_faces']]
    q['C_open_edges'],_=ctl.open_edges(src['C'])
    q['C_border_distance']=np.sqrt(legacy.cpu_math()['seams'].nearest_segments(q['C_points'],q['C_open_edges'])[0])
    rebuilt=np.einsum('ni,nij->nj',q['C_barycentric'],src['C'].astype(float)[q['C_point_faces']])
    if np.max(np.abs(rebuilt-q['C_points']))>1e-6*L:raise ValueError('Retessellation query owner mismatch')
    if not same(q['C_points'],oldq['C_points']) or not same(q['C_weights'],oldq['C_weights']):raise ValueError('Paired physical queries changed')
    return q


def prepare():
    dest=OUT/'registration.json'
    if dest.exists():
        r=load_registration()
        for q in r['requests']:verify(q['input_path'],q['input_sha256']);verify(q['condition_source'],q['condition_source_sha256'])
        for q in r['queries']:verify(q['path'],q['sha256'])
        return r
    event('PREPARE_BEGIN',model_calls=0,training_updates=0)
    matching=OUT/'coverage_audit/selected_train_dev_manifest.json'; pairs=read(matching)['pairs']
    oldreg=read(OLD/'evaluation/registration.json');oq={q['condition_id']:q for q in oldreg['queries'] if q['panel']=='NEW32'}
    legacy=old();package()
    from meshflow_control.source_winding import validate_source
    retess_pairs=[]
    for typ in ('SEAT_HALF','SEAT_FULL'):
        eligible=[p for p in pairs if p['condition_type']==typ and all(int(p[s+'_N'])+3*int(p[s+'_K'])<=256 for s in ('train','dev'))]
        if not eligible:raise ValueError('No within-N-contract subdivision pair '+typ)
        retess_pairs.append(eligible[0]['pair_id'])
    requests=[];queries=[];conditions=[];audits=[]
    for order,p in enumerate(pairs):
        for split in ('DEV','TRAIN'):
            s=split.lower();cid=p[s+'_condition_id'];pid=p[s+'_parent_id'];N=int(p[s+'_N']);K=int(p[s+'_K'])
            sp=resolve(p[s+'_condition_source']);src=arrays(sp);C=src['C'];validate_source(src,C)
            devrs=sorted([x for x in oldreg['requests'] if x['condition_id']==p['dev_condition_id'] and x['panel']=='NEW32'],key=lambda x:x['candidate'])
            assert len(devrs)==2
            if split=='DEV':
                qr0=oq[cid];verify(qr0['path'],qr0['sha256']);qa0=arrays(qr0['path']);seed=qr0['identity']['C_seed']
            else:qa0,seed=legacy.query_bank(C,cid)
            variants=['base','sub4'] if p['pair_id'] in retess_pairs else ['base']
            for variant in variants:
                vcid=cid+'__'+variant
                if variant=='base':vsrc=src;qa=qa0;vsp=sp;vN=N;vK=K;audit={'method':'unchanged frozen C'}
                else:
                    vsrc,audit=subdivide(src);vsp=OUT/'sources'/(vcid+'.npz');frozen_npz(vsp,vsrc);vK=4*K;vN=N+3*K;qa=child_queries(qa0,vsrc,legacy)
                vC=vsrc['C'];validate_source(vsrc,vC);qp=OUT/'queries'/(vcid+'.npz');frozen_npz(qp,qa)
                queries.append(dict(condition_id=vcid,base_condition_id=cid,parent_id=pid,panel=split,path=str(qp),sha256=sha(qp),query_count=2048,weight_denominator=float(qa['C_weights'].sum()),array_sha256={k:array_sha(v) for k,v in qa.items()},identity=dict(condition_id=vcid,C_seed=seed,C_sha256=array_sha(vC),L0=L)))
                region,tier=p['condition_type'].split('_')
                meta=dict(pair_id=p['pair_id'],condition_id=vcid,base_condition_id=cid,parent_id=pid,group_uid=pid,panel=split,split=split,condition_type=p['condition_type'],region_type=region,tier=tier,N=vN,K=vK,ratio=vK/vN,variant=variant,alpha=1.,condition_source=str(vsp),condition_source_sha256=sha(vsp),C_sha256=array_sha(vC),selection_order=order)
                conditions.append(meta);audits.append(dict(condition_id=vcid,**audit,N_before=N,N_after=vN,free_count_fixed=vN-vK==N-K,query_points_byte_preserved=same(qa['C_points'],qa0['C_points']),query_weights_byte_preserved=same(qa['C_weights'],qa0['C_weights'])))
                for dr in devrs:
                    verify(dr['input_path'],dr['input_sha256']);db=arrays(dr['input_path']);seedn=int(dr['noise_seed'])
                    # Original registration drew only free variables with MT19937.
                    free=np.random.RandomState(seedn).randn(N-K,3,3).astype(np.float32)
                    df=db['z'][:,int(dr['K']):].reshape(-1,3,3);m=min(len(free),len(df))
                    if not same(free[:m],df[:m]):raise ValueError('Original noise algorithm/prefix mismatch')
                    if split=='DEV' and not same(free,df):raise ValueError('Cached DEV free noise differs')
                    z=np.zeros((1,vN,9),np.float32);z[:,vK:]=free.reshape(1,-1,9)
                    rid='SEATDIAG__'+split+'__'+vcid+'__n'+str(dr['candidate']);ip=OUT/'inputs'/(rid+'.npz');frozen_npz(ip,C=vC,z=z,N=np.asarray(vN,np.int64),K=np.asarray(vK,np.int64))
                    action='CACHE' if split=='DEV' and variant=='base' else 'NEW'
                    q=dict(**meta,request_id=rid,base_request_id=split+'__'+cid+'__n'+str(dr['candidate']),candidate=dr['candidate'],action=action,input_path=str(ip),input_sha256=sha(ip),Gaussian_sha256=array_sha(z),free_Gaussian_sha256=array_sha(free),noise_seed=seedn,noise_provenance='original MT19937 free stream; TRAIN/DEV common prefix only when free counts differ; sub4 full free array identical',dev_request_id=dr['request_id'],common_train_dev_free_prefix=min(int(p['dev_free']),int(p['train_free'])),source_quality='SOURCE_AUDIT_SEPARATE_NO_UNIQUE_GT_RANKING')
                    if action=='CACHE':
                        for st in ('raw','post'):
                            rp=OLD/'evaluation'/st/'USER_EXPANDED_67'/dr['request_id']/'receipt.json'
                            q['cache_'+st+'_receipt']=str(rp);q['cache_'+st+'_receipt_sha256']=sha(rp)
                    requests.append(q)
    assert len(requests)==40 and sum(q['action']=='NEW' for q in requests)==24 and len(conditions)==20
    cp=read(OLD/'evaluation/checkpoints/USER_EXPANDED_67.json');d=read(OLD/'evaluation/D0/identity.json')
    cp['bytes']=resolve(cp['path']).stat().st_size
    d0=dict(path=d['checkpoint_path'],sha256=d['checkpoint_sha256'],bytes=d['checkpoint_bytes'])
    sources={str(p):sha(p) for p in [CODE/'common.py',CODE/'prepare.py',ROOT/'r2/src/meshflow_control/sampling.py',ROOT/'r2/src/meshflow_control/source_winding.py',ROOT/'r2/pilots/user_condition_finetune_v1/evaluation.py']}
    protocol=dict(training=0,backwards=0,model='EXP67 completed step6000, one fixed generator',sampling='Euler50 BF16 forward FP32 integration, all RAW then single D0 GEO_C source-winding export',matching='eight pairs selected from input geometry only; observational TRAIN/DEV comparison not causal generalization estimate',noise='two original DEV seeds; exact MT19937 free-prefix pairing; same complete free noise for sub4',subdivision='two earliest eligible matched pairs, HALF and FULL; each C face split4; N increases by3K with same free count; operational representation sensitivity not isolated token-attention causality',subdivision_pairs=retess_pairs,source_comparison='no unique source-chair GT scoring',selection_no_result_changes=True,visual_review='all RAW/POST; blue C/gray F; full views plus local side views; YES/NO/UNCERTAIN with notes; nonblind AI_PROXY',quality='unchanged full old quality CF FF I, AVMR PMR all thresholds, legal shared edges; no selecting favorable metrics',query='same physical2048 points and weights for base/sub4, point ownership remapped, normals/borders recomputed; vertex-based AVMR denominator changes and is not retess-invariant',failure='stop on reserved missing-return; no hidden retry; engineering recovery logged without extra forwards',do_not_train_commit_push_or_change_default=True)
    reg=dict(version=VERSION,status='FROZEN',created_utc=utc(),labels=[LABEL],checkpoint=cp,d0_checkpoint=d0,D0=d0,L=L,requests=requests,queries=queries,conditions=conditions,definition=protocol,budget=dict(RAW=40,POST=40,rollouts=24,generator_forwards_max=1200,generator_sampling_forwards=1200,D0_forwards_max=24,training_updates=0,backwards=0),preparation_source_hashes=sources,source_documents={str(p):sha(p) for p in (matching,OLD/'evaluation/registration.json')})
    write(OUT/'subdivision_audit.json',audits);csvwrite(OUT/'requests.csv',requests);csvwrite(OUT/'conditions.csv',conditions);write(dest,reg)
    event('PREPARE_COMPLETE',registration_sha256=sha(dest),logical_requests=40,new_rollouts=24,generator_F_max=1200,D0_F_max=24,models_loaded=0)
    return reg

if __name__=='__main__':
    r=prepare();print(json.dumps(dict(status=r['status'],requests=len(r['requests']),budget=r['budget']),ensure_ascii=False))