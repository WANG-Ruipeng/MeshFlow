"""One-time independent evaluation assets. GT never enters inference NPZ."""
from pathlib import Path
import hashlib,shutil
import numpy as np
from .io import *
from . import control
from .hashing import array_sha
VERSION='native_mesh_direct_context_readout_panels_v1'
START_SHA='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
NAMESPACE='NATIVE_MESH_DIRECT_CONTEXT_READOUT_V1|EVALUATION_NOISE|20261007'
def seed(*parts):return int.from_bytes(hashlib.sha256((NAMESPACE+'|'+'|'.join(map(str,parts))).encode()).digest()[:4],'little')
def copy_exact(src,dst,expected=None):
    src,dst=resolve(src),resolve(dst);h=digest(src)
    if expected and h!=expected:raise ValueError('Source SHA mismatch: '+str(src))
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if digest(dst)!=h:raise ValueError('Frozen copy changed: '+str(dst))
    else:shutil.copyfile(src,dst)
    return str(dst),h
def new_bank(C,N,s):
    K=len(C);z=np.zeros((1,N,9),np.float32)
    z[0,K:]=np.random.RandomState(s).standard_normal((N-K,9)).astype(np.float32)
    return dict(C=np.asarray(C,np.float32),z=z,N=np.asarray(N),K=np.asarray(K),known_mask=np.arange(N)<K,valid_mask=np.ones(N,bool))
def _p2():
    def seat(div):
        points=np.linspace(-1.2,1.2,div+1,dtype=np.float32);ts=[]
        for i in range(div):
            for j in range(div):
                a=[points[i],-.55,points[j]];b=[points[i],-.55,points[j+1]]
                c=[points[i+1],-.55,points[j+1]];d=[points[i+1],-.55,points[j]]
                ts += [[a,b,c],[a,c,d]]
        return np.asarray(ts,np.float32)
    a=[-1.2,-.55,1.2];b=[-1.2,1.85,1.2];c=[1.2,1.85,1.2];d=[1.2,-.55,1.2]
    back=np.asarray([[a,b,c],[a,c,d]],np.float32)
    return dict(seat2=seat(1),seat32=seat(4),back2=back,seat_back4=np.concatenate([seat(1),back]))
def prepare_panels(legacy_root,artifact_root):
    legacy=resolve(legacy_root);out=resolve(artifact_root)/'data/evaluation'
    source=legacy/'native_t1/runs/chair_capped500_half_v1'
    old=load_manifest(source/'manifests/OLD.json');fresh=load_manifest(source/'manifests/FRESH.json')
    if len(old)!=16 or len({r['uid'] for r in old})!=8:raise ValueError('Expected original DEV8x2')
    rows=[];conditions=[];receipts=[];seen={}
    for r in sorted(old,key=lambda r:(r['uid'],r['candidate'])):
        cid=r['condition_id'];C=arrays(r['condition_npz'])['C'];N,K=int(r['N']),int(r['K'])
        refdir=out/'evaluation_reference'/cid
        if cid not in seen:
            cp,ch=copy_exact(r['condition_npz'],refdir/'condition.npz',r['condition_sha256'])
            refassets={}
            for name in ('control.json','control.npz','quality.json','quality.npz'):
                p,h=copy_exact(source/'references'/cid/name,refdir/name);refassets[name]=dict(path=p,sha256=h)
            meta=dict(schema=VERSION,panel='P1',condition_id=cid,parent_id=r['uid'],N=N,K=K,L0=float(arrays(cp)['L0']),
                scale_source='Frozen original complete parent L0; scoring only',gamma_status=r.get('gamma_status','PASS'),
                condition_path=cp,condition_sha256=ch,assets=refassets,GT_available=True,
                interpretation='REFERENCE_COMPLETION_DIAGNOSTICS; observed DEV parents, no unseen-parent claim')
            write_json(refdir/'reference.json',meta,freeze=True);seen[cid]=meta;conditions.append(meta)
        refpath=refdir/'reference.json';bank=arrays(r['bank_npz'])
        ip,ih=copy_exact(r['bank_npz'],out/'inputs'/f'P1__{cid}__n{r["candidate"]}.npz',r['bank_sha256'])
        cache=r['historical_cache']['START']
        if cache['checkpoint_sha256']!=START_SHA:raise ValueError('Not retained START')
        verified={}
        for k in ('raw_path','result_path','joined_path','vectors_path','control_vectors_path','export_npz','export_ply'):
            if k in cache:
                hk=k.replace('_path','')+'_sha256' if k.endswith('_path') else k+'_sha256'
                expected=cache[hk];actual=digest(cache[k])
                if actual!=expected:raise ValueError('Historical cache hash mismatch '+k)
                verified[k]=dict(path=cache[k],sha256=actual)
        raw=arrays(cache['raw_path']);tri=raw['output'].reshape(N,3,3)
        if array_sha(raw['z'])!=array_sha(bank['z']) or tri[:K].tobytes()!=C.tobytes():raise ValueError('C/Gaussian not same')
        if raw['path'].shape!=(51,N,9) or not all(x[:K].tobytes()==C.reshape(K,9).tobytes() for x in raw['path']):raise ValueError('Historical 51C mismatch')
        cdir=out/'historical_start'/f'P1__{cid}__n{r["candidate"]}'
        copy_exact(cache['raw_path'],cdir/'raw.npz',cache['raw_sha256'])
        copy_exact(cache['joined_path'],cdir/'score.json',cache['joined_sha256'])
        copy_exact(cache['result_path'],cdir/'result.json',cache['result_sha256'])
        for key,name in (('vectors_path','quality_vectors.npz'),('control_vectors_path','control_vectors.npz')):
            copy_exact(cache[key],cdir/name,cache[key.replace('_path','')+'_sha256'])
        h=dict(raw_path=str(cdir/'raw.npz'),raw_sha256=digest(cdir/'raw.npz'),result_path=str(cdir/'result.json'),
            result_sha256=digest(cdir/'result.json'),score_path=str(cdir/'score.json'),score_sha256=digest(cdir/'score.json'),
            quality_vectors_path=str(cdir/'quality_vectors.npz'),control_vectors_path=str(cdir/'control_vectors.npz'),
            checkpoint_sha256=START_SHA,C_all51_bitwise=True,gaussian_bitwise=True,status='VERIFIED_CACHE')
        rows.append(dict(request_id=f'P1__{cid}__n{r["candidate"]}',panel='P1',cohort='OLD',condition_id=cid,parent_id=r['uid'],
            candidate=r['candidate'],N=N,K=K,input_path=ip,input_sha256=ih,reference_path=str(refpath),reference_sha256=digest(refpath),
            noise_seed=int(r['noise_seed']),noise_array_sha256=array_sha(bank['z']),C_array_sha256=array_sha(C),historical_start=h))
        receipts.append(dict(request_id=rows[-1]['request_id'],sources=verified,C_all51=True,Gaussian_bitwise=True))
    for cid,meta in seen.items():
        C=arrays(meta['condition_path'])['C'];N,K=meta['N'],meta['K'];refpath=out/'evaluation_reference'/cid/'reference.json'
        previous=[arrays(r['bank_npz'])['z'] for r in old+fresh if r['condition_id']==cid]
        hashes={array_sha(x) for x in previous}
        for n in (4,5):
            s=seed('P1',cid,n);bank=new_bank(C,N,s);h=array_sha(bank['z'])
            if h in hashes:raise ValueError('New Gaussian duplicates old 0..3/new stream')
            hashes.add(h);ip=out/'inputs'/f'P1__{cid}__n{n}.npz';save_npz(ip,bank)
            rows.append(dict(request_id=f'P1__{cid}__n{n}',panel='P1',cohort='NEW',condition_id=cid,parent_id=meta['parent_id'],
                candidate=n,N=N,K=K,input_path=str(ip),input_sha256=digest(ip),reference_path=str(refpath),reference_sha256=digest(refpath),
                noise_seed=s,noise_array_sha256=h,C_array_sha256=array_sha(C),new_vs_old_0_to_3=True))
    fixed_scale=float(np.sqrt(3)*2/.3762)
    for name,C in _p2().items():
        cid='P2_'+name;N=192;K=len(C);refdir=out/'evaluation_reference'/cid
        p=refdir/'condition.npz';save_npz(p,dict(C=C,N=np.asarray(N),K=np.asarray(K),L0=np.asarray(fixed_scale)))
        meta=dict(schema=VERSION,panel='P2',condition_id=cid,parent_id=cid,N=N,K=K,L0=fixed_scale,
            scale_source='Fixed diagonal of public model-coordinate canonical cube, side=2/dataset_std(.3762); never output bbox',
            condition_path=str(p),condition_sha256=digest(p),GT_available=False,gamma_status='NOT_APPLICABLE_NO_REQUIRED_INTERFACE',
            geometry_group='seat_surface' if name in ('seat2','seat32') else name,
            pressure_test='Two/few-face inputs and alternate tessellation; outside training context distribution')
        write_json(refdir/'reference.json',meta,freeze=True);conditions.append(meta)
        for n in range(4):
            s=seed('P2',cid,n);bank=new_bank(C,N,s);ip=out/'inputs'/f'{cid}__n{n}.npz';save_npz(ip,bank)
            rows.append(dict(request_id=f'{cid}__n{n}',panel='P2',cohort='OPEN',condition_id=cid,parent_id=cid,
                candidate=n,N=N,K=K,input_path=str(ip),input_sha256=digest(ip),reference_path=str(refdir/'reference.json'),
                reference_sha256=digest(refdir/'reference.json'),noise_seed=s,noise_array_sha256=array_sha(bank['z']),C_array_sha256=array_sha(C)))
    rows=sorted(rows,key=lambda x:(x['panel'],x['condition_id'],x['candidate']))
    for name,subset in [('requests',rows),('P1',[r for r in rows if r['panel']=='P1']),('P2',[r for r in rows if r['panel']=='P2'])]:
        write_json(out/(name+'.json'),dict(schema=VERSION,rows=subset,array_hash_definition='SHA256(str(dtype)+str(shape)+contiguous_bytes), original array_sha; see migrated hashing.py'),freeze=True)
    write_json(out/'conditions.json',conditions,freeze=True)
    receipt=dict(status='PASS',P1_requests=32,P1_OLD_verified_cache=16,P1_NEW=16,P2_requests=16,
        seed_namespace=NAMESPACE,P2_geometry=dict(seat_width=2.4,seat_depth=2.4,seat_y=-.55,back_z=1.2,back_y=[-.55,1.85],subdivision=32),
        source_receipts=receipts,old_repository_mutations=0,model_calls=0)
    write_json(out/'preparation_receipt.json',receipt,freeze=True)
    return receipt
