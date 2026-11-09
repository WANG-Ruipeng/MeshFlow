"""Final byte/index integrity check; no model or geometry evaluation."""
from pathlib import Path
import collections
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'experiments/r2_bench_control_transfer_v1'
CACHE={}

def path(p):
    p=str(p)
    if p.startswith('E:'):p='/mnt/e/'+p[3:].replace('\\','/')
    return Path(p)
def read(p):return json.loads(path(p).read_text('utf-8'))
def sha(p):
    p=path(p)
    if p not in CACHE:
        assert p.stat().st_size<100_000_000, 'No large weight hashing: '+str(p)
        CACHE[p]=hashlib.sha256(p.read_bytes()).hexdigest()
    return CACHE[p]
def verify(mapping):
    for p,h in mapping.items():assert sha(p)==h,(p,sha(p),h)
def arrays(p):
    with np.load(path(p),allow_pickle=False) as f:return {k:f[k] for k in f.files}
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def ply_identity(p,a):
    data=path(p).read_bytes();marker=b'end_header\n';end=data.index(marker)+len(marker)
    header=data[:end].decode('ascii');assert 'format binary_little_endian 1.0' in header
    n=len(a['vertices']);f=len(a['faces'])
    assert f'element vertex {n}\n' in header and f'element face {f}\n' in header
    assert len(data)==end+n*12+f*13
    v=np.frombuffer(data,dtype='<f4',offset=end,count=n*3).reshape(n,3)
    ff=np.frombuffer(data,dtype=np.dtype([('count','u1'),('indices','<i4',(3,))]),offset=end+n*12,count=f)
    assert np.array_equal(v,a['vertices']) and np.array_equal(ff['indices'],a['faces']) and np.all(ff['count']==3)
    return dict(path=str(p),bytes=len(data),sha256=sha(p),array_index_roundtrip=True)

def main():
    r=read(RUN/'registration.json');s=read(RUN/'evaluation/score_registration.json')
    e=read(RUN/'evaluation_inputs/manifest.json')
    verify(r['code_hashes']);verify(r['case_hashes']);verify(s['source_hashes']);verify(s['protected'])
    assert sha(r['manifest'])==r['manifest_sha256']
    assert sha(RUN/'input_schedule.json')==r['input_schedule_sha256']
    assert e['registration_sha256']==sha(RUN/'registration.json')
    snap=[]
    for p,h in s['source_hashes'].items():
        source=path(p)
        try:rel=source.relative_to(ROOT)
        except ValueError:rel=Path('external')/source.name
        copied=RUN/'scoring_source_snapshot'/rel
        assert sha(copied)==h
        snap.append(dict(source=str(source),snapshot=str(copied),sha256=h))
    rows=[];ply=[];noise_pairs=[];qby={q['request_id']:q for q in s['requests']}
    assert len(e['requests'])==24 and len(qby)==24
    for q in e['requests']:
        rid=q['request_id'];sq=qby[rid]
        assert (q['N'],q['K'],q['noise_seed'],q['condition_id'])==(sq['N'],sq['K'],sq['noise_seed'],sq['condition_id'])
        verify({q['inference_npz']:q['input_sha256'],q['source_npz']:q['source_sha256'],q['noise_npz']:q['noise_sha256']})
        source=arrays(q['source_npz']);noise=arrays(q['noise_npz'])['z'];k=q['K'];n=q['N']
        assert noise.shape==(n-k,3,3) and noise.dtype==np.float32
        armhash=[]
        for arm in ('PLAIN','SEAM'):
            rd=RUN/'evaluation'/arm/rid/'raw';pd=RUN/'evaluation'/arm/rid/'post'
            rr=read(rd/'receipt.json');pr=read(pd/'receipt.json')
            assert rr['status']==pr['status']=='COMPLETE' and rr['arm']==pr['arm']==arm
            assert rr['request_id']==pr['request_id']==rid and rr['noise_sha256']==q['noise_sha256']
            assert rr['generator_forwards']==50 and rr['D0_forwards']==0 and rr['anchored_all51_exact']
            assert pr['D0_forwards']==1 and pr['GEO_calls']==1
            assert rr['source_labels_read'] is False and pr['source_labels_read'] is False
            assert sha(rd/'raw.npz')==rr['raw_sha256']==pr['raw_sha256']
            assert sha(pd/'post.npz')==pr['post_sha256'];verify(pr['files'])
            assert sha(rd/'source_condition.npz')==q['source_sha256']
            raw=arrays(rd/'raw.npz');post=arrays(pd/'post.npz');traj=arrays(rd/'trajectory.npz')['state']
            fixed=raw['raw_anchor_ids']>=0
            initial=np.zeros((n,3,3),np.float32);initial[k:]=noise
            initial[fixed]=raw['C_vertices'][raw['raw_anchor_ids'][fixed]]
            assert traj.shape==(51,n,3,3) and same(traj[0],initial) and same(traj[-1],raw['raw'])
            assert all(same(st[fixed],initial[fixed]) for st in traj)
            assert same(raw['raw'][:k],source['C']) and same(post['C'],source['C_original'])
            assert np.array_equal(post['raw_slot_ids'],post['source_face_ids'])
            outputanchors=post['output_anchor_ids'];valid=outputanchors>=0
            compact=post['compact_C_vertex_to_post']
            assert len(set(compact.tolist()))==len(compact)
            assert same(post['vertices'][compact],raw['C_vertices'])
            assert np.array_equal(post['faces'][valid],compact[outputanchors[valid]])
            aid=read(pd/'anchor_identity.json')
            assert aid['status']=='PASS' and aid['C_vertex_ids_bijective'] and aid['final_anchor_vertex_identity_verified']
            assert aid==pr['anchors']
            for d,a,stem in ((rd,raw,'raw'),(pd,post,'post')):ply.append(ply_identity(d/(stem+'.ply'),a))
            rows.append(dict(arm=arm,request_id=rid,C_RAW_canonical_exact=True,C_POST_original_exact=True,
                all51_anchor_states_exact=True,original_noise_array_shared=True,POST_anchor_index_identity=True,
                RAW_sha256=rr['raw_sha256'],POST_sha256=pr['post_sha256'],noise_sha256=q['noise_sha256']))
            armhash.append(rr['noise_sha256'])
        assert armhash[0]==armhash[1]
        noise_pairs.append(dict(request_id=rid,noise_seed=q['noise_seed'],sha256=armhash[0],arms_same_saved_array=True))
    for arm in ('PLAIN','SEAM'):
        for stage in ('raw','post'):
            done=read(RUN/'evaluation'/arm/(stage+'_complete.json'))
            assert done['status']=='COMPLETE' and done['outputs']==24
            if 'files' in done:verify(done['files'])
    for name in ('score_completion.json','summary_completion.json'):
        done=read(RUN/'evaluation'/name)
        if 'files' in done:verify(done['files'])
        for item in done.get('receipts',[]):
            assert sha(item['path'])==item['sha256'];verify(read(item['path'])['files'])
    keep=read(RUN/'cleanup/final_rolling_retirement/cleanup_receipt.json')
    assert keep['status']=='COMPLETE' and keep['protected_before_after_identical']
    for p in keep['protected_checkpoints_after']:
        fp=Path(p['path'].replace('E:\\','/mnt/e/').replace('\\','/'))
        assert fp.exists() and fp.stat().st_size==p['bytes']
    assert 'torch' not in sys.modules
    result=dict(status='PASS',scope='Read-only final byte/index/metadata integrity; no model load, no geometry classification or distance scan.',
        registration_code_files=len(r['code_hashes']),registration_case_files=len(r['case_hashes']),
        original_score_source_files=len(s['source_hashes']),score_snapshot_files=len(snap),
        scoring_protected_files=len(s['protected']),all_registered_hashes_match=True,
        RAW_outputs=48,POST_outputs=48,PLY_files_verified=len(ply),trajectory_files_verified=48,
        noise_pairs=noise_pairs,outputs=rows,PLY=ply,score_source_snapshots=snap,
        distinct_small_files_hashed=len(CACHE),model_loads=0,model_forwards=0,new_metric_scans=0,
        checkpoint_policy='Final checkpoint content hashes reused from completed strict CPU audit and before/after rolling retirement; only existence and byte sizes checked now. No large weight rehash.',
        checkpoint_audit_sha256=sha(RUN/'training_audit/checkpoint_audit.json'),
        rolling_retirement_sha256=sha(RUN/'cleanup/final_rolling_retirement/cleanup_receipt.json'),
        source_sha256=sha(__file__))
    target=RUN/'final_integrity.json'
    if target.exists():raise FileExistsError(str(target))
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('noise_pairs','outputs','PLY','score_source_snapshots')}))

if __name__=='__main__':main()
