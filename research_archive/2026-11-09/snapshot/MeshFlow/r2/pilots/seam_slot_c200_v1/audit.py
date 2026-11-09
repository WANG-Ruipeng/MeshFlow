"""CPU-only paired input and immutable-source audit; no model imports."""
import argparse,json,time
from collections import Counter
import numpy as np
from . import common as C

def run():
    start=time.perf_counter();r=C.read(C.OUT/'registration.json');d=C.read(C.OUT/'data_manifest.json');e=C.read(C.OUT/'eval_manifest.json')
    exposure={a:Counter() for a in C.ARMS};coordinates={a:0 for a in C.ARMS};pairs=0
    for s in d['steps']:
        assert len(s['samples'][C.ARMS[0]])==len(s['samples'][C.ARMS[1]])==8
        for left,right in zip(s['samples'][C.ARMS[0]],s['samples'][C.ARMS[1]]):
            assert all(left[k]==right[k] for k in ('uid','alpha','sample_index'))
            samples=[]
            for arm,desc in zip(C.ARMS,(left,right)):
                assert C.sha(desc['path'])==desc['sha256'];x=C.arrays(desc['path']);samples.append(x)
                n,k=int(x['N']),int(x['K']);ids=x['source_face_ids'];cp=x['source_corner_permutations'];fixed=x['anchor_ids']>=0
                assert np.array_equal(np.sort(ids),np.arange(n));assert np.array_equal(np.sort(cp,1),np.tile(np.arange(3),(n,1)))
                assert np.array_equal(x['xt'][fixed],x['C_vertices'][x['anchor_ids'][fixed]])
                assert not x['u'][fixed].any()
                assert int(x['free_coordinate_count'])==3*int((~fixed).sum())==9*(n-k)-6*int(x['occupied_for_FM'].sum())
                assert np.max(np.abs(x['xt']+(1-x['t'])*x['u']-x['x1']))<2e-6
                exposure[arm][desc['uid']]+=1;coordinates[arm]+=int(x['free_coordinate_count'])
            assert np.array_equal(samples[0]['face_permutation'],samples[1]['face_permutation'])
            assert np.array_equal(samples[0]['C_vertices'],samples[1]['C_vertices']) and samples[0]['t']==samples[1]['t']
            pairs+=1
    assert pairs==1600
    assert all(set(c.values())=={400} for c in exposure.values())
    assert len(e['requests'])==16 and len({q['seed'] for q in e['requests']})==16
    for q in e['requests']:
        assert C.sha(q['noise_npz'])==q['noise_sha256']
        inp=C.arrays(q['inference_npz']);assert set(inp)=={'C_vertices','C_faces_original','C_faces_model','total_N'}
    changed=[]
    for p,h in r['protected_hashes'].items():
        if C.sha(p)!=h:changed.append(p)
    assert not changed,changed
    result=dict(status='PASS',paired_training_instances=pairs,OT_samples=3200,parent_exposure=exposure,
        total_free_scalars=coordinates,evaluation_requests=16,protected_files=len(r['protected_hashes']),
        source_original_corner_transport='checked per cached sample during prepare',
        model_forwards=0,GPU_calls=0,seconds=time.perf_counter()-start)
    C.write(C.OUT/'input_audit.json',result);return result

if __name__=='__main__':print(json.dumps(C.clean(run()),indent=2))
