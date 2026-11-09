"""Retained driver of the already-observed CPU fixture run; not run on import.

This file was saved after the original stdin execution; persistence did not rerun
OT or forwards. Only synthetic geometry and synthetic teacher pairing are used.
"""

def main():
    import os, sys, ast, json
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.chdir('/mnt/e/MeshFlow')
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import torch
    from experiments.mf_h1_local0.control_mvp48 import eval_teacher as T
    assert not torch.cuda.is_initialized()
    source = open('experiments/mf_h1_local0/jedge5_diagnosis/teacher.py', encoding='utf-8-sig').read()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_geometry')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<original_geometry_only>', 'exec'), namespace)
    records=[]
    for N,K in [(7,2),(128,26)]:
        rng = np.random.RandomState(13579+N)
        pre = rng.normal(size=(N-K,3,3)).astype(np.float32)
        C = rng.normal(size=(K,3,3)).astype(np.float32)
        gaussian = rng.normal(size=(N-K,3,3))
        before = gaussian.tobytes()
        sample = dict(uid='synthetic_'+str(N), patch_id='p0', N=N, K=K,
                      constraints=C, free_target=(pre*2).reshape(N-K,9), pre_ot_free=pre, bbox_diagonal=4.)
        arrays, identity = T.prepare_teacher_sample(sample, gaussian)
        T._unpack_bank((arrays, identity))
        assert identity['actual_OT_attempts']==identity['actual_OT_returns']==1
        assert gaussian.tobytes()==before
        assert np.array_equal(arrays['xt'][:K],C.reshape(K,9))
        oracle, endpoint = T.offline_teacher_diagnostics(arrays, identity, arrays['u'])
        assert oracle['FM_error']==0 and oracle['FM_coordinate_denominator']==9*(N-K)
        assert oracle['known_preserved'] and np.allclose(endpoint[K:], arrays['x1'][K:], atol=1e-6)
        shifted = arrays['u'].copy()
        shifted[K:] += .25
        shifted[:K] = 12345
        changed, endpoint2 = T.offline_teacher_diagnostics(arrays, identity, shifted)
        assert abs(changed['FM_error']-.0625)<1e-7 and changed['known_preserved']
        assert changed['geometry']['Yhat_free']==namespace['_geometry'](torch.from_numpy(endpoint2[K:].reshape(-1,3,3)),4.)
        invalid = shifted.copy(); invalid[K,0]=np.nan
        try:T.offline_teacher_diagnostics(arrays,identity,invalid)
        except FloatingPointError:pass
        else:raise AssertionError('Nonfinite velocity not rejected')
        arrays['xt'][K,0]+=1
        try:T._unpack_bank((arrays,identity))
        except ValueError:pass
        else:raise AssertionError('Bank mutation not rejected')
        records.append(dict(N=N,K=K,oracle_FM=oracle['FM_error'],perturbed_FM=changed['FM_error'],original_geometry_bitwise_equal=True))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PASS',fixtures=records,synthetic_OT_calls=2,real_OT_calls=0,model_forwards=0,CUDA_initialized=False)))


if __name__ == '__main__':
    main()
