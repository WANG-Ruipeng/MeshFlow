"""Mathematical state tests and optional read-only freeze/refit parameter probe."""
import hashlib
import json
import os
from pathlib import Path
import unittest
import numpy as np
from fm_state import encode_state, decode_state, STATE_DIM


def translation(x):
    T=np.broadcast_to(np.eye(4),(len(x),4,4)).copy();T[:,:3,3]=x
    return T


def congruence(Q,T):
    return T.transpose(0,2,1)@Q@T


def params_from_C(C,x=None,n=None):
    count=len(C)
    x=np.zeros((count,3)) if x is None else x
    n=np.tile([0.,0.,.4],(count,1)) if n is None else n
    Q=congruence(C,translation(-x))
    return dict(v=x,p=x.copy(),n=n,Q=Q,eigs=np.linalg.svd(Q[:,:3,:3],compute_uv=False))


class StateTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(510)
        self.x=rng.normal(size=(19,3))*.3
        factor=rng.normal(size=(19,4,4))
        self.C=factor@factor.transpose(0,2,1)
        self.n=rng.normal(size=(19,3))*.2
        self.params=params_from_C(self.C,self.x,self.n)

    def test_full_A_g_c_roundtrip_preserves_raw_normals(self):
        state,audit=encode_state(self.params)
        self.assertEqual(state.shape,(19,STATE_DIM))
        self.assertTrue(np.array_equal(state[:,:3],self.x))
        self.assertTrue(np.array_equal(state[:,3:6],self.n))
        decoded=decode_state(state,normalize_normals=False)
        self.assertTrue(np.array_equal(decoded['n'],self.n))
        self.assertTrue(np.array_equal(decoded['v'],decoded['p']))
        np.testing.assert_allclose(decoded['Q'],self.params['Q'],atol=1e-12,rtol=1e-13)
        C_back=congruence(decoded['Q'],translation(self.x))
        for sl in [(slice(None),slice(0,3),slice(0,3)),(slice(None),slice(0,3),3),(slice(None),3,3)]:
            np.testing.assert_allclose(C_back[sl],self.C[sl],atol=1e-12,rtol=1e-13)
        self.assertGreater(np.max(abs(self.C[:,:3,3])),.1)
        self.assertFalse(audit['ridge_added'])
        json.dumps(audit,allow_nan=False)

    def test_normalization_is_explicit_and_zero_normal_rejected(self):
        state,_=encode_state(self.params)
        self.assertTrue(np.array_equal(decode_state(state)['n'],self.n))
        unit=decode_state(state,normalize_normals=True)
        np.testing.assert_allclose(np.linalg.norm(unit['n'],axis=1),1.,atol=3e-16)
        self.assertFalse(np.array_equal(unit['n'],self.n))
        self.assertTrue(np.array_equal(unit['Q'],decode_state(state,normalize_normals=False)['Q']))
        state[0,3:6]=0
        for normalize in [False,True]:
            with self.assertRaises(ValueError):decode_state(state,normalize_normals=normalize)

    def test_principal_factor_symmetry_psd_and_energy_translation(self):
        state,_=encode_state(self.params)
        F=state[:,6:].reshape(-1,4,4)
        np.testing.assert_allclose(F,F.transpose(0,2,1),atol=5e-16)
        self.assertGreaterEqual(np.linalg.eigvalsh(F).min(),-1e-14)
        np.testing.assert_allclose(F@F.transpose(0,2,1),self.C,atol=1e-12,rtol=1e-13)
        local=np.tile([.2,-.1,.3,1.],(19,1));world=local.copy();world[:,:3]+=self.x
        a=np.einsum('ni,nij,nj->n',local,self.C,local)
        b=np.einsum('ni,nij,nj->n',world,self.params['Q'],world)
        np.testing.assert_allclose(a,b,atol=2e-14,rtol=1e-14)

    def test_only_machine_scale_negative_spectrum_is_clipped(self):
        eps=np.finfo(float).eps
        C=np.diag([1.,.3,.2,-4*eps])[None]
        state,audit=encode_state(params_from_C(C))
        self.assertEqual(audit['negative_eigenvalues_clipped'],1)
        self.assertEqual(audit['matrices_with_negative_eigenvalue_clipping'],1)
        self.assertEqual(audit['negative_eigenvalue_count_per_matrix'],[1])
        self.assertEqual(audit['maximum_negative_eigenvalue_magnitude_clipped'],4*eps)
        self.assertEqual(audit['positive_eigenvalue_floor'],0.)
        self.assertEqual(decode_state(state)['Q'][0,3,3],0.)
        C[0,3,3]=-1e-8
        with self.assertRaisesRegex(ValueError,'negative'):encode_state(params_from_C(C))

    def test_no_ridge_for_exact_rank_deficiency(self):
        C=np.diag([1.,.2,0.,0.])[None]
        state,audit=encode_state(params_from_C(C))
        decoded=decode_state(state)
        self.assertEqual(audit['negative_eigenvalues_clipped'],0)
        self.assertEqual(np.linalg.matrix_rank(decoded['Q'][0]),2)
        self.assertEqual(decoded['Q'][0,2,2],0.)
        self.assertEqual(decoded['Q'][0,3,3],0.)

    def test_material_asymmetry_nonfinite_and_zero_spatial_scale_rejected(self):
        p={k:v.copy() for k,v in self.params.items()};p['Q'][0,0,1]+=.01
        with self.assertRaisesRegex(ValueError,'asymmetric'):encode_state(p)
        state,_=encode_state(self.params);state[0,6]=np.inf
        with self.assertRaises(ValueError):decode_state(state)
        state,_=encode_state(params_from_C(np.diag([0.,0.,0.,1.])[None]))
        with self.assertRaisesRegex(ValueError,'zero normalization'):decode_state(state)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values=np.asarray(values).reshape(-1)
    return dict(count=int(values.size),min=float(values.min()),q05=float(np.quantile(values,.05)),
                median=float(np.median(values)),q95=float(np.quantile(values,.95)),max=float(values.max()),
                mean=float(values.mean()))


def reconstruction(a,b):
    error=np.linalg.norm((a-b).reshape(len(a),-1),axis=1)
    norm=np.linalg.norm(a.reshape(len(a),-1),axis=1)
    relative=np.divide(error,norm,out=np.zeros_like(error),where=norm!=0)
    return dict(max_absolute=float(np.max(abs(a-b))),relative_frobenius=distribution(relative),
                zero_reference_nonzero_error=int(np.sum((norm==0)&(error!=0))))


def probe(source):
    protocol=source/'FROZEN_PROTOCOL.json'
    objects=json.loads(protocol.read_text())['objects']
    if len(objects)!=24:raise ValueError('Expected original 24 clay object/frame targets')
    rows=[];magnitudes={arm:[] for arm in ['freeze','refit']}
    for object_id in sorted(objects):
        for arm in ['freeze','refit']:
            path=source/'runs'/(object_id+'__'+arm)/'parameters.npz'
            row=dict(object=object_id,arm=arm,parameters=str(path),parameters_sha256=sha(path))
            try:
                with np.load(path) as z:params={k:z[k].copy() for k in z.files}
                lengths=np.linalg.norm(params['n'],axis=1);magnitudes[arm].append(lengths)
                state,diagnostics=encode_state(params)
                decoded=decode_state(state,normalize_normals=False)
                normalized=decode_state(state,normalize_normals=True)
                C=congruence(params['Q'],translation(params['v']))
                C_back=congruence(decoded['Q'],translation(decoded['v']))
                factor=state[:,6:].reshape(-1,4,4)
                row.update(status='PASS',state_shape=list(state.shape),state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                    v_exact=bool(np.array_equal(decoded['v'],params['v'])),
                    p_decoded_equals_v=bool(np.array_equal(decoded['p'],params['v'])),
                    p_input_equals_v=bool(np.array_equal(params['p'],params['v'])),
                    raw_normal_exact=bool(np.array_equal(decoded['n'],params['n'])),
                    normal_magnitude=distribution(lengths),
                    explicit_normalization_delta_l2=distribution(np.linalg.norm(normalized['n']-params['n'],axis=1)),
                    normal_settings_Q_exact=bool(np.array_equal(decoded['Q'],normalized['Q'])),
                    raw_world_Q=reconstruction(params['Q'],decoded['Q']),
                    centered_Q=reconstruction(C,C_back),
                    factor_C=reconstruction(C,factor@factor.transpose(0,2,1)),
                    centered_A_max_abs=float(np.max(abs(C_back[:,:3,:3]-C[:,:3,:3]))),
                    centered_g_max_abs=float(np.max(abs(C_back[:,:3,3]-C[:,:3,3]))),
                    centered_c_max_abs=float(np.max(abs(C_back[:,3,3]-C[:,3,3]))),
                    eigs_max_abs=float(np.max(abs(decoded['eigs']-params['eigs']))),encoding_diagnostics=diagnostics)
                if not row['v_exact'] or not row['p_decoded_equals_v'] or not row['raw_normal_exact'] or not row['normal_settings_Q_exact']:
                    row['status']='FAIL'
            except Exception as exc:
                row.update(status='ERROR',error=repr(exc))
            rows.append(row)
    summary={}
    for arm in ['freeze','refit']:
        cases=[row for row in rows if row['arm']==arm];passed=[r for r in cases if r['status']=='PASS']
        summary[arm]=dict(requested=len(cases),passed=len(passed),failed=len(cases)-len(passed),
            normal_magnitude=distribution(np.concatenate(magnitudes[arm])) if magnitudes[arm] else None,
            clipped_eigenvalues=sum(r['encoding_diagnostics']['negative_eigenvalues_clipped'] for r in passed),
            maximum_world_Q_relative_error=max((r['raw_world_Q']['relative_frobenius']['max'] for r in passed),default=None),
            maximum_centered_Q_relative_error=max((r['centered_Q']['relative_frobenius']['max'] for r in passed),default=None),
            maximum_centered_Q_absolute_error=max((r['centered_Q']['max_absolute'] for r in passed),default=None),
            maximum_normal_normalization_change=max((r['explicit_normalization_delta_l2']['max'] for r in passed),default=None))
    return dict(passed=all(r['status']=='PASS' for r in rows),source_run=str(source),protocol_sha256=sha(protocol),
        requested=48,actual=len(rows),arms=['freeze','refit'],summary=summary,rows=rows,
        scope='Read-only parameter round trip only. No mesh extraction, training, model selection, or geometric validity claim.',
        normal_policy='encode raw n; strict replay normalize_normals=False; True separately measures explicit deployment change',
        target_preference='refit is reference-fitted target; freeze retained only as deliberately frozen-control diagnostic')


if __name__=='__main__':
    result=unittest.main(verbosity=2,exit=False).result
    destination=os.environ.get('PONQ_FM_STATE_VALIDATION_DIR')
    probe_result=None
    if destination:
        root=Path(destination);root.mkdir(parents=True,exist_ok=False)
        source=Path(os.environ.get('PONQ_FM_STATE_PROBE_SOURCE',str(Path(__file__).resolve().parents[3]/'output'/'ponq_clay_paths_20260926_01')))
        probe_result=probe(source)
        (root/'ROUNDTRIP_PROBE.json').write_text(json.dumps(probe_result,indent=2,allow_nan=False)+'\n')
        files=[Path(__file__),Path(__file__).with_name('fm_state.py')]
        record=dict(passed=result.wasSuccessful() and probe_result['passed'],tests_passed=result.wasSuccessful(),tests_run=result.testsRun,
            failures=len(result.failures),errors=len(result.errors),probe_passed=probe_result['passed'],probe_summary=probe_result['summary'],
            numpy_version=np.__version__,source_files=[dict(path=str(p),sha256=sha(p)) for p in files],
            scope='Synthetic mathematical tests plus 48 old parameter encodings; no mesh extraction or training.',
            failure_details=[dict(test=str(t),traceback=tb) for t,tb in result.failures+result.errors])
        (root/'VALIDATION.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    raise SystemExit(0 if result.wasSuccessful() and (probe_result is None or probe_result['passed']) else 1)