"""CPU-only synthetic artifact round-trip; never loads or executes a model."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import sample_worker as S
from . import policy as P
import torch
from meshflow_control.mesh_io import write_raw


class SampleContractTests(unittest.TestCase):
    def fixture(self, root, sigma):
        D = S.D
        n, k = 128, 2
        c = np.arange(18, dtype=np.float32).reshape(k, 3, 3) / np.float32(20)
        c[0, 0, 0] = -0.0
        z = np.zeros((1, n, 9), np.float32)
        z[:, k:] = np.linspace(-1, 1, (n-k)*9, dtype=np.float32).reshape(1, n-k, 9)
        eps = np.zeros((20, 1, n, 9), np.float32)
        eps[:, :, k:] = np.linspace(-2, 2, 20*(n-k)*9, dtype=np.float32).reshape(20, 1, n-k, 9)
        inputs = D.save_npz(root/'input.npz', C=c, z=z, N=np.asarray(n), K=np.asarray(k))
        transition = D.save_npz(root/'transition.npz', epsilon=eps, k=np.asarray(S.WINDOW, np.int64))
        q = dict(request_id='CPU_SYNTHETIC_CONTRACT_FIXTURE', N=n, K=k, noise_seed=1,
            input_path=str(inputs), input_sha256=D.sha(inputs), C_sha256=D.array_sha(c), Gaussian_sha256=D.array_sha(z),
            transition_noise_path=str(transition), transition_noise_sha256=D.sha(transition), transition_Gaussian_sha256=D.array_sha(eps))
        known = torch.zeros((1,n), dtype=torch.bool)
        known[:, :k] = True
        valid = torch.ones_like(known)
        free = valid & ~known
        state = torch.from_numpy(z.copy())
        state[:, :k] = torch.from_numpy(c.reshape(1,k,9))
        path, before, after, means, lp = [state[0].numpy().copy()], [], [], [], []
        with torch.no_grad():
            for step in range(50):
                mean = P.euler_mean(state, torch.full_like(state, .125), valid, known)
                if step in S.WINDOW:
                    action = P.apply_noise(mean, torch.from_numpy(eps[step-10]), sigma, free)
                    before.append(state.numpy().copy()); after.append(action.numpy().copy()); means.append(mean.numpy().copy())
                    if sigma:
                        lp.append(P.joint_log_prob(action, mean, sigma, free).item())
                else:
                    action = mean
                state = action
                path.append(state[0].numpy().copy())
        path = np.stack(path)
        dest = root/'cell'
        dest.mkdir()
        buffer = D.save_npz(dest/'window_buffer.npz', k=np.asarray(S.WINDOW, np.int64),
            state_before=np.stack(before), state_after=np.stack(after), mean_old=np.stack(means), epsilon=eps,
            sigma=np.full(20,sigma,np.float64), logp_old=np.asarray(lp,np.float64), free_dimension=np.asarray(9*(n-k),np.int64),
            known_mask=known.numpy(), valid_mask=valid.numpy(), N=np.asarray(n,np.int64), K=np.asarray(k,np.int64),
            y=np.asarray([n],np.int64), t=np.asarray(S.WINDOW,np.float32)/np.float32(50))
        write_raw(dest,path[-1],c,z,path,dict(nfe=50,known_state_checks=51,test_fixture=True,
            actual_model_forwards=0,interpretation='Synthetic arithmetic fixture only, not an experimental rollout'))
        receipt=dict(q,status='COMPLETE',label='CPU_SYNTHETIC_TEST_ONLY',model_identity={'synthetic':True},sigma=sigma,
            density_status='FULL_JOINT_GAUSSIAN' if sigma else 'DETERMINISTIC_NOT_DEFINED',
            window_buffer_path=str(buffer),window_buffer_sha256=D.sha(buffer))
        for key,name in (('raw','raw.npz'),('ply','raw.ply'),('result','run.json')):
            receipt[key+'_path'],receipt[key+'_sha256']=str(dest/name),D.sha(dest/name)
        return q,receipt

    def test_all_registered_sigmas_roundtrip_existing_RAW_contract(self):
        for sigma in S.SIGMAS:
            with self.subTest(sigma=sigma), tempfile.TemporaryDirectory(prefix='window_policy_cpu_fixture_') as temp:
                root=Path(temp)
                with patch.object(S.D,'OUT',root):
                    q,r=self.fixture(root,sigma)
                    facts=S.verify_window_receipt(q,r,model_identity={'synthetic':True})
                    self.assertTrue(all(facts.values()))

    def test_tampered_actual_action_rejected_even_with_rehashed_buffer(self):
        with tempfile.TemporaryDirectory(prefix='window_policy_cpu_fixture_') as temp:
            root=Path(temp)
            with patch.object(S.D,'OUT',root):
                q,r=self.fixture(root,.01)
                values=S.D.arrays(r['window_buffer_path'])
                values['state_after'][4,0,10,3]+=np.float32(.001)
                changed=S.D.save_npz(root/'tampered.npz',**values)
                r.update(window_buffer_path=str(changed),window_buffer_sha256=S.D.sha(changed))
                with self.assertRaisesRegex(ValueError,'Window artifact contract failed'):
                    S.verify_window_receipt(q,r)

    def test_tampered_transition_registration_rejected(self):
        with tempfile.TemporaryDirectory(prefix='window_policy_cpu_fixture_') as temp:
            root=Path(temp)
            with patch.object(S.D,'OUT',root):
                q,r=self.fixture(root,.01)
                q['transition_Gaussian_sha256']='0'*64
                with self.assertRaisesRegex(ValueError,'Transition noise array identity'):
                    S.verify_window_receipt(q,r)


if __name__=='__main__':
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    result=unittest.main(exit=False)
    assert not torch.cuda.is_initialized()
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
