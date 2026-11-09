"""CPU-only schedule/optimizer/recovery contract tests; no generator forward."""
import copy
import torch
import pytest
from meshflow_control.training import schedule40 as s


class ToyGenerator(torch.nn.Linear):
    readout_mode = 'none'


def objects(head=True):
    model = ToyGenerator(3, 2).eval()
    aux = torch.nn.Linear(2, 2).eval() if head else None
    return model, aux, s.make_optimizer(model, aux)


def seed_moments(opt, generator_step, head_step):
    for index, group in enumerate(opt.param_groups):
        count = generator_step if index == 0 else head_step
        for p in group['params']:
            opt.state[p] = dict(step=torch.tensor(float(count)), exp_avg=torch.full_like(p, .01), exp_avg_sq=torch.full_like(p, .02))


def test_boundaries_and_equal_alignment_budget():
    assert sum(s.active_alignment('EARLY', k) for k in range(1, 6001)) == 1000
    assert sum(s.active_alignment('LATE', k) for k in range(1, 6001)) == 1000
    assert not any(s.active_alignment('FM_ONLY', k) for k in range(1, 6001))
    assert s.active_alignment('EARLY', 1000) and not s.active_alignment('EARLY', 1001)
    assert not s.active_alignment('LATE', 5000) and s.active_alignment('LATE', 5001)
    assert sum(b-a for a,b in s.SEGMENTS.values()) == 13000
    with pytest.raises(ValueError): s.active_alignment('EARLY', 0)


def test_early_switch_preserves_generator_momentum_and_freezes_head():
    model, head, opt = objects()
    seed_moments(opt, 1000, 1000)
    head_before = copy.deepcopy(head.state_dict()); opt_head_before = s.tree_hash({i:opt.state[p] for i,p in enumerate(head.parameters())})
    assert s.validate_optimizer(model, head, opt, 'EARLY', 1000)
    for p in model.parameters(): p.grad = torch.full_like(p, .1)
    for p in head.parameters(): p.grad = None
    opt.step()
    assert s.validate_optimizer(model, head, opt, 'EARLY', 1001)
    assert s.tree_hash(head.state_dict()) == s.tree_hash(head_before)
    assert s.tree_hash({i:opt.state[p] for i,p in enumerate(head.parameters())}) == opt_head_before
    assert all(torch.count_nonzero(opt.state[p]['exp_avg']) for p in model.parameters())


def test_late_adds_fresh_head_without_resetting_generator():
    model, _, opt = objects(False)
    seed_moments(opt, 5000, 0)
    prior = s.tree_hash(opt.state_dict())
    restored = s.make_optimizer(model, None); restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert s.tree_hash(restored.state_dict()) == prior
    head = torch.nn.Linear(2, 2).eval()
    restored.add_param_group(dict(params=list(head.parameters()), lr=1e-4, betas=(.9,.95), weight_decay=0.))
    assert s.validate_optimizer(model, head, restored, 'LATE', 5000)
    for p in list(model.parameters()) + list(head.parameters()): p.grad = torch.full_like(p, .1)
    restored.step()
    assert s.validate_optimizer(model, head, restored, 'LATE', 5001)
    assert all(float(restored.state[p]['step']) == 1 for p in head.parameters())


def test_hash_includes_scalar_steps_and_detects_corruption():
    value = {'state': {0:{'step':torch.tensor(5000.), 'exp_avg':torch.zeros(3)}}}
    changed = copy.deepcopy(value); changed['state'][0]['step'] += 1
    assert s.tree_hash(value) != s.tree_hash(changed)
    assert s.tree_hash(value) == s.tree_hash(copy.deepcopy(value))


def test_fresh_optimizer_has_no_inherited_moments():
    model, head, opt = objects()
    assert not opt.state
    assert s.validate_optimizer(model, head, opt, 'EARLY', 0)
    with pytest.raises(ValueError): s.validate_optimizer(model, head, opt, 'EARLY', 1)


def test_head_seed_is_isolated_and_exact_across_activation_times():
    before = torch.get_rng_state().clone()
    early = s.AlignmentHead()
    assert torch.equal(torch.get_rng_state(), before)
    torch.manual_seed(42)
    late = s.AlignmentHead()
    assert s.state_hash(early.state_dict()) == s.state_hash(late.state_dict())
    assert torch.count_nonzero(early.W_O.weight) > 0


def test_atomic_checkpoint_roundtrip_and_rng_identity(tmp_path, monkeypatch):
    # A tiny parameter carrier exercises the real serialization and fork loader
    # without constructing or evaluating a 130M-parameter generator.
    monkeypatch.setattr(s, '_check_generator', lambda state: None)
    monkeypatch.setattr(s, 'build_model', lambda *a, **k: ToyGenerator(3,2).eval())
    model, _, opt = objects(False)
    seed_moments(opt, 5000, 0)
    saved = s._payload(model, None, opt, segment='COMMON', step=5000,
        stream_state={'batch_index':5000,'plan_sha256':'frozen'}, identities={'test':True}, lineage={'official':True}, receipt={})
    path = tmp_path / 'common.pt'
    receipt = s._write_checkpoint(path, saved, immutable=True)
    assert receipt['global_step'] == 5000 and receipt['sha256'] == s.digest(path)
    m, h, o, p = s.load_checkpoint(path, segment='FM_ONLY', device='cpu', fork=True)
    assert h is None and s.state_hash(m.state_dict()) == saved['generator_state_sha256']
    assert s.tree_hash(o.state_dict()) == saved['optimizer_state_sha256']
    assert s.tree_hash(p['rng']) == saved['rng_sha256']
    late, head, lateopt, p2 = s.load_checkpoint(path, segment='LATE', device='cpu', fork=True)
    assert s.validate_optimizer(late, head, lateopt, 'LATE', 5000)
    assert all(p not in lateopt.state for p in head.parameters())
    assert s.state_hash(head.state_dict()) == s.state_hash(s.AlignmentHead().state_dict())
    with pytest.raises(FileExistsError): s._write_checkpoint(path, saved, immutable=True)
    broken = copy.deepcopy(saved); broken['optimizer']['state'][0]['step'] += 1
    with pytest.raises(ValueError, match='hash'): s.validate_payload(broken)
