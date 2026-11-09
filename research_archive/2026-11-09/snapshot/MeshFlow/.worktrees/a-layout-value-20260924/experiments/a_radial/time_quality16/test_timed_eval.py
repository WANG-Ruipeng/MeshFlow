"""Mock expensive geometry to test audit accounting, uncached calls and gates."""
import numpy as np
import pytest
from . import timed_eval as te


class Support:
    def __init__(self): self.calls = []
    def query(self, field, directions):
        self.calls.append((field.copy(), directions.copy()))
        return np.full(258, 2.0)


def case(monkeypatch, *, quality_ok=True, exact_ok=True, rms=(0.97, 0.97)):
    u = np.zeros((258, 3)); u[:, 0] = 1
    faces = np.tile([0, 1, 2], (512, 1)).astype(np.int32)
    quality = {'face_count':512, 'degenerate_count':0,
               'min_triangle_angle_deg':10.0 if quality_ok else 9.0,
               'triangle_min_angle_p05_deg':20.0}
    monkeypatch.setattr(te, 'named_quality', lambda *_: dict(quality))
    monkeypatch.setattr(te, 'audit_mesh', lambda *_, exact: {'valid':exact_ok, 'exact_executed':True})
    calls=[]
    def metric(*_, points, seed):
        calls.append(seed)
        return {'seed':seed, 'points_per_direction':points,
                'rms_bbox_percent':rms[0 if seed==101 else 1], 'full_original_metric_payload':'retained'}
    monkeypatch.setattr(te, 'evaluate_mesh', metric)
    args=(u, faces, np.zeros(2050), Support(), np.zeros((2050,3)), np.tile([0,1,2],(4096,1)))
    baseline=[{'seed':s, 'points_per_direction':8192, 'rms_bbox_percent':1.0} for s in (101,202)]
    return args, calls, baseline


def test_baseline_fresh_radius_two_mc_and_no_cache(monkeypatch):
    args,calls,_=case(monkeypatch)
    before=args[0].copy()
    for _ in range(2):
        mesh,row=te.evaluate_candidate(*args,seeds=(101,202))
        assert row['gain_fractions']==[0.0,0.0] and not row['target_pass']
        assert row['metrics'][0]['full_original_metric_payload']=='retained'
        assert np.array_equal(mesh['vertices'],before*2)
        assert row['timing']['total_seconds']>=sum(row['timing']['mc_seconds'])
    assert calls==[101,202,101,202] and len(args[3].calls)==2
    assert np.array_equal(args[0],before)


@pytest.mark.parametrize('quality_ok,exact_ok', [(False,True),(True,False)])
def test_failed_quality_or_exact_still_gets_both_mc(monkeypatch,quality_ok,exact_ok):
    args,calls,baseline=case(monkeypatch,quality_ok=quality_ok,exact_ok=exact_ok)
    _,row=te.evaluate_candidate(*args, floor={'min_triangle_angle_deg':10.,'triangle_min_angle_p05_deg':20.},
                              baseline_metrics=baseline,seeds=(101,202))
    assert calls==[101,202] and not row['qualified'] and not row['target_pass']


def test_both_seeds_must_reach_predeclared_gain(monkeypatch):
    args,_,baseline=case(monkeypatch,rms=(0.95,0.985))
    kwargs=dict(floor={'min_triangle_angle_deg':10.,'triangle_min_angle_p05_deg':20.},baseline_metrics=baseline,seeds=(101,202))
    _,row=te.evaluate_candidate(*args,**kwargs)
    assert row['qualified'] and not row['target_pass']
    args,_,baseline=case(monkeypatch,rms=(0.979,0.979));kwargs['baseline_metrics']=baseline
    assert te.evaluate_candidate(*args,**kwargs)[1]['target_pass']


def test_pair_identity_rejected_before_geometry(monkeypatch):
    args,calls,baseline=case(monkeypatch);baseline[1]['seed']=999
    with pytest.raises(ValueError,match='seeds/count'):
        te.evaluate_candidate(*args,floor={'min_triangle_angle_deg':10.,'triangle_min_angle_p05_deg':20.},baseline_metrics=baseline,seeds=(101,202))
    assert not calls and not args[3].calls


def test_zero_denominator_is_undefined_and_exceptions_propagate(monkeypatch):
    args,calls,baseline=case(monkeypatch);baseline[0]['rms_bbox_percent']=0
    kw=dict(floor={'min_triangle_angle_deg':10.,'triangle_min_angle_p05_deg':20.},baseline_metrics=baseline,seeds=(101,202))
    _,row=te.evaluate_candidate(*args,**kw)
    assert row['gain_fractions'][0] is None and not row['target_pass']
    def fail(*_,**__):raise RuntimeError('preserve in runner ledger')
    monkeypatch.setattr(te,'evaluate_mesh',fail)
    with pytest.raises(RuntimeError,match='preserve'):te.evaluate_candidate(*args,**kw)


def test_original_degree_tolerance_is_unchanged(monkeypatch):
    args,_,baseline=case(monkeypatch)
    kw=dict(floor={'min_triangle_angle_deg':10.,'triangle_min_angle_p05_deg':20.},baseline_metrics=baseline,seeds=(101,202))
    def quality(delta):
        return {'face_count':512,'degenerate_count':0,'min_triangle_angle_deg':10.-delta,'triangle_min_angle_p05_deg':20.}
    monkeypatch.setattr(te,'named_quality',lambda *_:quality(0.5e-6))
    assert te.evaluate_candidate(*args,**kw)[1]['target_pass']
    monkeypatch.setattr(te,'named_quality',lambda *_:quality(2e-6))
    assert not te.evaluate_candidate(*args,**kw)[1]['target_pass']
