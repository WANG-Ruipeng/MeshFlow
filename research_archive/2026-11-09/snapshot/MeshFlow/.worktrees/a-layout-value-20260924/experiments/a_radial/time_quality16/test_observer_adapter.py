"""CPU tests for an observer-only transformation. No formal data or GPU use."""
import ast
import copy
import inspect
import numpy as np
import pytest
import torch

from . import observer_adapter as adapter
from ..exporter import RadialSupport
from ..legacy.geometry_task import fibonacci, hull_faces
from ..layout_value.optimizer import optimize_layout as original_optimizer, _np_quality


def test_AST_only_registered_insertion_and_keyword_argument():
    _, original = adapter._source_tree()
    observed = adapter.instrument_tree(original)
    assert adapter.validate_transform(original, observed)
    function = adapter._function(observed)
    record = adapter._record(function)
    assert isinstance(record.body[-3], ast.If)  # original snapshot branch
    assert 'snapshots.append' in ast.unparse(record.body[-3])
    assert ast.unparse(record.body[-2]) == ast.unparse(ast.parse(adapter.INSERTION_SOURCE).body[0])
    compiled = adapter.build_observed_optimizer()
    signature = inspect.signature(compiled)
    assert signature.parameters['on_record'].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters['on_record'].default is None
    original_signature = inspect.signature(original_optimizer)
    assert signature.replace(parameters=[v for k, v in signature.parameters.items()
                                         if k != 'on_record']) == original_signature
    identity = compiled.__observer_identity__
    assert identity['original_source_sha256'] == adapter.EXPECTED_SOURCE_SHA256
    assert identity['return_value_used_for_control'] is False


@pytest.mark.parametrize('change', ['other_statement', 'move_callback', 'row_alias'])
def test_AST_rejects_any_extra_edit(change):
    _, original = adapter._source_tree()
    observed = adapter.instrument_tree(original)
    function = adapter._function(observed)
    if change == 'other_statement':
        function.body.insert(1, ast.parse('max_steps = 0').body[0])
    elif change == 'move_callback':
        record = adapter._record(function)
        record.body.insert(-3, record.body.pop(-2))
    else:
        record = adapter._record(function)
        call = record.body[-2].body[0].value
        next(x for x in call.keywords if x.arg == 'row').value = ast.Name(id='row', ctx=ast.Load())
    with pytest.raises(ValueError):
        adapter.validate_transform(original, observed)


def test_changed_frozen_file_rejected_before_import(tmp_path, monkeypatch):
    changed = tmp_path / 'optimizer.py'
    changed.write_bytes(adapter.SOURCE_PATH.read_bytes() + b'\n# unregistered change\n')
    monkeypatch.setattr(adapter, 'SOURCE_PATH', changed)
    with pytest.raises(ValueError, match='source SHA256'):
        adapter.source_identity()


def _equal_semantics(a, b, key=None):
    # Only the original real-clock measurements are expected to differ.
    if key in {'elapsed_seconds', 'seconds', 'first_feasible_seconds'}:
        return
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            _equal_semantics(a[k], b[k], k)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _equal_semantics(x, y)
    else:
        assert a == b


def test_synthetic_four_steps_bitwise_parity_and_observer_cannot_mutate_state():
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    support = RadialSupport()
    axes = np.array([1.5, .8, 1.], dtype=np.float64)
    y = np.log(1. / np.sqrt(np.sum(support.u ** 2 / axes ** 2, axis=1)))
    u = fibonacci(258).astype(np.float64)
    faces = hull_faces(u).astype(np.int32)
    floor_quality = _np_quality(u * support.query(y, u)[:, None], faces)
    floor = {k: floor_quality[k] for k in ('min_angle_deg', 'angle_q05_deg')}
    inputs_before = [x.copy() for x in (y, u, faces)]
    kwargs = dict(device='cpu', max_steps=4, max_seconds=3600, seed=1493499,
                  snapshot_steps=tuple(range(5)))
    baseline = original_optimizer(y, u, faces, floor, **kwargs)
    observations = []
    def observer(**payload):
        assert set(payload) == set(adapter.CALLBACK_FIELDS)
        observations.append(copy.deepcopy(payload))
        assert not np.shares_memory(payload['current_u'], payload['delivered_u'])
        payload['current_u'][:] = 123.
        payload['delivered_u'][:] = -123.
        payload['faces'][:] = -1
        payload['row']['quality']['min_angle_deg'] = -999.
        payload['row']['internal_loss'] = -999.
        return {'stop': True, 'replace_output': np.zeros_like(u)}
    observed = adapter.optimize_layout_observed(y, u, faces, floor, on_record=observer, **kwargs)
    assert observed['stats']['iterations'] == baseline['stats']['iterations'] == 4
    _equal_semantics(baseline, observed)
    assert [x['iteration'] for x in observations] == list(range(5))
    for actual, before in zip((y, u, faces), inputs_before):
        assert np.array_equal(actual, before)
    by_iteration = {s['iteration']: s for s in baseline['snapshots']}
    best_u = None
    best_loss = float('inf')
    best_iteration = None
    for payload in observations:
        k = payload['iteration'];row = baseline['trace'][k]
        assert np.array_equal(payload['current_u'], by_iteration[k]['u'])
        _equal_semantics(payload['row'], row)
        if row['feasible'] and row['internal_loss'] < best_loss:
            best_loss = row['internal_loss'];best_u = by_iteration[k]['u'];best_iteration = k
        expected = payload['current_u'] if best_u is None else best_u
        assert np.array_equal(payload['delivered_u'], expected)
        assert payload['best_feasible_iteration'] == best_iteration
    assert not torch.cuda.is_initialized()