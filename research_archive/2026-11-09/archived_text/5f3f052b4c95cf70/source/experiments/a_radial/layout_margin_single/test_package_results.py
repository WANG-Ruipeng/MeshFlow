"""Small archival-contract tests; no training, deployment or new geometry study."""
import copy
import pytest
from .package_results import compare_prefix,SOURCE_NAMES,inside


def states():
    config={'initial_model_sha256':'shared'}
    trace=[{'update':i} for i in range(1,20001)]
    return ({'update':0,'config':config.copy(),'trace':[],'model_sha256':'shared'},
        {'update':5000,'config':config.copy(),'trace':copy.deepcopy(trace[:5000])},
        {'update':20000,'config':config.copy(),'trace':trace})


def test_fixed_saved_prefix_is_accepted():
    compare_prefix(*states())


def test_changed_historical_prefix_is_rejected():
    a,b,c=states();b['trace'][3]['update']=999
    with pytest.raises(ValueError,match='prefix'):compare_prefix(a,b,c)


def test_another_initializer_is_rejected():
    a,b,c=states();a['model_sha256']='different'
    with pytest.raises(ValueError,match='Initial model'):compare_prefix(a,b,c)


def test_budget_and_configuration_are_not_relabelled():
    a,b,c=states();c['update']=19999
    with pytest.raises(ValueError,match='budgets'):compare_prefix(a,b,c)
    a,b,c=states();b['config']['initial_model_sha256']='different'
    with pytest.raises(ValueError,match='configuration'):compare_prefix(a,b,c)


def test_explicit_source_scope_and_portable_path_safety(tmp_path):
    assert len(SOURCE_NAMES)==len(set(SOURCE_NAMES))
    assert all(not ('interpol' in n or n.startswith(('layout_balanced/','layout_longfit/','fm_tail/'))) for n in SOURCE_NAMES)
    assert {'layout_margin_single/train.py','layout_margin_single/review.py','legacy/deform.py','assets/support.npz'} <= set(SOURCE_NAMES)
    assert inside(tmp_path,'source/a.py')==tmp_path/'source/a.py'
    with pytest.raises(ValueError):inside(tmp_path,'../outside')
