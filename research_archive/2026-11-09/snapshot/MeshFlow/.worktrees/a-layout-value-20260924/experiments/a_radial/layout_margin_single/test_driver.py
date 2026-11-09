import pytest
from .run_all import validate_report

def example():
    p={'jobs':{'old_abcd':{'uid':'abcd','target_kind':'old'}}}
    report={'status':'COMPLETE','update':20000,'job_id':'old_abcd','uid':'abcd','target_kind':'old',
        'updates_this_invocation':20000,'parent_exposures':{'abcd':80000},'device':'cpu','torch_threads':1,
        'parameter_count':29907,'checkpoint_selection':'predeclared_final_update_only','cuda_rng_recorded':False,
        'step_sequence_sha256':'registered-sequence'}
    return p,report

def test_fixed_completion():
    p,r=example();assert validate_report(r,'old_abcd',p)=='registered-sequence'

@pytest.mark.parametrize('key,value',[('update',5000),('parent_exposures',{'abcd':79996}),
    ('target_kind','margin'),('device','cuda'),('cuda_rng_recorded',True)])
def test_reject_wrong_completion(key,value):
    p,r=example();r[key]=value
    with pytest.raises(ValueError):validate_report(r,'old_abcd',p)