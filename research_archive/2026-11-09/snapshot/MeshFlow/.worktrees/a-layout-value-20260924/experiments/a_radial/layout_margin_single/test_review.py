from .review import without_timing


def test_readback_ignores_only_runtime_not_quality_evidence():
    a={'valid':False,'audit_seconds':.1,'exact':{'pair_count':2,'nested_seconds':4.}}
    b={'valid':False,'audit_seconds':9.,'exact':{'pair_count':2,'nested_seconds':7.}}
    assert without_timing(a)==without_timing(b)
    b['exact']['pair_count']=0
    assert without_timing(a)!=without_timing(b)
