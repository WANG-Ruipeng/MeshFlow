"""CPU checks for curriculum authorization and existing trainer completion schema."""
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import prepare_recipe_curriculum as c
import pytest
from meshflow_control.data.io import json_hash

def auth():
    return dict(schema="chair_condition_recipe_factorial_training_authorization_v1",experiment="CHAIR_CONDITION_RECIPE_FACTORIAL_V1",
        namespace="fixed",out="/tmp/course",updates_per_recipe=6000,effective_batch=8,microbatch=1,
        authorized_recipes=list(c.COURSES),hybrid_schedules={x:"all" for x in c.COURSES})
def test_exact_budget_and_baseline_not_dispatched():
    a=auth(); c.authorization_contract(a,Path("/tmp/course"),"fixed")
    for kind in ("training","sampling"):
        w=c.wave(Path("/tmp/course"),Path("/tmp/auth.json"),kind,{})
        assert len(w["jobs"])==2 and w["max_workers"]==2
        assert {j["name"] for j in w["jobs"]}==set(c.COURSES)
        assert c.BASELINE not in str(w)
@pytest.mark.parametrize("key,value",[("updates_per_recipe",6001),("effective_batch",4),("namespace","other"),
    ("authorized_recipes",["C20_40_MIX"]),("hybrid_schedules",{"C20_40_MIX":"late","C40_20_MIX":"late"})])
def test_changed_design_rejected(key,value):
    a=auth();a[key]=value
    with pytest.raises(ValueError):c.authorization_contract(a,Path("/tmp/course"),"fixed")
def test_existing_real_receipt_schema_and_hash_function():
    repo=Path(__file__).resolve().parents[1]
    root=repo.parent/"MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1"
    d=c.read_json(root/"training/R2_MIX_H_ALL/completed.json")
    c.completion_contract(d,"R2_MIX_H_ALL",{"authorization_sha256":d["identities"]["authorization_sha256"]})
    from meshflow_control.training.recipe_factorial import protocol
    assert json_hash(protocol(d["recipe"],d["hybrid_schedule"],d["namespace"]))==d["protocol_sha256"]
    bad=copy.deepcopy(d);bad["step"]=5999
    with pytest.raises(ValueError):c.completion_contract(bad,"R2_MIX_H_ALL",{})
