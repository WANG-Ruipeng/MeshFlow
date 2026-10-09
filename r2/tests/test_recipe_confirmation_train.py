"""Small CPU metadata tests; no models, tensor weights, OT or forward calls."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from meshflow_control.training import recipe_confirmation as m
from meshflow_control.training import recipe_factorial as previous

NAMESPACE = "CHAIR_CONDITION_RECIPE_FACTORIAL_V1/confirmation/synthetic_test"


def registration(tmp_path, pair=None, schedules=None):
    pair = pair or ["R1_40_H_ALL", "R4_MIX_H_LATE"]
    schedules = schedules or {pair[0]: "all", pair[1]: "late"}
    source_hashes = {"src/meshflow_control/" + k: v for k, v in m.source_identity().items()}
    repo = Path(m.__file__).resolve().parents[3]
    source_hashes["tools/train_recipe_confirmation.py"] = m.digest(repo / "tools/train_recipe_confirmation.py")
    return dict(schema=m.AUTH_SCHEMA, experiment=m.EXPERIMENT, stage="independent_confirmation",
        namespace=NAMESPACE, selected_pair=pair, authorized_recipes=pair[:],
        hybrid_schedules=schedules, updates_per_recipe=6000, effective_batch=8,
        microbatch=1, max_training_recipes=2, out=str(tmp_path / "confirmation"),
        experiment_root=str(tmp_path), source_hashes=source_hashes)


def check(tmp_path, value, recipe="R1_40_H_ALL", schedule="all"):
    path=tmp_path/"authorization.json"
    m.atomic_json(path,value)
    return m.check_authorization(path,recipe,schedule,NAMESPACE,value["out"])


def test_math_initializer_optimizer_are_frozen_implementations():
    assert m.update is previous.update
    assert m.initialize_official is previous.initialize_official
    assert m.validate_optimizer is previous.validate_optimizer
    assert m.lambda_at_step is previous.lambda_at_step
    p=m.protocol("R3_40_H_LATE","late",NAMESPACE)
    assert p["generator_lr"] == 1e-5 and p["betas"] == [.9,.95]
    assert p["batch"] == 8 and p["microbatch"] == 1
    assert p["Geo_seed"] == 1010 and p["alignment"] is False and p["teacher"] is False
    assert p["FM_denominator"] == "whole effective batch valid free scalar count"
    assert p["precision"] == "BF16 forward; FP32 parameters and integration"
    assert p["sdpa"] == "math" and p["tf32"] is False
    assert p["no_optimizer_reset_at_lambda_switch"] is True
    assert p["independent_training_input_stream"] is True
    assert p["independent_model_initialization"] is False


def test_explicit_pair_authorization_passes(tmp_path):
    reg=registration(tmp_path)
    assert check(tmp_path,reg) == reg


@pytest.mark.parametrize("field,value", [
    ("schema",previous.AUTH_SCHEMA),
    ("stage","stage1"),
    ("selected_pair",["R1_40_H_ALL"]),
    ("selected_pair",["R1_40_H_ALL","R1_40_H_ALL"]),
    ("authorized_recipes",["R1_40_H_ALL"]),
    ("hybrid_schedules",{"R1_40_H_ALL":"all","R4_MIX_H_LATE":"all"}),
    ("updates_per_recipe",6001),
    ("effective_batch",4),
    ("microbatch",2),
    ("max_training_recipes",3),
])
def test_authorization_rejects_budget_or_pair_changes(tmp_path,field,value):
    reg=registration(tmp_path);reg[field]=value
    with pytest.raises(ValueError):
        check(tmp_path,reg)


def test_namespace_output_and_source_boundaries(tmp_path):
    with pytest.raises(ValueError):
        m.protocol("R1_40_H_ALL","all","CHAIR_CONDITION_RECIPE_FACTORIAL_V1/stage1")
    for badout in (str(tmp_path),str(tmp_path.parent/"elsewhere")):
        reg=registration(tmp_path);reg["out"]=badout
        with pytest.raises(ValueError):
            check(tmp_path,reg)
    reg=registration(tmp_path)
    reg["source_hashes"].pop("src/meshflow_control/data/recipe_confirmation.py")
    with pytest.raises(ValueError):
        check(tmp_path,reg)
    reg=registration(tmp_path)
    reg["source_hashes"]["src/meshflow_control/training/recipe_factorial.py"]="0"*64
    with pytest.raises(ValueError):
        check(tmp_path,reg)


def plan(reg):
    return dict(schema=m.PLAN_SCHEMA,namespace=reg["namespace"],
        independent_training_input_stream=True,independent_model_initialization=False,
        Geo_seed=1010,total_updates=6000,effective_batch_count=6000,
        selected_pair=reg["selected_pair"],hybrid_schedules=reg["hybrid_schedules"])


def test_plan_rejects_old_mother_stream_and_pair_mixing(tmp_path):
    reg=registration(tmp_path);p=plan(reg)
    assert m.validate_confirmation_plan(p,reg,"R1_40_H_ALL")
    for field,value in [
        ("namespace","CHAIR_CONDITION_RECIPE_FACTORIAL_V1/stage1"),
        ("independent_training_input_stream",False),("Geo_seed",777),
        ("base_stream_root","/old/stream"),("base_plan_sha256","old"),
        ("base_paired_inputs_sha256","old"),
        ("selected_pair",["R2_MIX_H_ALL","R3_40_H_LATE"])]:
        q=dict(p,**{field:value})
        with pytest.raises(ValueError):
            m.validate_confirmation_plan(q,reg,"R1_40_H_ALL")


def test_course_pair_requires_explicit_hybrid_policy(tmp_path):
    pair=["C20_40_MIX","R2_MIX_H_ALL"]
    reg=registration(tmp_path,pair,{pair[0]:"late",pair[1]:"all"})
    assert check(tmp_path,reg,recipe=pair[0],schedule="late")
    with pytest.raises(ValueError):
        m.protocol(pair[0],None,NAMESPACE)
    assert m.lambda_at_step(pair[0],4000,"late") == 0
    assert m.lambda_at_step(pair[0],4001,"late") == .25
    assert m.lambda_at_step(pair[0],4000,"all") == .25


def fake_payload(monkeypatch):
    # Patches are confined to the new module, never to the frozen implementation.
    monkeypatch.setattr(m,"validate_optimizer",lambda *args: None)
    monkeypatch.setattr(m,"optimizer_names",lambda model: ["synthetic"])
    monkeypatch.setattr(m,"_check_generator",lambda value: None)
    monkeypatch.setattr(m,"state_hash",m.json_hash)
    monkeypatch.setattr(m,"rng_state",lambda: {"synthetic_rng": [1,2,3]})
    model=SimpleNamespace(state_dict=lambda: {"synthetic": 1})
    opt=SimpleNamespace(state_dict=lambda: {"state": {"step":4000}})
    return m._payload(model,opt,recipe="R3_40_H_LATE",hybrid_schedule="late",
        namespace=NAMESPACE,step=4000,stream_state={"batch_index":4000,"namespace":NAMESPACE},
        identities={"selected_pair":["R1_40_H_ALL","R3_40_H_LATE"],"confirmation_namespace":NAMESPACE},
        lineage={"fresh_AdamW":True},receipt={"completed_updates":4000})


def test_complete_resume_metadata_and_lambda_boundary(monkeypatch):
    value=fake_payload(monkeypatch)
    assert value["schema"] == m.SCHEMA != previous.SCHEMA
    assert value["next_lambda"] == .25 and value["update_in_progress"] is False
    assert value["global_step"] == value["stream_state"]["batch_index"] == 4000
    assert m.validate_payload(value) is value
    for field,newvalue in [
        ("schema",previous.SCHEMA),("global_step",4001),("update_in_progress",True),
        ("next_lambda",0.),("aligner_state",{"unexpected":1}),
        ("teacher_state",{"unexpected":1}),("rng",{"synthetic_rng":[9]}),
        ("optimizer",{"state":{"step":0}}),("generator_state",{"synthetic":2})]:
        bad=copy.deepcopy(value);bad[field]=newvalue
        with pytest.raises(ValueError):
            m.validate_payload(bad)


def test_input_cursor_and_optimizer_step_are_required_for_payload(monkeypatch):
    fake_payload(monkeypatch)
    model=SimpleNamespace(state_dict=lambda: {"synthetic":1})
    opt=SimpleNamespace(state_dict=lambda: {})
    with pytest.raises(ValueError):
        m._payload(model,opt,recipe="R1_40_H_ALL",hybrid_schedule="all",
            namespace=NAMESPACE,step=4000,stream_state={"batch_index":3999},
            identities={},lineage={},receipt={})


def test_export_and_checkpoint_schemas_cannot_alias_stage1():
    assert m.SCHEMA != previous.SCHEMA
    assert m.INFERENCE_SCHEMA != previous.INFERENCE_SCHEMA
    assert m.AUTH_SCHEMA != previous.AUTH_SCHEMA
    assert m.CHECKPOINT_INTERVAL == 100 and m.TOTAL_UPDATES == 6000
