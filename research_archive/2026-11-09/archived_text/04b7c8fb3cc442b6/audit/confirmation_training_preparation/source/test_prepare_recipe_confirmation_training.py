"""CPU-only registration/job tests; never start models or OT."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

spec=importlib.util.spec_from_file_location("confirmation_training_preparation",Path(__file__).parents[1]/"tools/prepare_recipe_confirmation_training.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return path

def decision(tmp_path):
    root=tmp_path/"run";out=root/"confirmation";reg=root/"registrations"
    reg.mkdir(parents=True)
    for p in (root/"official.pt",root/"train_manifest.json",reg/"CONFIRMATION_DECISION.md"):p.write_text("fixture")
    d=dict(schema="recipe_confirmation_decision_v1",status="APPROVED",experiment=m.EXPERIMENT,
        stage="independent_confirmation",experiment_root=str(root),out=str(out),stream_root=str(out/"streams"),
        namespace=m.NAMESPACE,selected_pair=m.PAIR[:],hybrid_schedules={r:"all" for r in m.PAIR},
        official=str(root/"official.pt"),data_manifest=str(root/"train_manifest.json"),
        decision_markdown=str(reg/"CONFIRMATION_DECISION.md"),python=sys.executable,
        updates_per_recipe=6000,effective_batch=8,microbatch=1,max_training_recipes=2,max_workers=2,
        checkpoint_interval=100,max_resume_per_recipe=1,max_replay_completed_updates_per_failed_attempt=99,
        evaluation_noises=[8,9,10,11],default_changed=False,automatic_git_operation=False)
    p=write(reg/"confirmation_decision.json",d)
    return d,p

def test_valid_decision_and_independent_paths(tmp_path):
    d,p=decision(tmp_path)
    assert m.validate_decision(d,p)==d
    ap,jp,pp,rp=m.registration_paths(d)
    assert ap.name=="confirmation_training.json" and jp.parent==p.parent
    assert rp.name=="confirmation_training_ready.json"

@pytest.mark.parametrize("key,value",[
    ("status","PROPOSED"),("selected_pair",["R1_40_H_ALL","R2_MIX_H_ALL"]),
    ("namespace","CHAIR_CONDITION_RECIPE_FACTORIAL_V1/stage1"),
    ("hybrid_schedules",{"C20_40_MIX":"late","R2_MIX_H_ALL":"all"}),
    ("updates_per_recipe",6001),("effective_batch",4),("max_workers",3),
    ("max_resume_per_recipe",2),("max_replay_completed_updates_per_failed_attempt",100),
    ("evaluation_noises",[0,1,6,7]),("default_changed",True)])
def test_unapproved_scientific_or_budget_change_rejected(tmp_path,key,value):
    d,p=decision(tmp_path);d[key]=value
    with pytest.raises(ValueError):m.validate_decision(d,p)

def test_output_cannot_share_main_state(tmp_path):
    d,p=decision(tmp_path);d["out"]=d["experiment_root"]
    with pytest.raises(ValueError):m.validate_decision(d,p)

def test_freeze_idempotent_and_cannot_rewrite(tmp_path):
    p=tmp_path/"registration.json"
    first=m.freeze(p,{"x":1})
    assert m.freeze(p,{"x":1})==first
    with pytest.raises(ValueError):m.freeze(p,{"x":2})
    assert m.read(p)=={"x":1}

def test_authorization_preserves_explicit_pair_and_recovery_contract(tmp_path):
    d,p=decision(tmp_path)
    a=m.authorization(d,p,{"a":"b"})
    assert a["selected_pair"]==a["authorized_recipes"]==m.PAIR
    assert a["unique_updates"]==12000 and a["unique_microbatch_forwards"]==96000
    assert a["recovery_contract"]["maximum_resumes_per_recipe"]==1
    assert a["source_hashes"]=={"a":"b"}

def test_jobs_use_confirmation_worker_and_root_dispatcher(tmp_path):
    d,p=decision(tmp_path)
    a=m.registration_paths(d)[0];w=m.wave(d,a,{"input":"hash"})
    assert w["wave"]=="confirmation_training_input_repeat1" and w["max_workers"]==2
    assert [j["name"] for j in w["jobs"]]==m.PAIR
    for j in w["jobs"]:
        assert Path(j["status_root"]).is_relative_to(Path(d["out"]))
        cmd=j["command"]
        assert cmd[2].endswith("train_recipe_confirmation.py")
        assert cmd[cmd.index("--out")+1]==d["out"]
        assert cmd[cmd.index("--registration")+1]==str(a)
        assert "--resume" not in cmd and "--attempt-id" not in cmd
        assert len(j["required_completion_files"])==2

def paired_fixture():
    records=[]
    for i in range(48):
        records.append(dict(sample_index=i,origin_sample_index=i,step=i//8+1,uid="p"+str(i%8),
            task_id="task"+str(i),N=160,K=32,ratio=.2,alpha=1.0))
    doc=dict(schema="meshflow_recipe_confirmation_paired_inputs_v1",recipe=m.PAIR[0],
        namespace=m.NAMESPACE,selected_pair=m.PAIR,hybrid_schedule="all",plan_sha256="plan",
        data_manifest_sha256="data",implementation_sha256="implementation",historical_paired_inputs_reused=False,
        records=[dict(r,lambda_value=.25,free_coordinate_count=1152,
            shared_input_sha256="a"*64,input_bytehash="b"*64,label_bytehash="c"*64,ot_cache_key="d"*64) for r in records])
    return records,doc

def test_paired_order_and_origin_multiset(tmp_path):
    records,doc=paired_fixture()
    signatures=m.pair_contract(doc,m.PAIR[0],{},"plan","data","implementation",records,total=6)
    assert len(signatures)==48
    assert "step" not in signatures[0]
    reversed_records=list(reversed(records))
    with pytest.raises(ValueError,match="order/input"):
        m.pair_contract(doc,m.PAIR[0],{},"plan","data","implementation",reversed_records,total=6)

@pytest.mark.parametrize("field,value",[
    ("lambda_value",0),("free_coordinate_count",1),("sample_index",47),
    ("input_bytehash","bad"),("origin_sample_index",47),("step",6000)])
def test_paired_mutations_rejected(field,value):
    records,doc=paired_fixture();doc["records"][0][field]=value
    with pytest.raises(ValueError):
        m.pair_contract(doc,m.PAIR[0],{},"plan","data","implementation",records,total=6)

def test_legacy_paired_metadata_rejected():
    records,doc=paired_fixture();doc["historical_paired_inputs_reused"]=True
    with pytest.raises(ValueError,match="metadata"):
        m.pair_contract(doc,m.PAIR[0],{},"plan","data","implementation",records,total=6)

def test_pending_precompute_never_creates_dispatch_job(tmp_path,monkeypatch):
    d,p=decision(tmp_path)
    ap,jp,pp,rp=m.registration_paths(d);write(ap,{})
    monkeypatch.setattr(m,"register",lambda path:{"storage":{}})
    with pytest.raises(FileNotFoundError,match="PENDING"):
        m.finalize(p)
    assert not jp.exists() and not rp.exists()

def test_real_source_metadata_matches_worker_contract():
    from meshflow_control.training import recipe_confirmation as worker
    actual=m.source_hashes()
    expected={"src/meshflow_control/"+k:v for k,v in worker.source_identity().items()}
    assert set(expected).issubset(actual)
    assert all(actual[k]==v for k,v in expected.items())
    assert "tools/train_recipe_confirmation.py" in actual

def test_old_freeze_formats_are_all_checked(tmp_path,monkeypatch):
    repo=tmp_path/"repo";root=tmp_path/"run";regs=root/"registrations"
    sources={"src/one.py":"x","tools/compare_recipe_all_baselines.py":"y","tests/test_compare_recipe_all_baselines.py":"z","tools/new.py":"q"}
    for name,value in sources.items():
        p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    hashes={k:m.digest(repo/k) for k in sources}
    write(regs/"source_hashes.json",{"src/one.py":hashes["src/one.py"]})
    write(regs/"curriculum_additional_source_hashes.json",{"sources":{"tools/new.py":hashes["tools/new.py"]}})
    write(regs/"all_baselines_analysis_sources.json",dict(schema="v1",
        tool_path=r"E:\MeshFlow-Control\tools\compare_recipe_all_baselines.py",tool_sha256=hashes["tools/compare_recipe_all_baselines.py"],
        test_path=r"E:\MeshFlow-Control\tests\test_compare_recipe_all_baselines.py",test_sha256=hashes["tests/test_compare_recipe_all_baselines.py"]))
    write(regs/"all_baselines_analysis_v2_sources.json",{"sources":{"tools/new.py":hashes["tools/new.py"]}})
    monkeypatch.setattr(m,"REPO",repo)
    merged,files=m.prior_frozen_sources(root)
    assert merged==hashes and len(files)==4
