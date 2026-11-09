"""CPU-only confirmation sampling registration and real-format ledger fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).parents[1]
def module(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/"tools"/file)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prep=module("confirm_sampling_prep","prepare_recipe_confirmation_sampling.py")
audit=module("confirm_sampling_audit","audit_recipe_confirmation_sampling.py")

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value))
 return path

def auth(tmp_path):
 return dict(out=str(tmp_path/"confirmation"),namespace=prep.training_prep.NAMESPACE,
     selected_pair=list(prep.PAIR),hybrid_schedules={r:"all" for r in prep.PAIR})

def done_fixture(tmp_path):
 a=auth(tmp_path);r=prep.PAIR[0];ids=dict(plan_sha256="plan",paired_inputs_sha256="pair",data_manifest_sha256="data")
 lineage=dict(fresh_AdamW=True,new_optimizer_step=0,Geo_seed=1010,
     official_file_sha256=prep.training_prep.OFFICIAL_SHA,generator_initial_sha256=prep.training_prep.INITIAL_SHA,
     no_auxiliary_head_constructed=True,role_nonzero=0,geo_exit_nonzero=0)
 v=dict(status="COMPLETE",recipe=r,segment=r,step=6000,namespace=a["namespace"],hybrid_schedule="all",
     identities=dict(ids,authorization_sha256="auth",selected_pair=list(prep.PAIR),confirmation_namespace=a["namespace"]),
     lineage=lineage,physical_optimizer_updates_this_attempt=6000,successful_calls={"optimizer_update":6000})
 return a,r,ids,v

def test_pending_does_not_create_bindings_or_jobs(tmp_path,monkeypatch):
 a=auth(tmp_path)
 monkeypatch.setattr(prep.evaluation.base,"root",lambda out:Path(out))
 monkeypatch.setattr(prep.evaluation,"check_authorization",lambda *args:a)
 result=prep.prepare(tmp_path,tmp_path/"auth.json")
 assert result["status"]=="PENDING"
 assert not (tmp_path/"phases").exists()
 assert not (tmp_path/"registrations/confirmation_sampling_jobs.json").exists()

def test_both_final_endpoints_required(tmp_path):
 a=auth(tmp_path)
 write(tmp_path/"phases/confirm/registration.json",{})
 missing=prep.pending_endpoints(tmp_path,a)
 assert len(missing)==6
 for recipe in prep.PAIR:
  base=Path(a["out"])
  write(base/"training"/recipe/"completed.json",{"status":"COMPLETE"})
  for folder,suffix in [("checkpoints","_step6000.pt"),("inference","_generator_step6000.pt")]:
   p=base/folder/(recipe+suffix);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"not a loaded tensor")
 assert prep.pending_endpoints(tmp_path,a)==[]

def test_completion_exact_fresh_initialization_and_input_identity(tmp_path):
 a,r,ids,v=done_fixture(tmp_path)
 prep.completion_contract(v,r,a,"auth",ids)

@pytest.mark.parametrize("where,key,value",[
 ("top","step",5999),("top","namespace","old"),("top","hybrid_schedule","late"),
 ("top","physical_optimizer_updates_this_attempt",5999),
 ("lineage","fresh_AdamW",False),("lineage","new_optimizer_step",5000),
 ("lineage","Geo_seed",1011),("lineage","role_nonzero",1),
 ("identities","authorization_sha256","wrong"),("identities","plan_sha256","old")])
def test_completion_mutations_rejected(tmp_path,where,key,value):
 a,r,ids,v=done_fixture(tmp_path)
 (v if where=="top" else v[where])[key]=value
 with pytest.raises(ValueError):prep.completion_contract(v,r,a,"auth",ids)

def test_endpoint_hash_schema_and_generator_export_receipt(tmp_path):
 p=tmp_path/"fake.pt";p.write_bytes(b"tiny metadata test bytes; no tensor loaded")
 recipe=prep.PAIR[0];schema="chair_recipe_confirmation_generator_v1"
 r=dict(path=str(p.resolve()),schema=schema,global_step=6000,recipe=recipe,segment=recipe,
     protocol_sha256="protocol",generator_state_sha256="a"*64,sha256=prep.digest(p),
     aligner_state_sha256=None,optimizer_state_sha256=None,rng_sha256=None,stream_state_sha256=None)
 assert prep.endpoint_contract(r,p,recipe,schema,"protocol")==r
 bad=dict(r,optimizer_state_sha256="forbidden")
 with pytest.raises(ValueError,match="training state"):prep.endpoint_contract(bad,p,recipe,schema,"protocol")
 p.write_bytes(b"changed")
 with pytest.raises(ValueError,match="hash"):prep.endpoint_contract(r,p,recipe,schema,"protocol")

def test_sampling_jobs_use_shared_root_and_exact_adapter(tmp_path):
 w=prep.sampling_wave(tmp_path,{"source":"hash"})
 assert w["wave"]=="confirmation_sampling_input_repeat1" and len(w["jobs"])==2 and w["max_workers"]==2
 for j in w["jobs"]:
  c=j["command"]
  assert j["attempt_option"]=="--attempt"
  assert c[2].endswith("evaluate_recipe_confirmation.py")
  assert c[c.index("--phase")+1]=="confirm" and c[c.index("--out")+1]==str(tmp_path)
  assert "--controller-lock-owned" in c and "--resume" not in c
  assert "latest_checkpoint" not in j
  assert Path(j["status_root"]).is_relative_to(tmp_path/"phases/confirm/sampling")

def ledger_fixture(tmp_path,phase="confirm",error_step=None,wrong_step=None):
 context=dict(label=prep.PAIR[0],request_id="parent__C20__n8",attempt="initial",phase=phase,
     checkpoint_sha256="checkpoint",input_sha256="input")
 events=[dict(event="CELL_REGISTERED",context=context),
         dict(event="RESERVE",kind="rollout_attempt",ordinal=1,context=context)]
 for step in range(50):
  euler=wrong_step if wrong_step is not None and step==13 else step
  events.append(dict(event="RESERVE",kind="sampling_forward",ordinal=step+1,context=dict(context,euler_step=euler)))
  # Actual RETURN intentionally has neither context nor Euler step.
  events.append(dict(event="ERROR" if step==error_step else "RETURN",kind="sampling_forward"))
 events.append(dict(event="ROLLOUT_COMPLETE",nfe=50,known_all51_bitwise=True))
 p=tmp_path/"calls.jsonl"
 p.write_text("".join(json.dumps(e)+"\n" for e in events))
 return p,context

def test_confirm_real_format_returns_without_context(tmp_path):
 p,c=ledger_fixture(tmp_path)
 counts,errors,h=audit.ledger(p,c)
 assert not errors and counts["sampling_forward_returned"]==50 and counts["rollout_attempt_reserved"]==1
 assert len(h)==64

def test_wrong_phase_detected_even_if_counts_match(tmp_path):
 p,c=ledger_fixture(tmp_path,phase="course")
 counts,errors,h=audit.ledger(p,dict(c,phase="confirm"))
 assert counts["sampling_forward_returned"]==50 and any("context" in e for e in errors)

def test_wrong_euler_order_error_and_truncation_not_hidden(tmp_path):
 p,c=ledger_fixture(tmp_path,error_step=9,wrong_step=12)
 counts,errors,h=audit.ledger(p,c)
 assert counts["sampling_forward_failed"]==1 and "Euler step sequence" in errors
 p.write_bytes(p.read_bytes().rstrip(b"\n"))
 assert "truncated ledger tail" in audit.ledger(p,c)[1]

def test_partial_or_unregistered_output_is_not_complete(tmp_path):
 phase=tmp_path/"confirm"
 assert audit.readiness(phase)
 for recipe in audit.PAIR:
  write(phase/"sampling"/recipe/"completed.json",{"status":"COMPLETE"})
 assert "128 immutable output receipt files not yet present" in audit.readiness(phase)

def test_receipts_cannot_claim_reuse_or_external_raw_as_new(tmp_path):
    q=dict(request_id="p__C20__n8",candidate=8,noise_full_N_Gaussian_sha256="gaussian")
    r=dict(q,label=prep.PAIR[0],status="COMPLETE",generation_status="NEW_EULER50",new_rollouts=1,
           raw_path=str(tmp_path/"raw.npz"),ply_path=str(tmp_path/"raw.ply"),result_path=str(tmp_path/"run.json"))
    assert not audit.receipt_identity_errors(q,r,prep.PAIR[0],tmp_path,r)
    wrong=dict(r,raw_path=str(tmp_path.parent/"old/raw.npz"))
    assert "output_path_raw" in audit.receipt_identity_errors(q,wrong,prep.PAIR[0],tmp_path,wrong)
    reused=dict(r,generation_status="VERIFIED_HISTORICAL_RAW_REUSE",new_rollouts=0)
    assert "confirmation_requires_one_new_rollout" in audit.receipt_identity_errors(q,reused,prep.PAIR[0],tmp_path,reused)
