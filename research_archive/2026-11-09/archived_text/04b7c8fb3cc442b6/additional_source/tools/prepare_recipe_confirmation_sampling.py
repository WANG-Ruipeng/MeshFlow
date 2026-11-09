#!/usr/bin/env python3
"""Prepare the fixed confirmation sampling wave; never run a sampler."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_recipe_confirmation as evaluation
import prepare_recipe_confirmation_training as training_prep
read,digest,freeze=training_prep.read,training_prep.digest,training_prep.freeze
PAIR=tuple(training_prep.PAIR)
VERSION="recipe_confirmation_sampling_preparation_v1"
REPO=Path(__file__).resolve().parents[1]

def pending_endpoints(out,authorization):
    missing=[]
    phase=out/"phases/confirm"
    if not (phase/"registration.json").is_file():
        missing.append(str(phase/"registration.json"))
    training=Path(authorization["out"])
    for recipe in PAIR:
        done=training/"training"/recipe/"completed.json"
        if not done.is_file() or read(done).get("status")!="COMPLETE":
            missing.append(str(done)+": COMPLETE required")
        for path in (training/"checkpoints"/(recipe+"_step6000.pt"),
                     training/"inference"/(recipe+"_generator_step6000.pt")):
            if not path.is_file():missing.append(str(path))
    return missing

def completion_contract(value,recipe,auth,authorization_sha,expected_ids):
    for key,expected in dict(status="COMPLETE",recipe=recipe,segment=recipe,step=6000,
            namespace=auth["namespace"],hybrid_schedule="all").items():
        if value.get(key)!=expected:raise ValueError("Completed training differs: "+key)
    expected=dict(expected_ids,authorization_sha256=authorization_sha,selected_pair=list(PAIR),
                  confirmation_namespace=auth["namespace"])
    if any(value.get("identities",{}).get(k)!=v for k,v in expected.items()):
        raise ValueError("Completed input/authorization identities differ")
    lineage=value.get("lineage",{})
    for key,expected in dict(fresh_AdamW=True,new_optimizer_step=0,Geo_seed=1010,
            official_file_sha256=training_prep.OFFICIAL_SHA,
            generator_initial_sha256=training_prep.INITIAL_SHA,
            no_auxiliary_head_constructed=True,role_nonzero=0,geo_exit_nonzero=0).items():
        if lineage.get(key)!=expected:raise ValueError("Initialization differs: "+key)
    if value.get("physical_optimizer_updates_this_attempt")!=value.get("successful_calls",{}).get("optimizer_update"):
        raise ValueError("Completed attempt update ledger differs")

def endpoint_contract(record,path,recipe,schema,protocol_sha):
    if (record.get("path")!=str(path.resolve()) or record.get("schema")!=schema
        or record.get("global_step")!=6000 or record.get("recipe")!=recipe
        or record.get("segment")!=recipe or record.get("protocol_sha256")!=protocol_sha):
        raise ValueError("Endpoint receipt identity differs")
    if not isinstance(record.get("generator_state_sha256"),str) or len(record["generator_state_sha256"])!=64:
        raise ValueError("Endpoint generator-state receipt missing")
    if digest(path)!=record.get("sha256"):raise ValueError("Actual immutable endpoint hash differs")
    if record.get("aligner_state_sha256") is not None:raise ValueError("Unexpected auxiliary state")
    if schema.endswith("_generator_v1") and any(record.get(k) is not None for k in ("optimizer_state_sha256","rng_sha256","stream_state_sha256")):
        raise ValueError("Generator export receipt contains training state")
    return record

def sampling_wave(out,identities):
    jobs=[]
    for recipe in PAIR:
        folder=out/"phases/confirm/sampling"/recipe
        jobs.append(dict(name=recipe,phase="sampling",status_root=str(folder),
            progress_path=str(folder/"progress.json"),completed_path=str(folder/"completed.json"),
            attempt_option="--attempt",source_hashes=identities,
            command=[sys.executable,"-B",str(REPO/"tools/evaluate_recipe_confirmation.py"),
                "--stage","sample","--out",str(out),"--phase","confirm","--label",recipe,
                "--controller-lock-owned"]))
    return dict(schema="recipe_factorial_dispatch_wave_v1",wave="confirmation_sampling_input_repeat1",
        max_workers=2,jobs=jobs)

def prepare(out,authorization,verify=False):
    out=evaluation.base.root(out);authorization=Path(authorization).resolve()
    auth=evaluation.check_authorization(out,authorization)
    if auth["selected_pair"]!=list(PAIR) or auth["hybrid_schedules"]!={r:"all" for r in PAIR}:
        raise ValueError("This wave only supports the prospectively selected pair")
    missing=pending_endpoints(out,auth)
    if missing:return dict(status="PENDING",missing=missing,jobs_published=False,model_calls=0,rollouts=0)
    phase,registration,requests=evaluation.phase_registration(out,"confirm")
    if (registration.get("nfe_per_output")!=50 or registration.get("max_cell_attempts")!=1
        or registration.get("max_label_dispatch_attempts")!=2 or {q["candidate"] for q in requests}!={8,9,10,11}):
        raise ValueError("Fixed confirmation sampling budget differs")
    training=Path(auth["out"])
    ready_path=out/"registrations/confirmation_training_ready.json";ready=read(ready_path)
    if ready.get("status")!="READY_NOT_DISPATCHED" or ready.get("authorization_sha256")!=digest(authorization):
        raise ValueError("Training preparation authorization differs")
    for p,h in ready["input_and_source_hashes"].items():
        if digest(p)!=h:raise ValueError("Registered training input/source changed: "+p)
    source_hashes={str(REPO/k):v for k,v in auth["source_hashes"].items()}
    binding_pre=read(phase/"preparation_binding.json")
    source_hashes.update({str(REPO/k):v for k,v in binding_pre["sources"].items()})
    for path in (Path(__file__).resolve(),REPO/"tools/dispatch_recipe_factorial.py",
                 REPO/"tools/prepare_recipe_confirmation_training.py",authorization,ready_path,
                 phase/"registration.json",phase/"preparation_binding.json",Path(registration["requests_path"])):
        source_hashes[str(path)]=digest(path)
    source_hashes.update({item["path"]:item["sha256"] for item in binding_pre["files"].values()})
    endpoints=[]
    for recipe in PAIR:
        done_path=training/"training"/recipe/"completed.json";done=read(done_path)
        ids=dict(plan_sha256=digest(training/"streams/confirmation_plan.json"),
            paired_inputs_sha256=digest(training/"streams/paired_inputs"/(recipe+".json")),
            data_manifest_sha256=read(training/"streams/confirmation_plan.json")["data_manifest_sha256"])
        completion_contract(done,recipe,auth,digest(authorization),ids)
        expected_sources={k.removeprefix("src/meshflow_control/"):v for k,v in auth["source_hashes"].items() if k.startswith("src/meshflow_control/")}
        if done["identities"]["training_sources"]!=expected_sources:
            raise ValueError("Completed training source identities differ")
        initialization=training/"training"/recipe/"initialization_audit.json"
        if read(initialization)!=done["lineage"]:raise ValueError("Initialization receipt/lineage differs")
        checkpoint=training/"checkpoints"/(recipe+"_step6000.pt")
        export=training/"inference"/(recipe+"_generator_step6000.pt")
        cp=endpoint_contract(done["checkpoint"],checkpoint,recipe,"chair_recipe_confirmation_training_v1",done["protocol_sha256"])
        exp=endpoint_contract(done["inference_export"],export,recipe,"chair_recipe_confirmation_generator_v1",done["protocol_sha256"])
        if read(checkpoint.with_suffix(".json"))!=cp or cp["generator_state_sha256"]!=exp["generator_state_sha256"]:
            raise ValueError("Final checkpoint/export generator receipt differs")
        for path in (done_path,initialization,checkpoint.with_suffix(".json")):source_hashes[str(path)]=digest(path)
        endpoints.append(dict(recipe=recipe,checkpoint=cp,export=exp))
    # Both complete endpoints have passed before either binding is materialized.
    bindings=[]
    for recipe in PAIR:
        binding=evaluation.base.bind_checkpoint(out,"confirm",recipe)
        path=phase/"checkpoints"/(recipe+".json")
        source_hashes[str(path)]=digest(path);bindings.append(binding)
    prep=dict(version=VERSION,status="PREPARED_NOT_DISPATCHED",phase="confirm",selected_pair=list(PAIR),
        registered_outputs=128,nfe_per_output=50,expected_sampling_forwards=6400,
        noises=[8,9,10,11],max_cell_attempts=1,max_label_dispatch_attempts=2,
        authorization_path=str(authorization),authorization_sha256=digest(authorization),
        endpoints=endpoints,checkpoint_bindings=bindings,source_hashes=source_hashes,
        shared_controller_state=str(out/"audit/dispatch_state.json"),
        controller_contract="Reuse main dispatcher OOM downgrade/retry state; no per-wave reset",
        evidence_limit="File SHA and endpoint JSON verified; no tensor loading or model operation by this preparation",
        model_calls=0,rollouts=0)
    pp=out/"registrations/confirmation_sampling_preparation.json"
    jp=out/"registrations/confirmation_sampling_jobs.json"
    value=sampling_wave(out,source_hashes)
    if verify:
        if read(pp)!=prep or read(jp)!=value:raise ValueError("Frozen sampling preparation changed")
    else:
        freeze(pp,prep);freeze(jp,value)
    return dict(status="VERIFIED_READY_NOT_DISPATCHED" if verify else "PREPARED_NOT_DISPATCHED",
        jobs=str(jp),preparation=str(pp),registered_outputs=128,
        model_calls=0,rollouts=0,dispatcher_command=[sys.executable,"-B",str(REPO/"tools/dispatch_recipe_factorial.py"),
            "--out",str(out),"--jobs",str(jp)])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("prepare-job","verify-job"),default="prepare-job")
    p.add_argument("--out",default=evaluation.base.DEFAULT_OUT)
    p.add_argument("--authorization")
    a=p.parse_args(argv)
    auth=a.authorization or str(Path(a.out)/"registrations/confirmation_training.json")
    result=prepare(a.out,auth,a.stage=="verify-job")
    print(json.dumps(result,sort_keys=True),flush=True)
    return 2 if result["status"]=="PENDING" else 0

if __name__=="__main__":
    raise SystemExit(main())
