"""Archive completed logs and reuse the one independent nine-weight hash pass.

Run only after GPU work stops and audit_contextual_training.py reports PASS.
This module is stdlib-only and never loads a checkpoint or imports a model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

LOG_FILES=("training_log.csv","exposure.csv","diagnostics.jsonl","events.jsonl",
           "start_import_audit.json","status.json","latest.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):h.update(part)
    return h.hexdigest()


def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise RuntimeError("Existing artifact differs: "+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+"\n")


def build(run):
    run=Path(run).resolve()
    occupancy=run/"audit/budget/gpu_occupancy.jsonl"
    records=[json.loads(x) for x in occupancy.read_text().splitlines()]
    started={r["id"] for r in records if r["event"]=="START"}
    ended={r["id"] for r in records if r["event"]=="END"}
    if started-ended:raise RuntimeError("GPU work still active or unresolved; no final archive")
    audit_path=run/"audit/training_endpoint_independent.json"
    audit=read(audit_path)
    if audit["status"]!="PASS":raise RuntimeError("Final independent endpoint audit must PASS first")
    if digest(audit["audit_source"])!=audit["audit_source_sha256"]:
        raise RuntimeError("Audit source changed after final receipt")
    weights=[]
    for arm in "ABC":
        folder=run/"training"/arm
        status=read(folder/"status.json")
        if status["status"]!="COMPLETE" or status["step"]!=1000:raise RuntimeError("Arm not final: "+arm)
        checked=audit["checks"]["checkpoints"][arm]
        if checked["status"]!="PASS":raise RuntimeError("Endpoint metadata failed: "+arm)
        if [r["step"] for r in checked["training"]]!=[500,1000]:raise RuntimeError("Fixed snapshots missing")
        for record in checked["training"]:
            weights.append(dict(record,arm=arm,role="resumable_training",
                 schema="meshflow_contextual_alignment_training_v1",
                 includes_generator=True,includes_aligner=arm!="A",includes_optimizer=True))
        weights.append(dict(checked["export"],arm=arm,step=1000,role="inference_generator",
                 includes_generator=True,includes_aligner=False,includes_optimizer=False))
    if len(weights)!=9:raise RuntimeError("Expected six training and three generator files")
    for record in weights:
        p=Path(record["path"]);stat=p.stat()
        if stat.st_size!=record["bytes"] or stat.st_mtime_ns!=record["mtime_ns"]:
            raise RuntimeError("Weight changed after independent SHA verification: "+str(p))
        record["hash_verification"]="SHA256 computed by completed independent endpoint audit; unchanged size/mtime checked during manifest assembly"
    log_records=[]
    for arm in "ABC":
        for name in LOG_FILES:
            source=run/"training"/arm/name
            before=digest(source)
            destination=run/"training_logs"/arm/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                if digest(destination)!=before:raise RuntimeError("Existing log copy differs")
            else:
                shutil.copy2(source,destination)
            if digest(source)!=before or digest(destination)!=before:
                raise RuntimeError("Finalized log changed during copy")
            log_records.append(dict(arm=arm,source=str(source),path=str(destination),
                                    sha256=before,bytes=destination.stat().st_size))
    shared=dict(independent_audit_path=str(audit_path),independent_audit_sha256=digest(audit_path),
        assembly_source=str(Path(__file__).resolve()),assembly_source_sha256=digest(__file__),
        actual_model_calls=0,actual_checkpoint_loads_this_script=0,actual_weight_hash_passes_this_script=0,
        checkpoint_hashes_reused_from_independent_audit=9,
        explanation="The independent audit already performed the only full-file SHA pass and CPU metadata/state checks for each of six training and three export weights. This assembly reuses those hashes after checking recorded immutable file size/mtime; it does not load/hash large weights twice.")
    manifest=dict(schema="contextual_alignment_checkpoint_manifest_v1",status="PASS",
        source_start=read(run/"protocol.json")["start"],checkpoints=weights,
        counts=dict(resumable_training=6,inference_generator=3),
        default_model_changed=False,auxiliary_and_teacher_in_inference=False,**shared)
    logs=dict(schema="contextual_alignment_final_training_logs_v1",status="PASS",arms=list("ABC"),
              files=log_records,updates_per_arm=1000,exposures_per_arm=8000,**shared)
    freeze(run/"training_logs/manifest.json",logs)
    freeze(run/"checkpoint_manifest.json",manifest)
    result=dict(status="PASS",checkpoint_manifest=str(run/"checkpoint_manifest.json"),
        checkpoint_manifest_sha256=digest(run/"checkpoint_manifest.json"),
        training_logs_manifest=str(run/"training_logs/manifest.json"),
        training_logs_manifest_sha256=digest(run/"training_logs/manifest.json"),
        weight_files=9,copied_log_files=len(log_records),model_calls=0,checkpoint_loads=0)
    freeze(run/"audit/training_archive_receipt.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="CPU archival manifest, after all GPU jobs and final endpoint audit")
    parser.add_argument("--run",required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.run)),flush=True)
