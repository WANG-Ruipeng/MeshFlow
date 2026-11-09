from pathlib import Path
import json
from meshflow_control.artifacts import atomic_json,file_sha256
ROOT=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1")
REPO=Path("/mnt/e/MeshFlow-Control")
paths=[ROOT/"protocol.json",ROOT/"teacher_manifest.json",ROOT/"teacher/render_manifest.json",
       ROOT/"teacher/encoding_plan.json",ROOT/"training_stream/train_plan.json",
       ROOT/"training_stream/paired_inputs.json",ROOT/"data/evaluation/requests.json",
       ROOT/"data/evaluation/start_cached_outputs.json",ROOT/"preflight/preflight.json",
       ROOT/"audit/code_frozen_before_training.json"]
baseline=dict(schema="contextual_alignment_runtime_binding_v1",files={str(p):file_sha256(p) for p in paths},
             note="Registered before output generation; immutable protocol/data/teacher/code identities for final audit.",
             generator_calls=0,optimizer_updates=0)
p=ROOT/"audit/execution_binding.json"
if p.exists():
    assert json.loads(p.read_text())==baseline
else:atomic_json(p,baseline)
print(json.dumps(dict(status="PASS",identities=len(paths))))
