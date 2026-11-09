"""CPU-only checks of the new paired stream and frozen experiment identities."""
from pathlib import Path
import json
from meshflow_control.data.contextual_stream import HybridStream,make_plan,NAMESPACE
from meshflow_control.data.stream import make_plan as old_plan
from meshflow_control.data.io import read_json,digest
from meshflow_control.artifacts import atomic_json
ROOT=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1")
DATA=Path("/mnt/e/MeshFlow-Control-artifacts/data")
streams=[HybridStream(DATA,ROOT/"training_stream",a) for a in "ABC"]
new=streams[0].plan; old=old_plan(streams[0].data.manifest)
assert new["namespace"]!=old["namespace"]
assert len(new["records"])==8000 and len(set(r["uid"] for r in new["records"]))==32
assert all(v==250 for v in new["object_counts"].values())
assert all(len({r["uid"] for r in new["records"][i:i+8]})==8 and sum(r["alpha"]==1 for r in new["records"][i:i+8])==4 for i in range(0,8000,8))
assert not any(a["gaussian_seed"]==b["gaussian_seed"] for a,b in zip(new["records"],old["records"]))
paired=read_json(ROOT/"training_stream/paired_inputs.json")["records"]
batches=[s.next_effective_batch() for s in streams]
keys=("shared_input_sha256","input_bytehash","label_bytehash","ot_cache_key","free_coordinate_count")
for i in range(8):
    for k in keys: assert all(b[i][k]==paired[i][k] for b in batches)
s=streams[0].state_dict();clone=HybridStream(DATA,ROOT/"training_stream","A");clone.load_state_dict(s)
assert clone.batch_index==1 and clone.state_dict()==s
try:streams[1].load_state_dict(s)
except ValueError:pass
else:raise AssertionError("Cross-arm cursor should refuse")
rec=dict(status="PASS",namespace=NAMESPACE,plan_sha256=digest(ROOT/"training_stream/train_plan.json"),
    complete_plan_samples=8000,all_1000_batches_distinct8_and_exact4_alpha1=True,
    every_parent_exposures=250,new_namespace_and_no_positionwise_seed_collision=True,
    actual_matched_first_batch_A_B_C_equal=True,first_batch_matched_to_frozen_inputs=True,
    arm_specific_stream_restore=True,model_forward=0,model_backward=0,optimizer_updates=0,additional_OT=0)
atomic_json(ROOT/"audit/stream_checks.json",rec);print(json.dumps(rec))
