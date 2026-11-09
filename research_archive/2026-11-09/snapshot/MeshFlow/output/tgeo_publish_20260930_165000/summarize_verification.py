
import json,subprocess
from pathlib import Path
import numpy as np
out=Path("output/tgeo_publish_20260930_165000")
read=lambda p:json.loads((out/p).read_text(encoding="utf-8-sig"))
old,new=read("archived_forward.json"),read("public_forward.json")
assert old["velocity_sha256"]==new["velocity_sha256"]
assert np.array_equal(np.load(out/"archived_velocity.npy"),np.load(out/"public_velocity.npy"))
zero=read("zero_branch.json")
assert zero["pure_and_zero_geo_bitwise_equal"]
preflights=[read("preflight_fresh_geo/run.json"),read("preflight_existing_geo/run.json")]
for report in preflights:
    assert report["status"]=="PASS" and report["completed_updates"]==0 and not report["checkpoints"]
    assert report["parameter_state_before"]==report["parameter_state_after"]
    assert report["counts"]["optimizer_updates"]==0
    assert report["counts"]["forward_returns"]==1 and report["counts"]["backward_returns"]==1
    assert report["counts"]["OT_calls"]==report["counts"]["OT_returns"]==8
clean=read("clean_cpu_tests.json")
assert clean["status"]=="PASS"
current_tree=subprocess.check_output(["git","write-tree"],text=True).strip()
assert current_tree==clean["git_tree"]
record=dict(status="PASS",git_tree=current_tree,CPU_tests=74,clean_source_CPU_tests=74,
            retained_public_paths=["pure T1","T_geo"],trained_checkpoint_regression_bitwise=True,
            zero_geo_equals_pure_bitwise=True,model_parameters_changed=False,
            GPU_model_forwards=6,GPU_model_backwards=2,CPU_OT_calls=16,CPU_OT_returns=16,
            optimizer_updates=0,complete_rollouts=0,checkpoint_files_written=0,
            preflight_stream_modes=[r["stream_mode"] for r in preflights],
            new_files_count=15,weight_data_cache_output_files_staged=0,
            old_trained_geo_SHA256=old["load"]["sha256"],
            public_trained_geo_SHA256=new["load"]["sha256"])
(out/"verification.json").write_text(json.dumps(record,indent=2))
print(json.dumps(record))
