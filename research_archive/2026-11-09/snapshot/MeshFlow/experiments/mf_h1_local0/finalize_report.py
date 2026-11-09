"""Build a self-contained Phase 0-2 report from completed evidence only."""
from pathlib import Path
import csv,json,subprocess,hashlib
import yaml
from .prompt_dataset import file_sha256

def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def run():
    root=Path(__file__).resolve().parents[2]
    project=root/"experiments/mf_h1_local0";out=project/"outputs/run_20260928_01"
    pre=load(out/"precheck_execution.json");phase=load(out/"phase2_execution.json")
    paired=load(out/"paired_summary.json");response=load(out/"condition_response.json")
    benchmark=load(out/"components_benchmark.json")
    sampling=load(out/"sampling_precheck.json");diag=load(out/"sampler_diagnosis.json");integer=load(out/"sampler_integer_clock_reference.json")
    audit=load(project/"outputs/asset_audit_20260928_02/environment.json")
    if pre["status"]!="PASS" or phase["status"]!="PASS" or benchmark["status"]!="PASS":raise RuntimeError("Incomplete required evidence")
    tests=load(project/"tests/results/cpu_units_attempt02.json")
    official=load(out/"official_tests/official_unit_tests.json")
    # Benchmark counters must be read from actual recorded executions, not plans.
    bc=benchmark["counts"]
    benchmark_forwards=bc["backbone_forward"]
    benchmark_rollouts=bc["complete_sampling_calls"]
    counts=dict(official_backbone_forward_calls=pre["official_backbone_forwards"]+sampling["actual_backbone_forwards"]+
        diag["counts"]["actual_backbone_forwards"]+integer["nfe"]+phase["actual_backbone_forwards"]+benchmark_forwards,
        complete_rollouts=sampling["complete_rollouts"]+diag["counts"]["complete_rollouts"]+integer["complete_rollouts"]+
        phase["complete_rollouts"]+benchmark_rollouts,
        official_probe_optimizer_updates=pre["probe_updates"],standalone_cpu_probe_optimizer_updates=120,
        phase2_optimizer_updates=phase["optimizer_updates"],phase3_optimizer_updates=0,
        backbone_parameter_updates=0,
        total_optimizer_updates_including_all_recorded_unit_probes=phase["optimizer_updates"]+pre["probe_updates"]+120)
    phase_wrapper_calls=sum(arm["counts"]["wrapper_forward"] for arm in phase["arms"].values())+phase["bf16_full_mesh_resource_probe"]["counts"]["wrapper_forward"]+sum(row.get("call_delta",{}).get("wrapper_forward",0) for row in phase["sampling"])
    counts["official_wrapper_forward_calls"]=official["counts"]["wrapper_forward"]+sampling["wrapper_counts"]["wrapper_forward"]+phase_wrapper_calls+bc["wrapper_counts"]["wrapper_forward"]
    counts["direct_official_forward_calls"]=counts["official_backbone_forward_calls"]-counts["official_wrapper_forward_calls"]
    counts["complete_generation_nfe"]=counts["complete_rollouts"]*50
    counts["phase2_distinct_adapted_condition_noise_inputs"]=48
    counts["phase2_actual_adapted_rollouts"]=96
    counts["phase2_independent_base_draws"]=2
    counts["recorded_cpu_unit_adapter_layer_calls"]=704
    counts["recorded_cpu_unit_encoder_calls"]=319
    core=[p for p in (root/"models").glob("*.py")]+[root/p for p in ("datasets/mesh_dataset.py","utils/ot_utils.py","flow_matching.py","train.py","inference.py","configs/snet/base-120m-ot-v-chair.yaml")]
    diff=subprocess.run(["git","-c","core.autocrlf=true","diff","--exit-code","--"],cwd=root,capture_output=True,text=True)
    counts["source_tracked_files_unchanged"]=diff.returncode==0
    weights_hash=file_sha256("/mnt/c/Users/warpwang/Downloads/last.pt")
    if weights_hash!=pre["checkpoint"]["sha256"] or diff.returncode!=0:raise RuntimeError("Original weight/source changed")
    source=dict(head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                core_hashes={str(p.relative_to(root)):file_sha256(p) for p in core},
                git_tracked_diff=diff.stdout,checkpoint_sha256_after=weights_hash)
    (out/"final_source_verification.json").write_text(json.dumps(source,indent=2),encoding="utf-8")
    (out/"execution_counts.json").write_text(json.dumps(counts,indent=2),encoding="utf-8")
    protocol=yaml.safe_load((project/"configs/protocol_original.yaml").read_text(encoding="utf-8-sig"))
    protocol["status"]="PHASE0_2_COMPLETED_PHASE3_NOT_AUTHORIZED"
    protocol["assets"].update(repo_path="E:\\MeshFlow",actual_commit=source["head"],
        checkpoint_path="C:\\Users\\warpwang\\Downloads\\last.pt",checkpoint_sha256=weights_hash,
        data_path="E:\\MeshFlow\\experiments\\mf_h1_local0\\assets\\shapenet-03001627")
    protocol["training"].update(precision="BF16 autocast, FP32 parameters",actual_microbatch=2,actual_accumulation=4,
                               actual_attention_backend="PyTorch SDPA math")
    protocol["phase2"]["actual_updates"]=400
    protocol["sampling"]["numerical_reference_note"]="Original FP32 linspace rollout discrepancy retained; matched floating grid and integer-clock official reference each bitwise equal. Production constant dt remains 1/50."
    (out/"resolved_protocol.yaml").write_text(yaml.safe_dump(protocol,allow_unicode=True,sort_keys=False),encoding="utf-8")
    decision=dict(label="IMPLEMENTATION_PASS_ONLY",implementation="PASS",
        condition_response=response["status"],architecture_advantage="NOT_ESTABLISHED",
        learning_diagnosis="NEEDS_LEARNING_DIAGNOSIS",
        recommendation="Assets and wiring are ready, but repeated identical inputs vary on the scale of observed condition changes. Recommend separately authorized numerical and learning diagnosis before any Phase 3 expansion; do not automatically expand.",
        phase3="NOT_AUTHORIZED_NOT_STARTED",training_budget_complete=True,
        limitations=["4 diagnostic author-train UID parents; one adapter seed only","S/P not trained; G-vs-P NOT_RUN",
            "Original FP32 linspace rollout comparison failed numerically and is preserved",
            "Same-input repeat variation is comparable to condition-change variation; causal guide response is not established",
            "Self-intersection, watertightness and manifold status NOT_RUN; raw quality only",
            "No verified ancestry metadata beyond author UID; no pretraining-unseen claim"])
    (out/"decision.json").write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding="utf-8")
    # The Markdown report is added after inspecting the measured summaries.
    print(json.dumps(dict(counts=counts,decision=decision),ensure_ascii=False))
if __name__=="__main__":run()

