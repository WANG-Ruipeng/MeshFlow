"""Execute exactly the gated Phase 2 budget, then save all fixed-noise rollouts."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse,copy,csv,json,time,traceback,resource
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from .official_wrapper import load_official,OfficialWrapper,state_sha256
from .adapters import ControlAdapter
from .prompt_dataset import file_sha256,collate_prompts,collate_samples,RealChairStream,PROMPT_TYPES
from .train_adapter import train_phase2_arm,velocity_loss_sum
from .sample import euler_sample,save_raw
from .evaluate import guidance_metrics,raw_quality,completion_rms

def run(checkpoint,output):
    root=Path(__file__).resolve().parents[2];out=Path(output)
    dest=out/"phase2_execution.json"
    if dest.exists():raise FileExistsError(dest)
    pre=json.loads((out/"precheck_execution.json").read_text())
    units=json.loads((root/"experiments/mf_h1_local0/tests/results/cpu_units_attempt02.json").read_text())
    diagnosis=json.loads((out/"sampler_diagnosis.json").read_text())
    integer=json.loads((out/"sampler_integer_clock_reference.json").read_text())
    prior_sampling=json.loads((out/"sampling_precheck.json").read_text())
    standalone_ok=units.get("status")=="PASS"
    # Some test reporters use summary.status; never infer success from count alone.
    if not standalone_ok:standalone_ok=units.get("summary",{}).get("status")=="PASS"
    if pre["status"]!="PASS" or not standalone_ok or diagnosis["status"]!="PASS" or integer["status"]!="PASS" or prior_sampling["off_identity"]["status"]!="PASS":
        raise RuntimeError("A required precheck has not passed")
    manifest=json.loads((out/"split_manifest.json").read_text())
    pm=json.loads((out/"prompt_manifest.json").read_text())
    report=dict(status="RUNNING",phase3="NOT_AUTHORIZED",optimizer_updates=0,
                actual_backbone_forwards=0,actual_backbone_examples=0,complete_rollouts=0,
                complete_sampler_calls=0,arms={},sampling=[],numerical_reference_limitation=prior_sampling["reference_reconciliation"])
    current_stage=["load"];counts_by_stage={}
    start=time.perf_counter();wrapper=None
    def count(m,a):
        report["actual_backbone_forwards"]+=1;report["actual_backbone_examples"]+=len(a[0])
        st=counts_by_stage.setdefault(current_stage[0],dict(forwards=0,examples=0))
        st["forwards"]+=1;st["examples"]+=len(a[0])
    def write():
        report["counts_by_stage"]=counts_by_stage
        report["elapsed_seconds"]=time.perf_counter()-start
        report["cpu_peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    try:
        torch.set_num_threads(8)
        if file_sha256(checkpoint)!=pre["checkpoint"]["sha256"]:raise ValueError("Checkpoint changed")
        base,_=load_official(checkpoint,root/"configs/snet/base-120m-ot-v-chair.yaml","cuda")
        base.register_forward_pre_hook(count)
        if state_sha256(base)!=pre["base_state_sha256"]:raise ValueError("Backbone state changed")
        # One shared PyTorch SDPA math backend and BF16 autocast for all arms.
        precision="bf16";microbatch=2
        if not torch.cuda.is_bf16_supported():raise RuntimeError("BF16 not supported")
        audit=json.loads((root/"experiments/mf_h1_local0/outputs/asset_audit_20260928_02/environment.json").read_text())
        if audit["bf16_probe"]["status"]!="PASS":raise RuntimeError("BF16 probe not passed")
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(10)
        initial=copy.deepcopy(ControlAdapter("Q_center").state_dict())
        torch.save(initial,out/"initial_adapter_seed10.pt")
        config=dict(precision=precision,microbatch=microbatch,effective_batch=8,seed=10,
                    optimizer_updates_per_arm=200,total_updates=400,backend="PyTorch SDPA math",
                    query_chunk=256,scale=1,face_cfg=1,geometry_cfg=1,evaluation_raw_faces=512,
                    euler_updates=50,phase3_authorized=False)
        (out/"phase2_resolved_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
        gates=dict(strict_load="PASS",standalone="PASS",official="PASS",sampler="PASS")
        with sdpa_kernel(SDPBackend.MATH):
            # Actual full-N, microbatch=2 BF16 backward resource probe, no update.
            current_stage[0]="bf16_full_mesh_resource_probe"
            probe=ControlAdapter("Q_center").cuda();probe.load_state_dict(initial)
            wrapper=OfficialWrapper(base,probe).train()
            data=RealChairStream(manifest["train"],10).next_effective_batch(8)
            full_batch=collate_samples(data[:microbatch],"cuda")
            torch.cuda.reset_peak_memory_stats()
            loss=velocity_loss_sum(wrapper,full_batch,precision)/(int(full_batch["masks"].sum())*9)
            loss.backward();torch.cuda.synchronize()
            if not torch.isfinite(loss) or any(p.grad is not None for p in base.parameters()):raise RuntimeError("BF16 full-mesh backward failed")
            report["bf16_full_mesh_resource_probe"]=dict(status="PASS",loss=float(loss.detach()),faces=full_batch["num_faces"].tolist(),
                peak_cuda_allocated=torch.cuda.max_memory_allocated(),optimizer_updates=0,counts=dict(wrapper.counts))
            wrapper.close();wrapper=None;del probe,loss,full_batch;torch.cuda.empty_cache();write()
            q_report=None
            for arm in ("Q_center","G_closest"):
                current_stage[0]=arm+"/train_and_teacher"
                wrapper,result=train_phase2_arm(base,arm,manifest["train"],initial,out/arm,
                    pre["checkpoint"]["sha256"],file_sha256(out/"split_manifest.json"),file_sha256(out/"prompt_manifest.json"),
                    microbatch=microbatch,precision=precision,device="cuda",
                    reference_input_hashes=q_report["input_hashes"] if q_report else None,
                    reference_tensor_hashes=q_report["tensor_hashes"] if q_report else None,gates=gates)
                report["arms"][arm]={k:v for k,v in result.items() if k not in ("input_hashes","tensor_hashes")}
                report["optimizer_updates"]+=result["updates"]
                if arm=="Q_center":q_report=result
                else:
                    report["paired_input_hashes_equal"]=result["input_hashes"]==q_report["input_hashes"]
                    report["paired_tensor_hashes_equal"]=result["tensor_hashes"]==q_report["tensor_hashes"]
                wrapper.close();wrapper=None
                write()
            if report["optimizer_updates"]!=400:raise RuntimeError("Phase 2 update count mismatch")
            # Cache two base generations (same seed reused across the four parent
            # prompts, never count these as 24 independent base samples).
            current_stage[0]="base_bf16_generation"
            y=torch.tensor([512],device="cuda");mask=torch.ones((1,512),dtype=torch.bool,device="cuda")
            base_outputs={}
            for seed in (4101,4102):
                z=torch.randn((1,512,9),generator=torch.Generator().manual_seed(seed)).cuda()
                generated,stats=euler_sample(base,z,y,mask,precision=precision)
                base_outputs[seed]=generated[0].cpu().numpy()
                save_raw(generated[0],out/"raw"/f"base_bf16_seed{seed}",z[0],seed=seed)
                report["sampling"].append(dict(arm="base",seed=seed,**stats))
                report["complete_rollouts"]+=1;report["complete_sampler_calls"]+=1;write()
            adapters={}
            for arm in ("Q_center","G_closest"):
                adapter=ControlAdapter(arm).cuda()
                saved=torch.load(out/arm/"checkpoints/adapter_step0200.pt",map_location="cpu",weights_only=True)
                adapter.load_state_dict(saved["adapter"],strict=True);adapters[arm]=adapter.eval()
            wrapper=OfficialWrapper(base,adapters["Q_center"]).eval()
            jobs=[(arm,i,kind,seed,mode) for arm in ("Q_center","G_closest") for i in range(4)
                  for kind in PROMPT_TYPES for seed in (4101,4102) for mode in ("correct","cyclic_mismatch")]
            order=np.random.RandomState(20260928).permutation(len(jobs))
            records={(r["parent_index"],r["kind"]):r for r in pm["records"]}
            rows=[]
            for job_index in order:
                arm,i,kind,seed,mode=jobs[job_index]
                current_stage[0]=arm+"/generation"
                original=records[(i,kind)];input_index=i if mode=="correct" else (i+1)%4
                actual=records[(input_index,kind)]
                orig_arrays=dict(np.load(original["path"]));arrays=dict(np.load(actual["path"]))
                prompt={k:arrays[k] for k in ("vertices","arity","primitive_mask","points","point_primitive","point_mask")}
                batched=collate_prompts([prompt],"cuda")
                wrapper.adapter=adapters[arm]
                z=torch.randn((1,512,9),generator=torch.Generator().manual_seed(seed)).cuda()
                generated,stats=euler_sample(wrapper,z,y,mask,batched,precision=precision)
                raw=generated[0].cpu().numpy()
                name=f"{arm}_parent{i}_{kind}_seed{seed}_{mode}"
                save_raw(raw,out/"raw"/name,z[0],seed=seed,parent_uid=original["parent_uid"],condition_parent_uid=actual["parent_uid"],
                    prompt_vertices=arrays["vertices"],prompt_arity=arrays["arity"],prompt_points=arrays["points"],
                    point_primitive=arrays["point_primitive"],check_points=arrays["check_points"])
                if stats["call_delta"]["wrapper_forward"]!=50 or stats["call_delta"]["backbone_forward"]!=50 or stats["call_delta"]["adapter_layer"]!=200:
                    raise RuntimeError("Sampler bypassed adapter or added backbone calls")
                quality=raw_quality(raw)
                guide_actual=guidance_metrics(raw,arrays["check_points"],actual["bbox_diagonal"])
                guide_original=guidance_metrics(raw,orig_arrays["check_points"],original["bbox_diagonal"])
                row=dict(name=name,arm=arm,parent_index=i,parent_uid=original["parent_uid"],kind=kind,noise_seed=seed,
                    condition_mode=mode,condition_parent_index=input_index,**quality,
                    guide_actual_rms_bbox_pct=guide_actual["guide_rms_bbox_pct"],
                    guide_original_rms_bbox_pct=guide_original["guide_rms_bbox_pct"],
                    completion_symmetric_rms_bbox_pct=completion_rms(raw,orig_arrays["target"],original["bbox_diagonal"]),
                    sampling_seconds=stats["seconds"],nfe=stats["nfe"],peak_cuda_allocated=stats["peak_cuda_allocated"])
                rows.append(row)
                stats.update(arm=arm,name=name,guide_actual=guide_actual,guide_original=guide_original)
                report["sampling"].append(stats)
                report["complete_rollouts"]+=1;report["complete_sampler_calls"]+=1
                with (out/"per_output.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(row)+"\n")
                write()
                print(json.dumps(dict(stage="generated",completed=report["complete_rollouts"],name=name,guide_rms=row["guide_actual_rms_bbox_pct"])),flush=True)
            wrapper.close();wrapper=None
            # Baseline scoring reuse is explicitly metadata, not extra sampling.
            for i in range(4):
                for kind in PROMPT_TYPES:
                    record=records[(i,kind)];arrays=dict(np.load(record["path"]))
                    for seed in (4101,4102):
                        raw=base_outputs[seed];guide=guidance_metrics(raw,arrays["check_points"],record["bbox_diagonal"])
                        rows.append(dict(name=f"base_bf16_seed{seed}",arm="base",parent_index=i,parent_uid=record["parent_uid"],
                            kind=kind,noise_seed=seed,condition_mode="none",condition_parent_index=i,**raw_quality(raw),
                            guide_actual_rms_bbox_pct=guide["guide_rms_bbox_pct"],guide_original_rms_bbox_pct=guide["guide_rms_bbox_pct"],
                            completion_symmetric_rms_bbox_pct=completion_rms(raw,arrays["target"],record["bbox_diagonal"]),
                            sampling_seconds=None,nfe=50,peak_cuda_allocated=None))
            with (out/"per_output.csv").open("x",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            report["base_hash_after"]=state_sha256(base)
            report["base_hash_unchanged"]=report["base_hash_after"]==pre["base_state_sha256"]
            report["status"]="PASS" if report["base_hash_unchanged"] and report["paired_tensor_hashes_equal"] and all(s["status"]=="PASS" for s in report["sampling"]) else "FAIL"
    except BaseException as error:
        report.update(status="FAIL",error=repr(error),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        if wrapper:wrapper.close()
        write()
    print(json.dumps(dict(stage="phase2_finished",status=report["status"],updates=report["optimizer_updates"],rollouts=report["complete_rollouts"])),flush=True)
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();r=run(a.checkpoint,a.output);raise SystemExit(0 if r["status"]=="PASS" else 1)

