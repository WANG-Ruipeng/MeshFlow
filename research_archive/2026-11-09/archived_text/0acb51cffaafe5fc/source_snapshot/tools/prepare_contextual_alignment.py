"""Freeze explicit V1 protocol and input identity before GPU work."""
from pathlib import Path
import argparse,json,platform,subprocess
from meshflow_control.data.io import freeze_json,digest
from meshflow_control.data.contextual_stream import prepare_training_plan
from meshflow_control.accounting import resource_snapshot
from meshflow_control.training.context_budget import LIMITS
from meshflow_control.artifacts import atomic_json
p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--data",required=True);p.add_argument("--start",required=True)
a=p.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
start_sha=digest(a.start)
if start_sha!="d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80":raise ValueError("START hash differs")
plan=prepare_training_plan(a.data,root/"training_stream")
protocol=dict(task="CHAIR_CONTEXTUAL_ALIGNMENT_SIE_PILOT_V1",schema="contextual_alignment_protocol_v1",
 task_request_sha256=digest(root/"TASK_REQUEST.txt"),
 start=dict(path=a.start,sha256=start_sha,model_state_sha256="166c71fef888f26494fc4a0af1776b77a484efa72a6f3ce06552c604fbecc972",parameters=130401155,tensors=194,adamw_step=5000),
 data_manifest=dict(path=str(Path(a.data)/"train_manifest.json"),sha256=digest(Path(a.data)/"train_manifest.json")),
 train_plan=plan,arms={"A":"A_FM","B":"B_FREE_ALIGN","C":"C_KNOWN_ALIGN"},
 steps_per_arm=1000,primary_step=1000,save_steps=[500,1000],
 train=dict(batch=8,microbatch=1,distinct_parents=8,exact_alpha1=4,augmented=4,alpha_uniform=[.9,1.1],
            generator_lr=1e-5,head_lr=1e-4,betas=[.9,.95],weight_decay=0,generator_clip=1,head_clip=1,
            auxiliary_weight=.1,auxiliary_batch_denominator=4,FM_denominator="all valid free scalar coordinates over effective batch",
            capture="backbone.layers[3].attn forward output, inside projection included, before external gate/residual",
            head_init_seed=17006844,grad_observation_steps=[1,100,1000],parameter_update_steps=[1,100,500,1000],
            dtype="BF16 generator / FP32 head, geometry, parameters, integration",generator_mode="eval with autograd"),
 teacher=dict(model="google/siglip-base-patch16-256",feature="official vision pooler_output",dimensions=768,
              views=[45,135,225,315],elevation=20,image_size=256,augmented_alignment=False,
              target="full clean actual training target geometry; alpha=1 only",constant_mean_baseline=True),
 eval=dict(labels=["START","A_FM_1000","B_FREE_ALIGN_1000","C_KNOWN_ALIGN_1000"],
           P1=dict(parents=8,noises=4,old=[0,1],new=[6,7],new_seed_namespace="CHAIR_CONTEXTUAL_ALIGNMENT_SIE_PILOT_V1/eval/{uid}/{0,1}",
                   cohorts=["OLD","NEW","ALL"],S="REFERENCE_INTERFACE",E="REFERENCE_EXPOSURE"),
           P2=dict(conditions=["seat2","seat32","back2","seat_back4"],noises=4,S=None,E=None),
           expected_outputs=192,normal_new_euler50=160,expected_verified_START_cache=32,
           postprocessing=False,confirm_read=False),
 screen=dict(reference="A_FM_1000",S_all_mean_min_pp=1,S_parent_median_strict_gt_pp=0,
             S_improved_parent_min=5,S_new_mean_strict_gt_pp=0,I_all_increase_max_pp=1,E_all_drop_max_pp=1,
             threshold_crossing="INCOMPLETE; use conservative interval differences",
             severe_parent_I_increase_or_E_drop_pp=10,leave_one_parent="diagnostic only; never remove denominator"),
 budget=dict(**LIMITS,gpu_owner_seconds=21600),
 dispatch=dict(gpu_workers=1,lock="/tmp/wrp_meshflow_jedge5_diag_gpu0.lock",headroom="max(2048 MiB,15% total VRAM)",
               cpu_OT_workers=4,cpu_score_workers=4,reason="explicit single-GPU-worker experiment contract; independent CPU tasks overlap"),
 limitations=["CoAl-inspired mesh adaptation, no null-condition weighting","single shared training stream",
              "small observed DEV panel; NEW noise is not new-parent confirmation",
              "global multiview teacher may miss local interface defects","human usability NOT_RUN"],
 scientific_changes_require_new_version=True,automatic_default_change=False,automatic_git_commit_push=False)
freeze_json(root/"protocol.json",protocol)
atomic_json(root/"audit"/"resources_initial.json",resource_snapshot())
print(json.dumps(dict(protocol_sha256=digest(root/"protocol.json"),plan=plan)),flush=True)
