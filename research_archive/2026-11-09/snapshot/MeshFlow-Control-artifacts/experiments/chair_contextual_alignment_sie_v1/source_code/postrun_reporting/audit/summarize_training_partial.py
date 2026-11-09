"""Read completed A/B logs and existing teacher summary; no model or scoring calls."""
import csv,hashlib,json
from pathlib import Path
import sys

run=Path(sys.argv[1])
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def avg(rows,key):
 values=[float(r[key]) for r in rows if r[key]!=""]
 return sum(values)/len(values) if values else None
teacher=json.loads((run/"teacher_manifest.json").read_text())
stats=teacher["feature_statistics"]
evidence={"teacher":stats,"arms":{},"scope":"A/B only; C and generation quality are not assessed","model_calls":0,"source_sha256":{}}
for arm in "AB":
 folder=run/"training"/arm
 status=json.loads((folder/"status.json").read_text())
 assert status["status"]=="COMPLETE" and status["step"]==1000
 logs=list(csv.DictReader((folder/"training_log.csv").open()))
 assert len(logs)==1000
 diagnostics=[json.loads(v) for v in (folder/"diagnostics.jsonl").read_text().splitlines()]
 assert len(diagnostics)==1000
 e={"steps":len(logs),"first":logs[0],"last":logs[-1],
  "mean_all_steps":{k:avg(logs,k) for k in ("FM","raw_align","weighted_align","cosine_mean")},
  "last100_online_mean":{k:avg(logs[-100:],k) for k in ("FM","raw_align","weighted_align","cosine_mean","prediction_norm_mean")},
  "generator_clip_count":sum(r["generator_clip_triggered"]=="True" for r in logs),
  "aligner_clip_count":sum(r["aligner_clip_triggered"]=="True" for r in logs),
  "unused_generator_max":max(int(r["unused_generator_parameters"]) for r in logs),
  "unused_head_max":max(int(r["unused_aligner_parameters"]) for r in logs),
  "head_actual_update_min":min(float(r["aligner_actual_update_norm"]) for r in logs),
  "gradient_observations":[dict(step=q["step"],**o) for q in diagnostics for o in q["gradient_observations"]]}
 evidence["arms"][arm]=e
 for filename in ("status.json","training_log.csv","diagnostics.jsonl"):
  p=folder/filename;evidence["source_sha256"][str(p)]=digest(p)
evidence["source_sha256"][str(run/"teacher_manifest.json")]=digest(run/"teacher_manifest.json")
A,B=evidence["arms"]["A"],evidence["arms"]["B"]
weighted=stats["constant_training_frequency_mean"]["mean_alignment_loss"]
lines=[
"# Partial training evidence: A_FM and B_FREE_ALIGN",
"",
"Scope: both arms completed 1,000 updates. This note reads finalized training logs and the frozen teacher summary only. No model forward, teacher encoding, scoring, checkpoint loading or parameter update was performed. C_KNOWN_ALIGN and all output-quality conclusions remain outside this partial note.",
"",
"| Quantity | A_FM | B_FREE_ALIGN |",
"|---|---:|---:|",
f"| Mean FM across all 1,000 online updates | {A['mean_all_steps']['FM']:.9f} | {B['mean_all_steps']['FM']:.9f} |",
f"| Mean FM in online steps 901-1,000 | {A['last100_online_mean']['FM']:.9f} | {B['last100_online_mean']['FM']:.9f} |",
f"| Raw alignment loss at step 1 | NA | {float(B['first']['raw_align']):.9f} |",
f"| Raw alignment loss at step 1,000 (four samples) | NA | {float(B['last']['raw_align']):.9f} |",
f"| Raw alignment loss, online steps 901-1,000 | NA | {B['last100_online_mean']['raw_align']:.9f} |",
f"| Teacher cosine, online steps 901-1,000 | NA | {B['last100_online_mean']['cosine_mean']:.9f} |",
f"| Generator global-clip trigger count | {A['generator_clip_count']} | {B['generator_clip_count']} |",
f"| Auxiliary-head clip trigger count | NA | {B['aligner_clip_count']} |",
"",
"The FM windows are paired by the same frozen inputs. Their small difference is a training-fit observation, not a generation-quality or causal estimate. The step-1 FM values are identical before either arm's first update. A's alignment columns are unused placeholders and must not be interpreted as a zero-loss predictor.",
"",
"## Teacher comparison",
"",
f"The frozen teacher has {stats['unique_targets']} distinct complete-mesh targets from {teacher['images']} images/{teacher['teacher_batches']} batch forwards. Target pairwise cosine mean/median/min/max are {stats['pairwise_cosine']['mean']:.6f} / {stats['pairwise_cosine']['median']:.6f} / {stats['pairwise_cosine']['min']:.6f} / {stats['pairwise_cosine']['max']:.6f}. The targets are not numerically identical, but they are all chair-level global features.",
"",
f"A single normalized training-frequency-weighted teacher mean has frozen mean loss {weighted:.9f} across the actual 4,000 alpha=1 occurrences. B's mean online alignment loss across those same 4,000 occurrences is {B['mean_all_steps']['raw_align']:.9f}. This matches the occurrence denominator but averages predictions produced by changing checkpoints, so it is not a fixed-endpoint evaluation. The lower final-window loss ({B['last100_online_mean']['raw_align']:.9f}) is also online and uses a different 400-occurrence subset; do not treat that comparison as an exact matched-window constant-baseline experiment.",
"",
f"B's prediction norm before L2 normalization grows from {float(B['first']['prediction_norm_mean']):.3f} at step 1 to {float(B['last']['prediction_norm_mean']):.3f} at step 1,000; teacher norms remain approximately one. Cosine loss normalizes both vectors, so a larger raw prediction norm is not itself improved alignment or a changed 0.1 coefficient.",
"",
"## Sparse gradient observations",
"",
"These are the first unaugmented microbatch at each registered step, restricted to the coordinate input projection plus fourth-layer attention parameters. FM uses the whole eight-sample free-coordinate denominator; the auxiliary gradient already includes 0.1/4. They are not whole-generator or whole-batch gradient ratios.",
"",
"| Step | FM early-set norm | Weighted auxiliary norm | Auxiliary/FM norm | Gradient cosine |",
"|---:|---:|---:|---:|---:|"]
for o in B["gradient_observations"]:
 lines.append(f"| {o['step']} | {o['FM_norm']:.9g} | {o['weighted_aux_norm']:.9g} | {o['weighted_aux_norm']/o['FM_norm']:.6f} | {o['cosine']:.6f} |")
lines += ["",
f"The auxiliary early-set norm is nonzero at all three observations; both objectives report no unused member of that selected set. B's head has a nonzero actual parameter change at every logged update (minimum norm {B['head_actual_update_min']:.9g}); neither arm reports unused generator parameters. Together with the real preflight, this shows genuine auxiliary gradient connectivity beyond fitting only the head. The three near-zero signed gradient cosines do not establish either beneficial cooperation or harmful conflict over the full training trajectory.",
"",
"## Interpretation boundary",
"",
"Current evidence supports successful fitting of the specified training representation target and a live gradient path into the intended early generator. It does not establish S/I/E improvement, compatible use of C, preservation of a coherent chair body, diversity, superiority of known-side supervision, or a reason to replace START. Those require the registered endpoint generation, old/new-noise parent-paired metrics and actual complete visual review. The teacher has no local seam guarantee; training-target memorization or global semantics remain possible explanations.",
"",
"Machine-readable numbers and source hashes: training_interpretation_partial.json. All original logs remain unchanged.",
""]
out=run/"audit"
(out/"training_interpretation_partial.md").write_text("\n".join(lines),encoding="utf-8")
(out/"training_interpretation_partial.json").write_text(json.dumps(evidence,indent=2)+"\n")
print(json.dumps({"note":str(out/"training_interpretation_partial.md"),"sha256":digest(out/"training_interpretation_partial.md"),
 "A_mean_FM":A["mean_all_steps"]["FM"],"B_mean_FM":B["mean_all_steps"]["FM"],"B_online_all_align":B["mean_all_steps"]["raw_align"],
 "teacher_constant_loss":weighted}))
