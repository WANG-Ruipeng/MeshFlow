"""Final A/B/C online training evidence from immutable logs; no model/scorer calls."""
from pathlib import Path
import argparse,csv,hashlib,json,math

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def numeric(value):
    return None if value=="" else float(value)

def mean(rows,key):
    v=[numeric(r[key]) for r in rows]
    if any(x is None for x in v):return None
    return sum(v)/len(v)

def summarize(run):
    run=Path(run)
    teacher=json.loads((run/"teacher_manifest.json").read_text())
    result=dict(status="COMPLETE_ONLINE_TRAINING_SUMMARY",scope="Finalized A/B/C CSV and diagnostics only",
        interpretation="Online predictions and gradients at changing weights, not evaluation of a fixed checkpoint",
        generator_forwards_this_analysis=0,teacher_forwards_this_analysis=0,checkpoint_loads_this_analysis=0,
        metric_recomputations_this_analysis=0,teacher_statistics=teacher["feature_statistics"],
        teacher_images=teacher["images"],teacher_batches=teacher["teacher_batches"],arms={},source_sha256={})
    for arm in "ABC":
        folder=run/"training"/arm
        status=json.loads((folder/"status.json").read_text())
        assert status["status"]=="COMPLETE" and status["step"]==1000
        logs=list(csv.DictReader((folder/"training_log.csv").open()))
        diagnostics=[json.loads(s) for s in (folder/"diagnostics.jsonl").read_text().splitlines()]
        assert len(logs)==len(diagnostics)==1000
        assert [int(r["step"]) for r in logs]==list(range(1,1001))
        fields=("FM","raw_align","weighted_align","total_loss","cosine_mean","prediction_norm_mean",
                "teacher_norm_mean","generator_gradient_norm","aligner_gradient_norm","aligner_actual_update_norm")
        selected=[]
        for row in logs:
            if int(row["step"]) in (1,100,500,1000):
                selected.append(dict(step=int(row["step"]),
                    **{k:numeric(row[k]) for k in (*fields,"generator_actual_update_norm")},
                    generator_clip=row["generator_clip_triggered"]=="True",
                    aligner_clip=row["aligner_clip_triggered"]=="True"))
        observations=[]
        for d in diagnostics:
            for o in d["gradient_observations"]:
                observations.append(dict(step=d["step"],**o,
                    weighted_aux_to_FM_norm=None if o["FM_norm"]==0 else o["weighted_aux_norm"]/o["FM_norm"]))
        assert [o["step"] for o in observations]==([] if arm=="A" else [1,100,1000])
        aligned=sum(int(row["aligned_samples"]) for row in logs)
        assert aligned==(0 if arm=="A" else 4000)
        initial=json.loads((folder/"start_import_audit.json").read_text())
        result["arms"][arm]=dict(updates=1000,aligned_occurrences=aligned,
            mean_all_online={k:mean(logs,k) for k in fields},
            mean_last100_online={k:mean(logs[-100:],k) for k in fields},
            selected_steps=selected,gradient_observations=observations,
            generator_clip_count=sum(r["generator_clip_triggered"]=="True" for r in logs),
            aligner_clip_count=sum(r["aligner_clip_triggered"]=="True" for r in logs),
            generator_update_measurement_steps=[r["step"] for r in selected],
            actual_head_update_min=min(float(r["aligner_actual_update_norm"]) for r in logs),
            actual_head_update_max=max(float(r["aligner_actual_update_norm"]) for r in logs),
            unused_generator_max=max(int(r["unused_generator_parameters"]) for r in logs),
            unused_head_max=max(int(r["unused_aligner_parameters"]) for r in logs),
            final_old_adamw_step=int(logs[-1]["old_adamw_step"]),
            final_head_adamw_step=numeric(logs[-1]["aligner_adamw_step"]),
            head_initial_sha256=initial["aligner_initial_sha256"])
        for name in ("training_log.csv","diagnostics.jsonl","status.json","start_import_audit.json"):
            p=folder/name;result["source_sha256"][str(p)]=digest(p)
    result["source_sha256"][str(run/"teacher_manifest.json")]=digest(run/"teacher_manifest.json")
    result["B_C_initial_head_same"]=result["arms"]["B"]["head_initial_sha256"]==result["arms"]["C"]["head_initial_sha256"]
    assert result["B_C_initial_head_same"]
    result["teacher_constant_loss"]=teacher["feature_statistics"]["constant_training_frequency_mean"]["mean_alignment_loss"]
    result["generator_FM_online_mean_differences"]={lhs+"-"+rhs:
        result["arms"][lhs]["mean_all_online"]["FM"]-result["arms"][rhs]["mean_all_online"]["FM"]
        for lhs,rhs in (("B","A"),("C","A"),("C","B"))}
    result["analysis_source_sha256"]=digest(__file__)
    a=result["arms"]
    text=["# Complete online training evidence: A/B/C","",
        "All three arms completed the preregistered 1,000 updates. This analysis reads finalized CSV/diagnostics and teacher summaries only. It does not load checkpoints, execute a model, recompute geometry scores, or inspect output galleries. Final checkpoint and call-ledger audit remains a separate stage after GPU work.",
        "",
        "Online losses are observed at different successive weights and different fixed inputs. A final-window average is not a fixed-checkpoint evaluation. Training fits, gradient connectivity and actual parameter changes cannot establish S/I/E, compatibility with C, body quality, diversity, known-side superiority in generation, or model promotion.",
        "",
        "| Online quantity | A_FM | B_FREE_ALIGN | C_KNOWN_ALIGN |",
        "|---|---:|---:|---:|"]
    def cell(v):return "NA" if v is None else f"{v:.9f}"
    for label,key,window in [
        ("Mean FM, all 1,000 updates","FM","mean_all_online"),
        ("Mean FM, updates 901-1,000","FM","mean_last100_online"),
        ("Raw alignment mean, all 4,000 eligible occurrences","raw_align","mean_all_online"),
        ("Raw alignment mean, final 400 eligible occurrences","raw_align","mean_last100_online"),
        ("Weighted alignment mean, updates 901-1,000","weighted_align","mean_last100_online"),
        ("Cosine mean, updates 901-1,000","cosine_mean","mean_last100_online"),
        ("Prediction norm before normalization, updates 901-1,000","prediction_norm_mean","mean_last100_online")]:
        vals=[a[arm][window][key] if arm!="A" or key=="FM" else None for arm in "ABC"]
        text.append("| "+label+" | "+" | ".join(cell(v) for v in vals)+" |")
    text.append("| Generator clip trigger count | "+" | ".join(str(a[x]["generator_clip_count"]) for x in "ABC")+" |")
    text.append("| Aligner clip trigger count | NA | "+str(a["B"]["aligner_clip_count"])+" | "+str(a["C"]["aligner_clip_count"])+" |")
    text+=["","A's zero alignment log columns are unused placeholders, not a perfect alignment predictor. B/C use the same four unaugmented occurrences per update, identical initial head tensors, and the same 0.1/4 per-occurrence coefficient. Their token selector is the only prescribed head difference.",
        "",
        "## Teacher constant baseline","",
        f"The same frozen normalized teacher mean, weighted by all 4,000 registered eligible occurrences, has mean raw cosine loss {result['teacher_constant_loss']:.9f}. It uses the same occurrence denominator as each full-run online mean, but the latter is produced by changing checkpoints and includes early optimization. It is not an endpoint generalization test. Final-window occurrence frequencies differ from the full-run frequencies, so do not present the full-run constant baseline as an exactly matched final-window experiment.",
        "",
        f"The 32 targets are not identical: pairwise cosine mean {teacher['feature_statistics']['pairwise_cosine']['mean']:.6f}, min {teacher['feature_statistics']['pairwise_cosine']['min']:.6f}, max {teacher['feature_statistics']['pairwise_cosine']['max']:.6f}. Their global chair semantics do not guarantee sensitivity to local offsets, interfaces or conflicts.",
        "",
        "## Sparse gradient observations","",
        "Each observation is the first eligible microbatch at the registered step and only the fixed coordinate projection plus fourth attention parameter set. Auxiliary norms already include 0.1/4; FM uses the whole eight-example free-coordinate denominator. Ratios do not describe the full-generator or full-batch gradient balance.",
        "",
        "| Arm | Step | FM norm | Weighted auxiliary norm | Auxiliary/FM norm | Cosine |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm in "BC":
        for o in a[arm]["gradient_observations"]:
            text.append(f"| {arm} | {o['step']} | {o['FM_norm']:.9g} | {o['weighted_aux_norm']:.9g} | {o['weighted_aux_to_FM_norm']:.6f} | {o['cosine']:.6f} |")
    text+=["",
        "All recorded auxiliary observations have a live path into the selected early parameters; no unused members are reported for either objective. The hook preflight separately confirmed exact generator predictions, direct zero gradients on unselected messages, and no direct auxiliary path to later blocks. Small positive or negative cosine at six sparse points cannot establish persistent cooperation or interference.",
        "",
        "## Actual parameter updates and clipping","",
        "| Arm | Step | Generator actual delta norm | Head actual delta norm | Generator grad norm before clip | Head grad norm before clip |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm in "ABC":
        for s in a[arm]["selected_steps"]:
            text.append(f"| {arm} | {s['step']} | {s['generator_actual_update_norm']:.9g} | "+("NA" if arm=="A" else f"{s['aligner_actual_update_norm']:.9g}")+f" | {s['generator_gradient_norm']:.9g} | "+("NA" if arm=="A" else f"{s['aligner_gradient_norm']:.9g}")+" |")
    text+=["",
        "Generator update norms are measured only at the four preregistered steps; blank values elsewhere are intentional. Head update norms are measured every update. The clips are separate, so head gradient magnitude is not included in the generator clipping norm. All three logs report generator AdamW 6000 at the endpoint; B/C head AdamW is 1000. These are runtime records, to be independently confirmed from saved state without a model forward.",
        "",
        "Prediction norms are pre-normalization magnitudes and may grow even though the cosine criterion is normalized. That growth alone is not improved representation or a change in the fixed coefficient. The available evidence supports successful training-target fitting and real early-generator supervision, while generated control success remains to be tested by the registered old/new-noise S/I/E and complete actual visual review.",
        "",
        "Full numerical details and immutable input hashes: training_interpretation_complete.json. Original partial note and all original training logs remain intact.",""]
    dest=run/"audit"
    (dest/"training_interpretation_complete.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (dest/"training_interpretation_complete.md").write_text("\n".join(text),encoding="utf-8")
    print(json.dumps(dict(status=result["status"],path=str(dest/"training_interpretation_complete.md"),
        sha256=digest(dest/"training_interpretation_complete.md"),
        summary={arm:dict(meanFM=a[arm]["mean_all_online"]["FM"],last100FM=a[arm]["mean_last100_online"]["FM"],
          meanAlign=a[arm]["mean_all_online"]["raw_align"],last100Align=a[arm]["mean_last100_online"]["raw_align"],
          last100Cos=a[arm]["mean_last100_online"]["cosine_mean"],
          genClip=a[arm]["generator_clip_count"],headClip=a[arm]["aligner_clip_count"]) for arm in "ABC"})))
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    summarize(parser.parse_args().run)
