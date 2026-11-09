"""Fixed-budget seven-arm loss screen: unchanged offline geometry and selection.

No model, OT, training, repair, or geometry gradient is called by this module.
Full CF/FF auditing is only performed when the root explicitly calls evaluate_phase.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np

from ..tgeo_m10_evaluation import (
    load_bank, _input_audit, _path, _sha, _write, audit_new_seeds,
    score_output, audit_context, cross_binding, summarize_values, concentration,
    scalar_metrics, _counts, _summary, AREA_METRIC,
)
from ..t1_gated_factorial_evaluation import QUALITY_METRICS
from ..native_inpainting_metrics import array_sha
from ..guided_native_geometry_audit import _sha_array

MODELS=tuple("L"+str(j) for j in range(7))
DEVELOPMENT_SEEDS=tuple(range(9901,9909))
CONFIRMATION_SEEDS=tuple(range(10001,10009))
RMS="remainder_symmetric_RMS_bbox_pct"
GAMMA="boundary_RMS_bbox_pct"
PUBLIC_G0_STATE="6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392"
PROTOCOL=dict(
    method="exploratory fixed-budget direct geometry loss screen",
    initialization="User override: common G0 cumulative2500; seven fresh500-update arms end cumulative3000",
    baseline="Fresh L0 cumulative3000; prior G0 cumulative2500 is historical context only, excluded from ranking",
    aggregation="Within each parent/patch average registered noises, then equal mean of six cells",
    selection="Independent global best non-L0 RMS and Gamma slots; require strict improvement over L0; deduplicate, never fill with runner-up",
    exact_tie_break="Registered model order L1,L2,L3,L4,L5,L6",
    paired_sign="candidate minus reference; no marginal-median subtraction",
    confirmation="New noise on previously seen parents/patches, not new-object or independent-training-seed replication",
    quality="Report tradeoffs; no post-hoc composite score or additional quality selection threshold")


def audit_seeds(seeds,out):
    return audit_new_seeds(seeds,out)


def _seeds(r,phase):
    if phase not in ("development","confirmation"):raise ValueError("Unknown phase")
    protocol=r.get("protocol",{}); evaluation=protocol.get("evaluation",{})
    value=evaluation.get(phase,{}).get("seeds") if isinstance(evaluation.get(phase),dict) else None
    if value is None:value=evaluation.get(phase+"_seeds",protocol.get(phase+"_seeds"))
    if value is None:value=DEVELOPMENT_SEEDS if phase=="development" else CONFIRMATION_SEEDS
    seeds=tuple(int(s) for s in value)
    if len(seeds)!=8 or len(set(seeds))!=8:raise ValueError("Eight distinct registered seeds required")
    return seeds


def _expected(models,seeds):
    return {(m,p,j,s) for m in models for p in (0,1) for j in (0,1,2)
            for s in (seeds if j==0 else seeds[:2])}


def _identity(row):
    return row["model"],int(row["parent"]),int(row["patch"]),int(row["seed"])


def _seal(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _metric_reference_checks(metrics,bank,parent,patch):
    extra=metrics["t1_native_confirmation"]
    if metrics["K"]!=12 or metrics["total_faces"]!=112 or metrics["free_faces"]!=100 or extra["C_PRESERVED"]!="PASS":
        raise ValueError("C/K/N/free score identity differs")
    if extra["nan_coordinate_count"] or extra["inf_coordinate_count"] or not extra["all_coordinates_finite"]:
        raise ValueError("Nonfinite recorded geometry")
    for target in (0,1):
        prefix=f"parent{target}_patch{patch}_"
        if metrics["remainder"]["cross"][target]["target_sample_sha256"]!=array_sha(bank["arrays"][prefix+"pairwise_target_points"]):
            raise ValueError("Pairwise target512 differs from frozen reference")
        if metrics["common_unknown_R12"]["cross"][target]["target_point_sha256"]!=array_sha(bank["arrays"][prefix+"coverage_points"]):
            raise ValueError("Coverage target512 differs from frozen reference")
    prefix=f"parent{parent}_patch{patch}_"; boundary=metrics["guided_geometry_audit"]["boundary"]
    if boundary["point_sha256"]!=_sha_array(bank["arrays"][prefix+"boundary_points"]) or boundary["weight_sha256"]!=_sha_array(bank["arrays"][prefix+"boundary_weights"]):
        raise ValueError("Gamma points or weights differ")
    for scope,count in (("CF",1200),("FF",4950),("CC",66)):
        if extra["detector_views"][scope]["verified"]["tested_pairs"]!=count:
            raise ValueError("Independent pair denominator differs: "+scope)


def _raw(row,bank,parent):
    path=_path(row["raw_path"])
    if _sha(path)!=row["raw_sha256"]:raise ValueError("Raw file identity differs")
    with np.load(path,allow_pickle=False) as z:
        full=z["output"].copy();states=z["path"].copy()
    if full.dtype!=np.float32 or full.size!=112*9 or states.dtype!=np.float32 or states.size!=51*112*9:
        raise ValueError("Raw FP32 N112/51-state shape differs")
    full=full.reshape(112,3,3);states=states.reshape(51,112,3,3)
    C=bank["cases_by_key"][parent,12]["constraints"]
    if not np.isfinite(states).all() or states[-1].tobytes()!=full.tobytes() or any(x[:12].tobytes()!=C.tobytes() for x in states):
        raise ValueError("Nonfinite, changed C, or endpoint not final state")
    return full


def aggregate_results(rows,phase,models,seeds,*,candidate_status=None,pairings=None):
    """Complete eligible models only; pure scalar/JSON arithmetic."""
    rows=list(rows);models=tuple(models);seeds=tuple(seeds)
    if "L0" not in models or len(set(models))!=len(models):raise ValueError("Unique models and L0 required")
    index={_identity(x):x for x in rows}
    if set(index)!=_expected(models,seeds) or len(index)!=len(rows):raise ValueError("Incomplete/duplicate phase matrix")
    if any(x.get("phase")!=phase or x.get("K",12)!=12 for x in rows):raise ValueError("Phase/K mismatch")
    pairs=list(pairings) if pairings is not None else [(m,"L0") for m in models if m!="L0"]
    result=dict(schema="loss_screen_phase_summary_v1",phase=phase,models=list(models),seeds=list(seeds),
        patches={},macros={},candidate_status=candidate_status or {},protocol=PROTOCOL,geometry_calls=0,model_calls=0)
    for patch in (0,1,2):
        ss=seeds if patch==0 else seeds[:2]
        block=dict(patch=patch,seeds=list(ss),models={},paired={},bindings=[])
        for model in models:
            group=dict(parents={},binding={})
            for parent in (0,1):
                values=[dict(seed=s,**scalar_metrics(index[model,parent,patch,s]["metrics"])) for s in ss]
                names=[k for k in values[0] if k!="seed"]
                group["parents"][str(parent)]=dict(per_seed=values,metrics={k:_summary(values,k) for k in names})
            binding=[dict(seed=s,**cross_binding(index[model,0,patch,s]["metrics"],index[model,1,patch,s]["metrics"])) for s in ss]
            group["binding"]=dict(per_seed=binding,metrics={k:summarize_values(binding,k,higher_better=True) for k in ("Delta0","Delta1","B")},
                both_positive_count=sum(x["both_directions_positive"] for x in binding),concentration=concentration(binding))
            block["bindings"].extend(dict(model=model,patch=patch,seed=x["seed"],metrics=x) for x in binding)
            block["models"][model]=group
        for candidate,reference in pairs:
            pair=dict(candidate=candidate,reference=reference,parents={},binding={})
            for parent in ("0","1"):
                old=block["models"][reference]["parents"][parent]["per_seed"];new=block["models"][candidate]["parents"][parent]["per_seed"]
                values=[dict(seed=a["seed"],**{k:b[k]-a[k] if a[k] is not None and b[k] is not None else None for k in a if k!="seed"}) for a,b in zip(old,new)]
                names=[k for k in values[0] if k!="seed"]
                pair["parents"][parent]=dict(per_seed_candidate_minus_reference=values,metrics={k:_summary(values,k) for k in names},
                    improvement_counts={k:_counts(values,k) for k in names})
            old=block["models"][reference]["binding"]["per_seed"];new=block["models"][candidate]["binding"]["per_seed"]
            values=[dict(seed=a["seed"],**{k:b[k]-a[k] for k in ("Delta0","Delta1","B")}) for a,b in zip(old,new)]
            pair["binding"]=dict(per_seed_candidate_minus_reference=values,
                metrics={k:summarize_values(values,k,higher_better=True) for k in ("Delta0","Delta1","B")},
                higher_B_count=sum(x["B"]>0 for x in values),concentration=concentration(values))
            block["paired"][candidate+"_minus_"+reference]=pair
        result["patches"][str(patch)]=block
    names=list(next(iter(result["patches"]["0"]["models"].values()))["parents"]["0"]["metrics"])
    for model in models:
        macro={}
        for key in names:
            cells=[dict(parent=p,patch=j,n=len(seeds) if j==0 else 2,mean=result["patches"][str(j)]["models"][model]["parents"][str(p)]["metrics"][key]["mean"])
                for p in (0,1) for j in (0,1,2)]
            available=all(c["mean"] is not None and np.isfinite(c["mean"]) for c in cells)
            macro[key]=dict(six_cells=cells,macro_mean=float(np.mean([c["mean"] for c in cells])) if available else None,
                all_six_cells_available=available,cell_weight=1/6,output_weighting="First noise mean within cell, then six equal cells")
        result["macros"][model]=macro
    result["macro_pairs"]={}
    for candidate,reference in pairs:
        result["macro_pairs"][candidate+"_minus_"+reference]={key:dict(
            difference=result["macros"][candidate][key]["macro_mean"]-result["macros"][reference][key]["macro_mean"]
                if result["macros"][candidate][key]["macro_mean"] is not None and result["macros"][reference][key]["macro_mean"] is not None else None,
            six_cell_differences=[dict(parent=a["parent"],patch=a["patch"],difference=b["mean"]-a["mean"] if a["mean"] is not None and b["mean"] is not None else None)
                for a,b in zip(result["macros"][reference][key]["six_cells"],result["macros"][candidate][key]["six_cells"])])
            for key in names}
    json.dumps(result,allow_nan=False);return result


def _historical_reference(r,banks):
    record=r["assets"]["historical_G0_results"];path=_path(record["path"])
    if _sha(path)!=record["sha256"]:raise ValueError("Historical results identity changed")
    source=json.loads(path.read_text(encoding="utf-8-sig"))
    if source["status"]!="PASS_EXECUTION":raise ValueError("Historical G0 run incomplete")
    cp=source["arms"]["G0"]["checkpoints"]["500"]
    if cp.get("state_sha256",cp.get("generator_state_sha256"))!=PUBLIC_G0_STATE or cp["cumulative_updates"]!=2500:
        raise ValueError("Historical G0 tensor/update identity differs")
    prior=[x for x in source["scores"] if x["model"]=="G0"]
    raw={_identity(x):x for x in source["generated"] if x["model"]=="G0"}
    if len(prior)!=24 or {_identity(x) for x in prior}!=_expected(("G0",),DEVELOPMENT_SEEDS):
        raise ValueError("Historical G0 exact24 development matrix missing")
    rows=[];compact=[]
    for row in prior:
        j=int(row["patch"]);p=int(row["parent"]);_raw(raw[_identity(row)],banks[j],p)
        path=_path(row["metrics_path"])
        if _sha(path)!=row["metrics_sha256"]:raise ValueError("Historical metric file changed")
        payload=json.loads(path.read_text(encoding="utf-8"));metrics=payload["metrics"]
        if payload["model"]!="G0" or _identity(payload)!=_identity(row) or payload["raw_sha256"]!=raw[_identity(row)]["raw_sha256"]:
            raise ValueError("Historical score identity differs")
        _metric_reference_checks(metrics,banks[j],p,j)
        values=scalar_metrics(metrics)
        if values!=row["scalars"]:raise ValueError("Historical saved scalar definitions differ")
        identity=dict(phase="development",model="G0_initial",parent=p,patch=j,patch_group="original" if j==0 else "new",
            seed=int(row["seed"]),K=12,raw_path=row["raw_path"],raw_sha256=raw[_identity(row)]["raw_sha256"],reused=True,ranked=False)
        rows.append(dict(identity,metrics=metrics))
        compact.append(dict(identity,scalars=values,metrics_path=row["metrics_path"],metrics_sha256=row["metrics_sha256"]))
    return rows,compact,dict(status="PASS",source_results=record,source_model="G0",state_sha256=PUBLIC_G0_STATE,
        rows_reused=24,models_constructed=0,new_geometry_scores=0,source_checkpoint_path_required=False,
        scope="Historical cumulative2500 reference only; fresh L0 cumulative3000 is the baseline for ranking")


def _csv(path,rows):
    names=sorted({k for row in rows for k in row.get("scalars",{})})
    identity=["phase","model","parent","patch","seed","K","status","reused","ranked","raw_path","metrics_path"]
    with Path(path).open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=identity+names);writer.writeheader()
        for row in rows:writer.writerow({**{k:row.get(k) for k in identity},**row.get("scalars",{})})
    return dict(path=str(Path(path).resolve()),sha256=_sha(path),bytes=Path(path).stat().st_size)


def evaluate_phase(r,out,phase):
    """Explicit fresh scoring only after each planned model has finished its phase."""
    tick=time.perf_counter();out=_path(out);seeds=_seeds(r,phase)
    if r.get("summaries",{}).get(phase):raise RuntimeError("No automatic phase rescoring")
    models=MODELS if phase=="development" else tuple(["L0"]+r["selection"]["selected_models"])
    if phase=="confirmation" and _seal({k:v for k,v in r["selection"].items() if k not in ("selection_sha256","artifact")})!=r["selection"]["selection_sha256"]:
        raise ValueError("Frozen selection changed")
    generated=[x for x in r.get("generated",[]) if x.get("phase")==phase]
    if any(x["model"] not in models or x.get("reused",False) for x in generated):raise ValueError("Unexpected/reused main output")
    identities=[_identity(x) for x in generated]
    if len(identities)!=len(set(identities)) or not set(identities)<=_expected(models,seeds):raise ValueError("Unexpected or duplicate output identity")
    record=r["assets"]["evaluation_inputs"];banks=load_bank(record["path"],record["sha256"])
    directory=out/"audit"/phase;directory.mkdir(parents=True,exist_ok=False)
    counts=dict(model_calls=0,geometry_autograd_calls=0,optimizer_updates=0,OT_calls=0,repairs=0)
    input_records=[]
    for patch,bank in banks.items():
        bank["input_topology"]=_input_audit(bank,patch,counts)
        bank["context_audits"]={p:audit_context(bank["cases_by_key"][p,12]["constraints"],bank["cases_by_key"][p,12]["bbox_diagonal"],counts) for p in (0,1)}
        input_records.append(_write(directory/f"input_patch{patch}.json",dict(input_topology=bank["input_topology"],context_audits=bank["context_audits"],reference_audit=bank["reference_audit"])))
    full_rows=[];compact=[];statuses={};eligible=[]
    for model in models:
        arm=r.get("arms",{}).get(model,{})
        rows=[x for x in generated if x["model"]==model]
        sampling_status=r.get("sampling_phases",{}).get(phase,{}).get("models_audit",{}).get(model,{}).get("status")
        status=dict(sampling_status=sampling_status,training_status=arm.get("status","NOT_RECORDED"),completed_updates=arm.get("completed_updates"),
            output_count=len(rows),expected_outputs=24,errors=[])
        statuses[model]=status
        if sampling_status!="PASS" or arm.get("status")!="PASS" or arm.get("completed_updates")!=500 or {_identity(x) for x in rows}!=_expected((model,),seeds):
            status.update(status="INELIGIBLE_INCOMPLETE_OR_FAILED",reason="Require passing frozen-state sampling, fresh500-update endpoint, and complete registered24 outputs")
            if model=="L0":raise RuntimeError("Fresh baseline failed or incomplete")
            continue
        start_index=len(full_rows)
        for row in rows:
            started=time.perf_counter();p=int(row["parent"]);j=int(row["patch"])
            identity=dict(phase=phase,model=model,parent=p,patch=j,patch_group="original" if j==0 else "new",
                seed=int(row["seed"]),K=12,raw_path=row["raw_path"],raw_sha256=row["raw_sha256"],reused=False,ranked=True)
            try:
                raw=_raw(row,banks[j],p);metrics=score_output(raw,banks[j],p,ledger=counts)
                _metric_reference_checks(metrics,banks[j],p,j);values=scalar_metrics(metrics)
                if any(values[k] is None or not np.isfinite(values[k]) for k in (RMS,GAMMA,AREA_METRIC)):
                    raise ValueError("Nonfinite required endpoint metric")
                artifact=_write(directory/f"{model}_P{p}_patch{j}_seed{row['seed']}.json",dict(identity,status="PASS",metrics=metrics,scoring_seconds=time.perf_counter()-started))
                full_rows.append(dict(identity,metrics=metrics))
                compact.append(dict(identity,status="PASS",scalars=values,metrics_path=artifact["path"],metrics_sha256=artifact["sha256"],
                    metrics_bytes=artifact["bytes"],scoring_seconds=time.perf_counter()-started))
            except Exception as error:
                failure=dict(identity,status="FAIL_EVALUATION",error=repr(error),traceback=traceback.format_exc(),scoring_seconds=time.perf_counter()-started)
                artifact=_write(directory/f"{model}_P{p}_patch{j}_seed{row['seed']}_FAILED.json",failure)
                status["errors"].append(dict(seed=row["seed"],parent=p,patch=j,error=repr(error),artifact=artifact))
                compact.append(dict(identity,status="FAIL_EVALUATION",scalars={},metrics_path=artifact["path"],metrics_sha256=artifact["sha256"]))
            print(json.dumps(dict(event="loss_screen_offline_score",phase=phase,model=model,parent=p,patch=j,seed=row["seed"],scored_or_failed=len(compact))),flush=True)
        if status["errors"]:
            del full_rows[start_index:]
            status.update(status="INELIGIBLE_EVALUATION_FAILURE",reason="All observed rows retained; partial performance is not ranked")
            if model=="L0":raise RuntimeError("Baseline evaluation failed; cannot form screen")
        else:
            eligible.append(model);status["status"]="ELIGIBLE"
    summary=aggregate_results(full_rows,phase,eligible,seeds,candidate_status=statuses)
    if phase=="development":
        old,old_compact,audit=_historical_reference(r,banks)
        current=[x for x in full_rows if x["model"]=="L0"]
        reference=aggregate_results(current+old,"development",("L0","G0_initial"),seeds,pairings=[("L0","G0_initial")])
        r["historical_reference"]=dict(audit=audit,scores=old_compact,summary=reference,ranked=False)
    r.setdefault("scores",[]).extend(compact)
    r.setdefault("summaries",{})[phase]=summary
    csv_record=_csv(out/f"compact_per_output_{phase}.csv",compact)
    if phase=="development":_csv(out/"compact_historical_G0_initial.csv",r["historical_reference"]["scores"])
    r.setdefault("evaluation_ledger",{})[phase]=dict(status="PASS",counts=counts,seconds=time.perf_counter()-tick,
        input_audits=input_records,scored_or_failed=len(compact),eligible_models=eligible,candidate_status=statuses,
        complete_pairs_per_output=dict(CF=1200,FF=4950,CC_cached=66),immutable_context_audits=6,
        historical_reused_outputs=24 if phase=="development" else 0,historical_new_scores=0,compact_csv=csv_record)
    return r["evaluation_ledger"][phase]


def selection_from_summary(summary,checkpoints=None):
    """Prespecified exact ranking; quality remains visible, never a hidden gate."""
    if summary["phase"]!="development" or "L0" not in summary["models"]:raise ValueError("Complete development baseline required")
    candidates=[m for m in MODELS[1:] if m in summary["models"] and summary["candidate_status"].get(m,{}).get("status")=="ELIGIBLE"]
    slots={};selected=[]
    for slot,metric in (("remaining_RMS",RMS),("Gamma_RMS",GAMMA)):
        baseline=summary["macros"]["L0"][metric]["macro_mean"]
        ranked=sorted(candidates,key=lambda m:(summary["macros"][m][metric]["macro_mean"],MODELS.index(m)))
        ranking=[dict(model=m,macro_mean=summary["macros"][m][metric]["macro_mean"],
            minus_L0=summary["macros"][m][metric]["macro_mean"]-baseline,
            six_cells=summary["macros"][m][metric]["six_cells"]) for m in ranked]
        winner=ranked[0] if ranked and summary["macros"][ranked[0]][metric]["macro_mean"]<baseline else None
        slots[slot]=dict(metric=metric,L0_macro_mean=baseline,ranking=ranking,selected=winner,
            reason="STRICT_IMPROVEMENT_OVER_L0" if winner else "EMPTY_NO_STRICT_IMPROVEMENT")
        if winner and winner not in selected:selected.append(winner)
    return dict(schema="loss_screen_selection_v1",status="FROZEN",slots=slots,selected_models=selected,
        confirmation_models=["L0"]+selected,maximum_candidates=2,no_runner_up_backfill=True,
        exact_tie_break=PROTOCOL["exact_tie_break"],candidate_status=summary["candidate_status"],
        development_macros=summary["macros"],checkpoint_identities=checkpoints or {},
        selection_used_confirmation=False,quality_filter="Only numerical/C/N validity gate; all coverage/intersection/angle tradeoffs remain reported")


def select_candidates(r,out):
    if r.get("selection"):raise RuntimeError("Selection is one-shot")
    identities={m:r["arms"][m]["checkpoints"]["500"] for m in r["summaries"]["development"]["models"]}
    value=selection_from_summary(r["summaries"]["development"],identities)
    value["selection_sha256"]=_seal(value)
    value["artifact"]=_write(_path(out)/"selection.json",value)
    r["selection"]=value;return value


def run_cpu_tests():
    """One bounded synthetic scalar fixture suite; never a real geometry audit."""
    seeds=DEVELOPMENT_SEEDS;rows=[]
    for model in ("L0","L1","L2"):
        for p in (0,1):
            for j in (0,1,2):
                for seed in (seeds if j==0 else seeds[:2]):
                    own={"L0":10.,"L1":9.,"L2":11.}[model]+j
                    gamma={"L0":10.,"L1":8.,"L2":9.}[model]+p
                    cross=[dict(symmetric_rms_bbox_pct=own+(t!=p),predicted_to_target_rms_bbox_pct=own,target_to_predicted_rms_bbox_pct=own) for t in (0,1)]
                    common=[dict(rms_bbox_pct=own+(t!=p),fraction_target_points_within_1pct_bbox=.5) for t in (0,1)]
                    metrics=dict(K=12,variant="raw",parent_index=p,remainder=dict(own=cross[p],cross=cross),
                        whole=dict(surface=cross[p]),common_unknown_R12=dict(own=common[p],cross=common),
                        guided_geometry_audit=dict(boundary=dict(rms_bbox_pct=gamma,p95_bbox_pct=gamma,coverage_fraction=.5)),
                        t1_native_confirmation=dict(free_shape_quality=dict(total_area=1.)))
                    rows.append(dict(phase="development",model=model,parent=p,patch=j,seed=seed,K=12,metrics=metrics))
    statuses={m:dict(status="ELIGIBLE") for m in ("L0","L1","L2")}
    result=aggregate_results(rows,"development",("L0","L1","L2"),seeds,candidate_status=statuses)
    assert np.isclose(result["macros"]["L0"][RMS]["macro_mean"],11.)
    flat_mean=np.mean([x["metrics"]["remainder"]["own"]["symmetric_rms_bbox_pct"] for x in rows if x["model"]=="L0"])
    assert not np.isclose(flat_mean,11.)
    selected=selection_from_summary(result)
    assert selected["selected_models"]==["L1"] and selected["slots"]["Gamma_RMS"]["selected"]=="L1"
    result["macros"]["L1"][RMS]["macro_mean"]=10.;result["macros"]["L2"][GAMMA]["macro_mean"]=7.
    assert selection_from_summary(result)["selected_models"]==["L1","L2"]
    result["macros"]["L1"][RMS]["macro_mean"]=11.;result["macros"]["L2"][RMS]["macro_mean"]=11.
    result["macros"]["L1"][GAMMA]["macro_mean"]=20.;result["macros"]["L2"][GAMMA]["macro_mean"]=20.
    assert selection_from_summary(result)["selected_models"]==[]
    try:aggregate_results(rows+[rows[0]],"development",("L0","L1","L2"),seeds,candidate_status=statuses)
    except ValueError:pass
    else:raise AssertionError("Duplicate accepted")
    pair=result["patches"]["0"]["paired"]["L1_minus_L0"]["parents"]["0"]
    assert pair["improvement_counts"][RMS]["better"]==8
    return dict(status="PASS",checks=6,model_calls=0,geometry_calls=0,OT_calls=0,autograd_calls=0)
