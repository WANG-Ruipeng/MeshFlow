"""Fixed five-model interface pilot: unchanged independent endpoint scoring.

No model, OT, autograd, repair, training or parameter selection is implemented.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from ..tgeo_m10_evaluation import (
    load_bank,_input_audit,_sha,_write,score_output,audit_context,cross_binding,
    summarize_values,concentration,scalar_metrics,_counts,_summary,AREA_METRIC)
from ..loss_screen.evaluation import _metric_reference_checks,_raw

REPO=Path(__file__).resolve().parents[3]
MODELS=("L0","L3","L5","J35","JEdge5")
SEEDS=tuple(range(10101,10105))
PAIRS=(("L3","L0"),("L5","L0"),("J35","L5"),("J35","L3"),
       ("J35","L0"),("JEdge5","J35"),("JEdge5","L5"),("JEdge5","L0"))
EFFECTS={
 "D3_when5off":{"L3":1,"L0":-1},
 "D3_when5on":{"J35":1,"L5":-1},
 "D5_when3off":{"L5":1,"L0":-1},
 "D5_when3on":{"J35":1,"L3":-1},
 "I":{"J35":1,"L3":-1,"L5":-1,"L0":1}}
PROTOCOL=dict(phase="pilot",models=list(MODELS),outputs_per_model=24,
 cell_sample_count=4,cell_count=6,formal_outputs=120,
 aggregation="Per parent/patch mean of four matched seeds, then equal six-cell macro",
 paired_sign="candidate minus reference; paired medians never subtract marginal medians",
 factorial="Each seed first: J35-L3-L5+L0; fixed coefficient non-additivity, not intensity-controlled synergy",
 no_selection=True,no_old_scores_pooled=True,known_parent_patch_new_noise_only=True,
 independent_geometry="Unmodified complete CF1200/FF4950 with CC66 cached per C")

def _path(value):
    text=str(value).replace(chr(92),"/")
    if len(text)>2 and text[1]==":" and Path("/mnt").is_dir():
        return Path("/mnt/"+text[0].lower()+text[2:])
    return Path(value)

def _seeds(r):
    values=r["protocol"]["evaluation"]["seeds"]
    seeds=tuple(int(x) for x in values)
    if len(seeds)!=4 or len(set(seeds))!=4:raise ValueError("Four fixed distinct seeds required")
    return seeds

def _expected(models,seeds):
    return {(m,p,j,s) for m in models for p in (0,1) for j in (0,1,2) for s in seeds}

def _identity(x):
    return x["model"],int(x["parent"]),int(x["patch"]),int(x["seed"])

def audit_seeds(seeds,out):
    """Semantic read-only seed audit; supports the fixed four-noise design."""
    tick=time.perf_counter();seeds=tuple(int(s) for s in seeds)
    if len(seeds)!=4 or len(set(seeds))!=4:raise ValueError("Four distinct seeds required")
    roots=[REPO/"native_t1/runs",REPO/"experiments/mf_h1_local0/outputs",REPO/"output"]
    excluded=_path(out).resolve();wanted=set(seeds);hits=[];files=0;documents=0
    field=re.compile(r'"([^"\n]*(?:seed|seeds))"\s*:\s*(\[[^\]]*\]|-?\d+)',re.I)
    for root in roots:
        if not root.is_dir():continue
        for p in root.rglob("*"):
            if not p.is_file() or p.resolve().is_relative_to(excluded):continue
            files+=1
            for match in re.finditer(r"(?:seed|noise)[_-]?(\d+)",p.name,re.I):
                if int(match.group(1)) in wanted:hits.append(dict(path=str(p),kind="filename",seed=int(match.group(1))))
            if p.suffix.lower() not in (".json",".jsonl"):continue
            documents+=1;text=p.read_text(encoding="utf-8-sig")
            for match in field.finditer(text):
                value=json.loads(match.group(2))
                for seed in value if isinstance(value,list) else [value]:
                    if type(seed) is int and seed in wanted:hits.append(dict(path=str(p),kind="semantic_seed_field",field=match.group(1),seed=seed))
    return dict(status="PASS" if not hits else "FAIL_SEED_PREVIOUSLY_RECORDED",seeds=list(seeds),
        roots=[str(x) for x in roots],excluded_current_run=str(excluded),files_examined=files,
        json_documents_examined=documents,hits=hits,seconds=time.perf_counter()-tick,
        scope="Recorded semantic seed fields / seed-noise filenames; not arbitrary metric digits",
        model_calls=0,geometry_calls=0,optimizer_updates=0)

def _aggregate(rows,phase,models,seeds,*,candidate_status=None,pairings=None):
    """Complete eligible models only; pure scalar/JSON arithmetic."""
    rows=list(rows);models=tuple(models);seeds=tuple(seeds)
    if "L0" not in models or len(set(models))!=len(models):raise ValueError("Unique models and L0 required")
    index={_identity(x):x for x in rows}
    if set(index)!=_expected(models,seeds) or len(index)!=len(rows):raise ValueError("Incomplete/duplicate phase matrix")
    if any(x.get("phase")!=phase or x.get("K",12)!=12 for x in rows):raise ValueError("Phase/K mismatch")
    pairs=list(pairings) if pairings is not None else [(m,"L0") for m in models if m!="L0"]
    result=dict(schema="l3_l5_interface_pilot_summary_v1",phase=phase,models=list(models),seeds=list(seeds),
        patches={},macros={},candidate_status=candidate_status or {},protocol=PROTOCOL,geometry_calls=0,model_calls=0)
    for patch in (0,1,2):
        ss=seeds
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
            cells=[dict(parent=p,patch=j,n=len(seeds),mean=result["patches"][str(j)]["models"][model]["parents"][str(p)]["metrics"][key]["mean"])
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




def _effects(summary):
    names=list(summary["macros"]["L0"])
    effects=dict(per_cell={},macro={},binding={})
    for effect,coeff in EFFECTS.items():
        cells={}
        for j in (0,1,2):
            b=summary["patches"][str(j)]
            for p in ("0","1"):
                by_model={m:{x["seed"]:x for x in b["models"][m]["parents"][p]["per_seed"]} for m in coeff}
                values=[]
                for seed in summary["seeds"]:
                    v=dict(seed=seed)
                    for k in names:
                        parts=[(c,by_model[m][seed].get(k)) for m,c in coeff.items()]
                        v[k]=sum(c*x for c,x in parts) if all(x is not None for c,x in parts) else None
                    values.append(v)
                cells[f"P{p}_patch{j}"]=dict(parent=int(p),patch=j,n=4,coefficients=coeff,per_seed=values,
                    metrics={k:_summary(values,k) for k in names},
                    improvement_counts={k:_counts(values,k) for k in names})
        effects["per_cell"][effect]=cells
        effects["macro"][effect]={k:dict(six_cells=[dict(parent=c["parent"],patch=c["patch"],n=4,
            mean=c["metrics"][k]["mean"]) for c in cells.values()],
            macro_mean=float(np.mean([c["metrics"][k]["mean"] for c in cells.values()]))
                if all(c["metrics"][k]["mean"] is not None for c in cells.values()) else None)
            for k in names}
        bindings={}
        for j in (0,1,2):
            by_model={m:{x["seed"]:x for x in summary["patches"][str(j)]["models"][m]["binding"]["per_seed"]} for m in coeff}
            values=[dict(seed=s,**{k:sum(c*by_model[m][s][k] for m,c in coeff.items()) for k in ("Delta0","Delta1","B")}) for s in summary["seeds"]]
            bindings[str(j)]=dict(per_seed=values,metrics={k:summarize_values(values,k,higher_better=True) for k in ("Delta0","Delta1","B")},
                higher_B_count=sum(v["B"]>0 for v in values),concentration=concentration(values))
        effects["binding"][effect]=bindings
    return effects

def aggregate_results(rows,seeds):
    seeds=tuple(seeds)
    if len(seeds)!=4 or len(set(seeds))!=4:raise ValueError("Four fixed distinct seeds required")
    summary=_aggregate(rows,"pilot",MODELS,seeds,pairings=PAIRS)
    summary["factorial"]=_effects(summary)
    return summary

def _csv(path,fields,rows):
    with Path(path).open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return dict(path=str(Path(path).resolve()),sha256=_sha(path),bytes=Path(path).stat().st_size,rows=len(rows))

def export_tables(r,out):
    summary=r["summaries"]["pilot"];names=sorted(summary["macros"]["L0"])
    fields=["phase","model","parent","patch","seed","K","raw_path","metrics_path"]+names
    outputs=[{**{k:x.get(k) for k in fields[:8]},**x["scalars"]} for x in r["scores"]]
    records={"per_output":_csv(out/"compact_per_output.csv",fields,outputs)}
    rows=[]
    for j in (0,1,2):
        for name,pair in summary["patches"][str(j)]["paired"].items():
            for p,cell in pair["parents"].items():
                rows.extend(dict(kind="paired",effect=name,parent=int(p),patch=j,**v) for v in cell["per_seed_candidate_minus_reference"])
    for name,cells in summary["factorial"]["per_cell"].items():
        for cell in cells.values():rows.extend(dict(kind="factorial",effect=name,parent=cell["parent"],patch=cell["patch"],**v) for v in cell["per_seed"])
    records["paired_and_factorial"]=_csv(out/"compact_per_seed_effects.csv",["kind","effect","parent","patch","seed"]+names,rows)
    return records

def evaluate(r,out):
    """Explicit root-only scoring entry, all120 outputs first; no partial selection."""
    tick=time.perf_counter();out=_path(out);seeds=_seeds(r)
    if r.get("summaries",{}).get("pilot"):raise RuntimeError("No automatic scoring repeat")
    rows=list(r["generated"])
    if len(rows)!=120 or {_identity(x) for x in rows}!=_expected(MODELS,seeds):raise ValueError("Exact120 five-model matrix required")
    if any(x.get("phase")!="pilot" or x.get("K")!=12 for x in rows):raise ValueError("Pilot phase/K mismatch")
    if tuple(r["protocol"]["evaluation"]["models"])!=MODELS:raise ValueError("Fixed model order differs")
    phase=r.get("sampling_phases",{}).get("pilot",{})
    for m in MODELS:
        if phase.get("models_audit",{}).get(m,{}).get("status")!="PASS":raise ValueError("Require successful frozen-state sampling: "+m)
    record=r["assets"]["evaluation_inputs"];banks=load_bank(record["path"],record["sha256"])
    directory=out/"audit";directory.mkdir(parents=True,exist_ok=False)
    ledger=dict(model_calls=0,geometry_autograd_calls=0,optimizer_updates=0,OT_calls=0,repairs=0)
    inputs=[]
    for j,bank in banks.items():
        bank["input_topology"]=_input_audit(bank,j,ledger)
        bank["context_audits"]={p:audit_context(bank["cases_by_key"][p,12]["constraints"],bank["cases_by_key"][p,12]["bbox_diagonal"],ledger) for p in (0,1)}
        inputs.append(_write(directory/f"input_patch{j}.json",dict(input_topology=bank["input_topology"],context_audits=bank["context_audits"],reference_audit=bank["reference_audit"])))
    full=[];compact=[]
    for row in rows:
        started=time.perf_counter();m,p,j,s=_identity(row)
        raw=_raw(row,banks[j],p);metrics=score_output(raw,banks[j],p,ledger=ledger)
        _metric_reference_checks(metrics,banks[j],p,j);values=scalar_metrics(metrics)
        identity=dict(phase="pilot",model=m,parent=p,patch=j,seed=s,K=12,raw_path=row["raw_path"],raw_sha256=row["raw_sha256"])
        artifact=_write(directory/f"{m}_P{p}_patch{j}_seed{s}.json",dict(identity,metrics=metrics,scoring_seconds=time.perf_counter()-started))
        full.append(dict(identity,metrics=metrics));compact.append(dict(identity,scalars=values,metrics_path=artifact["path"],metrics_sha256=artifact["sha256"],metrics_bytes=artifact["bytes"],scoring_seconds=time.perf_counter()-started))
        print(json.dumps(dict(event="interface_pilot_score",model=m,parent=p,patch=j,seed=s,completed=len(compact),total=120)),flush=True)
    r["scores"]=compact;r.setdefault("summaries",{})["pilot"]=aggregate_results(full,seeds)
    r["evaluation_ledger"]=dict(status="PASS",counts=ledger,seconds=time.perf_counter()-tick,input_audits=inputs,
        scored_outputs=120,immutable_context_audits=6,complete_pairs_per_output=dict(CF=1200,FF=4950,CC_cached=66),historical_scores_reused=0)
    r["evaluation_ledger"]["tables"]=export_tables(r,out)
    return r["evaluation_ledger"]

def run_cpu_tests():
    """One scalar suite: no model, loss, geometry, OT or autograd."""
    rows=[]
    offsets={"L0":0.,"L3":1.,"L5":2.,"J35":2.5,"JEdge5":2.}
    for m in MODELS:
        for p in (0,1):
            for j in (0,1,2):
                for i,s in enumerate(SEEDS):
                    own=10.+j+p+i/10+offsets[m];cross=[dict(symmetric_rms_bbox_pct=own+(target!=p),predicted_to_target_rms_bbox_pct=own,target_to_predicted_rms_bbox_pct=own) for target in (0,1)]
                    common=[dict(rms_bbox_pct=own+(target!=p),fraction_target_points_within_1pct_bbox=.5) for target in (0,1)]
                    metric=dict(K=12,variant="raw",parent_index=p,remainder=dict(own=cross[p],cross=cross),whole=dict(surface=cross[p]),common_unknown_R12=dict(own=common[p],cross=common),guided_geometry_audit=dict(boundary=dict(rms_bbox_pct=own,p95_bbox_pct=own,coverage_fraction=.5)),t1_native_confirmation=dict(free_shape_quality=dict(total_area=1.)))
                    rows.append(dict(phase="pilot",model=m,parent=p,patch=j,seed=s,K=12,metrics=metric))
    v=aggregate_results(rows,SEEDS);key="remainder_symmetric_RMS_bbox_pct"
    assert len(rows)==120
    assert all(len(v["patches"][str(j)]["models"][m]["parents"][str(p)]["per_seed"])==4 for j in (0,1,2) for m in MODELS for p in (0,1))
    assert np.isclose(v["macros"]["L0"][key]["macro_mean"],11.65)
    assert len(v["patches"]["0"]["paired"])==8
    assert np.isclose(v["factorial"]["macro"]["I"][key]["macro_mean"],-.5)
    assert all(np.isclose(x[key],-.5) for cell in v["factorial"]["per_cell"]["I"].values() for x in cell["per_seed"])
    assert np.isclose(v["patches"]["1"]["paired"]["JEdge5_minus_J35"]["parents"]["0"]["metrics"][key]["median"],-.5)
    assert v["patches"]["1"]["paired"]["JEdge5_minus_J35"]["parents"]["0"]["improvement_counts"][key]["better"]==4
    try:aggregate_results(rows+[rows[0]],SEEDS)
    except ValueError:pass
    else:raise AssertionError("Duplicate accepted")
    try:aggregate_results(rows[:-1],SEEDS)
    except ValueError:pass
    else:raise AssertionError("Incomplete matrix accepted")
    return dict(status="PASS",checks=10,model_calls=0,geometry_calls=0,OT_calls=0,autograd_calls=0)
