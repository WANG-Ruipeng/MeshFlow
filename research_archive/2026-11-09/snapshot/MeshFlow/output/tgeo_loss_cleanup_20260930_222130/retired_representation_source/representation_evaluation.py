"""Fixed-bank, read-only full CPU evaluation of analytic geometry representation alignment.

The old complete evaluator is called unchanged. No model, sampling, training,
OT, autograd, geometry repair, or output-based patch selection occurs here.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np

from experiments.mf_h1_local0.a_then_fc_evaluation import scalar_metrics, _counts, _summary, AREA_METRIC
from experiments.mf_h1_local0.t1_gated_factorial_evaluation import QUALITY_METRICS
from experiments.mf_h1_local0.t1_native_confirmation_evaluation import (
    score_output, audit_context, cross_binding, summarize_values, concentration,
    PROTOCOL as ORIGINAL_METRIC_PROTOCOL,
)

REPO = Path(__file__).resolve().parents[1]
MODELS = ("Geo_initial", "G0", "G_Angular", "G_AngularScale")
PAIRINGS = (("G0", "Geo_initial"), ("G_Angular", "G0"), ("G_AngularScale", "G_Angular"), ("G_AngularScale", "G0"))
SEEDS = tuple(range(9901, 9909))
EXPECTED_BANK_SHA = "2a27cf5d576272947b6944f71b7f17ab32ca1489d43d267204303c114a82248e"
PARENT_UIDS = ("2af09bd8df40506c9e646678ef50aa3d", "5b0e833cf2ea465e42bd82e95587d8af")
ERROR_METRIC = "remainder_symmetric_RMS_bbox_pct"
PROTOCOL = dict(original_metric_protocol=ORIGINAL_METRIC_PROTOCOL,
    model_order=list(MODELS), pairings=[f"{a}_minus_{b}" for a,b in PAIRINGS],
    pair_sign="candidate minus reference; lower error/count and higher coverage/angle/binding are favorable",
    method="analytic-geometry representation alignment pilot; no external teacher, topology guarantee, or method reproduction claim",
    patches="Original patch0 and frozen additional patches1/2 are separate; never pool them or claim new-parent generalization",
    reference="All saved reference points are verified bitwise against the unchanged scorer before any endpoint scoring",
    binding="Four cells use the identical fixed 100 free slots and target banks; positive binding was already established",
    automatic_winner="NOT_DEFINED", geometry_repair="NONE")


def _path(value):
    p = Path(value)
    if p.exists(): return p
    text = str(value).replace(chr(92), "/")
    if len(text)>2 and text[1]==":": return Path("/mnt/"+text[0].lower()+text[2:])
    if text.startswith("/mnt/") and len(text)>7: return Path(text[5].upper()+":"+text[6:])
    return p


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def _array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str((value.dtype,value.shape)).encode()+value.tobytes()).hexdigest()


def _write(path,value):
    path=Path(path)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return dict(path=str(path.resolve()),sha256=_sha(path),bytes=path.stat().st_size)


def audit_seeds(seeds, out):
    """Read semantic seed records and seed/noise filenames, not metric digits."""
    tick=time.perf_counter(); seeds=tuple(int(s) for s in seeds)
    if len(seeds)!=8 or len(set(seeds))!=8: raise ValueError("Eight distinct registered seeds required")
    roots=[REPO/"native_t1/runs",REPO/"experiments/mf_h1_local0/outputs",REPO/"output"]
    excluded=_path(out).resolve(); wanted=set(seeds); hits=[]; files=0; documents=0
    field=re.compile(r'"([^"\n]*(?:seed|seeds))"\s*:\s*(\[[^\]]*\]|-?\d+)',re.I)
    for root in roots:
        if not root.is_dir(): continue
        for p in root.rglob("*"):
            if not p.is_file() or p.resolve().is_relative_to(excluded): continue
            files+=1
            for match in re.finditer(r"(?:seed|noise)[_-]?(\d+)",p.name,re.I):
                if int(match.group(1)) in wanted: hits.append(dict(path=str(p),kind="filename",seed=int(match.group(1))))
            if p.suffix.lower() not in (".json",".jsonl"): continue
            documents+=1
            text=p.read_text(encoding="utf-8-sig")
            for match in field.finditer(text):
                value=json.loads(match.group(2)); values=value if isinstance(value,list) else [value]
                for seed in values:
                    if type(seed) is int and seed in wanted:
                        hits.append(dict(path=str(p),kind="semantic_seed_field",field=match.group(1),seed=seed))
    return dict(status="PASS" if not hits else "FAIL_SEED_PREVIOUSLY_RECORDED",seeds=list(seeds),
        roots=[str(p) for p in roots],excluded_current_run=str(excluded),files_examined=files,
        json_documents_examined=documents,hits=hits,seconds=time.perf_counter()-tick,
        scope="Recorded seed fields / seed-noise filenames; hashes and numerical metrics are not seed evidence",
        model_calls=0,geometry_calls=0,optimizer_updates=0)


def load_bank(path, expected_sha=EXPECTED_BANK_SHA):
    """Read the unchanged SHA-pinned bank through the frozen offline helper."""
    from experiments.mf_h1_local0.tgeo_m10_evaluation import load_bank as original
    return original(path,expected_sha)


def _input_audit(bank,patch,ledger):
    from experiments.mf_h1_local0.tgeo_m10_evaluation import _input_audit as original
    return original(bank,patch,ledger)


def _seed_map(seeds):
    seeds=tuple(map(int,seeds))
    if len(seeds)!=8 or len(set(seeds))!=8: raise ValueError("Require eight distinct original-patch seeds")
    return {0:seeds,1:seeds[:2],2:seeds[:2]}


def aggregate_results(rows,seeds=SEEDS):
    """Only scalar arithmetic; no geometry calls, no pooling across patches."""
    rows=list(rows); by_patch=_seed_map(seeds)
    expected={(m,p,j,s) for m in MODELS for p in (0,1) for j,ss in by_patch.items() for s in ss}
    index={(r["model"],int(r["parent"]),int(r["patch"]),int(r["seed"])):r for r in rows}
    if set(index)!=expected or len(rows)!=len(expected) or any(int(r.get("K",12))!=12 for r in rows):
        raise ValueError("Require complete unique 96-row registered matrix")
    result=dict(schema="geometry_rep_alignment_summary_v1",patches={},original_patch="0",new_patches=["1","2"],
        protocol=PROTOCOL,geometry_calls=0,model_calls=0)
    for patch,ss in by_patch.items():
        block=dict(patch=patch,seeds=list(ss),patch_group="original" if patch==0 else "new",models={},paired={},bindings=[])
        for model in MODELS:
            group=dict(parents={},binding={})
            for parent in (0,1):
                values=[dict(seed=s,**index[model,parent,patch,s].get("scalars",scalar_metrics(index[model,parent,patch,s]["metrics"]))) for s in ss]
                if any(r[AREA_METRIC] is None for r in values): raise ValueError("Full scorer area diagnostic missing")
                names=[k for k in values[0] if k!="seed"]
                stats={k:_summary(values,k) for k in names}
                group["parents"][str(parent)]=dict(per_seed=values,metrics=stats,worst_RMS=stats[ERROR_METRIC]["worst"])
            binding=[dict(seed=s,**cross_binding(index[model,0,patch,s]["metrics"],index[model,1,patch,s]["metrics"])) for s in ss]
            block["bindings"].extend(dict(model=model,patch=patch,seed=b["seed"],K=12,metrics=b) for b in binding)
            group["binding"]=dict(per_seed=binding,metrics={k:summarize_values(binding,k,higher_better=True) for k in ("Delta0","Delta1","B")},
                both_positive_count=sum(b["both_directions_positive"] for b in binding),concentration=concentration(binding))
            block["models"][model]=group
        for candidate,reference in PAIRINGS:
            pair=dict(candidate=candidate,reference=reference,parents={},binding={},equal_new_update_budget=reference!="Geo_initial")
            for parent in ("0","1"):
                old=block["models"][reference]["parents"][parent]["per_seed"]
                new=block["models"][candidate]["parents"][parent]["per_seed"]
                values=[dict(seed=a["seed"],**{k:b[k]-a[k] if b[k] is not None and a[k] is not None else None for k in a if k!="seed"}) for a,b in zip(old,new)]
                names=[k for k in values[0] if k!="seed"]; counts={k:_counts(values,k) for k in names}
                pair["parents"][parent]=dict(per_seed_candidate_minus_reference=values,
                    metrics={k:_summary(values,k) for k in names},improvement_counts=counts,
                    remainder_better_count=counts[ERROR_METRIC]["better"],remainder_tie_count=counts[ERROR_METRIC]["tie"],
                    remainder_worse_count=counts[ERROR_METRIC]["worse"],
                    tradeoffs=[dict(seed=v["seed"],remainder_RMS_change=v[ERROR_METRIC],
                        quality_metrics_increased=[k for k in QUALITY_METRICS if v.get(k) is not None and v[k]>0]) for v in values])
            old=block["models"][reference]["binding"]["per_seed"]; new=block["models"][candidate]["binding"]["per_seed"]
            binding=[dict(seed=a["seed"],**{k:b[k]-a[k] for k in ("Delta0","Delta1","B")}) for a,b in zip(old,new)]
            pair["binding"]=dict(per_seed_candidate_minus_reference=binding,metrics={k:summarize_values(binding,k,higher_better=True) for k in ("Delta0","Delta1","B")},
                higher_B_count=sum(b["B"]>0 for b in binding),concentration=concentration(binding))
            block["paired"][f"{candidate}_minus_{reference}"]=pair
        result["patches"][str(patch)]=block
    json.dumps(result,allow_nan=False)
    return result


def evaluate(r,out):
    """Score every registered endpoint once and save full independent audits."""
    from experiments.mf_h1_local0.native_inpainting_metrics import array_sha
    from experiments.mf_h1_local0.guided_native_geometry_audit import _sha_array
    tick=time.perf_counter(); out=_path(out); rows=list(r["generated"])
    by_patch=_seed_map(SEEDS)
    expected={(m,p,j,s) for m in MODELS for p in (0,1) for j,ss in by_patch.items() for s in ss}
    actual={(x["model"],int(x["parent"]),int(x["patch"]),int(x["seed"])) for x in rows}
    if actual!=expected or len(rows)!=96: raise ValueError("Incomplete registered generated matrix")
    record=r["assets"]["evaluation_inputs"]; banks=load_bank(record["path"],record["sha256"])
    audit_dir=out/"audit"
    audit_dir.mkdir(exist_ok=False)
    counts=dict(model_calls=0,geometry_autograd_calls=0,optimizer_updates=0,OT_calls=0,repairs=0)
    input_records=[]
    for patch,bank in banks.items():
        bank["input_topology"]=_input_audit(bank,patch,counts)
        bank["context_audits"]={p:audit_context(bank["cases_by_key"][p,12]["constraints"],bank["cases_by_key"][p,12]["bbox_diagonal"],counts) for p in (0,1)}
        input_records.append(_write(audit_dir/f"input_patch{patch}.json",dict(patch=patch,input_topology=bank["input_topology"],context_audits=bank["context_audits"],reference_audit=bank["reference_audit"])))
    full_rows=[]; compact=[]
    for number,row in enumerate(rows,1):
        start=time.perf_counter(); p=int(row["parent"]); patch=int(row["patch"]); seed=int(row["seed"])
        bank=banks[patch]; raw_path=_path(row["raw_path"])
        if _sha(raw_path)!=row["raw_sha256"]: raise ValueError("Raw file identity changed")
        with np.load(raw_path,allow_pickle=False) as raw:
            output=raw["output"].copy(); path=raw["path"].copy()
        C=bank["cases_by_key"][p,12]["constraints"]
        if output.dtype!=np.float32 or output.size!=112*9 or path.dtype!=np.float32 or path.size!=51*112*9:
            raise ValueError("Raw endpoint/51-state FP32 shape differs")
        output=output.reshape(112,3,3); path=path.reshape(51,112,3,3)
        if not np.isfinite(path).all() or output.tobytes()!=path[-1].tobytes() or any(state[:12].tobytes()!=C.tobytes() for state in path):
            raise ValueError("Nonfinite, endpoint mismatch, or immutable C changed")
        metrics=score_output(output,bank,p,ledger=counts)
        for target in (0,1):
            prefix=f"parent{target}_patch{patch}_"
            if metrics["remainder"]["cross"][target]["target_sample_sha256"]!=array_sha(bank["arrays"][prefix+"pairwise_target_points"]):
                raise ValueError("Actual scoring target512 differs from frozen reference")
            if metrics["common_unknown_R12"]["cross"][target]["target_point_sha256"]!=array_sha(bank["arrays"][prefix+"coverage_points"]):
                raise ValueError("Actual coverage target512 differs from frozen reference")
        prefix=f"parent{p}_patch{patch}_"; boundary=metrics["guided_geometry_audit"]["boundary"]
        if boundary["point_sha256"]!=_sha_array(bank["arrays"][prefix+"boundary_points"]) or boundary["weight_sha256"]!=_sha_array(bank["arrays"][prefix+"boundary_weights"]):
            raise ValueError("Actual Gamma samples/weights differ from frozen reference")
        identity=dict(model=row["model"],parent=p,patch=patch,patch_group="original" if patch==0 else "new",seed=seed,K=12,raw_path=str(raw_path),raw_sha256=row["raw_sha256"])
        values=scalar_metrics(metrics)
        if values[AREA_METRIC] is None: raise ValueError("Unchanged scorer omitted free area")
        duration=time.perf_counter()-start
        artifact=_write(audit_dir/f'{row["model"]}_P{p}_patch{patch}_seed{seed}.json',dict(**identity,metrics=metrics,scoring_seconds=duration))
        full_rows.append(dict(**identity,metrics=metrics,scalars=values))
        compact.append(dict(**identity,scalars=values,metrics_path=artifact["path"],metrics_sha256=artifact["sha256"],metrics_bytes=artifact["bytes"],scoring_seconds=duration,
            C_all51_bitwise=True,all_coordinates_finite=True,total_faces=112,free_faces=100))
        print(json.dumps(dict(event="offline_score",completed=number,total=96,**identity,seconds=duration)),flush=True)
    summary=aggregate_results(full_rows,SEEDS)
    csv_path=out/"compact_per_output.csv"
    scalar_names=list(compact[0]["scalars"])
    with csv_path.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["model","parent","patch","patch_group","seed","K"]+scalar_names+["raw_path","metrics_path"])
        writer.writeheader()
        for row in compact: writer.writerow({**{k:row[k] for k in ("model","parent","patch","patch_group","seed","K","raw_path","metrics_path")},**row["scalars"]})
    r["scores"]=compact; r["summary"]=summary
    r["evaluation_ledger"]=dict(status="PASS",counts=counts,seconds=time.perf_counter()-tick,scored_outputs=96,
        input_audits=input_records,reference_archive=record,complete_pairs_per_output=dict(CF=1200,FF=4950,CC_cached=66),
        immutable_context_audits=6,generated_topology_audits=96,all_reference_hashes_verified=True,
        compact_csv=dict(path=str(csv_path.resolve()),sha256=_sha(csv_path),bytes=csv_path.stat().st_size),
        scope="Static independent full CF/FF plus cached CC; exact FP32 source/inherited topology. No repair/model/OT/autograd.",
        exact_predicate_certification="NOT_RUN",continuous_collision_safety="NOT_RUN")
    return r["evaluation_ledger"]


def run_cpu_tests():
    """Minimal saved-scalar fixtures only; no geometry or model calls."""
    rows=[]
    for patch,seeds in _seed_map(SEEDS).items():
        for model in MODELS:
            for parent in (0,1):
                for n,seed in enumerate(seeds):
                    own={"Geo_initial":10.,"G0":9.,"G_Angular":8.,"G_AngularScale":7.}[model]+n*.125
                    cross=[dict(symmetric_rms_bbox_pct=own+(0 if p==parent else 2.),predicted_to_target_rms_bbox_pct=own,target_to_predicted_rms_bbox_pct=own) for p in (0,1)]
                    common=[dict(rms_bbox_pct=own+(0 if p==parent else 2.),fraction_target_points_within_1pct_bbox=.5) for p in (0,1)]
                    metrics=dict(K=12,variant="raw",parent_index=parent,remainder=dict(own=cross[parent],cross=cross),whole=dict(surface=cross[parent]),
                        common_unknown_R12=dict(own=common[parent],cross=common),t1_native_confirmation=dict(free_shape_quality=dict(total_area=3.+n*.25)))
                    rows.append(dict(model=model,parent=parent,patch=patch,seed=seed,K=12,metrics=metrics))
    result=aggregate_results(rows)
    assert len(rows)==96 and sum(len(x["bindings"]) for x in result["patches"].values())==48
    for patch,count in (("0",8),("1",2),("2",2)):
        for name,value in (("G0_minus_Geo_initial",-1.),("G_Angular_minus_G0",-1.),("G_AngularScale_minus_G_Angular",-1.),("G_AngularScale_minus_G0",-2.)):
            group=result["patches"][patch]["paired"][name]["parents"]["0"]
            assert np.isclose(group["metrics"][ERROR_METRIC]["mean"],value,rtol=0,atol=1e-12)
            assert group["remainder_better_count"]==count
            assert group["improvement_counts"][AREA_METRIC]["better"] is None
    for bad in (rows[:-1],rows+[rows[0]],[dict(row,patch=9) if i==0 else row for i,row in enumerate(rows)]):
        try: aggregate_results(bad)
        except ValueError: pass
        else: raise AssertionError("Incomplete/duplicate/wrong-patch matrix accepted")
    a=np.array([0,0,0,0,100,100,100,100]); b=np.array([0,0,100,100,100,100,100,100])
    assert summarize_values([dict(seed=s,value=float(x)) for s,x in zip(SEEDS,b-a)],"value")["median"]==0
    assert np.median(b)-np.median(a)==50
    return dict(status="PASS",checks=6,model_calls=0,geometry_calls=0,autograd_calls=0,optimizer_updates=0)
