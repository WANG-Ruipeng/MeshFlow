"""Synthetic scalar-table analysis only; no model, geometry or frozen sources."""
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import pytest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("analyze_recipe_factorial_results",REPO/"tools/analyze_recipe_factorial_results.py")
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def panel():
    requests=[];rows=[]
    # Per-parent recipe means: R1=10, R2=8, R3=7, R4=3 plus the same
    # parent/noise offsets. MIX main=-3, LATE main=-4, interaction=-2.
    intercept={"START5000":11,"HIST_FM6000":12,"R1_40_H_ALL":10,
               "R2_MIX_H_ALL":8,"R3_40_H_LATE":7,"R4_MIX_H_LATE":3}
    for p in range(8):
        for ratio in (.2,.4):
            cohort="A20" if ratio==.2 else "ORIGINAL6" if p<6 else "EXTENDED2"
            for noise in a.NOISES:
                request=dict(request_id=f"p{p}_C{int(ratio*100)}_n{noise}",parent_id=f"p{p}",
                             condition_id=f"p{p}_C{int(ratio*100)}",candidate=noise,ratio=ratio,
                             condition_cohort=cohort,N=128,K=round(ratio*128))
                requests.append(request)
                for label in a.LABELS:
                    rows.append(dict(request,label=label,status="PASS",contracts={"C_all51_bitwise":True},
                                     metrics={"CF_proper_affected_free_fraction":intercept[label]+p+noise/10,
                                              "FF_illegal_affected_free_fraction":20-intercept[label]+p,
                                              "ConflictCov_C_status":"PARTIAL" if p==0 else "PASS",
                                              "universal_missing_metric":None,"topology_boolean":False}))
    registration=dict(kind="main",labels=list(a.LABELS),expected_outputs=384)
    receipt=dict(status="COMPLETE",registered=384,scored=384)
    return registration,requests,rows,receipt


def test_validates_complete_panel_and_refuses_duplicates_or_missing_noise():
    reg,qs,rows,receipt=panel()
    a.validate_panel(reg,qs,rows,receipt)
    with pytest.raises(ValueError,match="Complete64"):
        a.validate_panel(reg,qs,rows[:-1],receipt)
    broken=[*rows[:-1],rows[0]]
    with pytest.raises(ValueError,match="Duplicate"):
        a.validate_panel(reg,qs,broken,receipt)
    badreceipt=dict(receipt,status="PARTIAL_BLOCKED")
    with pytest.raises(ValueError,match="Complete scoring"):
        a.validate_panel(reg,qs,rows,badreceipt)


def test_noise_then_equal_parent_and_six_two_strata():
    _,qs,rows,_=panel();key="CF_proper_affected_free_fraction"
    parents=a.parent_means(rows,qs,[key]);summary=a.model_summary(parents)
    get=lambda cohort:next(r for r in summary if r["label"]=="R1_40_H_ALL" and r["ratio"]==.4 and r["condition_cohort"]==cohort)
    assert get("ALL")["mean"]==pytest.approx(13.85)
    assert get("ORIGINAL6")["expected"]==6 and get("EXTENDED2")["expected"]==2
    assert get("ORIGINAL6")["mean"]==pytest.approx(12.85)
    assert get("EXTENDED2")["mean"]==pytest.approx(16.85)
    assert not any(r["ratio"] not in (.2,.4) for r in summary)


def test_missing_metric_never_shrinks_parent_or_noise_denominator():
    _,qs,rows,_=panel();key="CF_proper_affected_free_fraction"
    row=next(r for r in rows if r["label"]=="R1_40_H_ALL" and r["parent_id"]=="p0" and r["ratio"]==.4 and r["candidate"]==7)
    row["metrics"][key]=None
    parents=a.parent_means(rows,qs,[key]);summary=a.model_summary(parents)
    p=next(r for r in parents if r["label"]=="R1_40_H_ALL" and r["parent_id"]=="p0" and r["ratio"]==.4)
    assert p["mean"] is None and p["expected"]==4 and p["available"]==3
    s=next(r for r in summary if r["label"]=="R1_40_H_ALL" and r["ratio"]==.4 and r["condition_cohort"]=="ALL")
    assert s["mean"] is None and s["expected"]==8 and s["available"]==7


def test_main_effects_and_interaction_use_correct_factorial_weights():
    _,qs,rows,_=panel();metric="CF_proper_affected_free_fraction"
    parents=a.parent_means(rows,qs,[metric]);detail,summary=a.factorial_tables(parents,[metric])
    get=lambda effect:next(r for r in summary if r["ratio"]==.2 and r["condition_cohort"]=="ALL" and r["effect"]==effect)
    assert get("MIX_main_effect")["mean"]==pytest.approx(-3)
    assert get("LATE_main_effect")["mean"]==pytest.approx(-4)
    assert get("interaction_difference_of_differences")["mean"]==pytest.approx(-2)
    assert get("MIX_at_H_ALL")["mean"]==pytest.approx(-2)
    assert get("MIX_at_H_LATE")["mean"]==pytest.approx(-4)
    assert get("MIX_main_effect")["expected"]==8
    assert len(detail)==16*7


def test_pair_distributions_keep_each_noise_and_each_parent():
    _,qs,rows,_=panel();key="CF_proper_affected_free_fraction"
    parents=a.parent_means(rows,qs,[key]);noise,pp,summary=a.paired_tables(rows,qs,parents,[key])
    assert len(noise)==15*64 and len(pp)==15*16
    s=next(r for r in summary if r["first"]=="R1_40_H_ALL" and r["second"]=="R2_MIX_H_ALL" and r["ratio"]==.2)
    assert s["mean"]==pytest.approx(2) and s["positive_count"]==8
    assert len(s["parent_deltas"])==8 and s["sign_counts_are_descriptive_not_significance"]


def test_tradeoffs_do_not_collapse_opposing_costs_or_source_GT():
    _,qs,rows,_=panel()
    for row in rows:row["metrics"]["GT_source_MSE"]=1000 if row["label"]=="R2_MIX_H_ALL" else 0
    keys=["CF_proper_affected_free_fraction","FF_illegal_affected_free_fraction","GT_source_MSE"]
    parents=a.parent_means(rows,qs,keys);_,pairs,_=a.paired_tables(rows,qs,parents,keys)
    table=a.tradeoff_table(pairs)
    chosen=[r for r in table if r["candidate"]=="R2_MIX_H_ALL" and r["reference"]=="R1_40_H_ALL" and r["parent_id"]=="p0" and r["ratio"]==.2]
    assert {r["metric"]:r["movement"] for r in chosen}=={"CF_proper_affected_free_fraction":"LOWER","FF_illegal_affected_free_fraction":"HIGHER"}
    assert not any(r["metric"]=="GT_source_MSE" for r in table)
    assert all(r["overall_model_judgment"]=="NOT_INFERRED" for r in table)


def test_catalog_preserves_statuses_and_all_null_metrics():
    _,_,rows,_=panel();catalog={r["metric"]:r for r in a.metric_catalog(rows)}
    assert not catalog["universal_missing_metric"]["numeric_summary"]
    assert catalog["universal_missing_metric"]["value_status_counts"]=={"NA_NULL":384}
    assert not catalog["topology_boolean"]["numeric_summary"]
    assert catalog["ConflictCov_C_status"]["value_status_counts"]['"PARTIAL"']==48


def save_panel(out):
    reg,qs,rows,receipt=panel();phase=out/"phases/main";evaluation=phase/"evaluation"
    qp=out/"evaluation/requests.json";a.write_json(qp,dict(rows=qs))
    reg.update(requests_path=str(qp),requests_sha256=a.digest(qp))
    a.write_json(phase/"registration.json",reg);a.write_json(evaluation/"per_output.json",dict(rows=rows))
    a.write_json(evaluation/"scoring_receipt.json",receipt)
    (out/"PROTOCOL_CLARIFICATIONS.md").write_text("Synthetic protocol fixture: historical pools differ.\n",encoding="utf8")
    a.write_json(out/"audit/condition_construction_audit.json",dict(
        synthetic_fixture=True,actual_training_completion_claimed=False,
        fixed_pool=dict(task_counts_by_family=dict(MVP_LEGACY20=104,DOMAIN_NEW20=104,DOMAIN_NEW40=203)),
        planned_exposure=dict(count_basis="PLANNED, NOT completed updates or actual exposure")))
    conditions=[];pairs=[]
    by={q["condition_id"]:q for q in qs}
    for label in a.LABELS:
        for cid,q in by.items():
            conditions.append(dict(label=label,condition_id=cid,parent_id=q["parent_id"],ratio=q["ratio"],condition_cohort=q["condition_cohort"],expected_pairs=6,available_pairs=6,D_free_mean=.25))
            for na,nb in itertools.combinations(a.NOISES,2):
                pairs.append(dict(label=label,condition_id=cid,parent_id=q["parent_id"],ratio=q["ratio"],candidate_a=na,candidate_b=nb,D_free_pair=.25,status="PASS"))
    a.write_csv(evaluation/"diversity_per_condition.csv",conditions);a.write_csv(evaluation/"diversity_pairs.csv",pairs)
    return evaluation


def test_end_to_end_writes_only_analysis_and_reuses_immutable_receipt(tmp_path):
    out=tmp_path/"experiments"/a.ROOT_NAME;save_panel(out)
    before={str(p):a.digest(p) for p in out.rglob("*") if p.is_file()}
    result=a.analyze(out)
    assert result["status"]=="COMPLETE" and result["new_model_calls"]==result["new_scoring_calls"]==0
    assert result["overall_score"] is None and not result["automatic_promotion"]
    dest=Path(result["analysis_dir"])
    assert (dest/"REPORT.md").exists() and (dest/"paired_parent_summary.csv").exists()
    assert len(a.read_json(dest/"all_metrics.json")["rows"])==384
    provenance=a.read_json(dest/"provenance.json")["input_files"]
    for key,path in (("protocol_clarifications",out/"PROTOCOL_CLARIFICATIONS.md"),
                     ("condition_construction_audit",out/"audit/condition_construction_audit.json")):
        assert provenance[key]==dict(path=str(path),sha256=a.digest(path))
    context=a.read_json(dest/"condition_construction_context.json")
    assert context["protocol_clarification_text"]==(out/"PROTOCOL_CLARIFICATIONS.md").read_text(encoding="utf8")
    assert context["original_condition_construction_audit"]==a.read_json(out/"audit/condition_construction_audit.json")
    report=(dest/"REPORT.md").read_text(encoding="utf8")
    assert report.index("完整历史条件池配方")<report.index("本报告只分析")
    assert "104个旧MVP任务和104个Domain新任务" in report and "203个任务全部来自Domain" in report
    assert "未隔离等曝光下的时机效应" in report
    interpretation=a.read_json(dest/"decision_evidence.json")["condition_pool_interpretation"]
    assert not interpretation["pure_ratio_causal_effect_identified"]
    assert not interpretation["ratio_only_interaction_identified"]
    assert not interpretation["planned_exposure_is_completed_exposure"]
    assert all(a.digest(p)==h for p,h in before.items())
    assert a.analyze(out)==result
    (dest/"REPORT.md").write_text("changed",encoding="utf8")
    with pytest.raises(ValueError,match="Immutable"):
        a.analyze(out)


def test_refuses_absent_real_results_without_creating_analysis(tmp_path):
    out=tmp_path/"experiments"/a.ROOT_NAME
    with pytest.raises(FileNotFoundError):a.analyze(out)
    assert not (out/"analysis").exists()


def test_diversity_rejects_duplicate_pairs_even_when_total_row_count_matches(tmp_path):
    out=tmp_path/"experiments"/a.ROOT_NAME;ev=save_panel(out)
    rows=a.read_json(ev/"per_output.json")["rows"]
    conditions=a.read_csv(ev/"diversity_per_condition.csv");pairs=a.read_csv(ev/"diversity_pairs.csv")
    a.validate_diversity(conditions,pairs,rows)
    pairs[1]=pairs[0]
    with pytest.raises(ValueError,match="Duplicate"):
        a.validate_diversity(conditions,pairs,rows)


def test_missing_condition_audit_blocks_report_instead_of_omitting_limitations(tmp_path):
    out=tmp_path/"experiments"/a.ROOT_NAME;save_panel(out)
    (out/"audit/condition_construction_audit.json").unlink()
    with pytest.raises(FileNotFoundError):a.analyze(out)
    assert not (out/"analysis").exists()
