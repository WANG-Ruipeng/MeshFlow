"""Synthetic tables only; no real run, checkpoint, scoring or model access."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result
a=module("all_baselines",ROOT/"tools/compare_recipe_all_baselines.py")
fixtures=module("course_table_fixtures",ROOT/"tests/test_analyze_recipe_curriculum_results.py")
METRIC=fixtures.METRIC

def panel():
    values=list(fixtures.panel())
    for row in values[3]+values[4]:
        row["metrics"]["free_angle_p05_degrees"]=row["metrics"][METRIC]
        row["metrics"]["descriptive_count"]=123.
    return values

def rows():
    p=panel()
    return p[2],a.validate_rows(*p)

def test_all512_and_frozen_globals_are_unchanged():
    old_labels=a.base.LABELS;old_course=a.course.LABELS;old_pairs=a.course.PAIRS
    qs,rr=rows()
    result=a.aggregate(rr,qs)
    assert len(rr)==512 and len({(r["label"],r["request_id"]) for r in rr})==512
    assert a.base.LABELS==old_labels and a.course.LABELS==old_course and a.course.PAIRS==old_pairs
    assert len(old_labels)==6 and len(old_course)==3
    assert all(r["registered_outputs"]==512 for r in result["metric_catalog"])
    assert {r["label"] for r in result["model_summary"]}==set(a.LABELS)

def test_only_six_course_vs_practical_baselines_and_pair_orientation():
    qs,rr=rows();result=a.aggregate(rr,qs)
    assert set((r["first"],r["second"]) for r in result["paired_noise_metrics"])==set(a.PAIRS)
    assert len(a.PAIRS)==6
    for item in result["paired_parent_summary"]:
        if item["metric"]==METRIC:
            expected=-3 if item["first"]=="C20_40_MIX" else 1
            assert item["mean"]==pytest.approx(expected)
            assert item["improved_parents"]==(item["expected"] if expected<0 else 0)
            assert item["worsened_parents"]==(item["expected"] if expected>0 else 0)
            assert item["tied_parents"]==0
            assert item["delta_direction"]=="first minus second"

def test_both_ratios_full_original_extended_equal_parent_summary():
    qs,rr=rows();result=a.aggregate(rr,qs)
    summaries=[r for r in result["model_summary"] if r["metric"]==METRIC]
    assert len(summaries)==8*2*3
    for item in summaries:
        expected=a.SCOPES[item["parent_stratum"]]
        assert item["expected"]==item["available"]==expected
        assert len(item["parent_ids"])==expected
        assert len(set(item["parent_ids"]))==expected
    # Noise mean=(0+1+6+7)/4/10=.35; parent mean original6=2.5, extended2=6.5.
    for item in summaries:
        if item["label"]=="START5000":
            offset={"ALL":3.5,"ORIGINAL6":2.5,"EXTENDED2":6.5}[item["parent_stratum"]]
            assert item["mean"]==pytest.approx(11+offset+.35)
    assert {r["ratio"] for r in summaries if r["parent_stratum"]=="EXTENDED2"}=={.2,.4}

def test_metric_directions_higher_unknown_and_NA_are_preserved():
    qs,rr=rows();result=a.aggregate(rr,qs)
    target=next(r for r in result["paired_parent_summary"] if r["first"]=="C20_40_MIX"
                and r["second"]=="START5000" and r["metric"]=="free_angle_p05_degrees"
                and r["ratio"]==.2 and r["parent_stratum"]=="ALL")
    assert target["direction"]=="higher" and target["worsened_parents"]==8
    for item in result["paired_parent_summary"]:
        if item["metric"]=="descriptive_count":
            assert item["improved_parents"] is None and item["worsened_parents"] is None
            assert item["exact_zero_count"]==item["expected"]
        if item["metric"] in ("all_NA","status_field","boolean_field"):
            assert not item["numeric_summary"] and item["mean"] is None
    catalog={r["metric"]:r for r in result["metric_catalog"]}
    assert catalog["all_NA"]["value_status_counts"]=={"NA_NULL":512}
    assert not catalog["status_field"]["numeric_summary"]
    assert not catalog["boolean_field"]["numeric_summary"]

def test_missing_one_noise_keeps_fixed_parent_and_scope_denominators():
    qs,rr=rows()
    target=next(r for r in rr if r["label"]=="C20_40_MIX" and r["parent_id"]=="p0"
                and r["ratio"]==.2 and r["candidate"]==0)
    target["metrics"][METRIC]=None
    result=a.aggregate(rr,qs)
    parent=next(r for r in result["paired_parent_metrics"] if r["first"]=="C20_40_MIX"
                and r["second"]=="START5000" and r["parent_id"]=="p0" and r["ratio"]==.2 and r["metric"]==METRIC)
    assert parent["noise_expected"]==4 and parent["noise_available"]==3 and parent["delta"] is None
    summary=next(r for r in result["paired_parent_summary"] if r["first"]=="C20_40_MIX"
                 and r["second"]=="START5000" and r["ratio"]==.2 and r["metric"]==METRIC and r["parent_stratum"]=="ALL")
    assert summary["mean"] is None and summary["expected"]==8 and summary["available"]==7
    assert summary["improved_parents"]==7 and summary["missing_parents"]==1

@pytest.mark.parametrize("field",["input","query","missing"])
def test_eight_label_bindings_and_missing_reject(field):
    p=panel()
    # START was outside the old course/R2-only identity checks.
    row=next(r for r in p[3] if r["label"]=="START5000")
    if field=="input":row["identity"]["input_sha256"]="changed"
    elif field=="query":row["query_sha256"]="changed"
    else:p[3].remove(row)
    with pytest.raises(ValueError):a.validate_rows(*p)

def test_end_to_end_synthetic_readonly_and_resume(tmp_path):
    out=fixtures.save_panel(tmp_path/"experiments"/a.base.ROOT_NAME)
    before={p:a.base.digest(p) for p in out.rglob("*") if p.is_file()}
    receipt=a.analyze(out)
    assert receipt["status"]=="COMPLETE" and receipt["preserved_outputs"]==512
    assert receipt["new_model_calls"]==receipt["new_scoring_calls"]==0
    assert receipt["composite_score"] is None and receipt["automatic_selection"] is False
    assert all(a.base.digest(p)==h for p,h in before.items())
    dest=Path(receipt["analysis_dir"])
    assert len(a.base.read_json(dest/"all_metrics.json")["rows"])==512
    assert len(a.base.read_csv(dest/"per_output.csv"))==512
    assert a.analyze(out)==receipt
    (dest/"README.md").write_text("tampered",encoding="utf8")
    with pytest.raises(ValueError,match="Immutable"):a.analyze(out)

def test_pending_has_no_output_and_no_real_data_reads(tmp_path):
    out=tmp_path/"experiments"/a.base.ROOT_NAME
    receipt=a.analyze(out)
    assert receipt["status"]=="PENDING" and receipt["missing_files"]
    assert not out.exists()

def test_actual_course_request_file_hash_must_match(tmp_path):
    out=fixtures.save_panel(tmp_path/"experiments"/a.base.ROOT_NAME)
    regpath=out/"phases/course/registration.json";reg=a.base.read_json(regpath)
    bad=out/"evaluation/changed_requests.json";a.base.write_json(bad,{"rows":[]})
    reg["requests_path"]=str(bad);a.base.write_json(regpath,reg)
    with pytest.raises(ValueError,match="requests file differs"):a.analyze(out)

def test_direction_counts_tie_and_uncertainty_are_metric_only():
    result=a.direction_counts([-2,0,3,None],"lower_uncertainty")
    assert result["expected_parents"]==4 and result["comparable_parents"]==3
    assert (result["improved_parents"],result["tied_parents"],result["worsened_parents"])==(1,1,1)
    assert result["quality_certificate"] is False
