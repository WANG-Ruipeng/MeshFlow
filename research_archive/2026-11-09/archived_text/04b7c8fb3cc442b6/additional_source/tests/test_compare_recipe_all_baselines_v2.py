"""V2 registered-identity counterexamples on synthetic saved tables only."""
import copy
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
a=module("all_baselines_v2",ROOT/"tools/compare_recipe_all_baselines_v2.py")
f=module("course_fixtures_v2",ROOT/"tests/test_analyze_recipe_curriculum_results.py")

def panel():
    values=list(f.panel());requests=values[2];rows=values[3]+values[4]
    queries={}
    for q in requests:
        q["input_sha256"]=q["request_id"]
        q["C_sha256"]="C:"+q["condition_id"]
        queries[q["condition_id"]]=dict(
            condition_id=q["condition_id"],parent_id=q["parent_id"],ratio=q["ratio"],
            C_sha256=q["C_sha256"],query_sha256=q["condition_id"],query_count=2048,
            all_four_noises_share_same_bank=True,weight_denominator=2048,identity={"L0":9.})
    for row in rows:
        row["identity"].update(request_id=row["request_id"],label=row["label"],query_sha256=row["query_sha256"])
    registration=dict(status="FROZEN",identity={"requests_sha256":"shared"},rows=list(queries.values()))
    return values,rows,registration

def test_full512_registered_bindings_and_frozen_v1():
    p,rows,q=panel()
    assert len(a.v1.validate_rows(*p))==512
    proof=a.registered_bindings(rows,p[2],q,"shared")
    assert proof["input_query_binding"]=="PASS" and proof["registered_outputs"]==512
    assert a.base.digest(a.v1.__file__)==a.V1_SHA

def test_shared_wrong_input_passes_horizontal_v1_but_is_blocked_by_registration():
    p,rows,q=panel()
    for r in rows:r["identity"]["input_sha256"]="same-wrong-all-models"
    assert len(a.v1.validate_rows(*p))==512
    with pytest.raises(ValueError,match="registered request"):
        a.registered_bindings(rows,p[2],q,"shared")

def test_shared_wrong_query_passes_horizontal_v1_but_is_blocked_by_registration():
    p,rows,q=panel()
    for r in rows:
        r["query_sha256"]="same-wrong-query"
        r["identity"]["query_sha256"]="same-wrong-query"
    assert len(a.v1.validate_rows(*p))==512
    with pytest.raises(ValueError,match="registered query"):
        a.registered_bindings(rows,p[2],q,"shared")

@pytest.mark.parametrize("change",["inner_query","inner_label","query_C","denominator","query_registration","missing"])
def test_binding_counterexamples(change):
    p,rows,q=panel()
    if change=="inner_query":rows[0]["identity"]["query_sha256"]="different-inner"
    elif change=="inner_label":rows[0]["identity"]["label"]="wrong"
    elif change=="query_C":q["rows"][0]["C_sha256"]="wrong-C"
    elif change=="denominator":
        for row in rows:row["query_weight_denominator"]=1
    elif change=="query_registration":q["identity"]["requests_sha256"]="wrong-requests"
    else:rows.pop()
    with pytest.raises(ValueError):a.registered_bindings(rows,p[2],q,"shared")

def save(tmp_path):
    out=f.save_panel(tmp_path/"experiments"/a.base.ROOT_NAME)
    qp=out/"evaluation/requests.json";requests=a.base.read_json(qp)["rows"]
    queries={}
    for q in requests:
        q["input_sha256"]=q["request_id"];q["C_sha256"]="C:"+q["condition_id"]
        cid=q["condition_id"]
        if cid not in queries:
            path=out/"evaluation/query_banks"/cid/"synthetic.query"
            path.parent.mkdir(parents=True);path.write_text("synthetic fixed query "+cid)
            queries[cid]=dict(condition_id=cid,parent_id=q["parent_id"],ratio=q["ratio"],
                C_sha256=q["C_sha256"],query_path=str(path),query_sha256=a.base.digest(path),
                query_count=2048,all_four_noises_share_same_bank=True,weight_denominator=2048,identity={"L0":9.})
    a.base.write_json(qp,dict(rows=requests));sha=a.base.digest(qp)
    for phase in ("main","course"):
        rp=out/"phases"/phase/"registration.json";reg=a.base.read_json(rp)
        reg["requests_sha256"]=sha;a.base.write_json(rp,reg)
        scores=out/"phases"/phase/"evaluation/per_output.json";data=a.base.read_json(scores)
        for r in data["rows"]:
            r["query_sha256"]=queries[r["condition_id"]]["query_sha256"]
            r["identity"].update(request_id=r["request_id"],label=r["label"],query_sha256=r["query_sha256"])
        a.base.write_json(scores,data)
    a.base.write_json(out/"evaluation/query_registrations"/(sha+".json"),
        dict(status="FROZEN",identity={"requests_sha256":sha},rows=list(queries.values())))
    return out,queries

def test_guarded_end_to_end_preserves_frozen_aggregation_and_input_files(tmp_path):
    out,queries=save(tmp_path);before={p:a.base.digest(p) for p in out.rglob("*") if p.is_file()}
    receipt=a.analyze(out)
    assert receipt["status"]=="COMPLETE" and receipt["preserved_outputs"]==512
    assert receipt["new_model_calls"]==receipt["new_scoring_calls"]==0
    guard=a.base.read_json(receipt["binding_receipt_path"])
    assert guard["binding"]["registered_query_conditions"]==16
    assert not guard["changed_metric_definitions"] and not guard["changed_numerical_aggregation"]
    assert all(a.base.digest(p)==sha for p,sha in before.items())
    assert a.analyze(out)==receipt
    # All table arithmetic is the original frozen implementation.
    summary=a.base.read_csv(Path(receipt["analysis_dir"])/"model_summary.csv")
    assert len(summary)==8*2*3*5

def test_actual_registered_query_hash_changed_blocks_before_aggregation(tmp_path):
    out,queries=save(tmp_path)
    Path(next(iter(queries.values()))["query_path"]).write_text("tampered")
    with pytest.raises(ValueError,match="Actual registered query"):a.analyze(out)
    assert not (out/"analysis").exists()

def test_pending_does_not_create_analysis(tmp_path):
    out=tmp_path/"experiments"/a.base.ROOT_NAME
    assert a.analyze(out)["status"]=="PENDING"
    assert not out.exists()
