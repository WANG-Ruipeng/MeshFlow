"""Synthetic CPU tests only: metadata/renderer adapter, no model/scorer/query generation."""
import copy
import importlib
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
c=importlib.import_module("compare_recipe_curriculum")

def table_fixture():
    qs=[]
    for parent in range(8):
        for ratio in (.2,.4):
            for n in (0,1,6,7):
                qs.append(dict(request_id=f"p{parent}_C{int(100*ratio)}__n{n}",parent_id=f"p{parent}",
                    condition_id=f"p{parent}_C{int(100*ratio)}",candidate=n,ratio=ratio,
                    condition_cohort="A20" if ratio==.2 else "ORIGINAL6" if parent<6 else "EXTENDED2",
                    N=128,K=round(ratio*128),input_sha256=f"input-{parent}-{ratio}-{n}"))
    def rows(labels):
        return [dict(q,label=label,status="PASS",contracts={"C_preserved":True},
                     identity={"input_sha256":q["input_sha256"]},query_sha256="query-"+q["condition_id"],
                     registered_free_faces=q["N"]-q["K"],registered_C_queries=2048,
                     query_weight_denominator=1.25,L0=9.2) for label in labels for q in qs]
    mr=dict(labels=list(c.ev.LABELS),expected_outputs=384,requests_sha256="same")
    cr=dict(labels=list(c.COURSES),expected_outputs=128,requests_sha256="same",kind="course")
    md=dict(status="COMPLETE",registered=384,scored=384);cd=dict(status="COMPLETE",registered=128,scored=128)
    return mr,cr,qs,rows(c.ev.LABELS),rows(c.COURSES),md,cd

def test_complete192_cells_share_original_inputs_and_query():
    args=table_fixture();snapshot=copy.deepcopy(args)
    by=c.validate_saved_panel(*args)
    assert len(by)==192 and CounterLabels(by)=={l:64 for l in c.LABELS}
    assert args==snapshot

def CounterLabels(by):
    from collections import Counter
    return Counter(k[0] for k in by)

@pytest.mark.parametrize("damage",["missing","duplicate","input","query","status","partial_receipt","wrong_labels"])
def test_incomplete_or_mismatched_saved_evidence_rejected(damage):
    a=list(table_fixture())
    if damage=="missing":a[4].pop()
    elif damage=="duplicate":a[4][-1]=copy.deepcopy(a[4][0])
    elif damage=="input":a[4][0]["identity"]["input_sha256"]="changed"
    elif damage=="query":a[4][0]["query_sha256"]="changed"
    elif damage=="status":a[4][0]["contracts"]["C_preserved"]=False
    elif damage=="partial_receipt":a[6]["scored"]=127
    elif damage=="wrong_labels":a[1]["labels"]=[c.BASELINE,c.COURSES[0]]
    with pytest.raises(ValueError):c.validate_saved_panel(*a)

def test_original_bbox_overflow_triggers_even_inside_padded_cube():
    tri=np.array([[[0,0,0],[1.02,0,0],[0,1,0]]],float)
    audit=c.compare_bounds(tri,[[0,0,0],[1,1,1]])
    assert audit["outside_original_bounds"] and not audit["outside_padded_axis_cube"]
    tri[0,1,0]=2
    assert c.compare_bounds(tri,[[0,0,0],[1,1,1]])["outside_padded_axis_cube"]
    with pytest.raises(ValueError):c.compare_bounds(tri,[[1,0,0],[0,1,1]])

def bundle_fixture(overflow=False):
    qs=table_fixture()[2]
    tri=np.array([[[.2,.2,.2],[.8,.2,.2],[.2,.8,.2]]],np.float32)
    loaded={(l,q["request_id"]):tri.copy() for l in c.LABELS for q in qs}
    if overflow:loaded[c.COURSES[0],qs[0]["request_id"]][0,1,0]=1.02
    return dict(requests=qs,loaded=loaded,inputs={q["request_id"]:tri.copy() for q in qs},
                cameras={q["condition_id"]:{"bounds":[[0.,0.,0.],[1.,1.,1.]],"raw_hashes":{"baseline":"frozen"}} for q in qs})

def renderer_spy(calls):
    def render(path,columns,bounds,title,neutral):
        calls.append(dict(path=Path(path),bounds=copy.deepcopy(bounds),title=title,neutral=neutral,columns=[x[0] for x in columns]))
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"synthetic-render-placeholder")
    return render

@pytest.mark.parametrize("overflow,expected_extra",[(False,0),(True,8)])
def test_exact128_primary_and_complete_affected_condition_supplement(tmp_path,overflow,expected_extra):
    bundle=bundle_fixture(overflow);before=copy.deepcopy(bundle["cameras"]);calls=[]
    manifest=c.render_bundle(bundle,tmp_path,renderer_spy(calls))
    assert manifest["page_count"]==128 and manifest["supplemental_page_count"]==expected_extra
    assert len(calls)==128+expected_extra and bundle["cameras"]==before
    primary=[r for r in calls if "fixed_main" in r["path"].parts]
    assert len(primary)==128 and all(r["bounds"]==[[0.,0.,0.],[1.,1.,1.]] for r in primary)
    assert all(r["columns"][0]=="Input C" and set(r["columns"][1:])=={"Method 1","Method 2","Method 3"} for r in calls)
    extra=[r for r in calls if "supplement_complete" in r["path"].parts]
    if overflow:
        assert {r["path"].stem.rsplit("__n",1)[1] for r in extra}=={"0","1","6","7"}
        assert {r["neutral"] for r in extra}=={False,True}
        assert all(r["bounds"]==extra[0]["bounds"] for r in extra)
        assert extra[0]["bounds"][1][0]==float(np.float32(1.02))
        assert all("FRAMING LIMIT" in r["title"] for r in primary if "p0_C20" in r["path"].name)
    # Resume verifies bytes/identities and must never call the renderer again.
    again=c.render_bundle(bundle,tmp_path,lambda *a:pytest.fail("Repeated render"))
    assert again==manifest
    html=(tmp_path/"INDEX.html").read_text()
    assert html.count("<a href=")==128+expected_extra

def test_missing_noise_refuses_complete_condition(tmp_path):
    bundle=bundle_fixture();bundle["requests"].pop()
    with pytest.raises(ValueError,match="All four noises"):
        c.render_bundle(bundle,tmp_path,renderer_spy([]))
    assert not list(tmp_path.iterdir())

def test_changed_or_partial_page_is_not_silently_overwritten(tmp_path):
    p=tmp_path/"page.png";p.write_bytes(b"orphan")
    with pytest.raises(RuntimeError,match="partial render"):
        c.render_registered_page(p,[],[],False,False,{},lambda *a:pytest.fail("overwrite"))
    p.unlink();spy=renderer_spy([])
    c.render_registered_page(p,[],[[0,0,0],[1,1,1]],"x",False,{"input":"one"},spy)
    with pytest.raises(ValueError,match="changed"):
        c.render_registered_page(p,[],[],False,False,{"input":"two"},spy)

def test_compare_requires_collect_barrier_before_writing(tmp_path,monkeypatch):
    def block(*args):raise ValueError("Course scoring incomplete")
    monkeypatch.setattr(c,"collect",block)
    with pytest.raises(ValueError,match="incomplete"):c.compare(tmp_path)
    assert not list(tmp_path.iterdir())

def test_cpu_adapter_has_no_model_scoring_or_query_creation_calls():
    import ast
    tree=ast.parse(Path(c.__file__).read_text())
    forbidden={"sample_label","completed_rows","score_one","prepare_queries","sample_mesh_surface","gaussian","train_recipe"}
    called={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else "" for n in ast.walk(tree) if isinstance(n,ast.Call)}
    assert not called&forbidden


@pytest.mark.parametrize("damage",[None,"raw","checkpoint","query"])
def test_score_bound_to_exact_existing_raw_checkpoint_and_queries(damage):
    q=table_fixture()[2][0]
    r=dict(q,label=c.BASELINE,raw_sha256="raw",ply_sha256="ply",result_sha256="result",checkpoint_sha256="weights")
    score=dict(label=c.BASELINE,query_sha256="query",identity=dict(raw_sha256="raw",ply_sha256="ply",result_sha256="result",checkpoint_sha256="weights",query_sha256="query"))
    binding={"checkpoint_sha256":"weights"}
    if damage=="raw":score["identity"]["raw_sha256"]="changed"
    elif damage=="checkpoint":binding["checkpoint_sha256"]="changed"
    elif damage=="query":score["identity"]["query_sha256"]="changed"
    if damage:
        with pytest.raises(ValueError):c.verify_saved_binding(q,r,score,binding,"query")
    else:c.verify_saved_binding(q,r,score,binding,"query")
