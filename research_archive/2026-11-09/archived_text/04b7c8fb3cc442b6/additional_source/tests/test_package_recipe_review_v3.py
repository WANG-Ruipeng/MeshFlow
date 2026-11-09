from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import pytest
_spec=importlib.util.spec_from_file_location("review_bundle",Path(__file__).parents[1]/"tools/package_recipe_review_v3.py")
m=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(m)

def write_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj),encoding="utf8")
def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8-sig",newline="")as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def fixture(tmp_path):
    run=tmp_path/"chair_condition_recipe_factorial_v1"
    combined=[];models=[];query_paths=[]
    for phase,labels in m.PHASE_LABELS.items():
        noise=(8,9,10,11)if phase=="confirm"else(0,1,6,7)
        requests=[dict(request_id=f"p{p}_C{ratio}_n{n}")for p in range(8)for ratio in(20,40)for n in noise]
        request_path=run/("phases/confirm/evaluation/requests.json"if phase=="confirm"else"evaluation/requests.json")
        write_json(request_path,dict(rows=requests))
        sha=m.sha(request_path)
        query_path=run/"evaluation/query_registrations"/(sha+".json")
        write_json(query_path,dict(status="FROZEN",identity=dict(requests_sha256=sha),rows=[{}]*16))
        query_paths.append(query_path)
        write_json(run/"phases"/phase/"registration.json",dict(labels=list(labels),requests_path=str(request_path),requests_sha256=sha))
        rows=[dict(label=label,request_id=q["request_id"],scientific_value="0.12300")for label in labels for q in requests]
        write_csv(run/"phases"/phase/"evaluation/per_output.csv",rows)
        write_json(run/"phases"/phase/"evaluation/scoring_receipt.json",dict(status="COMPLETE",scored=len(rows),registered=len(rows)))
        combined.extend(dict(evidence_phase=phase,model_instance_id=phase+"/"+r["label"],**r)for r in rows)
        models.extend(dict(instance=phase+"/"+label,phase=phase,label=label,registered_outputs=64)for label in labels)
        visual_labels=labels+(("R2_MIX_H_ALL",)if phase=="course"else())
        visual=[dict(model_label=label,request_id=q["request_id"],observation="Synthetic original note")for label in visual_labels for q in requests]
        root=run/"audit/visual_review"/(phase+"_complete");write_csv(root/"per_output.csv",visual)
        if phase=="main":
            coverage=dict(status="COMPLETE",expected_and_actual=dict(unique_outputs=384,requests=64,original_pngs=128),all_128_original_png_hashes="PASS")
        else:
            coverage=dict(status="PASS",requests=64,unique_outputs=len(visual),primary_pngs=128,
                all_image_bytes_sha256_verified=True,supplemental_pngs=56 if phase=="course"else 0)
            if phase=="course":coverage.update(reused_R2_outputs=64,new_course_outputs=128)
        write_json(root/"coverage.json",coverage)
    write_csv(run/"analysis/recipe_factorial_delivery_tables_v1/complete_per_output_640.csv",combined)
    write_json(run/"analysis/model_index.json",dict(status="COMPLETE",instances=models))
    write_json(run/"evaluation/confirmation_plan.json",dict(frozen=True))
    return run

def test_complete_future_fixture_counts_and_required_members(tmp_path):
    run=fixture(tmp_path);result=m.validate_final_evidence(run)
    assert result["phase_rows"]=={"main":384,"course":128,"confirm":128}
    assert result["combined_rows"]==640 and result["model_instances"]==10
    assert result["visual_rows"]=={"main":384,"course":192,"confirm":128}
    required=set(result["required_archive_members"])
    assert "evaluation/requests.json"in required and "evaluation/confirmation_plan.json"in required
    assert len([p for p in required if p.startswith("evaluation/query_registrations/")])==2
    assert "source_snapshot/src/meshflow_control/evaluation/control.py"in required

@pytest.mark.parametrize("case",["phase_missing","phase_duplicate","wrong_combined_string","combined_duplicate","model9",
    "visual_course128","visual_duplicate","coverage_pending","query_missing","receipt_incomplete","wrong_instance"])
def test_final_gates_reject_missing_or_miscounted_evidence(tmp_path,case):
    run=fixture(tmp_path)
    if case in("phase_missing","phase_duplicate"):
        p=run/"phases/confirm/evaluation/per_output.csv";rows=m.read_csv(p)
        if case=="phase_missing":rows.pop()
        else:rows[-1]=dict(rows[0])
        write_csv(p,rows)
    elif case in("wrong_combined_string","combined_duplicate","wrong_instance"):
        p=run/"analysis/recipe_factorial_delivery_tables_v1/complete_per_output_640.csv";rows=m.read_csv(p)
        if case=="wrong_combined_string":rows[0]["scientific_value"]="0.123"
        elif case=="wrong_instance":rows[0]["model_instance_id"]="other/model"
        else:rows[-1]=dict(rows[0])
        write_csv(p,rows)
    elif case=="model9":
        p=run/"analysis/model_index.json";obj=m.read(p);obj["instances"].pop();write_json(p,obj)
    elif case in("visual_course128","visual_duplicate"):
        p=run/"audit/visual_review/course_complete/per_output.csv";rows=m.read_csv(p)
        if case=="visual_course128":rows=rows[:128]
        else:rows[-1]=dict(rows[0])
        write_csv(p,rows)
    elif case=="coverage_pending":
        p=run/"audit/visual_review/confirm_complete/coverage.json";obj=m.read(p);obj["status"]="PENDING";write_json(p,obj)
    elif case=="query_missing":next((run/"evaluation/query_registrations").glob("*.json")).unlink()
    else:
        p=run/"phases/confirm/evaluation/scoring_receipt.json";obj=m.read(p);obj["registered"]=127;write_json(p,obj)
    with pytest.raises((ValueError,OSError)):m.validate_final_evidence(run)

def test_only_small_input_definitions_selected(tmp_path):
    run=fixture(tmp_path);paths=m.evaluation_metadata_paths(run)
    (run/"evaluation/inputs").mkdir();(run/"evaluation/inputs/noise.npz").write_bytes(b"not-included")
    (run/"evaluation/query_banks").mkdir();(run/"evaluation/query_banks/points.npz").write_bytes(b"not-included")
    assert set(m.evaluation_metadata_paths(run))==set(paths)
    assert all(p.suffix==".json"for p in paths)

@pytest.mark.parametrize("path",["FINAL_REPORT_DRAFT_ZH.md","analysis/draft/notes.md","audit/old_DRAFT.txt",
    "weights/model.pt","raw/output.npz","raw/mesh.ply","checkpoint.safetensors"])
def test_drafts_and_raw_weights_excluded(path):
    assert not m.selected_file_allowed(path)

def zip_fixture(tmp_path,wrong_content=False,omit=False):
    data=b"actual scoring source"
    metadata=dict(files=[dict(archive_path="source_snapshot/score.py",bytes=len(data),sha256=hashlib.sha256(data).hexdigest())])
    generated={"BUNDLE_MANIFEST.json":json.dumps(metadata).encode(),"code.diff":b"diff","READ_FIRST.txt":b"readme"}
    target=tmp_path/"review.zip"
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED)as z:
        if not omit:z.writestr("source_snapshot/score.py",b"changed source bytes"if wrong_content else data)
        for name,value in generated.items():z.writestr(name,value)
    return target,metadata,generated

def test_all_zip_member_hashes_and_crc(tmp_path):
    args=zip_fixture(tmp_path)
    result=m.verify_archive(*args,{"source_snapshot/score.py","code.diff"})
    assert result==dict(all_members_sha256_verified=True,crc_verified=True,verified_members=4)

@pytest.mark.parametrize("case",["changed","missing","mandatory"])
def test_archive_validation_catches_crc_valid_bad_member_or_missing(tmp_path,case):
    args=zip_fixture(tmp_path,wrong_content=case=="changed",omit=case=="missing")
    mandatory={"missing_mandatory.csv"}if case=="mandatory"else{"code.diff"}
    with pytest.raises(ValueError):m.verify_archive(*args,mandatory)

def test_v3_explicit_license_and_self_source_are_mandatory():
    assert "source_external_snapshot/third_party/trimesh/LICENSE" in m.CORE_MANDATORY
    assert "additional_source/tools/package_recipe_review_v3.py" in m.CORE_MANDATORY
    assert m.selected_file_allowed("source_external_snapshot/third_party/trimesh/LICENSE")

def test_v3_all_nonmain_validation_functions_remain_identical_to_v2():
    import ast
    tools=Path(__file__).parents[1]/"tools"
    trees=[ast.parse((tools/name).read_text()) for name in ("package_recipe_review_v2.py","package_recipe_review_v3.py")]
    functions=[{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef) and n.name!="main"} for t in trees]
    assert functions[0]==functions[1]
    source=(tools/"package_recipe_review_v3.py").read_text()
    assert 'add(run/"source_external_snapshot/third_party/trimesh/LICENSE")' in source
    assert 'add(source_repo/"tools/package_recipe_review_v3.py","additional_source/tools/package_recipe_review_v3.py")' in source
    assert "deduplicate" not in source
