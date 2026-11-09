from pathlib import Path
p=Path("/mnt/e/MeshFlow-Control/tools/package_recipe_review.py")
s=p.read_text(encoding="utf-8-sig")
s=s.replace('ROOT_NAME="chair_condition_recipe_factorial_v1"','ROOT_NAME="chair_condition_recipe_factorial_v1"\nVERSION="recipe_factorial_review_bundle_v2"')
s=s.replace('for p in (run/"training").glob("*/completed.json"):add(p)\n    for p in (run/"training").glob("*/initialization_audit.json"):add(p)\n    for p in (run/"training").glob("*/attempts/*/final_status.json"):add(p)', '''for training in (run/"training",run/"confirmation/training"):
        for pattern in ("*/completed.json","*/initialization_audit.json","*/attempts/*/final_status.json"):
            for p in training.glob(pattern):add(p)
    for name in ("precompute_receipt.json","precompute_progress.json"):
        add(run/"confirmation/streams"/name)
    for p in (run/"confirmation/registrations").glob("*.json"):add(p)
    for p in (run/"confirmation/streams").glob("*.json"):
        if p.name in ("precompute_receipt.json","precompute_progress.json"):continue
        excluded.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason="Full independent training stream retained locally"))
    for p in (run/"confirmation/streams/paired_inputs").glob("*.json"):
        excluded.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason="Full independent paired inputs retained locally"))''')
s=s.replace('add(source_repo/"tools/package_recipe_review.py","additional_source/tools/package_recipe_review.py")','''add(source_repo/"tools/package_recipe_review.py","additional_source/tools/package_recipe_review.py")
    add(source_repo/"tools/package_recipe_review_v2.py","additional_source/tools/package_recipe_review_v2.py")''')
s=s.replace('schema="recipe_factorial_review_bundle_v1"','schema=VERSION')
s=s.replace('if (run/"phases/course/evaluation/per_output.csv").exists() and "phases/course/evaluation/per_output.csv" not in names:raise ValueError("Course rows missing")','''for phase in ("course","confirm"):
            if (run/"phases"/phase/"evaluation/per_output.csv").exists() and "phases/"+phase+"/evaluation/per_output.csv" not in names:raise ValueError(phase+" rows missing")
        if final.get("confirmation_status")=="COMPLETE" and "phases/confirm/evaluation/per_output.csv" not in names:
            raise ValueError("Completed confirmation evidence missing")''')
Path("/mnt/e/MeshFlow-Control/tools/package_recipe_review_v2.py").write_text(s,encoding="utf8")
print("Prepared separate package v2; no packaging yet")
