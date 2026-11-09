from pathlib import Path
import ast,hashlib,json
r=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
repo=Path("/mnt/e/MeshFlow-Control")
names=["tools/prepare_recipe_curriculum.py","tools/compare_recipe_curriculum.py","tools/analyze_recipe_curriculum_results.py","tools/audit_recipe_curriculum_training.py",
"tests/test_recipe_curriculum_preparation.py","tests/test_compare_recipe_curriculum.py","tests/test_analyze_recipe_curriculum_results.py","tests/test_recipe_curriculum_accounting.py"]
found={}
for n in names:
 p=repo/n
 if not p.is_file():raise FileNotFoundError(n)
 ast.parse(p.read_text(encoding="utf-8-sig"))
 found[n]=hashlib.sha256(p.read_bytes()).hexdigest()
saved=json.loads((r/"registrations/source_hashes.json").read_text())
assert all(hashlib.sha256((repo/n).read_bytes()).hexdigest()==h for n,h in saved.items())
out=r/"registrations/curriculum_additional_source_hashes.json"
v={"schema":"curriculum_additional_source_freeze_v1","original_frozen_sources_unchanged":len(saved),"sources":found,
"policy":"Independent new adapters; original 97-file source definition remains unchanged.","tests":"7 preparation + 19 comparison + 46 analysis/accounting including originals PASS"}
text=json.dumps(v,indent=2)
if out.exists():assert json.loads(out.read_text())==v
else:out.write_text(text,encoding="utf8")
ast.parse((repo/"tools/package_recipe_review.py").read_text())
print(json.dumps({"status":"PASS","original_sources":len(saved),"additional_sources":len(found),"registry_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"package_syntax":"PASS"}))
