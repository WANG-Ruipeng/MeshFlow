import csv,json,hashlib,datetime
from pathlib import Path
r=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
out={}
for a in ("C20_40_MIX","C40_20_MIX"):
 p=r/"training"/a/"attempts/initial/training_log.csv"
 with p.open() as f: rows=[x for x in csv.DictReader(f) if int(x["step"]) in (1999,2000,2001,2002)]
 assert len(rows)==4
 for x in rows:
  k=int(x["step"]);is20=(a=="C20_40_MIX")==(k<=2000)
  assert int(x["generator_adamw_step"])==k and float(x["lambda_value"])==.25
  assert int(x["ratio20_samples"])==(8 if is20 else 0)
  assert int(x["ratio40_samples"])==(0 if is20 else 8)
  assert float(x["total_loss"])==float(x["FM"])
 out[a]=dict(source_path=str(p),rows=rows)
result=dict(status="PASS",scope="Actual immutable completed log rows at first curriculum transition; no model/checkpoint load.",checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=out)
p=r/"audit/curriculum_boundary_2000_actual.json"
if p.exists():raise RuntimeError("Already saved")
p.write_text(json.dumps(result,indent=2),encoding="utf8")
print(json.dumps(dict(status="PASS",path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
