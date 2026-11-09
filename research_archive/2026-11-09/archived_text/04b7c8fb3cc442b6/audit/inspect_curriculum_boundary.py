import csv,json
from pathlib import Path
r=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
out={}
for a in ("C20_40_MIX","C40_20_MIX"):
 p=r/"training"/a/"attempts/initial/training_log.csv"
 with p.open() as f:
  reader=csv.DictReader(f)
  rows=[x for x in reader if int(x["step"]) in (1999,2000,2001,2002)]
 out[a]=rows
print(json.dumps(out))
