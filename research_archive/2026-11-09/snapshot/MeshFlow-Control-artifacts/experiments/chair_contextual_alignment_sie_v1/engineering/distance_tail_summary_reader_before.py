"""Report-only p95 of already-saved frozen 2048-point distances; no new queries/model calls."""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];D=R/"analysis/distance_tails";D.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,rows):
 with p.open("w",newline="",encoding="utf8") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[];src={}
for p in sorted((R/"evaluation/rows").glob("*/*.json")):
 r=json.loads(p.read_text());m=r["metrics"]
 x={k:r[k] for k in ("request_id","label","panel","cohort","parent_id","condition_id","candidate")}
 x.update(GT_to_F_p95_2048_L0=None,F_to_GT_p95_2048_L0=None,status="NA_NOT_APPLICABLE",query_count=None)
 if r["panel"]=="P1":
  v=Path(r["quality_vectors_path"]);h=sha(v);assert r["hashes"][str(v)]==h
  src[str(v)]=h
  with np.load(v,allow_pickle=False) as a:
   for key,metric in (("GT_to_F_d2_2048","GT_to_F"),("F_to_GT_d2_2048","F_to_GT")):
    d=a[key];assert d.shape==(2048,) and np.isfinite(d).all() and (d>=0).all()
    value=float(np.mean(d)/float(r["L0"])**2)
    assert np.isclose(value,m[metric+"_MSE2048_L0"],rtol=1e-12,atol=1e-14)
    x[metric+"_p95_2048_L0"]=float(np.quantile(np.sqrt(d)/float(r["L0"]),.95,method="linear"))
  x.update(status="PASS",query_count=2048)
 rows.append(x)
assert len(rows)==192 and sum(x["status"]=="PASS" for x in rows)==128
write(D/"per_output_distance_p95.csv",rows)
parents=[];groups=defaultdict(list)
for r in rows:
 for cohort in (("ALL",r["cohort"]) if r["panel"]=="P1" else ("OPEN",)):
  groups[(r["label"],r["panel"],cohort,r["parent_id"])].append(r)
for (label,panel,cohort,parent),g in sorted(groups.items()):
 x=dict(label=label,panel=panel,cohort=cohort,parent_id=parent,noises=len(g),status="PASS" if panel=="P1" else "NA_NOT_APPLICABLE")
 for k in ("GT_to_F_p95_2048_L0","F_to_GT_p95_2048_L0"):x[k]=float(np.mean([r[k] for r in g])) if panel=="P1" else None
 parents.append(x)
write(D/"per_parent_distance_p95.csv",parents)
groups=defaultdict(list);summary=[]
for r in parents:groups[(r["label"],r["panel"],r["cohort"])].append(r)
for (label,panel,cohort),g in sorted(groups.items()):
 x=dict(label=label,panel=panel,cohort=cohort,parent_count=len(g),status="PASS" if panel=="P1" else "NA_NOT_APPLICABLE")
 for k in ("GT_to_F_p95_2048_L0","F_to_GT_p95_2048_L0"):x[k]=float(np.mean([r[k] for r in g])) if panel=="P1" else None
 summary.append(x)
write(D/"summary_distance_p95.csv",summary)
receipt=dict(status="PASS",scope="Additional requested reference-distance-tail descriptive summary only; original metrics/screens unchanged.",definition="numpy.quantile(sqrt(saved squared surface distance)/L0, .95, method=linear); directions separate; average noise inside parent, then equal parents",saved_queries_only=True,P1_outputs=128,P2_NA_outputs=64,new_queries=0,geometry_distance_calls=0,model_calls=0,MSE_reproduced_count=256,source_vectors=src,script_sha256=sha(__file__),outputs={p.name:sha(p) for p in D.glob("*.csv")})
(D/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({k:v for k,v in receipt.items() if k!="source_vectors"},indent=2))
