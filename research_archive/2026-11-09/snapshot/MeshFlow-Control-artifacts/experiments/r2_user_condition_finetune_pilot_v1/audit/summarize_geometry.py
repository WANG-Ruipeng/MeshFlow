"""Derive descriptive tables from all frozen CPU scores; no new geometry/model calls."""
from pathlib import Path
from collections import defaultdict
import json,csv,math
import numpy as np
OUT=Path(__file__).resolve().parents[1]
def write(p,x):
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def csvwrite(p,rows):
 with p.open("w",newline="",encoding="utf8") as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 scores=json.loads((OUT/"evaluation/scores_all.json").read_text())
 if len(scores["rows"])!=84:raise ValueError("All84 score attempts required")
 rows=scores["rows"]
 numeric=sorted({k for r in rows for k,v in r["metrics"].items() if isinstance(v,(int,float)) and not isinstance(v,bool)})
 summary=[]
 for panel in ("NEWDEV","OLDREG"):
  regions=("ALL","seat","back") if panel=="NEWDEV" else ("ALL",)
  for region in regions:
   for label in ("R2_FROZEN","R2_CONT_OLD","R2_CONT_USER"):
    rr=[r for r in rows if r["panel"]==panel and r["label"]==label and (region=="ALL" or r["region_type"]==region)]
    for metric in numeric:
     valid=[r for r in rr if r["status"]=="PASS" and isinstance(r["metrics"].get(metric),(int,float)) and math.isfinite(r["metrics"][metric])]
     byparent=defaultdict(list)
     for r in valid:byparent[r["parent_id"]].append(r["metrics"][metric])
     values=[r["metrics"][metric] for r in valid]
     summary.append(dict(panel=panel,region=region,label=label,metric=metric,registered_outputs=len(rr),valid_outputs=len(valid),
       parents=len({r["parent_id"] for r in rr}),mean_equal_parent=float(np.mean([np.mean(x) for x in byparent.values()])) if len(valid)==len(rr) else None,
       median_output=float(np.median(values)) if values else None,p95_output=float(np.quantile(values,.95)) if values else None,
       min_output=min(values) if values else None,max_output=max(values) if values else None,
       worst_max_request_id=max(valid,key=lambda r:r["metrics"][metric])["request_id"] if valid else None))
 pairs=[];index={(r["label"],r["request_id"]):r for r in rows}
 for ref in ("R2_CONT_OLD","R2_FROZEN"):
  for panel in ("NEWDEV","OLDREG"):
   for metric in numeric:
    grouped=defaultdict(list)
    for r in rows:
     if r["label"]!="R2_CONT_USER" or r["panel"]!=panel:continue
     base=index[ref,r["request_id"]]
     a,b=r["metrics"].get(metric),base["metrics"].get(metric)
     if r["status"]==base["status"]=="PASS" and isinstance(a,(float,int)) and isinstance(b,(float,int)) and math.isfinite(a) and math.isfinite(b):
      grouped[r["condition_id"]].append((r["candidate"],a-b,r["request_id"]))
    expected=6 if panel=="NEWDEV" else 8
    complete=[(k,x) for k,x in grouped.items() if len(x)==2]
    means=[np.mean([v[1] for v in x]) for k,x in complete]
    pairs.append(dict(panel=panel,comparison="R2_CONT_USER minus "+ref,metric=metric,
      expected_conditions=expected,valid_paired_conditions=len(complete),
      mean_condition_delta=float(np.mean(means)) if len(complete)==expected else None,
      conditions_lower=int(sum(v < -1e-12 for v in means)),conditions_equal=int(sum(abs(v)<=1e-12 for v in means)),conditions_higher=int(sum(v>1e-12 for v in means)),
      conditions_both_noises_lower=int(sum(all(v[1]<-1e-12 for v in x) for k,x in complete)),
      conditions_both_noises_higher=int(sum(all(v[1]>1e-12 for v in x) for k,x in complete)),
      largest_positive_delta=max(means) if means else None,largest_negative_delta=min(means) if means else None,
      interpretation="Descriptive numeric lower/higher only; direction is metric-specific, not general quality or semantic success",
      tie_epsilon=1e-12))
 csvwrite(OUT/"evaluation/geometry_distributions.csv",summary)
 csvwrite(OUT/"evaluation/paired_direction_counts.csv",pairs)
 write(OUT/"evaluation/descriptive_summary.json",dict(status="COMPLETE",score_rows=len(rows),model_calls=0,
  new_geometry_tests=0,statistical_significance_claim=False,distributions=summary,pairs=pairs))
 print(json.dumps(dict(status="COMPLETE",distributions=len(summary),paired_metrics=len(pairs))))
if __name__=="__main__":main()
