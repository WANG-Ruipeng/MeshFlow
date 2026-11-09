"""Validate and combine actual image review records, retaining original anonymous observations."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib,argparse,re
R=Path(__file__).resolve().parents[1];V=R/"visual_review"
def js(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cr(p):
 with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def wr(p,rows):
 fields=list(dict.fromkeys(k for x in rows for k in x))
 with Path(p).open("w",encoding="utf8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["blind","final"],required=True);args=ap.parse_args()
pages=js(R/"audit/report_ready/gallery_pages_blinded.json")["pages"]
by={p["request_id"]:p for p in pages}
expected={(p["request_id"],m) for p in pages for m in p["anonymous_order"]}
def collect(names,stage):
 allrows=[];files={}
 for name in names:
  path=V/name;rows=cr(path);assert len(rows)==64,(name,len(rows));files[str(path)]=sha(path)
  for row in rows:
   assert row["review_kind"]=="AI_VISUAL_REVIEW" and row["human_usability"]=="NOT_RUN"
   assert row["review_status"].startswith(("REVIEWED","COMPLETE")),row["review_status"]
   assert row["observed_utc"] and row["reviewer"] and row["notes"]
   views=json.loads(row["view_scope"]) if row["view_scope"].lstrip().startswith("[") else row["view_scope"].split("|");assert set(views)=={"front_3q","back_3q","front","side","wire"}
   page=by[row["request_id"]]
   field="blind" if stage=="blind" else "reference"
   assert row["page_sha256"]==page[field+"_sha256"]==sha(page[field+"_page"])
   allrows.append(row)
 keys=[(x["request_id"],x["anonymous_method"]) for x in allrows];assert len(set(keys))==len(keys)
 goal=expected if stage=="blind" else {(p["request_id"],m) for p in pages if p["panel"]=="P1" for m in p["anonymous_order"]}
 assert set(keys)==goal
 return sorted(allrows,key=lambda r:(r["request_id"],r["anonymous_method"])),files
blind,files=collect(["blind_coupling.csv","blind_scoring.csv","blind_diagnostics.csv"],"blind")
if args.stage=="blind":
 wr(V/"blind_review_combined.csv",blind)
 receipt=dict(status="PASS",kind="ACTUAL_SOURCE_HIDDEN_REVIEW_FROZEN",blind_output_rows=len(blind),blind_pages=48,files=files,combined_sha256=sha(V/"blind_review_combined.csv"),frozen_utc=datetime.now(timezone.utc).isoformat(),reference_review_authorized_next=True,method_blinding="Anonymous labels; not strict reviewer blinding because scoring reviewer had prior mapping exposure",human_usability="NOT_RUN")
 dump(V/"source_hidden_freeze.json",receipt);print(json.dumps(receipt));raise SystemExit()
freeze=js(V/"source_hidden_freeze.json");assert freeze["status"]=="PASS" and freeze["files"]==files
assert freeze["combined_sha256"]==sha(V/"blind_review_combined.csv")
ref,ref_files=collect(["reference_coupling.csv","reference_scoring.csv"],"reference")
def stamp(value):
 value=value.replace(" UTC","+00:00").replace("Z","+00:00")
 value=re.sub(r"(\.\d{6})\d+",r"\1",value)
 return datetime.fromisoformat(value)
assert all(stamp(r["observed_utc"])>=stamp(freeze["frozen_utc"]) for r in ref), "Reference review preceded global source-hidden freeze"
wr(V/"reference_review_combined.csv",ref)
mapping=js(R/"gallery/method_mapping.json")["mapping"];alias_to_label={a:l for l,a in mapping.items()}
refby={(r["request_id"],r["anonymous_method"]):r for r in ref};merged=[]
for row in blind:
 x=dict(row);x["label"]=alias_to_label[row["anonymous_method"]];r=refby.get((row["request_id"],row["anonymous_method"]))
 x["reference_review_status"]=r["review_status"] if r else "NA_NOT_APPLICABLE"
 for k in ("source_specific_difference","alternative_valid_completion","notes","observed_utc","reviewer","page_sha256"):
  x["reference_"+k]=r.get(k,"") if r else ""
 x["review_protocol"]="Anonymous-labelled, source hidden first; not strict method-blinded evaluation"
 merged.append(x)
wr(R/"visual_review.csv",merged)
receipt=dict(status="PASS",kind="ACTUAL_AI_VISUAL_REVIEW_COMPLETE",blind_output_rows=192,reference_output_rows=128,blind_pages=48,reference_pages=32,unique_final_outputs=192,files={**files,**ref_files},final_csv_sha256=sha(R/"visual_review.csv"),source_hidden_freeze_sha256=sha(V/"source_hidden_freeze.json"),reference_combined_sha256=sha(V/"reference_review_combined.csv"),method_mapping_sha256=sha(R/"gallery/method_mapping.json"),finished_utc=datetime.now(timezone.utc).isoformat(),method_blinding="Anonymous labels only; scoring reviewer previously exposed to mapping; no strict method-blind claim",human_usability="NOT_RUN",scientific_image_modification=False,model_calls=0,scoring_calls=0)
dump(V/"final_review_receipt.json",receipt);print(json.dumps(receipt))
