"""Audited correction of an auxiliary span unit label; no geometric re-score."""
from pathlib import Path
import json, hashlib
import run
out=Path(__file__).resolve().parent
receipt=out/"span_metadata_correction.json"
if receipt.exists():
    raise RuntimeError("Correction already applied; do not overwrite receipt")
rows=[]
for p in sorted((out/"cells").glob("*.json")):
    old=p.read_bytes()
    x=json.loads(old)
    if x["status"]!="COMPLETE":
        continue
    stats=x["stats"]
    value=stats["source_C_span_over_L"]
    if "source_C_span_model" in stats:
        raise RuntimeError("Unexpected partial/mixed metadata correction")
    stats["source_C_span_model"]=value
    stats["source_C_span_over_L"]=[v/x["job"]["L"] for v in value]
    row=dict(cell=p.name,old_sha256=hashlib.sha256(old).hexdigest(),
             original_mislabeled_value=value,L=x["job"]["L"])
    run.write(p,x)
    row["new_sha256"]=run.sha(p)
    rows.append(row)
run.aggregate(run.read(out/"manifest.json"))
run.write(receipt,dict(reason="Auxiliary span_over_L originally stored model units",
    primary_features_changed=False,geometry_recomputed=False,model_calls=0,
    originals_recoverable="Set source_C_span_over_L to original_mislabeled_value, remove source_C_span_model",
    rows=rows))
print("CORRECTED_AUXILIARY_METADATA",len(rows))
