"""Build a portable source provenance index; no model imports or asset mutation."""
from pathlib import Path
import argparse,hashlib,json
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--repo",required=True);a=p.parse_args()
repo=Path(a.repo);source=Path(a.source)
entries=[]
for fname,key in (("model_source_map.json","migrations"),("data_training_source_map.json","files"),("evaluation_source_map.json","files")):
    data=json.loads((repo/"docs"/fname).read_text())
    for e in data[key]:
        entries.append(dict(source=e.get("source_path",e.get("source")),source_sha256=e["source_sha256"],
            destination=e.get("destination",e.get("new_path")),reason=e["reason"]))
extra=[
("native_t1/runtime.py","src/meshflow_control/runtime.py","Preserve numerical runtime; add external lock, resource and six-hour owner accounting."),
("native_t1/artifacts.py","src/meshflow_control/artifacts.py","Retain deterministic hashes and atomic metadata writer."),
("native_t1/chair_checkpoint.py","src/meshflow_control/checkpoints.py","Original START identity contract; new explicit old parameter/AdamW mapping and strict resumable candidate schema."),
("native_t1/chair_sampling.py","src/meshflow_control/sampling.py","Retain original Gaussian, free-only Euler50 and clamp; add accounting and lossless external exports.")]
for old,new,reason in extra:
    entries.append(dict(source=old,source_sha256=sha(source/old),destination=new,reason=reason))
for e in entries:
    old=source/e["source"]
    if not old.exists() or sha(old)!=e["source_sha256"]:raise RuntimeError("Original source changed: "+e["source"])
    e["destination_sha256"]=sha(repo/e["destination"])
body=["# Source map","","Source fork: https://github.com/WANG-Ruipeng/MeshFlow, commit ae91eaa6f882a7d311233fa32053b7cd5de7ebe0.",
"Only necessary model, data and scoring modules were migrated. The original workspace was read-only.",
"Model and scoring component JSON maps preserve the detailed extraction audit. The data/training map records local retained source snapshots.",
"Paths below are relative to their respective source or destination repository. Runtime never imports those old paths.","",
"| Source | SHA256 of source | Destination | Adaptation |","|---|---|---|---|"]
for e in entries:body.append("| "+e["source"]+" | "+e["source_sha256"]+" | "+e["destination"]+" | "+e["reason"].replace("|","/")+" |")
body+=["","New readout, unified interfaces, lossless triangle IO, versioned checkpoints, frozen panel metadata, orchestration and tests are task-specific additions. External weights/data are not distributed by this code license.",
"Copyright/license: LICENSE (original MeshFlow MIT) and THIRD_PARTY_NOTICES.md; source NVIDIA notices are retained.",
"Destination hashes and source identities are in source_map.json. They describe the delivery code; experiment runtime hashes are also preserved externally."]
(repo/"docs/source_map.md").write_text("\n".join(body)+"\n",encoding="utf-8")
(repo/"docs/source_map.json").write_text(json.dumps(dict(source_head="ae91eaa6f882a7d311233fa32053b7cd5de7ebe0",files=entries),indent=2)+"\n",encoding="utf-8")
print(json.dumps(dict(status="PASS",source_entries=len(entries))))
