"""Pure CPU identity binding for an already completed preflight; no model imports."""
from pathlib import Path
import json
from meshflow_control.artifacts import file_sha256

def validate(root,start,preflight):
    root=Path(root);gate=json.loads(Path(preflight).read_text())
    if gate["status"]!="PASS":raise RuntimeError("Actual GPU preflight has not passed")
    identity=gate["identity"]
    checks={"source_start_sha256":file_sha256(start),"data_manifest_sha256":file_sha256(root/"data/train_manifest.json"),
        "plan_sha256":file_sha256(root/"training_stream/train_plan.json")}
    for name,value in checks.items():
        if identity[name]!=value:raise RuntimeError("Preflight identity differs: "+name)
    import meshflow_control
    package=Path(meshflow_control.__file__).parent
    for rel,h in identity["scientific_sources"].items():
        if file_sha256(package/rel)!=h:raise RuntimeError("Preflight scientific source changed: "+rel)
    for rel,h in identity["reference_hashes"].items():
        if file_sha256(root/"preflight/references"/rel)!=h:raise RuntimeError("Official reference changed: "+rel)
    return dict(**checks,preflight_sha256=file_sha256(preflight),
        requests_sha256=file_sha256(root/"data/evaluation/requests.json"),scientific_sources=identity["scientific_sources"])
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    for k in ("artifacts","start","preflight"):p.add_argument("--"+k,required=True)
    a=p.parse_args();print(json.dumps(validate(a.artifacts,a.start,a.preflight),indent=2))
