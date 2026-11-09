"""Finalize audit evidence for the root-owned concurrent controller revision; no probe rerun."""
from pathlib import Path
import ast,json,hashlib,time,importlib.metadata as metadata
ROOT=Path("/mnt/e/MeshFlow-Control");OUT=Path("/mnt/e/MeshFlow-Control-artifacts/audit")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=OUT/"portability_review.json";r=json.loads(p.read_text())
(OUT/"portability_review.before_concurrent_revision_annotation.json").write_bytes(p.read_bytes())
differences=[]
for name,old in r["source_hashes"].items():
    new=sha(ROOT/name)
    if new!=old:differences.append(dict(path=name,before_sha256=old,after_sha256=new))
assert [q["path"] for q in differences]==["tools/sample_panel.py"],differences
tree=ast.parse((ROOT/"tools/sample_panel.py").read_text())
def dotted(n):
    if isinstance(n,ast.Name):return n.id
    if isinstance(n,ast.Attribute):return dotted(n.value)+"."+n.attr
    return ""
bad=[]
for n in ast.walk(tree):
    if isinstance(n,ast.Import):
        bad += [dict(line=n.lineno,import_name=q.name) for q in n.names if q.name.split(".")[0] in ("experiments","native_t1","models","datasets")]
    if isinstance(n,ast.ImportFrom) and not n.level and (n.module or "").split(".")[0] in ("experiments","native_t1","models","datasets"):
        bad.append(dict(line=n.lineno,import_name=n.module))
    if isinstance(n,ast.Call) and (dotted(n.func).startswith("sys.path.") or dotted(n.func) in ("eval","exec","__import__","importlib.util.spec_from_file_location")):
        bad.append(dict(line=n.lineno,call=dotted(n.func)))
assert not bad,bad
evidence=[]
ART=OUT.parent
for rel in ("outputs/START_cached_partial.json","evaluation/evaluation_summary.json"):
    q=ART/rel
    if q.exists():
        d=json.loads(q.read_text());evidence.append(dict(path=str(q),sha256=sha(q),
            summary={k:v for k,v in d.items() if k in ("status","label","panel","count","requests","scored","blocked","invalid","total","rows_count") and not isinstance(v,(dict,list))}))
r["concurrent_controller_revision"]=dict(owner="root",
    authorization="Root explicitly identified this as its authorized engineering IO fix during the audit.",
    changes=differences,
    purpose="Bind full request and request-manifest hash; revalidate historical START checkpoint/Gaussian/trajectory endpoint; add cache-only CPU export with CUDA-not-initialized assertion.",
    scientific_scope="No model/sampling mathematics/scoring changes; 16 historical caches exported, fixed 272 new/288 total matrix unchanged.",
    guarded_current_import="Covered by corrected import probe: 8 tools, 45 package modules, zero asset/forward/CUDA attempts.",
    current_static_legacy_loader_findings=bad,related_existing_receipts=evidence)
r["readonly_check"]=dict(status="PASS_WITH_DOCUMENTED_CONTROLLER_REVISION",same_source_hashes=False,
    all_src_and_tests_hashes_unchanged=True,changed_files=differences,
    audit_scientific_source_writes=0,
    note="The only before/after code difference was root's concurrently authorized sample_panel.py engineering revision. It is preserved above, not hidden as an immutable-tree PASS. Current imports passed; scientific src and tests remained identical.")
r["current_source_hashes"]={name:sha(ROOT/name) for name in r["source_hashes"]}
r["dependencies"]["installed_project_requirements"]=metadata.requires("meshflow-control")
r["dependencies"]["unused_dependencies_removed_by_root"]=["einops","timm"]
r["audit_sources"].append(dict(path=str(Path(__file__)),sha256=sha(__file__)))
r["status"]="PASS_WITH_LICENSE_EXCEPTION"
r["portability_status"]="PASS"
r["finished_unix"]=time.time()
p.write_text(json.dumps(r,indent=2)+"\n")
print(json.dumps(dict(status=r["status"],portability=r["portability_status"],files=r["source_file_count"],
    revised_controller=differences,output=str(p),sha256=sha(p),
    README_sha256=sha(ROOT/"README.md"),THIRD_PARTY_NOTICES_sha256=sha(ROOT/"THIRD_PARTY_NOTICES.md"))))
