"""Finish the bounded audit after preserving the first result; repeat only the failed import probe."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time,ast,re
import importlib.metadata as metadata
from packaging.requirements import Requirement
from packaging.version import Version
ROOT=Path("/mnt/e/MeshFlow-Control");OUT=Path("/mnt/e/MeshFlow-Control-artifacts/audit")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=OUT/"portability_review.json";previous=p.read_bytes()
(OUT/"portability_review.initial_completed.json").write_bytes(previous)
r=json.loads(previous)
r["prior_attempts"]=[dict(path=str(OUT/n),sha256=sha(OUT/n)) for n in ("portability_attempt1.json","portability_review_source.attempt1.py","portability_attempt2.json","portability_review_source.attempt2.py","portability_review.initial_completed.json")]
# Exclude a web URL from the drive-letter heuristic; retain its original evidence.
s=r["static_import_audit"];s["excluded_non_path_literals"]=[q for q in s["absolute_personal_paths"] if q["value"].startswith(("https://","http://"))]
s["absolute_personal_paths"]=[q for q in s["absolute_personal_paths"] if not q["value"].startswith(("https://","http://"))]
s["status"]="PASS" if not s["hazards"] and not s["absolute_personal_paths"] else "FAIL"
# Current repository dependencies: no package installation.
declared=ast.literal_eval(re.search(r"(?m)^dependencies\s*=\s*(\[.*\])",(ROOT/"pyproject.toml").read_text()).group(1))
checks=[]
for spec in declared:
    q=Requirement(spec)
    try:v=metadata.version(q.name)
    except metadata.PackageNotFoundError:v=None
    checks.append(dict(requirement=spec,installed=v,satisfied=v is not None and Version(v) in q.specifier))
r["dependencies"].update(declared=declared,checks=checks,pyproject_sha256=sha(ROOT/"pyproject.toml"))
r["dependencies"]["local_tool_imports"]={"validate_binding":"tools/validate_binding.py"}
r["dependencies"]["third_party_import_roots"].pop("validate_binding",None)
modules=[]
for q in sorted((ROOT/"src/meshflow_control").rglob("*.py")):
    if q.name=="__main__.py":continue
    parts=list(q.relative_to(ROOT/"src").with_suffix("").parts)
    if parts[-1]=="__init__":parts.pop()
    modules.append(".".join(parts))
tools=[str(q) for q in sorted((ROOT/"tools").glob("*.py"))]
code=f"""import sys,json,importlib,runpy,builtins
import torch,numpy as np
assert not torch.cuda.is_initialized()
torch.set_num_threads(2)
calls=[]
def forbidden(*a,**k):
 calls.append(str(a[:1]));raise RuntimeError('Import attempted asset load, CUDA initialization or model forward')
torch.load=forbidden;np.load=forbidden
torch.cuda.init=forbidden;torch.cuda._lazy_init=forbidden
torch.nn.Module._call_impl=forbidden
original_open=builtins.open
def guarded_open(file,*a,**k):
 if isinstance(file,(str,bytes)) and str(file).lower().endswith(('.pt','.pth','.ckpt','.npz','.npy','.ply','.obj')):return forbidden(file)
 return original_open(file,*a,**k)
builtins.open=guarded_open
for module in {modules!r}:importlib.import_module(module)
# Direct 'python tools/name.py' supplies this exact new-repository script directory.
sys.path.insert(0,{str(ROOT/'tools')!r})
for path in {tools!r}:runpy.run_path(path,run_name='portability_import_probe')
bad={{n:getattr(m,'__file__',None) for n,m in sys.modules.copy().items() if getattr(m,'__file__',None) and str(m.__file__).startswith('/mnt/e/MeshFlow/')}}
assert not bad,bad;assert not calls,calls;assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS',module_count={len(modules)},tool_count={len(tools)},asset_or_forward_or_CUDA_attempts=calls,legacy_modules=bad,cuda_initialized=torch.cuda.is_initialized(),package_path=__import__('meshflow_control').__file__)))
"""
env=os.environ.copy();env.pop("PYTHONPATH",None);env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
begin=time.perf_counter();proc=subprocess.run([sys.executable,"-B","-c",code],cwd=OUT,env=env,text=True,capture_output=True)
stdout=OUT/"portability_imports_corrected.stdout.txt";stderr=OUT/"portability_imports_corrected.stderr.txt"
stdout.write_text(proc.stdout);stderr.write_text(proc.stderr)
probe=dict(name="imports",returncode=proc.returncode,wall_seconds=time.perf_counter()-begin,stdout=str(stdout),stdout_sha256=sha(stdout),stderr=str(stderr),stderr_sha256=sha(stderr),
    correction="Use only new tools directory as direct-script import context; source repository imports remain forbidden.")
if proc.returncode:probe["error_tail"]=proc.stderr[-3000:]
r["prior_import_probe"]=next(q for q in r["probes"] if q["name"]=="imports")
r["probes"]=[probe if q["name"]=="imports" else q for q in r["probes"]]
r["followup_scope"]="Repeated only guarded module/tool import probe; successful help/config/real CPU inspect results reused with their original hashes, no second START load."
r["license_documentation"]=dict(status="DISCLOSED",files=[dict(path=n,sha256=sha(ROOT/n)) for n in ("README.md","THIRD_PARTY_NOTICES.md")],
    note="Authorized prose correction discloses retained NVIDIA notice and unverified separate permission; no scientific code or experiment changed.")
after={n:sha(ROOT/n) for n in r["source_hashes"]}
r["readonly_check"]["same_source_hashes"]=after==r["source_hashes"]
r["readonly_check"]["status"]="PASS" if after==r["source_hashes"] else "FAIL"
r["audit_sources"]=[r.pop("audit_source"),dict(path=str(Path(__file__)),sha256=sha(__file__))]
okay=s["status"]=="PASS" and all(q["returncode"]==0 for q in r["probes"]) and all(q["satisfied"] for q in checks) and r["documentation_links"]["status"]=="PASS" and all(m["status"]=="PASS" for m in r["source_maps"]) and r["readonly_check"]["status"]=="PASS"
r["portability_status"]="PASS" if okay else "ISSUES_FOUND"
r["status"]="PASS_WITH_LICENSE_EXCEPTION" if okay else "ISSUES_FOUND"
r["finished_unix"]=time.time()
p.write_text(json.dumps(r,indent=2)+"\n")
print(json.dumps(dict(status=r["status"],portability=r["portability_status"],import_probe_returncode=proc.returncode,dependencies=checks,readonly=r["readonly_check"],output=str(p),sha256=sha(p))))
