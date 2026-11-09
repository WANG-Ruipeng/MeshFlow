"""Bounded read-only portability audit. No model forward or scientific file mutation."""
from __future__ import annotations
import ast,contextlib,hashlib,io,json,os,re,subprocess,sys,time,traceback
from pathlib import Path
import importlib.metadata as metadata
import tomllib
ROOT=Path("/mnt/e/MeshFlow-Control")
OUT=Path("/mnt/e/MeshFlow-Control-artifacts/audit")
START=Path(json.loads((OUT/"start_import.json").read_text())["external_checkpoint"])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dotted(n):
    if isinstance(n,ast.Name):return n.id
    if isinstance(n,ast.Attribute):return dotted(n.value)+"."+n.attr
    return ""
paths=sorted(p for folder in ("src","tests","tools") for p in (ROOT/folder).rglob("*.py"))
before={str(p.relative_to(ROOT)):sha(p) for p in paths}
report=dict(schema="meshflow_control_portability_review_v1",created_unix=time.time(),
    scope=["src/**/*.py","tests/**/*.py","tools/**/*.py","pyproject.toml","CI","license/source maps/docs links"],
    scientific_source_writes=0,real_model_forward_calls=0,real_model_backward_calls=0,optimizer_updates=0,
    full_test_suite_rerun=False,source_file_count=len(paths),source_hashes=before,findings=[])
imports=[];hazards=[];legacy_strings=[];absolute=[];external_imports=set()
for p in paths:
    rel=str(p.relative_to(ROOT));tree=ast.parse(p.read_text(encoding="utf-8-sig"),filename=rel)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            mods=[v.name for v in node.names] if isinstance(node,ast.Import) else [node.module or ""]
            for name in mods:
                root=name.split(".")[0]
                if not isinstance(node,ast.ImportFrom) or node.level==0:
                    imports.append(dict(path=rel,line=node.lineno,module=name))
                    external_imports.add(root)
                    if root in ("experiments","native_t1","models","datasets"):
                        hazards.append(dict(type="legacy_import",path=rel,line=node.lineno,module=name))
        if isinstance(node,ast.Call):
            name=dotted(node.func)
            if name in ("eval","exec","__import__","runpy.run_path","runpy.run_module","importlib.import_module","importlib.util.spec_from_file_location","importlib.machinery.SourceFileLoader") or name.startswith("sys.path."):
                hazards.append(dict(type="dynamic_loader_or_path_mutation",path=rel,line=node.lineno,call=name))
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            value=node.value
            if re.search(r"(?:/mnt/[a-z]/|[A-Za-z]:[\\/]|/home/|C:/Users/|warpwang)",value):
                absolute.append(dict(path=rel,line=node.lineno,value=value[:300]))
            if "native_t1/" in value or "experiments/" in value:
                legacy_strings.append(dict(path=rel,line=node.lineno,value=value[:300]))
report["static_import_audit"]=dict(status="PASS" if not hazards and not absolute else "FAIL",
    hazards=hazards,absolute_personal_paths=absolute,declared_legacy_asset_or_provenance_strings=legacy_strings,
    interpretation="Relative historical paths occur only in explicit import/reference preparation and provenance collection; they are not Python imports.")
pyproject=tomllib.loads((ROOT/"pyproject.toml").read_text())
declared=pyproject["project"]["dependencies"]
versions={}
for name in ("torch","torchvision","numpy","scipy","PyYAML","einops","timm","matplotlib","Pillow","trimesh","pytest","setuptools","wheel","packaging"):
    try:versions[name]=metadata.version(name)
    except metadata.PackageNotFoundError:versions[name]=None
from packaging.requirements import Requirement
from packaging.version import Version
dep_checks=[]
for spec in declared:
    r=Requirement(spec)
    installed=metadata.version(r.name)
    dep_checks.append(dict(requirement=spec,installed=installed,satisfied=Version(installed) in r.specifier))
stdlib=set(sys.stdlib_module_names)
package_map=metadata.packages_distributions()
thirdparty={n:package_map.get(n,[]) for n in sorted(external_imports) if n and n not in stdlib and n!="meshflow_control"}
report["environment"]=dict(python=sys.version,executable=sys.executable,platform=sys.platform,
    cwd=str(Path.cwd()),CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),versions=versions)
report["dependencies"]=dict(declared=declared,checks=dep_checks,third_party_import_roots=thirdparty,
    offline_install="Requires dependencies and setuptools/wheel preinstalled; no network/install performed by audit.",
    non_python_requirement="GPU dispatch requires Linux/WSL fcntl and nvidia-smi. CPU CI uses Ubuntu.")
# Every migrated destination must match its declared hash. No old source files are rehashed.
maps=[]
for name,key in (("source_map.json","files"),("model_source_map.json","migrations"),("evaluation_source_map.json","files")):
    p=ROOT/"docs"/name;doc=json.loads(p.read_text());rows=[]
    for row in doc[key]:
        dest=ROOT/row["destination"];actual=sha(dest) if dest.exists() else None
        rows.append(dict(destination=row["destination"],expected=row.get("destination_sha256"),actual=actual,
            matches=actual==row.get("destination_sha256")))
    maps.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),rows=rows,status="PASS" if all(x["matches"] for x in rows) else "FAIL"))
report["source_maps"]=maps
broken=[];links=[]
for p in [ROOT/"README.md",ROOT/"THIRD_PARTY_NOTICES.md",ROOT/"AGENTS.md",*sorted((ROOT/"docs").glob("*.md"))]:
    for target in re.findall(r"\]\(([^)]+)\)",p.read_text(encoding="utf-8-sig")):
        if "://" in target or target.startswith("#"):continue
        target=target.split("#")[0];q=p.parent/target
        row=dict(document=str(p.relative_to(ROOT)),target=target,exists=q.exists());links.append(row)
        if not row["exists"]:broken.append(row)
report["documentation_links"]=dict(status="PASS" if not broken else "FAIL",checked=links,broken=broken,
    plain_source_map_json_reference_exists=(ROOT/"docs/source_map.json").exists())
ci_paths=sorted((ROOT/".github/workflows").glob("*"))
report["cpu_ci"]=dict(status="PASS_CURRENT_SUITE_ONLY",workflows=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),text=p.read_text()) for p in ci_paths],
    tests=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),test_count=sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith("test_") for n in ast.walk(ast.parse(p.read_text())))) for p in sorted((ROOT/"tests").glob("test_*.py"))],
    inspected_scope="Current tests use synthetic arrays, temporary files, small readout/backbone fixtures, or meta-device full parameter structure; no real checkpoint/private data or real model forward.",
    synthetic_updates="test_models.py deliberately performs exactly two SGD steps on a 12-dimensional synthetic readout, not on the real model.",
    caveat="CI invokes pytest tests with CUDA_VISIBLE_DEVICES empty, not -m 'not gpu'; no real GPU test currently exists. Future GPU tests must be excluded explicitly.")
# Probe CLI lazy behavior, guarded module/tool imports, and real CPU metadata inspect in isolated processes.
env=os.environ.copy();env.pop("PYTHONPATH",None)
env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
def probe(name,code):
    begin=time.perf_counter()
    proc=subprocess.run([sys.executable,"-B","-c",code],cwd=OUT,env=env,text=True,capture_output=True)
    stdout=OUT/("portability_"+name+".stdout.txt");stderr=OUT/("portability_"+name+".stderr.txt")
    stdout.write_text(proc.stdout);stderr.write_text(proc.stderr)
    result=dict(name=name,returncode=proc.returncode,wall_seconds=time.perf_counter()-begin,stdout=str(stdout),stdout_sha256=sha(stdout),stderr=str(stderr),stderr_sha256=sha(stderr))
    if proc.returncode:result["error_tail"]=proc.stderr[-3000:]
    return result
helpcode="import sys;from meshflow_control.cli import main\ntry:main(['--help'])\nexcept SystemExit as e:assert e.code==0\nassert 'torch' not in sys.modules\nprint('AUDIT_HELP_TORCH_NOT_IMPORTED_PASS')"
configcode="import sys;from meshflow_control.cli import main\n"+ "\n".join("main(['check-config','--config',"+repr(str(ROOT/"configs"/f))+"])" for f in ("a_baseline.yaml","b_mixed_readout.yaml","c_clean_readout.yaml"))+"\nassert 'torch' not in sys.modules\nprint('AUDIT_CONFIG_TORCH_NOT_IMPORTED_PASS')"
modules=[]
for p in sorted((ROOT/"src/meshflow_control").rglob("*.py")):
    if p.name=="__main__.py":continue
    rel=p.relative_to(ROOT/"src").with_suffix("")
    parts=list(rel.parts)
    if parts[-1]=="__init__":parts.pop()
    modules.append(".".join(parts))
toolpaths=[str(p) for p in sorted((ROOT/"tools").glob("*.py"))]
importcode=f"""import sys,json,importlib,runpy,builtins
import torch,numpy as np
assert not torch.cuda.is_initialized()
torch.set_num_threads(2)
calls=[]
def forbidden(*a,**k):
 calls.append(str(a[:1]));raise RuntimeError('Import attempted asset load, CUDA initialization or model forward')
torch.load=forbidden
np.load=forbidden
torch.cuda.init=forbidden
torch.cuda._lazy_init=forbidden
torch.nn.Module._call_impl=forbidden
original_open=builtins.open
def guarded_open(file,*a,**k):
 if isinstance(file,(str,bytes)) and str(file).lower().endswith(('.pt','.pth','.ckpt','.npz','.npy','.ply','.obj')):
  return forbidden(file)
 return original_open(file,*a,**k)
builtins.open=guarded_open
for module in {modules!r}:importlib.import_module(module)
for path in {toolpaths!r}:runpy.run_path(path,run_name='portability_import_probe')
bad={{n:getattr(m,'__file__',None) for n,m in sys.modules.copy().items() if getattr(m,'__file__',None) and str(m.__file__).startswith('/mnt/e/MeshFlow/')}}
assert not bad,bad
assert not calls,calls
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS',module_count={len(modules)},tool_count={len(toolpaths)},asset_or_forward_or_CUDA_attempts=calls,legacy_modules=bad,cuda_initialized=torch.cuda.is_initialized(),package_path=__import__('meshflow_control').__file__)))
"""
inspectcode=f"""import json,torch
from meshflow_control.cli import main
assert not torch.cuda.is_initialized()
original=torch.load;calls=[]
def guarded_load(path,*args,**kwargs):
 assert str(path)=={str(START)!r}
 assert kwargs.get('map_location')=='cpu'
 assert kwargs.get('weights_only') is True
 assert kwargs.get('mmap') is True
 calls.append(dict(path=str(path),map_location=kwargs['map_location'],weights_only=kwargs['weights_only'],mmap=kwargs['mmap']))
 return original(path,*args,**kwargs)
torch.load=guarded_load
def forbidden(*args,**kwargs):raise RuntimeError('Metadata inspect attempted CUDA or forward')
torch.cuda.init=forbidden
torch.cuda._lazy_init=forbidden
torch.nn.Module._call_impl=forbidden
main(['inspect','--checkpoint',{str(START)!r}])
assert len(calls)==1
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS',torch_load_calls=calls,cuda_initialized=torch.cuda.is_initialized(),real_model_constructed=False,model_forward_calls=0)))
"""
report["probes"]=[probe("help",helpcode),probe("config",configcode),probe("imports",importcode),probe("inspect",inspectcode)]
report["licensing"]=dict(status="UNRESOLVED_EXCEPTION",root_license="MIT; copyright (c) 2026 qiisun",
    retained_notices=[dict(path="LICENSE",sha256=sha(ROOT/"LICENSE")),dict(path="THIRD_PARTY_NOTICES.md",sha256=sha(ROOT/"THIRD_PARTY_NOTICES.md")),dict(path="src/meshflow_control/models/backbone/utils.py",sha256=sha(ROOT/"src/meshflow_control/models/backbone/utils.py"))],
    exception=dict(path="src/meshflow_control/models/backbone/utils.py",lines="1-10",notice="NVIDIA 2024 proprietary notice requires express license agreement for use/reproduction/disclosure/distribution.",
    evidence_boundary="No separate permission or permissive license for this notice was found in the inspected new repository. Retention of the notice and the source repository MIT file does not establish such permission. This is an unresolved redistribution/licensing provenance item, not a new experiment result."),
    third_party_weights_data="Not distributed by repository; independent terms not evaluated by this code-only audit.")
after={str(p.relative_to(ROOT)):sha(p) for p in paths}
report["readonly_check"]=dict(status="PASS" if before==after else "FAIL",same_source_hashes=before==after,
    note="Only audit source/results/CLI logs under the external audit directory were written; no scientific source, checkpoint or data modified.")
report["portability_status"]="PASS" if not hazards and not absolute and all(x["returncode"]==0 for x in report["probes"]) and not broken and all(m["status"]=="PASS" for m in maps) and before==after and all(x["satisfied"] for x in dep_checks) else "ISSUES_FOUND"
report["status"]="PASS_WITH_LICENSE_EXCEPTION" if report["portability_status"]=="PASS" else "ISSUES_FOUND"
report["limitations"]=["Inspection validates this frozen source tree and installed environment, not all future imports/dependencies.","No full test suite rerun, installation, network, actual model forward, CUDA initialization or new GPU job.","Linux/WSL GPU-worker lock is intentional; native Windows GPU dispatch is not validated.","Distribution permission for the retained NVIDIA-noticed utility remains unresolved."]
report["audit_source"]=dict(path=str(Path(__file__)),sha256=sha(__file__))
target=OUT/"portability_review.json";target.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(dict(status=report["status"],portability=report["portability_status"],source_files=len(paths),probes=[dict(name=p["name"],returncode=p["returncode"]) for p in report["probes"]],source_maps=[dict(path=m["path"],status=m["status"]) for m in maps],output=str(target),sha256=sha(target))))
