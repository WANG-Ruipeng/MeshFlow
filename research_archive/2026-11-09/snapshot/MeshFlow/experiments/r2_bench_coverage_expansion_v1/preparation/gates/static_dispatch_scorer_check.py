"""Read-only preflight of dispatch source and retained scientific scorer hashes."""
import ast,hashlib,json,time
from pathlib import Path
ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_bench_coverage_expansion_v1/preparation/gates'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resolve(p):return Path(str(p).replace('E:\\MeshFlow','/mnt/e/MeshFlow').replace('E:/MeshFlow','/mnt/e/MeshFlow'))
dispatch=ROOT/'r2/pilots/bench_coverage_expansion_v1/dispatch.py'
text=dispatch.read_text();tree=ast.parse(text)
assigned={n.targets[0].id:ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
assert assigned['estimate']=='4500.0',assigned
assert assigned['headroom']=='max(2048.0, 0.1 * total)',assigned
assert assigned['budget']=='1.2 * estimate + headroom',assigned
assert "reg['updates'] != 1500" in ast.unparse(tree)
assert "actual_gpu_workers=1" in text
old=ROOT/'experiments/r2_bench_adaptation_strength_v1/evaluation/score_registration.json'
reg=json.loads(old.read_text());checked={};mismatches=[]
for p,h in reg['source_hashes'].items():
    path=resolve(p);actual=sha(path);checked[str(path)]=dict(expected=h,actual=actual)
    if actual!=h:mismatches.append(str(path))
result=dict(status='PASS' if not mismatches else 'SCORER_HASH_MISMATCH',dispatch_AST='PASS',dispatch_sha256=sha(dispatch),planning_worker_MiB=4500.,headroom='max(2048MiB,10%totalVRAM)',required_free_on16303MiB_GPU=1.2*4500.+max(2048.,.1*16303),authorized_gpu_workers=1,registered_updates_guard=1500,historical_scorer_sources=len(checked),historical_registration_sha256=sha(old),scorer_hashes=checked,mismatches=mismatches,model_calls=0,GPU_calls=0,OT_calls=0,time_unix=time.time())
(OUT/'dispatch_scorer_static_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='scorer_hashes'},indent=2))
if mismatches:raise SystemExit(1)
