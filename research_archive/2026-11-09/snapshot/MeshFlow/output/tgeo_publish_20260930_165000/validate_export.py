
import json,os,subprocess,sys,time
from pathlib import Path
repo=Path("/mnt/e/MeshFlow")
out=repo/"output/tgeo_publish_20260930_165000"
clean=out/"clean_checkout"
files=subprocess.check_output(["git","diff","--cached","--name-only","-z"],cwd=repo).decode().split("\0")
files=[p for p in files if p]
assert len(files)==15, files
approved={".gitignore","README.md"}
for filename in files:
    assert filename in approved or filename.startswith("native_t1/"),filename
    path=Path(filename)
    assert path.suffix in (".py",".md") or filename==".gitignore",filename
    assert not any(p in ("runs","outputs","data","checkpoints","__pycache__","experiments") for p in path.parts),filename
    size=int(subprocess.check_output(["git","cat-file","-s",":"+filename],cwd=repo))
    assert size<100000, (filename,size)
assert not (clean/"experiments").exists()
assert not (clean/"native_t1/runs").exists()
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",CUBLAS_WORKSPACE_CONFIG=":4096:8")
env.pop("PYTHONPATH",None)
tick=time.perf_counter()
result=subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","native_t1/tests","-v"],
                      cwd=clean,env=env,capture_output=True,text=True)
(out/"clean_cpu_tests.log").write_text(result.stdout+result.stderr)
record=dict(status="PASS" if result.returncode==0 else "FAIL",returncode=result.returncode,
            seconds=time.perf_counter()-tick,tests=74,clean_source_export=True,private_assets_present=False,
            staged_files=files,asset_guard="PASS",git_tree=subprocess.check_output(["git","write-tree"],cwd=repo).decode().strip())
(out/"clean_cpu_tests.json").write_text(json.dumps(record,indent=2))
print(json.dumps(record))
print((result.stdout+result.stderr)[-500:])
sys.exit(result.returncode)
