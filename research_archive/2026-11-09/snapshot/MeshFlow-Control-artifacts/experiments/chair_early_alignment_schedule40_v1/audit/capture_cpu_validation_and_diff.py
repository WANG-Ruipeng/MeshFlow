from pathlib import Path
import subprocess,json,time,hashlib,difflib
root=Path('/mnt/e/MeshFlow-Control'); out=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1')
tests=['tests/test_schedule40_data.py','tests/test_schedule40_train.py','tests/test_schedule40_dispatch_eval.py','tests/test_schedule40_accounting.py']
missing=[p for p in tests if not (root/p).is_file()]
if missing:
    found=sorted(p.name for p in (root/'tests').glob('*schedule40*')); raise RuntimeError((missing,found))
cmd=['/mnt/e/MeshFlow-Control-artifacts/venv/bin/python','-B','-m','pytest','-q']+[str(root/p) for p in tests]
tick=time.perf_counter(); result=subprocess.run(cmd,capture_output=True,text=True,cwd=root)
(out/'audit/cpu_validation.log').write_text(result.stdout+result.stderr)
(out/'audit/cpu_validation.json').write_text(json.dumps(dict(status='PASS' if result.returncode==0 else 'FAIL',command=cmd,returncode=result.returncode,wall_seconds=time.perf_counter()-tick,tests=tests,GPU_training_or_sampling_calls=0),indent=2))
print(result.stdout)
if result.returncode: raise SystemExit(result.returncode)
files=['src/meshflow_control/data/schedule40.py','src/meshflow_control/training/schedule40.py','tools/train_schedule40.py','tools/run_schedule40.py','tools/evaluate_schedule40.py','tools/analyze_schedule40.py']+tests
parts=[]; hashes={}
for name in files:
    p=root/name; raw=p.read_bytes(); hashes[name]=hashlib.sha256(raw).hexdigest()
    parts.append('diff --git a/'+name+' b/'+name+'\nnew file mode 100644\n'+''.join(difflib.unified_diff([],raw.decode('utf8').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+name)))
(out/'code.diff').write_text('\n'.join(parts))
(out/'audit/new_source_hashes.json').write_text(json.dumps(hashes,indent=2))
for name in ['tools/analyze_schedule40.py']+tests:
    p=out/'source_code'/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((root/name).read_bytes())
print(json.dumps(dict(new_source_files=len(files),diff_bytes=(out/'code.diff').stat().st_size)))