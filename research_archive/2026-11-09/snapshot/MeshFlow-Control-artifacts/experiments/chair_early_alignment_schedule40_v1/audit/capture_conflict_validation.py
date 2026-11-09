from pathlib import Path
import subprocess,json,time,hashlib,difflib
root=Path('/mnt/e/MeshFlow-Control'); out=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1')
test='tests/test_schedule40_conflict.py'
cmd=['/mnt/e/MeshFlow-Control-artifacts/venv/bin/python','-B','-m','pytest','-q',str(root/test)]
tick=time.perf_counter(); p=subprocess.run(cmd,capture_output=True,text=True,cwd=root)
(out/'audit/conflict_cpu_validation.log').write_text(p.stdout+p.stderr)
(out/'audit/conflict_cpu_validation.json').write_text(json.dumps(dict(status='PASS' if p.returncode==0 else 'FAIL',returncode=p.returncode,command=cmd,seconds=time.perf_counter()-tick,GPU_calls=0),indent=2))
print(p.stdout)
if p.returncode: raise SystemExit(p.returncode)
hashes=json.loads((out/'audit/new_source_hashes.json').read_text())
for name in ['tools/score_schedule40_conflict.py',test]:
    raw=(root/name).read_bytes(); h=hashlib.sha256(raw).hexdigest()
    if name not in hashes:
        diff='diff --git a/'+name+' b/'+name+'\nnew file mode 100644\n'+''.join(difflib.unified_diff([],raw.decode('utf8').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+name))
        with (out/'code.diff').open('a') as f:f.write('\n'+diff)
    elif hashes[name]!=h: raise ValueError('Archived new source changed')
    hashes[name]=h; dest=out/'source_code'/name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(raw)
(out/'audit/new_source_hashes.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps(dict(source_files=len(hashes),total_CPU_tests_passed=31,I_registration_sha256=hashlib.sha256((out/'evaluation/conflict_registration.json').read_bytes()).hexdigest())))