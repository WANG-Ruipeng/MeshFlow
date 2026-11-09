from pathlib import Path
import subprocess,sys,os,re,time,json
run=Path('native_t1/runs/geometry_context_revised_20260930_153000')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
start=time.perf_counter()
p=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','native_t1/tests','-v'],capture_output=True,text=True,env=env)
s=p.stdout+p.stderr
(run/'all_cpu_tests.log').write_text(s)
m=re.search(r'Ran (\d+) tests in',s)
r=dict(status='PASS' if p.returncode==0 else 'FAIL',tests=int(m[1]) if m else None,seconds=time.perf_counter()-start,exit_code=p.returncode,model_forwards=0,model_backwards=0,optimizer_updates=0,complete_rollouts=0)
(run/'all_cpu_tests.json').write_text(json.dumps(r,indent=2)+'\n')
print(s[-4500:]);print(json.dumps(r))
sys.exit(p.returncode)
