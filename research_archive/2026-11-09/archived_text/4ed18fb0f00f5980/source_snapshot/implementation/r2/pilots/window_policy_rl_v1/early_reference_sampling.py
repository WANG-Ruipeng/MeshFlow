"""Only the already registered 88 frozen final-reference RAW, one auxiliary worker."""
from .bootstrap import bind,ROOT,OUT
D,E=bind()
from .run import snapshot,stamp,phase_jobs,verify_completed,PREFIX
from pathlib import Path
import json,subprocess,sys,time

def run():
    m,stages=phase_jobs('evaluation')
    run_dir=OUT/'dispatch'/('early_reference_'+stamp());run_dir.mkdir(parents=True)
    completed=[];start=time.monotonic()
    decision=dict(status='REGISTERED_ENGINEERING_OVERLAP',existing_budget='88 of the fixed132 final RAW; zero extra outputs',workers=1,
      measured_sampling_reserved_MiB=614,conservative_process_estimate_MiB=1126,
      dependency='Frozen reference weights and predeclared final noises independent of RL; no scoring/review/selection until final RAW complete',
      source_files={str(Path(__file__)):D.sha(__file__)},training_config_sha256=D.sha(OUT/'training_config.json'))
    D.write(run_dir/'decision.json',decision)
    for stage in stages:
        if stage['label'] not in ('FULL_DET','FULL_STOCH'):continue
        for offset in range(0,len(stage['requests']),16):
            name=stage['label']+f'_{offset:03d}';path=OUT/'jobs/evaluation'/('sample_'+name+'.json')
            ledger=OUT/'dispatch/evaluation'/('sample_'+name)
            spec=dict(stage);spec['requests']=stage['requests'][offset:offset+16]
            spec.update(model_mode='reference',source_binding=m['source_binding'],registration_path=str(OUT/'registration.json'),
              registration_sha256=D.sha(OUT/'registration.json'),manifest_path=str(OUT/'inputs/manifest.json'),ledger_dir=str(ledger),cpu_threads=1,adapter_init_seed=2026101501)
            D.write(path,spec)
            if (ledger/'completion.json').exists():verify_completed(path,ledger);continue
            if ledger.exists():raise RuntimeError('Existing incomplete reference sample: no retry')
            while True:
                rp,res=snapshot('early_reference_'+name)
                avail=next((int(x.split()[1]) for x in res['meminfo'].splitlines() if x.startswith('MemAvailable:')),0)
                head=max(2048.,.1*res['_memory_total_mib'])
                # Reserve an additional4GiB for the next already-running training phase.
                if res['_memory_free_mib']-head>=4096+1.2*1126 and avail>=8*1024*1024 and res['load_average'][0]<12 and res['_util']<85:break
                print(json.dumps(dict(event='WAIT_REFERENCE_RESOURCES',label=name,resources=str(rp))),flush=True);time.sleep(15)
            D.write(run_dir/(name+'_admission.json'),dict(resources=str(rp),workers=1,headroom_MiB=head,reserved_for_training_MiB=4096))
            begin=time.monotonic()
            with (run_dir/(name+'.log')).open('x',encoding='utf8') as log:
                result=subprocess.run([sys.executable,'-B','-m',PREFIX+'sample_worker','--job',str(path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Reference sampling failure; no automatic replay: '+str(path))
            verify_completed(path,ledger)
            ending,_=snapshot('early_reference_'+name+'_finished')
            row=dict(job=str(path),outputs=len(spec['requests']),seconds=time.monotonic()-begin,final_resources=str(ending))
            completed.append(row);D.write(run_dir/(name+'_complete.json'),row);print(json.dumps(dict(event='REFERENCE_RAW_COMPLETE',**row)),flush=True)
    D.write(run_dir/'completion.json',dict(status='COMPLETE',rows=completed,seconds=time.monotonic()-start,concurrency='One auxiliary frozen-reference process overlapped with dependent RL phases; not an isolated speed benchmark'))
if __name__=='__main__':run()
