from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
root=Path(__file__).resolve().parents[2]
folder=Path(__file__).resolve().parent
balanced=root/'output/a_balanced_20260925_01';target=root/'output/a_warm_start_20260925_01'
source=root/'experiments/a_radial/system_probe/warm_start.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(name,value):
    p=folder/name;temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(p)
source_hash=sha(source);assert not target.exists() and not (folder/'STARTED.json').exists()
put('STARTED.json',{'pid':os.getpid(),'fixed_source_sha256':source_hash,'target':str(target),'tasks':20,'max_steps_each':16,'automatic_retries':False,'only_CPU':True})
started=time.monotonic();stage='WAIT_FINALS_AND_DIAGNOSTICS';env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=''
try:
    while True:
        assert not (balanced/'TRAINING_FAILURE.json').exists(),'Balanced training failed'
        ready=False;d=balanced/'TRAIN_DIAGNOSTICS.json'
        if (balanced/'TRAINING_COMPLETE.json').exists() and d.exists():
            data=json.loads(d.read_text());ready=data['attempt_count']==120 and len(data['rows'])==120 and not data['errors']
        put('STATUS.json',{'stage':stage,'waiting_seconds':time.monotonic()-started,'ready':ready})
        if ready:break
        assert time.monotonic()-started<10800,'Wait exceeded fixed3h cap, no study attempted'
        time.sleep(30)
    assert sha(source)==source_hash,'Frozen warm_start implementation changed'
    commands=[['prepare','--long-run',str(root/'output/a_longfit_20260925_01'),'--balanced-run',str(balanced),'--out',str(target)],['run','--run',str(target)]]
    for args in commands:
        stage=args[0];command=[sys.executable,'-B','-m','experiments.a_radial.system_probe.warm_start',*args]
        put('STATUS.json',{'stage':stage,'command':command,'automatic_retries':False})
        with (folder/(stage+'.log')).open('x') as log:result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        put(stage+'_EXIT.json',{'returncode':result.returncode,'command':command})
        assert result.returncode==0,'Warm-start child failed; logs and artifacts retained'
    assert sha(source)==source_hash
    put('COMPLETE.json',{'status':'CPU_STUDY_COMPLETED_REVIEW_PENDING','target':str(target),'source_sha256':source_hash,'summary_sha256':sha(target/'SUMMARY.json')})
    put('STATUS.json',{'stage':'COMPLETE_REVIEW_PENDING'})
except BaseException as exc:
    put('FAILED.json',{'stage':stage,'error':repr(exc),'traceback':traceback.format_exc(),'no_automatic_retry':True});raise