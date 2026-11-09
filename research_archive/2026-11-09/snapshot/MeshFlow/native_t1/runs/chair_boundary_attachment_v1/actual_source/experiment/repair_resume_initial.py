"""Pre-dispatch review fixes: preserve safe-stop and verify all cached endpoints."""
from pathlib import Path
code=Path(__file__).parent
p=code/'worker.py';s=p.read_text()
s=s.replace("state=dict(status='DONE',stage=name,wall_seconds=time.perf_counter()-started,result=result)","worker_status=result['status'] if stage=='train' and result['status']!='DONE' else 'DONE'\n            state=dict(status=worker_status,stage=name,wall_seconds=time.perf_counter()-started,result=result)")
p.write_text(s)
p=code/'sampling.py';s=p.read_text()
s=s.replace("    if not todo:return dict(status='DONE',arm=arm,outputs=len(jobs),new_this_worker=0)","    for row in jobs:\n        completed=out/'outputs'/row['request_id']/'result.json'\n        if completed.exists():\n            saved=read_json(completed)\n            if saved['status']!='SAMPLED':raise RuntimeError('Existing endpoint is failed/unresolved; no automatic replay')\n            assert saved['request_id']==row['request_id'] and saved['input_sha256']==row['input_sha256']\n            assert saved['checkpoint_sha256']==pointer['checkpoint_sha256']\n            assert digest(resolve(saved['raw_path']))==saved['raw_sha256']\n    if not todo:return dict(status='DONE',arm=arm,outputs=len(jobs),new_this_worker=0)")
p.write_text(s)
p=code/'run_all.py';s=p.read_text()
s=s.replace("    return read_json(status)\n\n\ndef run", "    state=read_json(status)\n    if state['status']!='DONE':raise RuntimeError('Worker stopped safely before completion: '+name+' '+state['status'])\n    return state\n\n\ndef run")
p.write_text(s)
print('Resume review fixes complete; zero actual calls')
