"""Engineering coordinator only: completed-result CPU scoring and final views.

Never generates or changes a score definition. A failed scoring/rendering stage
stops with its evidence intact. Restart skips validated complete stages; it does
not silently retry a previously failed stage.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
    os.environ[_key]='2'
from . import evaluation as ev
from . import analysis


def utc():return datetime.now(timezone.utc).isoformat()


def run(out=ev.OUT,*,render_final=False,allow_failed_resume=False):
    out=ev.assert_out(out);status_path=out/'status/score_watch.json';log=out/'logs/score_watch.log'
    log.parent.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    old=ev.read_json(status_path) if status_path.exists() else {}
    if old.get('status')=='BLOCKED' and not allow_failed_resume:
        raise RuntimeError('Previous watcher failure retained; explicit engineering resume required')
    attempt=int(old.get('attempt',0))+1
    state=dict(version='gradbudget_cpu_watch_v1',attempt=attempt,pid=os.getpid(),started_utc=utc(),status='RUNNING',
        scoring_workers=2,threads_per_worker=2,max_model_workers=0,model_calls=0,new_GT_queries=0,
        sources={name:ev.digest(Path(ev.__file__).parent/(name+'.py')) for name in ('evaluation','analysis','visualization')},
        coordinator_sha256=ev.digest(__file__),completed=[],pending=[],active=None,render_final=render_final)
    def save():
        state.update(updated_utc=utc(),wall_seconds=time.perf_counter()-started);ev.write_json(status_path,state)
    def note(event,**fields):
        item=dict(utc=utc(),attempt=attempt,event=event,**fields)
        with log.open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
        print(json.dumps(item,ensure_ascii=False),flush=True)
    def frozen_sources():
        if any(ev.digest(Path(ev.__file__).parent/(name+'.py'))!=sha for name,sha in state['sources'].items()):
            raise ValueError('Frozen scoring/analysis/visualization source changed during watcher')
    def completed(label,panel):
        folder=out/'evaluation'/label/panel;path=folder/'summary.json';rp=folder/'per_output.json'
        if not path.exists() or not rp.exists():return None
        summary=ev.read_json(path);rows=ev.read_json(rp);requests=ev.manifest(out,panel)
        if summary.get('status')!='COMPLETE' or len(rows)!=16 or any(r.get('status')!='PASS' for r in rows):return None
        if {r['request_id'] for r in rows}!={r['request_id'] for r in requests}:raise ValueError('Complete panel IDs changed')
        # Per-row cache checks are read-only, including all declared vector hashes.
        for q in requests:
            dest=folder/'scoring/joined'/(q['request_id']+'.json')
            if not dest.exists():raise ValueError('Complete panel lost joined result')
            r=ev.evaluate_request(q,out_dir=out,label=label)
            if r.get('status')!='PASS':raise ValueError('Completed score failed identity reuse')
        return dict(label=label,panel=panel,status='COMPLETE',outputs=16,per_output_sha256=ev.digest(rp),
            summary_sha256=ev.digest(path),historical_scores_reused=summary.get('historical_scores_reused',0),
            new_scored_outputs=summary.get('new_scored_outputs',0),first_scoring_invocation_seconds=summary.get('invocation_seconds'))
    def ready(label,panel):
        missing=[]
        for request in ev.manifest(out,panel):
            path,result,raw=ev.result_for(out,label,request)
            if result.get('status') in ('FAIL','FAILED','BLOCKED'):
                raise RuntimeError('Generation failed; no silent retry: '+str(path)+' '+str(result.get('reason')))
            if result.get('status')!='PASS':missing.append(request['request_id']);continue
            if not raw.exists() or not result.get('raw_sha256') or ev.digest(raw)!=result['raw_sha256']:
                raise ValueError('Completed result lacks its exact RAW: '+str(path))
        return not missing,len(missing)
    plan=[(label,panel) for label in ev.LABELS for panel in ev.PANELS]
    try:
        note('WATCH_START',render_final=render_final)
        done={}
        for label,panel in plan:
            rec=completed(label,panel)
            if rec:
                done[(label,panel)]=rec;note('PANEL_REUSED',label=label,panel=panel,outputs=16)
        last_heartbeat=0.
        while len(done)<len(plan):
            frozen_sources();pending=[];progress=False
            for label,panel in plan:
                if (label,panel) in done:continue
                isready,missing=ready(label,panel)
                if not isready:pending.append(dict(label=label,panel=panel,missing_completed_results=missing));continue
                state.update(active=dict(label=label,panel=panel),completed=list(done.values()),pending=pending);save()
                note('SCORE_BEGIN',label=label,panel=panel,workers=2)
                tick=time.perf_counter();summary=ev.score_set(out,label,panel,workers=2)
                note('SCORE_RETURN',label=label,panel=panel,status=summary['status'],completed=summary['completed'],wall_seconds=time.perf_counter()-tick)
                if summary['status']!='COMPLETE':
                    raise RuntimeError('CPU score panel incomplete; preserved rows, no automatic retry: '+label+'/'+panel)
                rec=completed(label,panel)
                if not rec:raise ValueError('Scoring returned complete without durable complete artifacts')
                done[(label,panel)]=rec;progress=True;state['active']=None
            state.update(completed=list(done.values()),pending=pending,completed_cells=16*len(done));save()
            if time.perf_counter()-last_heartbeat>=60:
                note('WAIT',completed_cells=16*len(done),pending_panels=len(plan)-len(done));last_heartbeat=time.perf_counter()
            if not progress:time.sleep(5)
        frozen_sources();state.update(status='AGGREGATING',active=None,completed_cells=288,pending=[]);save()
        result=analysis.run(out)
        if result['status']!='COMPLETE':raise ValueError('All score panels complete but aggregation incomplete')
        state['analysis']=dict(path=str(out/'analysis/results.json'),sha256=ev.digest(out/'analysis/results.json'),
            question_A_status=result['question_A_status'],question_B_status=result['question_B_status'])
        note('ALL_SCORED',cells=288,question_A_status=result['question_A_status'],question_B_status=result['question_B_status'])
        if render_final:
            from . import visualization
            cameras,camera_path=visualization.prepare_cameras(out)
            if cameras['status']!='FINAL_COMPLETE':raise ValueError('Final shared camera missing registered meshes')
            state['camera_manifest']=str(camera_path);state['galleries']=[]
            for panel in ev.PANELS:
                state.update(status='RENDERING',active=dict(panel=panel,labels=list(ev.LABELS)));save();note('RENDER_BEGIN',panel=panel)
                gallery=visualization.generate(out,ev.LABELS,panel)
                if gallery['status']!='COMPLETE' or gallery['camera_status']!='FINAL_COMPLETE':raise ValueError('Final gallery incomplete')
                state['galleries'].append({key:gallery[key] for key in ('panel','index','contact_sheet','camera_status','wall_seconds','unique_mesh_panels','overlay_panels')})
                note('RENDER_RETURN',panel=panel,index=gallery['index'],wall_seconds=gallery['wall_seconds'])
        state.update(status='COMPLETE',active=None,completed_utc=utc());save();note('WATCH_COMPLETE',cells=288)
        return state
    except BaseException as exc:
        state.update(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc());save();note('WATCH_BLOCKED',error=repr(exc))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ev.OUT));p.add_argument('--render-final',action='store_true');p.add_argument('--allow-failed-resume',action='store_true');a=p.parse_args()
    run(a.out,render_final=a.render_final,allow_failed_resume=a.allow_failed_resume)
