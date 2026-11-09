"""One six-column CPU renderer, overlapping completed per-request scores only."""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path
from . import common as C
from . import render as R
from .evaluate import locations, tasks_from

def main():
    logdir=C.OUT/'render_stream';done=logdir/'complete.json';attempt=logdir/'attempt.json'
    if done.exists():
        result=C.read(done);C.verify(result['files']);print(json.dumps(result));return
    if attempt.exists():raise RuntimeError('Existing render-stream attempt; no concurrent/implicit second writer')
    logdir.mkdir(parents=True,exist_ok=True)
    C.write(attempt,dict(status='RUNNING',pid=os.getpid(),time=time.time(),renderer_sha256=C.sha(R.__file__),
        source_sha256=C.sha(__file__),workers=1,threads_per_worker=1,wait_limit_seconds=1800))
    C.resource_snapshot('render_stream_before')
    start=time.monotonic();m,tasks=tasks_from();mh=C.sha(C.OLD/'manifest.json');finished=set()
    try:
        while len(finished)<len(tasks):
            if time.monotonic()-start>1800:raise TimeoutError('CPU geometry/render readiness exceeded30minutes')
            failures=[p for name in ('geometry','cpu','scores') for p in (C.OUT/name).rglob('failure.json')]
            if failures:raise RuntimeError('Upstream CPU failure exists: '+str(failures[0]))
            made=False
            for t in tasks:
                rid=t['request_id']
                if rid in finished:continue
                loc=locations(t);receipts=[loc['geometry']/'complete.json',loc['cpu']/'complete.json']
                if not all(p.exists() for p in receipts):continue
                for p in receipts:
                    receipt=C.read(p)
                    if receipt['status']!='COMPLETE':raise RuntimeError('Upstream receipt is not complete: '+str(p))
                    C.verify(receipt['files'])
                r=R.one(t,mh);finished.add(rid);made=True
                with (logdir/'pages.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(request_id=rid,finished=len(finished),image_sha256=r['image_sha256'],time=time.time()))+'\n')
                print('STREAM_RENDER',len(finished),rid,flush=True)
            if not made:time.sleep(2)
        result=R.run();C.resource_snapshot('render_stream_finished')
        C.write(done,dict(status='COMPLETE',pages=len(finished),workers=1,elapsed_seconds=time.monotonic()-start,
            model_calls=0,new_scoring_calls=0,files={str(C.OUT/'gallery/COMPLETE.json'):C.sha(C.OUT/'gallery/COMPLETE.json'),
                                                  str(logdir/'pages.jsonl'):C.sha(logdir/'pages.jsonl')}))
        print('STREAM_COMPLETE',len(finished),flush=True)
    except BaseException as error:
        C.write(logdir/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),finished=len(finished),
            elapsed_seconds=time.monotonic()-start,no_automatic_retry=True));raise

if __name__=='__main__':main()
