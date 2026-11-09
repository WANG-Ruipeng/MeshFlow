"""One render worker overlaps the one existing scoring worker, max two CPUs."""
from . import common as C,render as R
import time

def run():
    plan=R.prepare();dh=C.sha(C.OUT/'gallery/manifest.json');done=C.OUT/'gallery/completion.json'
    if done.exists():r=C.read(done);C.verify(r['files']);return r
    C.write(C.OUT/'gallery/stream_dispatch.json',dict(utc=C.utc(),workers=1,threads_per_worker=1,resources=C.resources(),
        other_worker='Frozen scorer actual worker=1; total at most2',model_calls=0,score_calls=0))
    tick=time.perf_counter();remaining=list(plan['pages']);finished={}
    while remaining:
        progressed=False
        for page in remaining[:]:
            rid=page['request']['request_id'];receipt=C.OUT/'outputs'/rid/'complete.json'
            if (receipt.parent/'failure.json').exists():raise RuntimeError('Scorer failed; renderer retained completed pages')
            if not receipt.exists():continue
            C.verify(C.read(receipt)['files'])
            r=R.one((page,dh));finished[page['page_id']]=r;remaining.remove(page);progressed=True
            print('RENDERED',page['page_id'],flush=True)
        if not progressed:time.sleep(3)
    # Ensure the scorer's full collection finishes before final gallery seal.
    while not (C.OUT/'completions/geometry_scores.json').exists():time.sleep(3)
    C.verify(C.read(C.OUT/'completions/geometry_scores.json')['files'])
    return R.finish(plan,[finished[p['page_id']] for p in plan['pages']],1,time.perf_counter()-tick,streaming=True)

if __name__=='__main__':print(run())
