"""Measured rendering of the 24 registered FINAL comparison pages, <=2 workers.

This helper is not a scientific dispatch controller. Root invokes it only
after resource review and all required producers are complete. Existing pages
are verified by frozen runtime.render_one; incomplete pages are never erased.
"""
from .. import runtime as R
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import os
import time


def worker(jobs_path,job_id):
    page=R.job_dir(job_id)/'gallery/CLEAN_FINAL_POST.json';existed=page.exists()
    before=R.resources();tick=time.perf_counter()
    result=R.render_one(jobs_path,job_id,'CLEAN_FINAL_POST')
    after=R.resources()
    return dict(job_id=job_id,verified_existing=existed,page=str(page),page_sha256=R.sha(page),
        image_path=result['image_path'],image_sha256=result['image_sha256'],
        process_peak_RSS_KiB=after['peak_RSS_KiB'],peak_scope='Fresh render subprocess lifetime peak',
        pid=os.getpid(),seconds=time.perf_counter()-tick,resources_before=before,resources_after=after)


def run(jobs_path):
    reg,rh,jobs=R.load_jobs(jobs_path)
    selected=[j for j in reg['jobs'] if j['label']!='RAW_ALL']
    if len(selected)!=24:raise ValueError('Expected exactly 24 registered comparison pages')
    root=R.OUT/'render_queue';root.mkdir(exist_ok=True)
    done=root/'complete.json'
    if done.exists():
        result=R.read(done)
        if result['registration_sha256']!=rh:raise ValueError('Render registration changed')
        R.verify(result['files']);return result
    # Check every prerequisite before creating an attempt; no backend invocation.
    for job in reg['jobs']:
        receipt=R.verify_result(R.read(R.job_dir(job['job_id'])/'complete.json'))
        if receipt['registration_sha256']!=rh:raise ValueError('Producer registration differs')
    number=1
    while (root/f'attempt_{number:03d}.json').exists():number+=1
    lock=root/'active.lock'
    with lock.open('x',encoding='utf-8') as handle:handle.write(str(os.getpid()))
    tick=time.perf_counter();files={};records=[];workers=1
    try:
        R.write_new(root/f'attempt_{number:03d}.json',dict(status='STARTED',pid=os.getpid(),utc=R.utc(),
            registration_sha256=rh,maximum_workers=2,threads_per_worker=1,helper_sha256=R.sha(__file__),
            resources=R.resources(),automatic_retry=False,scientific_calls=0))
        def accept(record):
            files[record['page']]=record['page_sha256'];files[record['image_path']]=record['image_sha256']
            records.append(record)
            print(record['job_id']+(' VERIFIED' if record['verified_existing'] else ' RENDERED'),flush=True)
        pending=list(selected)
        # Choose an actually unrendered authorized page for representative memory.
        first=next((j for j in pending if not (R.job_dir(j['job_id'])/'gallery/CLEAN_FINAL_POST.json').exists()),pending[0])
        pending.remove(first)
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            measured=pool.submit(worker,jobs_path,first['job_id']).result();accept(measured)
            now=R.resources();peak=measured['process_peak_RSS_KiB']*1024
            margin=max(2*(1<<30),.1*now['ram_total_bytes'])
            slots=int(max(0,now['ram_available_bytes']-margin)//(1.2*peak)) if peak>0 else 1
            workers=min(2,slots,max(1,now['cpu_logical']//2))
            if workers<1:raise RuntimeError('Insufficient measured render memory headroom')
            R.write_new(root/f'measured_dispatch_{number:03d}.json',dict(first=first['job_id'],chosen_workers=workers,
                observed_peak_bytes=peak,resources=now,first_was_existing=measured['verified_existing']))
            active={}
            while pending or active:
                while pending and len(active)<workers:
                    now=R.resources();margin=max(2*(1<<30),.1*now['ram_total_bytes'])
                    if now['ram_available_bytes']-margin<1.2*peak:break
                    job=pending.pop(0);future=pool.submit(worker,jobs_path,job['job_id']);active[future]=job
                if not active:raise RuntimeError('Resource headroom disappeared; preserve unstarted render queue')
                finished,_=wait(active,return_when=FIRST_COMPLETED)
                for future in finished:
                    record=future.result();accept(record);active.pop(future)
                    peak=max(peak,record['process_peak_RSS_KiB']*1024)
        records.sort(key=lambda r:next(i for i,j in enumerate(selected) if j['job_id']==r['job_id']))
        result=dict(status='COMPLETE',registration_sha256=rh,pages=24,cells=96,records=records,
            workers=workers,seconds_this_attempt=time.perf_counter()-tick,resources_final=R.resources(),
            model_calls=0,GEO_calls=0,scoring_calls=0,ratings_generated=0,files=files)
        R.write_new(done,result);return result
    except BaseException as error:
        R.write_new(root/f'failure_{number:03d}.json',dict(utc=R.utc(),error=repr(error),
            completed_pages_this_attempt=records,recovery='Review partial page; never automatically delete/replay it',
            model_calls=0,GEO_calls=0,scoring_calls=0))
        raise
    finally:
        # Remove only this process's explicit bookkeeping lock, never assets.
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',required=True);a=p.parse_args()
    r=run(a.jobs);print({k:r[k] for k in ('status','pages','cells','workers')})
