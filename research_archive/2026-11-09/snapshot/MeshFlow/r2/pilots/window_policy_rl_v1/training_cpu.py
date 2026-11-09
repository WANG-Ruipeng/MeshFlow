"""Four single-thread CPU workers for one authorized eight-rollout RL batch.

This wrapper changes scheduling only. Each cell is the unchanged maintained
postprocess._cpu_cell, with identical scorer calls, stages and result schema.
The controller must record fresh resource admission before invoking this module.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import time
from . import postprocess as P
D,E=P.bind()

VERSION='window_policy_rl_training_cpu_four_workers_v1'
WORKERS=4


def prepare(jobs,*,manifest_path,registration_path,stages,resource_snapshot_path):
    """CPU-only frozen input validation; no scorer invocation or new geometry."""
    P.torch_free()
    if not stages or len(set(stages))!=len(stages) or any(s not in ('post','score') for s in stages):
        raise ValueError('Only original post/score stages are supported')
    stages=tuple(s for s in ('post','score') if s in stages)
    if len(jobs)!=1 or jobs[0]['label']!='RL_TRAIN' or len(jobs[0].get('requests',[]))!=8:
        raise ValueError('This scheduling profile is exactly one training group of eight')
    qs=jobs[0]['requests']
    if len({q['condition_id'] for q in qs})!=1 or len({q['batch_index'] for q in qs})!=1:
        raise ValueError('Eight requests must belong to one frozen on-policy condition/batch')
    stage=P._checked(jobs[0]['stage_dir'])
    if not stage.is_relative_to((P.OUT/'training').resolve()):
        raise ValueError('Four-worker profile is scoped to this experiment training')
    rows,identity=P.prepare_jobs(jobs,manifest_path=manifest_path,registration_path=registration_path)
    for row in rows:P._proposal(row)
    resource=D.resolve(resource_snapshot_path)
    if not resource.is_file():raise ValueError('Controller resource snapshot is required')
    identity.update(phase='cpu',stages=list(stages),orchestration_version=VERSION,
        orchestration_source_sha256=D.sha(__file__),workers=WORKERS,CPU_threads_per_worker=1,
        pool_start_method='spawn',result_order='pool.map input request order',
        resource_snapshot_path=str(resource),resource_snapshot_sha256=D.sha(resource),
        scientific_definition='Unmodified postprocess._cpu_cell and original score_one; scheduling only',
        source_hashes={str(Path(__file__)):D.sha(__file__),str(Path(P.__file__)):D.sha(P.__file__),str(Path(E.__file__)):D.sha(E.__file__)})
    P.torch_free()
    return rows,identity,stages


def reject_partial_cells(rows,stages):
    """A new phase cannot silently adopt or replay a partly processed batch."""
    for row in rows:
        post=Path(row['out_dir']);score=Path(row['score_dir'])
        if 'post' in stages and post.exists():
            raise RuntimeError('POST directory already exists without this completed ledger; explicit recovery required: '+str(post))
        if 'post' not in stages:
            receipt=D.read(post/'receipt.json')
            if receipt.get('status')!='COMPLETE':raise ValueError('Score-only requires completed POST')
            P._verify(receipt['files'])
        if 'score' in stages and score.exists():
            raise RuntimeError('Score directory already exists without this completed ledger; explicit recovery required: '+str(score))


def run_cpu(jobs,*,manifest_path,registration_path,ledger_dir,resource_snapshot_path,
            stages=('post','score'),workers=WORKERS):
    if workers!=WORKERS:raise ValueError('This separately admitted profile fixes four workers, one thread each')
    rows,identity,stages=prepare(jobs,manifest_path=manifest_path,registration_path=registration_path,
        stages=stages,resource_snapshot_path=resource_snapshot_path)
    ledger=P._checked(ledger_dir)
    saved=P._completed(ledger,identity)
    if saved is not None:
        if (len(saved['rows'])!=8 or [r['request_id'] for r in saved['rows']]!=[r['request']['request_id'] for r in rows]
                or saved['workers']!=WORKERS or saved['model_calls']!=0):
            raise ValueError('Completed four-worker batch receipt coverage differs')
        P.torch_free()
        return saved
    reject_partial_cells(rows,stages)
    D.write(ledger/'started.json',identity)
    tick=time.perf_counter()
    try:
        tasks=[(row,stages,identity['registration_sha256']) for row in rows]
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            results=list(pool.map(P._cpu_cell,tasks))
        expected=[r['request']['request_id'] for r in rows]
        if len(results)!=8 or [r['request_id'] for r in results]!=expected:
            raise RuntimeError('Deterministic result coverage/order differs')
        if any(r['model_calls']!=0 for r in results):raise RuntimeError('CPU worker unexpectedly reported a model call')
        files={}
        for row in results:
            for path,sha in row['files'].items():
                if path in files and files[path]!=sha:raise ValueError('Conflicting output file identities')
                files[path]=sha
        P._verify(files);P.torch_free()
        result=dict(status='COMPLETE',version=VERSION,identity=identity,rows=results,requests=len(rows),
            stages=list(stages),files=files,workers=WORKERS,CPU_threads_per_worker=1,
            model_calls=0,model_calls_this_invocation=0,torch_imported=False,
            seconds=time.perf_counter()-tick,semantic_review='NOT_ASSIGNED; requires actual separate image review',
            scheduling_only=True,original_cell_source_sha256=D.sha(P.__file__))
        D.write(ledger/'completion.json',result)
        return result
    except BaseException as error:
        D.write(ledger/'failure.json',dict(status='STOPPED',error=repr(error),automatic_retry=False,
            model_calls_this_invocation=0,scheduling_only=True))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True)
    parser.add_argument('--phase',choices=('cpu','verify-inputs'),default='cpu')
    parser.add_argument('--workers',type=int,choices=(WORKERS,),default=None)
    parser.add_argument('--stages',default=None)
    args=parser.parse_args();spec=D.read(args.job)
    workers=args.workers if args.workers is not None else spec.get('workers',WORKERS)
    if workers!=WORKERS:raise ValueError('Job must explicitly select or default to four workers')
    common=dict(jobs=spec['jobs'],manifest_path=spec['manifest_path'],registration_path=spec['registration_path'],
        resource_snapshot_path=spec['resource_snapshot_path'],stages=args.stages.split(',') if args.stages else spec.get('stages',('post','score')))
    if args.phase=='verify-inputs':
        rows,identity,stages=prepare(**common)
        result=dict(status='PASS',requests=len(rows),identity=identity,model_calls_this_invocation=0)
    else:result=run_cpu(**common,ledger_dir=spec['ledger_dir'],workers=workers)
    print(json.dumps({k:result[k] for k in ('status','requests','workers','model_calls_this_invocation',
        'seconds','cache_verified') if k in result},sort_keys=True),flush=True)


if __name__=='__main__':main()
