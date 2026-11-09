"""Unchanged archived evaluator with a new isolated endpoint IO runner."""
from .common import ROOT,EVALUATION,EVALUATION_SHA,read_json,write_json,file_sha256
import importlib.util
from pathlib import Path
import time
import sys
import numpy as np

REFERENCE=ROOT/'native_t1/runs/edge_factorial_20261001_123000/source_snapshot/experiments/mf_h1_local0/edge_factorial_pilot'

def reference_evaluator():
    for name,filename in (('edge_metric','edge_metric.py'),('_reference_evaluator','evaluation.py')):
        key=__package__+'.'+name
        if key not in sys.modules:
            spec=importlib.util.spec_from_file_location(key,REFERENCE/filename)
            module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module)
    return sys.modules[__package__+'._reference_evaluator']

def banks_with_context_audit(out, *, initialize=False):
    ref=reference_evaluator();banks=ref.load_bank(EVALUATION,EVALUATION_SHA)
    cache=Path(out)/'registration/static_context_audit.json'
    if cache.exists():
        saved=read_json(cache)
        if (saved.get('status') != 'PASS' or saved.get('source_sha256') != file_sha256(REFERENCE/'evaluation.py')
                or saved.get('evaluation_inputs_sha256') != EVALUATION_SHA):
            raise ValueError('Static context cache identity differs')
        for j,bank in banks.items():bank['context_audits']={int(p):v for p,v in saved['contexts'][str(j)].items()}
    else:
        if not initialize:
            raise RuntimeError('Static context cache is absent; root must initialize it once before scoring workers')
        ledger={};contexts={};tick=time.perf_counter()
        for j,bank in banks.items():
            bank['context_audits']={p:ref.audit_context(bank['cases_by_key'][p,12]['constraints'],bank['cases_by_key'][p,12]['bbox_diagonal'],ledger) for p in (0,1)}
            contexts[str(j)]=bank['context_audits']
        write_json(cache,dict(status='PASS',contexts=contexts,ledger=ledger,seconds=time.perf_counter()-tick,
            source_sha256=file_sha256(REFERENCE/'evaluation.py'),evaluation_inputs_sha256=EVALUATION_SHA,model_forwards=0,OT_calls=0))
    return banks

def score_one(raw,bank,parent,patch,ledger):
    ref=reference_evaluator();raw=np.asarray(raw).reshape(112,3,3)
    # The existing scorer permits explicit NOT_RUN attribution while preserving
    # all distance/CF/FF/shape checks. This task does not request topology repair.
    topo=dict(status='NOT_RUN',reason='No additional topology certification in this diagnostic',ledger={})
    value=ref.score_output(raw,bank,parent,ledger=ledger,topology=topo)
    ref._metric_reference_checks(value,bank,parent,patch)
    value['edge_factorial_boundary']=ref._boundary_extension(raw,bank,parent,patch,value,ledger)
    return value,ref.scalar_metrics(value)

def score_rows(out,rows,phase):
    if not isinstance(rows,list) or not rows:
        raise ValueError('An explicit nonempty list of registered endpoint rows is required')
    if not isinstance(phase,str) or not phase or any(x in phase for x in ('/','\\','..')):
        raise ValueError('Scoring phase must be a single isolated directory name')
    out=Path(out);folder=out/'scores'/phase
    folder.mkdir(parents=True,exist_ok=True)
    resultfile=folder/'results.json'
    if resultfile.exists():raise FileExistsError(resultfile)
    banks=banks_with_context_audit(out);ledger={};saved=[];tick=time.perf_counter()
    result=dict(status='RUNNING',phase=phase,rows=saved,ledger=ledger,model_forwards=0,repairs=0)
    write_json(resultfile,result)
    try:
        for i,row in enumerate(rows):
            key=row.get('key') or f"{phase}_{i:04d}"
            if not isinstance(key,str) or not key or any(x in key for x in ('/','\\','..')):
                raise ValueError('Score key must be a single filename stem')
            relative=row.get('path') or row.get('raw_path')
            path=reference_evaluator()._path(relative)
            if not path.is_absolute():path=out/path
            if not path.is_file():raise FileNotFoundError(path)
            expected=[row[k] for k in ('sha256','raw_sha256') if row.get(k)]
            if not expected:raise ValueError('Raw file SHA256 is required')
            actual=file_sha256(path)
            if any(actual!=value for value in expected):raise ValueError('Raw file changed')
            with np.load(path,allow_pickle=False) as z:
                raw=z['raw'].copy() if 'raw' in z else z['output'].copy()
            parent,patch=int(row['parent']),int(row['patch'])
            t=time.perf_counter();metrics,scalars=score_one(raw,banks[patch],parent,patch,ledger)
            detail=folder/(key+'.json');write_json(detail,metrics)
            saved.append(dict(**row,score_path=str(detail.relative_to(out)),scalars=scalars,
                cross_remainder=metrics['remainder']['cross'],seconds=time.perf_counter()-t))
            result['completed']=len(saved);result['seconds']=time.perf_counter()-tick
            write_json(resultfile,result)
            print('score',phase,len(saved),'/',len(rows),flush=True)
        result['status']='PASS'
    except BaseException as exc:
        result['status']='FAIL';result['error']=repr(exc);raise
    finally:
        result['seconds']=time.perf_counter()-tick;write_json(resultfile,result)
    return result


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description="Read-only fixed endpoint CPU geometry scoring")
    parser.add_argument('--out',required=True)
    parser.add_argument('--phase',required=True)
    parser.add_argument('--rows-json',required=True)
    args=parser.parse_args(argv)
    manifest=read_json(args.rows_json)
    rows=manifest if isinstance(manifest,list) else manifest.get('rows')
    if rows is None:raise ValueError('Rows JSON must be a list or an object with a rows list')
    score_rows(args.out,rows,args.phase)

if __name__=='__main__': main()
