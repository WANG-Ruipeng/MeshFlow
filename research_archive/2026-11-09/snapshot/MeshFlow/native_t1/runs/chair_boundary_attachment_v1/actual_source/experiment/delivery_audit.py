"""Final read-only protected-file audit and small actual-source delivery."""
from pathlib import Path
import difflib
import shutil
import subprocess
from .common import *


def run(out=OUT):
    out=Path(out)
    from .protocol import verify
    verify(out,training=True)
    protected={}
    for filename in ('registration/source_hashes_before.json','protocol/protected_sources_before.json','protocol/evaluation_sources_before.json'):
        values=read_json(out/filename)
        if not isinstance(values,dict):raise ValueError('Unexpected source hash schema')
        protected.update(values)
    protected[str(START)]=START_SHA
    checked=[]
    for value,expected in protected.items():
        path=resolve(value)
        # Registration hashes map actual files directly; do not walk the warehouse.
        observed=digest(path)
        checked.append(dict(path=str(path),before=expected,after=observed,unchanged=observed==expected))
    if not all(x['unchanged'] for x in checked):raise RuntimeError('Protected source/data/cache changed')
    raw=[]
    for request in read_json(out/'requests.json'):
        p=out/'outputs'/request['request_id']/'result.json'
        if not p.exists():
            raw.append(dict(request_id=request['request_id'],status='BLOCKED'));continue
        result=read_json(p)
        entry={k:result.get(k) for k in ('request_id','base_request_id','label','arm','panel','uid','condition_id','candidate','status','raw_path','raw_sha256','checkpoint_sha256')}
        if result['status'] in ('SAMPLED','CACHED'):
            entry['verified_raw_sha256']=digest(resolve(result['raw_path']))
            assert entry['verified_raw_sha256']==entry['raw_sha256']
            entry['NFE']=result['metadata']['nfe'];entry['C_all51_bitwise']=result['metadata']['known_all51_bitwise']
            assert entry['NFE']==50 and entry['C_all51_bitwise']
        raw.append(entry)
    atomic_json(out/'raw_integrity_index.json',raw)
    from experiments.mf_h1_local0.chair_control_usability_quick_v1.scoring import write_csv
    write_csv(out/'raw_integrity_index.csv',raw)
    sources={str(p.relative_to(ROOT)):digest(p) for p in sorted(CODE.glob('*.py'))}
    sources.update(read_json(out/'protocol/protected_sources_before.json'))
    actual=out/'actual_source';actual.mkdir(parents=True,exist_ok=True)
    diff=[]
    for p in sorted(CODE.glob('*.py')):
        dst=actual/'experiment'/p.name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dst)
        diff+=difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(ROOT)))
    # A small runnable scoring source collection, including the unchanged original
    # quality/classification helpers that supplied actual reported definitions.
    copied=[]
    extra=['experiments/mf_h1_local0/chair_control_usability_quick_v1/scoring.py',
        'experiments/mf_h1_local0/evaluate.py','experiments/mf_h1_local0/native_inpainting_metrics.py',
        'experiments/mf_h1_local0/native_patch_seams.py','experiments/mf_h1_local0/guided_native_geometry_audit.py',
        'experiments/mf_h1_local0/guided_native_geometry_geometry.py',
        'experiments/mf_h1_local0/control_mvp48/eval_scoring.py',
        'experiments/mf_h1_local0/gpu_guidance_evaluation.py',
        'native_t1/runs/edge_factorial_20261001_123000/source_snapshot/experiments/mf_h1_local0/edge_factorial_pilot/edge_metric.py']
    for name in extra:
        src=ROOT/name;dst=actual/'original_scoring'/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst);sources[name]=digest(src);copied.append(dict(original=name,copy=str(dst.relative_to(out)),sha256=digest(src)))
    (out/'code.diff').write_text(''.join(diff),encoding='utf-8')
    atomic_json(out/'source_hashes.json',sources)
    atomic_json(out/'actual_source/README.json',dict(experiment='Current executed experiment sources; old revisions retained under engineering/',original_scoring=copied,
        dependency_note='These small files document actual scoring code. Run with the existing MeshFlow workspace/environment; no new dependency installation.'))
    git={}
    for name,args in [('HEAD',['rev-parse','HEAD']),('status',['status','--short']),('tracked_diff',['diff','--stat'])]:
        r=subprocess.run(['git','-c','core.autocrlf=true',*args],cwd=ROOT,text=True,capture_output=True,timeout=20)
        git[name]=dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
    atomic_json(out/'delivery_git_audit.json',dict(policy='Match existing Windows core.autocrlf=true using command-only override; protected file bytes separately SHA256 verified',results=git))
    assert git['tracked_diff']['returncode']==0 and not git['tracked_diff']['stdout']
    summary=dict(status='PASS',protected_files=len(checked),all_unchanged=True,rows=checked,git=git,
        model_forward_calls=0,backward_calls=0,optimizer_updates=0,new_generation=0,
        no_automatic_commit_or_default_change=True,source_hash_count=len(sources),raw_index_rows=len(raw))
    atomic_json(out/'protected_after_audit.json',summary)
    return {k:v for k,v in summary.items() if k not in ('rows','git')}


if __name__=='__main__':print(run())
