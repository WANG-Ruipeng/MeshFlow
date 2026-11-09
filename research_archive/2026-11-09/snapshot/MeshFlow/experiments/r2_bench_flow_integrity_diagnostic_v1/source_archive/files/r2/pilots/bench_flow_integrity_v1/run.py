"""Recover completed stages and seal/verify evidence. Never initiates new science."""
from .common import *
import argparse,difflib,os,re,shutil
from datetime import datetime,timezone

def archive():
    dest=OUT/'source_archive'; receipt=dest/'receipt.json'
    if receipt.exists():
        r=read(receipt)
        for p,h in r['archived_sha256'].items():assert sha(p)==h
        return r
    sources=set(Path(__file__).parent.glob('*.py'))
    sources.update(p for p in (OUT/'audits').rglob('*.py'))
    inherited={resolve(r['path']) for r in read(OUT/'audits/source_contract/manifest.json')['source_inventory']}
    legacy=ROOT.parent/'MeshFlow-Control/src/meshflow_control/evaluation'
    inherited.update(legacy/name for name in ('intersections.py','seams.py'))
    copied={};hashes={};patch=[]
    for p in sorted(sources|inherited,key=str):
        relative=p.relative_to(ROOT) if p.is_relative_to(ROOT) else Path('external')/p.relative_to(ROOT.parent)
        output=dest/'files'/relative;output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():assert sha(output)==sha(p)
        else:shutil.copyfile(p,output)
        copied[str(output)]=sha(output);hashes[str(p)]=sha(p)
        if p in sources:
            patch.extend(difflib.unified_diff([],p.read_text('utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+relative.as_posix()))
    diff=dest/'code.diff';diff.write_text(''.join(patch),encoding='utf-8',newline='\n');copied[str(diff)]=sha(diff)
    r=dict(status='COMPLETE',new_sources=len(sources),copied_sources=len(hashes),source_sha256=hashes,
        archived_sha256=copied,diff_scope='Only this diagnostic new source files; not the whole dirty working tree',
        old_source_modified=False,weights_copied=False,git_actions=False)
    write_new(receipt,r);return r

def verify():
    complete=read(OUT/'COMPLETE.json')
    for p,h in complete['artifacts_sha256'].items():assert sha(p)==h,p
    print(dict(status='COMPLETE_VERIFIED',files=len(complete['artifacts_sha256']),model_calls=0,updates=0))
    return complete

def seal():
    if (OUT/'COMPLETE.json').exists():return verify()
    final=read(OUT/'audits/final_check.json');assert final['status'] in ('PASS','AUDIT_PASS')
    reg=read(OUT/'registration.json')
    for p,h in reg['protected'].items():assert sha(p)==h,p
    assert read(OUT/'trajectory/risk_gate.json')['status']=='NOT_TRIGGERED'
    assert read(OUT/'dispatch/completion.json')['forwards']==200
    assert read(OUT/'audits/source_contract/summary.json')['production_OT_calls']==16
    assert read(OUT/'probe_analysis_v1/COMPLETE.json')['registered_forward_artifacts']==200
    archive()
    for x in re.findall(r'\]\(([^)]+)\)',(OUT/'REPORT.md').read_text('utf-8')):
        if '://' not in x:assert (OUT/x).exists(),x
    import psutil
    live=[]
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            if p.pid!=os.getpid() and any(s in ('r2.pilots.bench_flow_integrity_v1.probe','r2.pilots.bench_flow_integrity_v1.dispatch') for s in p.info['cmdline'] or []):live.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not live,live
    write_new(OUT/'resources_final.json',dict(status='NO_EXPERIMENT_GPU_WORKERS',resources=resources(),live_experiment_workers=live))
    files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.pyc',) and '.partial' not in p.name]
    done=dict(status='COMPLETE',experiment='R2_BENCH_FLOW_INTEGRITY_DIAGNOSTIC_V1',completed_utc=datetime.now(timezone.utc).isoformat(),
        counts=dict(CPU_OT=16,existing_complete_trajectories=108,frozen_probe_forwards=200,backwards=0,updates=0,
            head_forwards=0,new_rollouts=0,D0=0,new_mesh_scoring=0,failed_scientific_calls=0,replayed_scientific_calls=0),
        Euler100='NOT_RUN_GATE_NOT_TRIGGERED; numerical integration error not excluded',
        decision='No global FT-damaged-flow evidence; local new-condition gains plus limited transfer; preserve RAW support through backend is priority',
        artifacts_sha256={str(p):sha(p) for p in files},source_archive_sha256=sha(OUT/'source_archive/receipt.json'),
        old_assets_modified=False,default_changed=False,commit=False,push=False,automatic_next_experiment=False)
    write_new(OUT/'COMPLETE.json',done)
    print({k:v for k,v in done.items() if k!='artifacts_sha256'});return done

def status():
    if (OUT/'COMPLETE.json').exists():return verify()
    phases={'registration':'registration.json','CPU_contract':'audits/source_contract/summary.json',
        'trajectories':'trajectory/completion.json','probes':'dispatch/completion.json',
        'probe_analysis':'probe_analysis_v1/COMPLETE.json','final_audit':'audits/final_check.json'}
    result={k:read(OUT/p).get('status') if (OUT/p).exists() else 'PENDING' for k,p in phases.items()}
    print(dict(stages=result,model_calls=0,note='See RUNBOOK; no unresolved scientific attempts are automatically replayed'))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args()
    seal() if a.seal else status()
