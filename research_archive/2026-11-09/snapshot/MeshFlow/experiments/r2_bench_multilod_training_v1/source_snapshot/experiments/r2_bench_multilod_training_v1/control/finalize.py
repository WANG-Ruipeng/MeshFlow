"""Read-only identity/call audit and small source archive; never runs models/scorers."""
from pathlib import Path
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_bench_multilod_training_v1'
ARMS=('PLAIN1500','MULTILOD1500')
HEX=re.compile(r'^[0-9a-f]{64}$')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_once(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()
    if p.exists():
        if p.read_bytes()!=data:raise RuntimeError(f'No overwrite: {p}')
    else:
        with p.open('xb') as f:f.write(data)

def collect_maps(obj, expected):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if isinstance(k,str) and k.startswith('/mnt/') and isinstance(v,str) and HEX.fullmatch(v):
                if k in expected and expected[k]!=v:raise ValueError(f'Conflicting frozen hashes: {k}')
                expected[k]=v
            else:collect_maps(v,expected)
    elif isinstance(obj,list):
        for v in obj:collect_maps(v,expected)

def audit():
    reg=read(OUT/'training_registration.json')
    expected={reg['source_checkpoint']:reg['source_checkpoint_sha256'],reg['manifest']:reg['manifest_sha256'],reg['eval_manifest']:reg['eval_manifest_sha256'],reg['source_registration']:reg['source_registration_sha256'],reg['source_data_registration']:reg['source_data_registration_sha256']}
    docs=['training_registration.json','evaluation/score_registration.json','training_summary/complete.json','results/completion.json','results/report_metrics_v1/completion.json','gallery/complete.json']
    calls={};weights={};phases={}
    for arm in ARMS:
        status=read(OUT/'training'/arm/'status.json')
        assert status['status']=='COMPLETE' and status['local_step']==1500 and status['global_step']==27000
        c=status['calls']
        assert c['attempted']==c['successful']==dict(train_forward=12000,train_backward=12000,optimizer_update=1500)
        assert not c['failed'] and not c['pending_call_ids'] and status['head_calls']==status['probe_calls']==0
        calls[arm]=c
        weights[arm]={k:status[k] for k in ('checkpoint','inference_export')}
        for receipt in weights[arm].values():expected[receipt['path']]=receipt['sha256']
        raw=read(OUT/'evaluation'/arm/'raw_complete_full.json')
        post=read(OUT/'evaluation'/arm/'post_complete_full.json')
        assert raw['status']==post['status']=='COMPLETE'
        assert raw['total_new_rollouts']==46 and raw['total_new_generator_forwards']==2300 and raw['total_new_head_forwards']==0
        assert post['new_D0_forwards']==46 and post['new_GEO_calls']==46
        phases[arm]=dict(raw_outputs=46,logical_vector_field_calls=2300,D0_calls=46,training_seconds=status['elapsed_seconds'],sampling_seconds=raw['seconds'],D0_GEO_seconds=post['seconds'])
        docs.extend([f'training/{arm}/status.json',f'evaluation/{arm}/raw_complete_full.json',f'evaluation/{arm}/post_complete_full.json'])
    for p in sorted((OUT/'evaluation/completions').glob('*.json')):docs.append(str(p.relative_to(OUT)))
    for rel in docs:collect_maps(read(OUT/rel),expected)
    for key in ('source_checkpoint','manifest','eval_manifest','source_registration','source_data_registration'):
        assert reg[key] in expected
    verification=[]
    for name,wanted in sorted(expected.items()):
        p=Path(name)
        if not p.is_file():raise FileNotFoundError(p)
        got=sha(p)
        if got!=wanted:raise ValueError(f'Frozen asset changed: {p}: {wanted} != {got}')
        verification.append(dict(path=name,sha256=got,bytes=p.stat().st_size))
    result=read(OUT/'results/completion.json');gallery=read(OUT/'gallery/complete.json')
    assert result['rows']==414 and result['new_score_rows']==276 and result['reused_score_rows']==138 and len(result['metric_fields'])==177
    assert gallery['pages']==138 and gallery['ratings_generated']==0
    receipt=dict(schema='bench_multilod_final_identity_audit_v1',status='PASS',scope='File hashes and recorded completed budgets; no new model/scoring/geometry execution',verified_files=verification,verified_file_count=len(verification),weights=weights,training_calls=calls,phases=phases,rows=414,metric_fields=177,visual_review_status='SEPARATE_GATE',model_calls=0,scoring_calls=0,failures=0,docs={rel:sha(OUT/rel) for rel in docs})
    write_once(OUT/'closeout/identity_audit.json',receipt)
    print(json.dumps(dict(status='PASS',verified_files=len(verification),new_model_calls=0,new_scoring_calls=0),ensure_ascii=False))
    return receipt

def finalize():
    identity=read(OUT/'closeout/identity_audit.json')
    visual=read(OUT/'visual_review/summary_v1/completion.json')
    assert identity['status']=='PASS' and visual['status']=='COMPLETE' and visual['cells']==414 and visual['pages']==138
    assert (OUT/'REPORT.md').is_file()
    expected={}
    collect_maps(visual,expected)
    for p,h in expected.items():
        if sha(p)!=h:raise ValueError(f'Visual source changed: {p}')
    sources={}
    for x in identity['verified_files']:
        p=Path(x['path'])
        if p.suffix=='.py' and p.is_relative_to(ROOT):sources[p]=x['sha256']
    for base in (ROOT/'r2/pilots/bench_multilod_training_v1',OUT/'control'):
        for p in base.glob('*.py'):sources[p]=sha(p)
    archived=[]
    for p,h in sorted(sources.items()):
        data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=h:raise ValueError(f'Source changed: {p}')
        target=OUT/'source_snapshot'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==data
        else:
            with target.open('xb') as f:f.write(data)
        archived.append(dict(source=str(p),archive=str(target.relative_to(OUT)),sha256=h,bytes=len(data)))
    write_once(OUT/'closeout/source_hashes.json',archived)
    deliverables=['REPORT.md','METHODS.md','RECOVERY.md','training_registration.json','closeout/identity_audit.json','closeout/source_hashes.json','training_summary/complete.json','training_summary/training_loss.png','results/completion.json','results/per_output.csv','results/report_metrics_v1/completion.json','gallery/complete.json','gallery/index.html','visual_review/protocol.json','visual_review/summary_v1/completion.json','visual_review/summary_v1/actual_observations.csv','visual_review/summary_v1/baseline_review_differences.csv']
    receipt=dict(schema='R2_BENCH_MULTILOD_TRAINING_V1_COMPLETE',status='COMPLETE',decision='No universal promotion; preserve paired data/geometry signal and contradictory visual/geometric costs',new_optimizer_updates=3000,new_train_forwards=24000,new_train_backwards=24000,new_raw_outputs=92,new_sampling_vector_field_calls=4600,new_D0_calls=92,new_score_rows=276,reused_score_rows=138,full_metric_fields=177,actual_visual_cells=414,actual_visual_pages=138,reviewer_kind='AI_PROXY',blind=False,predictor_head_training=0,failed_or_replayed_scientific_calls=0,default_changed=False,cleanup_performed=False,commit_or_push=False,source_archive_files=len(archived),deliverables={p:sha(OUT/p) for p in deliverables},identity_audit='closeout/identity_audit.json',weights=identity['weights'])
    write_once(OUT/'completion.json',receipt)
    print(json.dumps(dict(status='COMPLETE',source_files=len(archived),new_model_calls=0),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('audit','finalize'));args=parser.parse_args()
    audit() if args.mode=='audit' else finalize()
