"""Read-only endpoint/cost/artifact audit. No model imports or scoring."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from r2.pilots.bench_three_mechanisms_v1 import common as C

def main():
    train=[]
    for arm,target in C.UPDATES.items():
        r=C.read(C.OUT/'training'/arm/'status.json')
        assert r['status']=='COMPLETE' and r['step']==r['target']==target
        assert not r['calls']['pending_call_ids'] and not r['calls']['failed']
        C.verify({r['checkpoint']['path']:r['checkpoint']['sha256']})
        train.append(dict(arm=arm,updates=target,forwards=r['calls']['successful']['train_forward'],
            backwards=r['calls']['successful']['train_backward']))
    rl=C.read(C.OUT/'rl/verification.json');assert rl['status']=='PASS' and rl['opportunities']==16
    evals=[]
    for label in C.NEW_LABELS:
        folder=C.OUT/'evaluation'/label
        raw=C.read(folder/'raw_complete_full.json');post=C.read(folder/'post_complete_full.json')
        assert raw['status']==post['status']=='COMPLETE' and raw['outputs']==post['outputs']==12
        assert post['new_D0_forwards']==post['new_GEO_calls']==12
        C.verify(raw['files']);C.verify(post['files'])
        for receipt in list(raw['files'])+list(post['files']):C.verify(C.read(receipt)['files'])
        evals.append(dict(label=label,RAW=12,D0=12,GEO=12,generator_forwards=raw['total_new_generator_forwards']))
    scores=C.read(C.OUT/'results/completion.json');assert scores['status']=='COMPLETE' and scores['rows']==252
    assert len(scores['metric_fields'])==177
    C.verify(scores['files']);C.verify(scores['source_rows'])
    gallery=C.read(C.OUT/'gallery/complete.json');assert gallery['status']=='COMPLETE' and gallery['pages']==36
    C.verify(gallery['files'])
    actual=[]
    for name in ('review_part_01','review_part_02','review_part_03'):
        doc=C.read(C.OUT/'visual_review/completed'/(name+'.json'))
        assert doc['status']=='COMPLETE'
        actual.extend(doc['rows'])
    assert len(actual)==252
    assert len({(r['request_id'],r['label'],r['stage']) for r in actual})==252
    assert all(r.get('actually_viewed') for r in actual)
    C.verify(C.read(C.REGISTRATION)['data'])
    result=dict(status='COMPLETE',supervised=train,RL=rl,evaluation=evals,
        new_RAW=72,new_D0=72,new_GEO=72,score_rows=252,metric_fields=177,gallery_pages=36,
        actual_AI_PROXY_cells=252,visual_is_human_review=False,extra_engineering_forward=1,
        supervised_forwards=sum(r['forwards'] for r in train),supervised_backwards=sum(r['backwards'] for r in train),
        evaluation_generator_forwards=sum(r['generator_forwards'] for r in evals),
        no_default_change=True,commit=False,push=False,additional_model_calls=0)
    C.write(C.OUT/'COMPLETION.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('supervised','RL','evaluation')}))

if __name__=='__main__':main()
