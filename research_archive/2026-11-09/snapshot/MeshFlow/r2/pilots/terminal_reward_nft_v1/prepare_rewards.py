"""Join actual reviews with immutable geometry; freeze the training exposures."""
from __future__ import annotations
import argparse, csv
from . import data as D
from . import evaluation as E

def run(ri):
    stage=D.OUT/f'training/round_{ri:02d}';m=D.read(D.OUT/'inputs/manifest.json')
    requests=[q for q in m['training_requests'] if q['round_index']==ri]
    reviews=D.read(stage/'reviews_actual.json');reviews=reviews['rows'] if isinstance(reviews,dict) else reviews
    if len(reviews)!=32:raise ValueError('All32 actual reviews required; UNKNOWN permitted')
    byid={v['request_id']:v for v in reviews}
    if set(byid)!={q['request_id'] for q in requests}:raise ValueError('Review request set differs')
    gallery=D.read(stage/'gallery/manifest.json');im={r['request_id']:r for r in gallery['images']}
    rows=[]
    for q in requests:
        rid=q['request_id'];review=byid[rid]
        if review['image_sha256']!=im[rid]['image_sha256']:raise ValueError('Actual review image identity mismatch')
        if D.sha(im[rid]['image_path'])!=review['image_sha256']:raise ValueError('Image changed')
        score=E.read(stage/'scores'/rid/'POST/score.json')
        r=E.geometry_review_reward(score,review)
        rows.append(dict(request_id=rid,condition_id=q['condition_id'],parent_id=q['parent_id'],**r))
    result=E.group_rewards(rows);D.write(stage/'rewards.json',result)
    flat=[]
    for r in result['rows']:
        row={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
        for k in ('components','quality_factors','review_labels'):
            row.update({k+'__'+key:v for key,v in r[k].items()})
        flat.append(row)
    csvp=stage/'rewards.csv'
    if not csvp.exists():
        with csvp.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=sorted({k for r in flat for k in r}));w.writeheader();w.writerows(flat)
    usable=sum(g['usable'] for g in result['groups'])
    if ri==0 and usable==0:
        D.write(stage/'reward_gate.json',dict(status='STOP_REWARD_BLOCKED',informative_groups=0,reason='Predeclared first batch lacks reliable within-C reward contrast; no fabricated signal/no training'))
        print('STOP_REWARD_BLOCKED');return
    D.write(stage/'reward_gate.json',dict(status='PASS',informative_groups=usable))
    sched=D.read(D.OUT/'inputs/training_stream_schedule.json')[ri]['items']
    rq={q['request_id']:q for q in requests};reward={r['request_id']:r for r in result['rows']};stream=[]
    for item in sched:
        rid=item['request_id'];q=rq[rid];r=reward[rid];receipt=D.read(stage/'raw'/rid/'receipt.json')
        stream.append(dict(item,raw_path=receipt['raw_path'],raw_sha256=receipt['raw_sha256'],K=q['K'],N=q['N'],r=r['r_NFT'],known_reward=r['preference_known']))
    D.write(stage/'training_stream.json',stream)
    proof=[stage/'reviews_actual.json',stage/'gallery/manifest.json',stage/'rewards.json',stage/'reward_gate.json',stage/'training_stream.json',D.OUT/'inputs/training_stream_schedule.json',D.OUT/'inputs/manifest.json']
    proof += [stage/'scores'/q['request_id']/'POST/score.json' for q in requests]
    D.write(stage/'learning_input_seal.json',dict(status='PASS',hashes={str(p):D.sha(p) for p in proof}))
    print('REWARD_READY',ri,'informative_groups',usable,'known',sum(r['preference_known'] for r in result['rows']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True);a=p.parse_args();run(a.round)
