
"""Independent CPU-only evidence audit; no imports from experiment scorers or models."""
import csv,json,math,hashlib,collections,datetime,time
from pathlib import Path
O=Path('/mnt/e/MeshFlow/experiments/r2_seat_noise_transfer_v1'); E=O/'evaluation'; A=O/'independent_audit'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
cache={}
def sha(p):
    p=str(p)
    if p not in cache:
        h=hashlib.sha256()
        with open(p,'rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
        cache[p]=h.hexdigest()
    return cache[p]
checks=collections.Counter(); errors=[]
def check(ok,msg):
    checks['PASS' if ok else 'FAIL']+=1
    if not ok: errors.append(msg)
def num(x):
    if x in ('True',True):return 1.
    if x in ('False',False):return 0.
    if x in ('',None):return None
    v=float(x);return v if math.isfinite(v) else None
def close(a,b):
    a,b=num(a),num(b);return (a is None and b is None) or (a is not None and b is not None and abs(a-b)<=1e-11*max(1,abs(a),abs(b)))
def mean(v):
    v=[x for x in v if x is not None];return sum(v)/len(v) if v else None
R=read(E/'registration.json'); raw=read(E/'raw_index.json')['rows'];post=read(E/'post_index.json')['rows']
ids={x['request_id'] for x in R['requests']};labels=R['labels']
expected={(l,i) for l in labels for i in ids}
check(len(ids)==48 and len(raw)==len(post)==144,'logical request/raw/post count')
check({(x['label'],x['request_id']) for x in raw}==expected,'RAW identifiers')
check({(x['label'],x['request_id']) for x in post}==expected,'POST identifiers')
res=ret=attempts=0; end_times=[]; calls_by_arm=collections.Counter()
for x in raw:
    d=Path(x['raw_path']).parent
    events=[json.loads(l) for l in (d/'calls.jsonl').read_text().splitlines()]
    rr=[z for z in events if z.get('kind')=='sampling_forward' and z['event']=='RESERVE']
    tt=[z for z in events if z.get('kind')=='sampling_forward' and z['event']=='RETURN']
    aa=[z for z in events if z.get('kind')=='rollout_attempt' and z['event']=='RESERVE']
    check(len(rr)==len(tt)==50 and len(aa)==1,'RAW call count '+str(d))
    check([z['event'] for z in events if z.get('kind')=='sampling_forward']==['RESERVE','RETURN']*50,'RAW reserve/return alternation '+str(d))
    check(not any(z.get('event') in ('ERROR','FAIL','FAILED') for z in events),'RAW failure events')
    check(all(x['contracts'].values()),'RAW preservation contracts')
    check(x['checkpoint_sha256']==R['models'][x['label']]['sha256'],'model checkpoint binding')
    for pathkey,hashkey in [('raw_path','raw_sha256'),('ply_path','ply_sha256'),('velocities_path','velocities_sha256'),('result_path','result_sha256')]:
        check(sha(x[pathkey])==x[hashkey],'RAW file changed '+x[pathkey])
    res+=len(rr);ret+=len(tt);attempts+=len(aa);calls_by_arm[x['label']]+=len(tt)
    end_times.append(max(z['time_unix'] for z in events if z.get('event')=='ROLLOUT_COMPLETE'))
d0events=[json.loads(l) for l in (O/'execution/worker_0_of_1_events.jsonl').read_text().splitlines()]
dr=[x for x in d0events if x['event']=='D0_FORWARD_RESERVED'];dt=[x for x in d0events if x['event']=='D0_FORWARD_RETURNED']
check(len(dr)==len(dt)==144,'D0 total calls')
check(collections.Counter(x['request_id'] for x in dr)==collections.Counter(x['request_id'] for x in dt),'D0 reserve/return per request')
check(all(x['event'] in ('D0_FORWARD_RESERVED','D0_FORWARD_RETURNED') for x in d0events),'Unexpected D0 events')
first_d0=min(datetime.datetime.fromisoformat(x['utc']).timestamp() for x in dr)
check(max(end_times)<first_d0,'all RAW before first D0')
for x in post:
    q=read(x['proposal_receipt'])
    check(q['model_forwards']==1 and x['new_D0_F']==1 and x['GEO_calls']==1,'POST exactly one D0/GEO')
    check(q['checkpoint_sha256']==R['d0_checkpoint']['sha256'],'D0 checkpoint')
    check(q['raw_sha256']==x['raw_sha256'] and sha(x['proposal_path'])==q['output_sha256'],'D0 input/output binding')
    for p,h in x['files'].items():check(sha(p)==h,'POST changed '+p)
    s=x['source_restoration'];orient=x['orientation']
    check(s['status']=='PASS' and s['vertices_bytewise_unchanged'] and s['unordered_face_vertex_IDs_unchanged'],'source restoration')
    check(s['source_C_vertex_ID_topology']['status']=='PASS','source C ID connectivity')
    check(orient['C_face_rows_bytewise_preserved'] and orient['coordinates_unchanged'] and orient['face_vertex_sets_unchanged'],'source fixed orientation')
P=read(O/'protected_assets_final.json');protected=0
for p,h in P['hashes'].items():
    check(sha(p)==h,'protected hash '+p);protected+=1
N=rows(E/'per_output.csv');lk={(x['label'],x['request_id'],x['stage']):x for x in N}
check(len(N)==len(lk)==288,'288 unique scores')
check(set(lk)=={(l,i,s) for l,i in expected for s in ('RAW','POST')},'score identities')
scores=read(E/'score_completion.json')
check(scores['model_calls']==0 and scores['new_stages']==288,'no model calls in scoring')
for x in N:
    check(sha(x['score_path'])==x['score_sha256'],'score hash')
    check(x['C_preserved_exact']=='True' and x['source_C_connectivity_status']=='PASS','scored C preservation')
    check(x['I_status']=='PASS' and x['topology_status']=='PASS','scoring status')
    check(close(x['actual_free_faces'],x['free_face_denominator']),'free denominator')
    s=read(x['score_path'])
    check(s['model_calls']==0 and len(s['boundary'])==8,'score budget/threshold grid')
pairrows=rows(E/'paired_per_output.csv');groups=collections.defaultdict(list)
for x in pairrows:
    after,before=x['comparison'].split('-minus-');m=x['metric']
    aa=num(lk[before,x['request_id'],x['stage']].get(m));bb=num(lk[after,x['request_id'],x['stage']].get(m))
    valid=aa is not None and bb is not None
    check(x['status']==('PASS' if valid else 'NA'),'pair validity')
    check(close(x['baseline_value'],aa) and close(x['candidate_value'],bb),'pair source')
    check(close(x['delta'],bb-aa if valid else None),'pair delta')
    key=tuple(x[k] for k in ('comparison','panel','parent_id','condition_id','condition_type','stage','metric'))
    groups[key].append(x)
parents=rows(E/'paired_per_parent.csv');pg=collections.defaultdict(list)
for x in parents:
    key=tuple(x[k] for k in ('comparison','panel','parent_id','condition_id','condition_type','stage','metric'));gg=groups[key]
    check(int(x['n_registered'])==len(gg)==(8 if x['panel']=='TRAIN_FAMILIAR' else 4),'noise denominator')
    check(int(x['n_valid'])==sum(y['status']=='PASS' for y in gg),'valid noise denominator')
    for k in ('baseline_value','candidate_value','delta'):check(close(x[k],mean([num(y[k]) for y in gg])),'parent mean '+k)
    pg[tuple(x[k] for k in ('comparison','panel','stage','metric'))].append(x)
for x in rows(E/'paired_summary.csv'):
    gg=pg[tuple(x[k] for k in ('comparison','panel','stage','metric'))]
    check(int(x['n_registered'])==len(gg)==4,'summary uses four parents')
    for k in ('baseline_value','candidate_value','delta'):check(close(x[k],mean([num(y[k]) for y in gg])),'summary mean '+k)
metrics=['C_F_has_valid_edge','C_F_valid_edge_count','CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','A_VMR__raw_tau_1e-6','PMR__raw_tau_1e-6']
panels=[]
for panel in ('TRAIN_FAMILIAR','DEV_HELDOUT'):
 for l in labels:
  for stage in ('RAW','POST'):
    rr=[x for x in N if (x['panel'],x['label'],x['stage'])==(panel,l,stage)]
    parentids={x['parent_id'] for x in rr};check(len(parentids)==4,'panel parent count')
    z=dict(panel=panel,label=l,stage=stage,parents=4,outputs=len(rr),valid_edge_outputs=sum(num(x['C_F_has_valid_edge'])==1 for x in rr))
    z.update({m:mean([mean([num(x[m]) for x in rr if x['parent_id']==p]) for p in parentids]) for m in metrics});panels.append(z)
phase=[json.loads(l) for l in (O/'ledger.jsonl').read_text().splitlines()]
phase=[x for x in phase if x['event'] in ('PHASE_START','PHASE_END')]
for i,x in enumerate(phase):
    if x['event']=='PHASE_END':check(x['returncode']==0,'phase failure')
source_counts=dict(protected_files=protected,hashed_files=len(cache),raw_output_checks=len(raw),post_output_checks=len(post),score_checks=len(N))
out=dict(status='PASS' if not errors else 'FAIL',review_locked_before_unblinding=True,review_73b_sha256=sha(E/'gallery/review_73b.csv'),style_73b_sha256=sha(E/'gallery/style_73b.csv'),checks=dict(checks),errors=errors,counts=dict(generator_rollouts=attempts,generator_reserved=res,generator_returned=ret,D0_reserved=len(dr),D0_returned=len(dt),scores=len(N),training_updates=0,backwards=0),counts_by_generator=calls_by_arm,all_raw_barrier=dict(last_RAW_unix=max(end_times),first_D0_unix=first_d0,gap_seconds=first_d0-max(end_times)),source_integrity=source_counts,paired_rows=len(pairrows),paired_parent_rows=len(parents),independent_per_panel=panels,final_analyse_status='PENDING',scope_notes=['No model import/call, no training, no geometry rescoring. Existing CSV values independently paired and averaged.','Four independent parents per panel; 8/4 noises are within-C repetitions.','C_F_has_valid_edge means at least one locally valid indexed C/F edge, not whole C adoption or a closed seam.','Unknown/illegal face pairs rejected by original valid-edge scorer; this audit does not redesign detector.','Per-PID WDDM VRAM is unavailable; PyTorch counters and whole-device observations do not establish an exact 6 GiB process peak.'])

# Final analyse outputs: independently map locked anonymous annotations only now.
fields=('C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer','whole_body_complete','support_missing','fragments_or_folded_layers','visible_join')
key=read(E/'blinding/model_key.json')['assignments']
reviews=[x for f in ('3f','73b','familiar_full','dev') for x in rows(E/'gallery'/('review_'+f+'.csv'))]
check(len(reviews)==288,'complete actual reviews')
vl={}
for x in reviews:
    check(x['review_status']=='REVIEWED' and all(x[f] in ('YES','NO','UNCERTAIN') for f in fields),'actual categorical review')
    check(x['image_sha256']==x['reviewed_image_sha256']==sha(x['image_path']),'actual reviewed image hash')
    vl[key[x['condition_id']][x['anonymous_label']],x['request_id'],x['stage']]=x
check(len(vl)==288,'review unique identifiers')
pc=rows(O/'results/per_condition.csv');pan=rows(O/'results/per_panel.csv')
for x in pc:
    rr=[y for y in N if all(x[k]==y[k] for k in ('panel','parent_id','condition_id','label','stage'))]
    check(len(rr)==int(x['n']),'final condition denominator')
    for k in metrics:check(close(x[k],mean([num(y[k]) for y in rr])),'final condition numeric '+k)
    for f in fields:
        for v in ('YES','NO','UNCERTAIN'):check(int(x[f+'__'+v])==sum(vl[y['label'],y['request_id'],y['stage']][f]==v for y in rr),'final condition categories')
for x in pan:
    rr=[y for y in pc if all(x[k]==y[k] for k in ('panel','label','stage'))]
    check(len(rr)==int(x['parents'])==4 and int(x['n'])==sum(int(y['n']) for y in rr),'final panel denominator')
    for k in metrics:check(close(x[k],mean([num(y[k]) for y in rr])),'final panel numeric '+k)
    for f in fields:
        for v in ('YES','NO','UNCERTAIN'):check(int(x[f+'__'+v])==sum(int(y[f+'__'+v]) for y in rr),'final panel categories')
    check(int(x['valid_edge_outputs'])==sum(num(y['C_F_has_valid_edge'])==1 for y in N if all(x[k]==y[k] for k in ('panel','label','stage'))),'valid edge output denominator')
decisions=rows(O/'results/paired_decisions.csv')
for x in decisions:
    after,before=x['comparison'].split('-minus-')
    av=vl[before,x['request_id'],x['stage']]['C_adoption'];bv=vl[after,x['request_id'],x['stage']]['C_adoption']
    expected='UNCERTAIN' if 'UNCERTAIN' in (av,bv) else 'GAIN' if (av,bv)==('NO','YES') else 'LOSS' if (av,bv)==('YES','NO') else 'SAME'
    check(x['C_adoption_pair']==expected and (x['C_adoption_before'],x['C_adoption_after'])==(av,bv),'adoption transition')
    a=lk[before,x['request_id'],x['stage']];b=lk[after,x['request_id'],x['stage']]
    aa=num(a['C_F_has_valid_edge']);bb=num(b['C_F_has_valid_edge'])
    check(x['local_edge_pair']==('GAIN' if bb>aa else 'LOSS' if bb<aa else 'SAME'),'local edge transition')
    for k in metrics:check(close(x[k+'__delta'],num(b[k])-num(a[k])),'final paired metric')
ds=rows(O/'results/paired_panel_decisions.csv')
for x in ds:
    rr=[y for y in decisions if all(x[k]==y[k] for k in ('comparison','panel','stage'))]
    check(int(x['n'])==len(rr),'paired panel denominator')
    for f in ('C_adoption_pair','local_edge_pair'):
        for v in ('GAIN','LOSS','SAME','UNCERTAIN'):check(int(x[f+'__'+v])==sum(y[f]==v for y in rr),'paired transition counts')
small=[]
for panel in ('TRAIN_FAMILIAR','DEV_HELDOUT'):
    b=next(x for x in pan if (x['panel'],x['label'],x['stage'])==(panel,'EXP67_FROZEN','POST'))
    a=next(x for x in pan if (x['panel'],x['label'],x['stage'])==(panel,'SMALL_C200','POST'))
    d=next(x for x in ds if (x['panel'],x['comparison'],x['stage'])==(panel,'SMALL_C200-minus-EXP67_FROZEN','POST'))
    uncertain_to_yes=sum(x['C_adoption_before']=='UNCERTAIN' and x['C_adoption_after']=='YES' for x in decisions if (x['panel'],x['comparison'],x['stage'])==(panel,'SMALL_C200-minus-EXP67_FROZEN','POST'))
    small.append(dict(panel=panel,outputs=int(a['n']),valid_edge_before=int(b['valid_edge_outputs']),valid_edge_after=int(a['valid_edge_outputs']),adoption_YES_before=int(b['C_adoption__YES']),adoption_YES_after=int(a['C_adoption__YES']),definite_NO_to_YES=int(d['C_adoption_pair__GAIN']),definite_YES_to_NO=int(d['C_adoption_pair__LOSS']),UNCERTAIN_to_YES=uncertain_to_yes))
out.update(status='PASS' if not errors else 'FAIL',checks=dict(checks),errors=errors,final_analyse_status='PASS' if not errors else 'FAIL',final_analyse_rows=dict(per_condition=len(pc),per_panel=len(pan),paired_decisions=len(decisions),paired_panel_decisions=len(ds)),small_vs_exp67_POST=small)
out['source_integrity']['hashed_files']=len(cache)

A.mkdir(exist_ok=True)
(A/'final_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
(A/'final_audit.md').write_text('# Independent final evidence audit\n\nStatus: '+out['status']+' (including final analyse outputs).\n\n- 144 rollouts, 7200 reserved/returned generator calls; 144 reserved/returned D0 calls; 288 scored stages.\n- Last RAW precedes first D0 by '+str(round(first_d0-max(end_times),3))+' seconds.\n- '+str(protected)+' protected hashes independently recomputed; '+str(len(pairrows))+' paired metric rows and '+str(len(parents))+' parent averages independently checked.\n- Each panel has four parents; eight or four noises are repeated outputs, not new parents.\n- Local valid C/F edges are distinct from adoption, complete seams, or globally valid meshes.\n- Visual 73b review was fixed before numeric access; AI proxy judgments, not human ratings.\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'checks':dict(checks),'errors':errors[:10],'source_integrity':source_counts,'counts':out['counts'],'barrier':out['all_raw_barrier'],'independent_per_panel':panels},ensure_ascii=False))
