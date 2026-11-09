"""Independent CPU-only final audit. No model/scorer imports or geometry rescoring.

Default is readiness only, which does not open numeric scores or the blind key.
--final is gated by complete saved anonymous reviews before reading the key.
"""
import argparse, ast, collections, csv, datetime, hashlib, json, math, os, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
O=ROOT/'experiments/r2_c_reference_frame_v1'; E=O/'evaluation'; A=O/'independent_audit'
OLD=ROOT/'experiments/r2_seat_noise_transfer_v1'
QUICK=ROOT/'experiments/r2_seat_quick_learnability_v1'
LABELS=('EXP67_FROZEN','SMALL_C200','CREF_C0','CREF_C200')
FIELDS=('C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer','whole_body_complete','support_missing','fragments_or_folded_layers','visible_join')
METRICS=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction','A_VMR__raw_tau_1e-6','PMR__raw_tau_1e-6','C_F_has_valid_edge','C_F_valid_edge_count','C_F_valid_opposite_winding_edge_count','opposite_winding_reached_free_area_fraction','actual_free_faces')
cache={}; checks=collections.Counter(); errors=[]
def path(p):
    s=str(p).replace('\\','/')
    if os.name!='nt' and len(s)>2 and s[1:3]==':/':s='/mnt/'+s[0].lower()+s[2:]
    if os.name=='nt' and s.startswith('/mnt/') and len(s)>7:s=s[5].upper()+':/'+s[7:]
    return Path(s)
def read(p):return json.loads(path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def arrays(p):
    with np.load(path(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def events(p):return [json.loads(x) for x in path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def sha(p):
    p=path(p).resolve()
    if p not in cache:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
        cache[p]=h.hexdigest()
    return cache[p]
def check(ok,msg):
    checks['PASS' if ok else 'FAIL']+=1
    if not ok:errors.append(msg)
def verify(mapping,prefix='hash'):
    for p,h in mapping.items():check(sha(p)==h,prefix+': '+str(p))
def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def num(x):
    if x in ('True',True):return 1.
    if x in ('False',False):return 0.
    if x in ('',None):return None
    x=float(x);return x if math.isfinite(x) else None
def close(a,b):
    a,b=num(a),num(b)
    return (a is None and b is None) or (a is not None and b is not None and abs(a-b)<=1e-11*max(1,abs(a),abs(b)))
def mean(v):
    v=[num(x) for x in v];v=[x for x in v if x is not None]
    return sum(v)/len(v) if v else None
def save(name,value):
    p=(A/name).resolve()
    if not p.is_relative_to(A.resolve()):raise ValueError('Audit output outside owned directory')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def readiness():
    required=[O/'registration.json',E/'registration.json',O/'training/CREF_C200/status.json',
      E/'raw_index.json',E/'post_index.json',E/'per_output.csv',E/'score_completion.json',
      E/'gallery/AI_PROXY_review_template.csv',E/'gallery/AI_PROXY_noise_style_template.csv',
      E/'gallery/review_completion.json',O/'results/per_condition.csv',O/'results/per_panel.csv',
      O/'results/paired_decisions.csv',O/'results/paired_panel_decisions.csv',O/'summary.json',O/'REPORT.md']
    required += [E/'gallery'/f'{kind}_{who}.csv' for kind in ('review','style') for who in ('3f','73b','familiar_full','dev')]
    missing=[str(p) for p in required if not p.is_file()]
    return dict(status='READY_FOR_FINAL_AUDIT' if not missing else 'PENDING',missing=missing,
      model_calls=0,quality_scores_read=False,blind_key_read=False)

def locked_visual_reviews():
    template=rows(E/'gallery/AI_PROXY_review_template.csv')
    expected={(x['request_id'],x['anonymous_label'],x['stage']) for x in template}
    reviewfiles=[E/'gallery'/f'review_{w}.csv' for w in ('3f','73b','familiar_full','dev')]
    stylefiles=[E/'gallery'/f'style_{w}.csv' for w in ('3f','73b','familiar_full','dev')]
    reviews=[x for p in reviewfiles for x in rows(p)]
    check(len(reviews)==len(expected)==384,'384 actual review rows')
    check({(x['request_id'],x['anonymous_label'],x['stage']) for x in reviews}==expected,'visual review identities')
    for x in reviews:
        check(x['review_status']=='REVIEWED' and all(x[f] in ('YES','NO','UNCERTAIN') for f in FIELDS),'categorical review completed')
        check(x['image_sha256']==x['reviewed_image_sha256']==sha(x['image_path']),'actual reviewed page hash')
        check(bool(x.get('notes','').strip()),'actual review notes')
    styles=[x for p in stylefiles for x in rows(p)]
    check(len(styles)==len({(x['condition_id'],x['anonymous_label']) for x in styles})==32,'32 actual style rows')
    for x in styles:
        check(x['review_status']=='REVIEWED' and x['reviewed_image_sha256']==sha(x['image_path']),'actual style review/hash')
    check(len({x['image_path'] for x in reviews})==48 and len({x['image_path'] for x in styles})==8,'48 pages and 8 sheets')
    if errors:raise RuntimeError('Visual review gate failed before unblinding: '+repr(errors[:5]))
    return reviews,styles,{str(p):sha(p) for p in reviewfiles+stylefiles}

def audit_training(TR):
    original=read(QUICK/'training/SMALL_C200/initialization.json')
    init=read(O/'training/CREF_C200/initialization.json')
    for k in ('loaded_generator_sha256','loaded_optimizer_sha256','loaded_rng_sha256'):
        check(init[k]==original[k],'same original SMALL initialization '+k)
    check(init['restored_global_step']==6000 and init['new_parameters']==0,'initial step/no new parameters')
    check(init['hyperparameters']==original['hyperparameters'],'same AdamW hyperparameters')
    plan=read(O/'plans/CREF_C200.json');baseline=read(QUICK/'plans/SMALL_C200.json')
    check(plan['records']==baseline['records'] and len(plan['records'])==1600,'all1600 records identical')
    check(sha(O/'plans/SMALL_C200_reused.json')==sha(QUICK/'plans/SMALL_C200.json'),'baseline plan byte-identical')
    local=read(O/'preparation/local_stream_manifest.json')['records']
    ll={x['sample_index']:x for x in local}
    check(len(local)==len(ll)==1600,'1600 local stream identities')
    status=read(O/'training/CREF_C200/status.json')
    check((status['status'],status['local_step'],status['global_step'])==('COMPLETE',200,6200),'fixed200 endpoint')
    log=events(O/'training/CREF_C200/calls.jsonl')
    begin=[x for x in log if x['event']=='CALL_BEGIN'];done=[x for x in log if x['event']=='CALL_DONE']
    target={'train_forward':1600,'train_backward':1600,'optimizer_update':200}
    check(dict(collections.Counter(x['kind'] for x in begin))==dict(collections.Counter(x['kind'] for x in done))==target,'exact train F/B/update counts')
    check(len(log)==6800 and [x['event'] for x in log]==['CALL_BEGIN','CALL_DONE']*3400,'no unresolved/retry training calls')
    for b,d in zip(begin,done):
        check((b['call_id'],b['kind'],b['context'])==(d['call_id'],d['kind'],d['context']),'training reservation return identity')
    for step in range(1,201):
        rr=plan['records'][8*(step-1):8*step]
        check(sorted(collections.Counter(x['uid'] for x in rr).values())==[2]*4 and sum(x['alpha']==1 for x in rr)==4,'per-batch exposure')
        calls=[x for x in begin if x['context']['local_step']==step]
        check([x['kind'] for x in calls]==['train_forward','train_backward']*8+['optimizer_update'],'per-update exact ordered math')
        for x in calls:
            check(x['context']['global_step']==6000+step,'cumulative clock')
            if x['kind']!='optimizer_update':
                c=x['context'];s=ll[c['sample_index']]
                check(c['input_bytehash']==s['input_bytehash'] and c['label_bytehash']==s['label_bytehash'],'actual stream input/label ledger')
    check(status['calls']['successful']==target and not status['calls']['failed'] and not status['calls']['pending_call_ids'],'status completed ledger')
    ot=sorted((O/'ot/CREF_C200').glob('*.json'));ot=[p for p in ot if not p.name.endswith('.attempt.json')]
    check(len(ot)==200,'200 isolated OT batch receipts')
    for p in ot:
        z=read(p);check(z['status']=='PASS' and z['actual_OT_attempts']==z['actual_OT_returns']==8,'OT matching count')
        check(sha(p.with_suffix('.npz'))==z['sha256'],'immutable OT bytes')
    for name in ('checkpoint','inference_export'):
        b=status[name];check(sha(b['path'])==b['sha256'],'final checkpoint bytes '+name)
        check((b['local_step'],b['global_step'])==(200,6200),'final checkpoint step '+name)
    return dict(updates=200,train_forward=1600,train_backward=1600,OT_returns=1600,initialization_hashes={k:init[k] for k in ('loaded_generator_sha256','loaded_optimizer_sha256','loaded_rng_sha256')})

def area_center(c):
    cross=np.cross(c[:,1]-c[:,0],c[:,2]-c[:,0])
    ar=np.sqrt(np.sum(cross*cross,axis=1,dtype=np.float32))*np.float32(.5)
    return np.sum(ar[:,None]*np.mean(c,axis=1,dtype=np.float32),axis=0,dtype=np.float32)/np.sum(ar,dtype=np.float32)
def ply_triangles(p):
    with path(p).open('rb') as f:
        header=[]
        while True:
            s=f.readline().decode('ascii').strip();header.append(s)
            if s=='end_header':break
            if len(header)>100:raise ValueError('PLY header')
        nv=int(next(x for x in header if x.startswith('element vertex ')).split()[-1])
        nf=int(next(x for x in header if x.startswith('element face ')).split()[-1])
        v=np.frombuffer(f.read(nv*12),dtype='<f4').copy().reshape(nv,3)
        ff=np.frombuffer(f.read(nf*13),dtype=np.dtype([('n','u1'),('v','<i4',(3,))],align=False))
        check(len(ff)==nf and bool((ff['n']==3).all()) and not f.read(1),'binary triangle PLY')
        return v[ff['v']]
def audit_outputs(R):
    qs={q['request_id']:q for q in R['requests']};raw=read(E/'raw_index.json')['rows'];post=read(E/'post_index.json')['rows']
    expected={(l,i) for l in LABELS for i in qs}
    check(len(qs)==48 and len(raw)==len(post)==192,'192 logical RAW/POST')
    check({(x['label'],x['request_id']) for x in raw}=={(x['label'],x['request_id']) for x in post}==expected,'output identifiers')
    oldidx={s:{(x['label'],x['request_id']):x for x in read(OLD/f'evaluation/{s}_index.json')['rows']} for s in ('raw','post')}
    bindings=dict(R['models']);bindings['CREF_C200']=read(E/'bindings/CREF_C200.json')
    check(bindings['CREF_C0']['sha256']==bindings['EXP67_FROZEN']['sha256'],'CREF0 uses exact EXP67 checkpoint')
    for label in LABELS[2:]:
        ident=read(E/'execution'/label/'generator_0_1_identity.json')
        check(ident['file_sha256']==bindings[label]['sha256'],'actual loaded generator file identity')
        expected_state=read(O/'training/CREF_C200/status.json')['inference_export']['generator_state_sha256'] if label=='CREF_C200' else read(O/'registration.json')['source_generator_sha256']
        check(ident['generator_state_sha256']==expected_state,'actual loaded generator state, not EMA')
    end=[];new=0;returned=0;newpost=0
    for x in raw:
        q=qs[x['request_id']];n,k=q['N'],q['K'];isnew=x['label'] in LABELS[2:]
        if not isnew:check(x==oldidx['raw'][x['label'],x['request_id']],'RAW cache exact receipt')
        verify({x[p+'_path']:x[p+'_sha256'] for p in ('raw','ply','velocities','result')},'RAW bytes')
        check(x['checkpoint_sha256']==bindings[x['label']]['sha256'],'RAW checkpoint binding')
        for key in ('input_sha256','C_sha256','Gaussian_sha256','condition_source_sha256'):check(x[key]==q[key],'RAW frozen request '+key)
        inp=arrays(q['input_path']);a=arrays(x['raw_path']);c=inp['C'];w=a['trajectory'].reshape(51,n,3,3);r=a['raw'].reshape(n,3,3)
        check(same(a['z'],inp['z']) and same(a['C'].reshape(k,3,3),c),'saved C and Gaussian')
        check(all(same(v[:k],c) for v in w) and same(r,w[-1]),'world all51 C/final RAW')
        check(same(ply_triangles(x['ply_path']),r),'RAW PLY exact')
        check(np.isfinite(w).all() and w.dtype==np.float32,'RAW finite FP32')
        if isnew:
            new+=1;lc=a['local_trajectory'].reshape(51,n,3,3);center=a['center'];v=arrays(x['velocities_path'])['velocity'].reshape(50,n,3,3)
            check(same(center,area_center(c)),'C-only area center')
            check(all(same(t[:k],c-center) for t in lc),'local all51 C')
            check(same(lc[0,k:].reshape(1,n-k,9),inp['z'][:,k:]),'local original noise start')
            check(all(same(lc[i+1,k:],lc[i,k:]+v[i,k:]*np.float32(1/50)) for i in range(50)),'exact local Euler')
            check(all(same(ww[k:],ll[k:]+center) for ww,ll in zip(w,lc)),'world restore every free state')
            check(v.dtype==np.float32 and not np.any(v.view(np.uint32)&0xffff),'BF16 velocity lossless storage')
            ev=events(path(x['raw_path']).parent/'calls.jsonl');rr=[z for z in ev if z.get('kind')=='sampling_forward' and z['event']=='RESERVE'];tt=[z for z in ev if z.get('kind')=='sampling_forward' and z['event']=='RETURN']
            check(len(ev)==103 and len(rr)==len(tt)==50,'one Euler50 ledger')
            check([z['event'] for z in ev if z.get('kind')=='sampling_forward']==['RESERVE','RETURN']*50,'sampling reserve-return alternation')
            check([z['context']['euler_step'] for z in rr]==list(range(50)),'Euler step identities')
            check(sum(z['event']=='RESERVE' and z.get('kind')=='rollout_attempt' for z in ev)==1,'one rollout attempt')
            check(x['new_rollouts']==1 and x['new_sampling_F']==50,'new receipt counts')
            returned+=len(tt);end.append(next(z['time_unix'] for z in ev if z['event']=='ROLLOUT_COMPLETE'))
        else:check(same(w[0,k:].reshape(1,n-k,9),inp['z'][:,k:]),'cached original world noise')
    for x in post:
        isnew=x['label'] in LABELS[2:];q=qs[x['request_id']]
        if not isnew:check(x==oldidx['post'][x['label'],x['request_id']],'POST cache exact receipt')
        verify(x['files'],'POST bytes');p=read(x['proposal_receipt'])
        check(p['model_forwards']==1 and p['checkpoint_sha256']==R['d0_checkpoint']['sha256'],'one official D0 per output')
        check(p['raw_sha256']==x['raw_sha256'] and sha(x['proposal_path'])==p['output_sha256'],'D0 proposal binding')
        check(sha(x['post_path'])==x['post_sha256'],'POST hash')
        src=arrays(q['condition_source']);a=arrays(x['post_path']);c=src['C_original'].reshape(q['K'],3,3)
        check(same(a['output'][:q['K']].reshape(q['K'],3,3),c),'POST original source C')
        check(same(a['vertices'][a['faces']],a['output']),'POST actual indexed triangles')
        s=x['source_restoration'];o=x['orientation']
        check(s['status']=='PASS' and s['vertices_bytewise_unchanged'] and s['unordered_face_vertex_IDs_unchanged'],'source restoration preserves geometry')
        check(o['C_face_rows_bytewise_preserved'] and o['coordinates_unchanged'] and o['face_vertex_sets_unchanged'],'fixed-C free orientation')
        if isnew:
            newpost+=1;check(x['new_D0_F']==x['GEO_calls']==1,'new one D0/GEO')
    ev=[]
    for p in (O/'execution').rglob('*events.jsonl'):ev.extend(events(p))
    dr=[x for x in ev if x['event']=='D0_FORWARD_RESERVED'];dt=[x for x in ev if x['event']=='D0_FORWARD_RETURNED']
    check(len(dr)==len(dt)==96,'96 D0 reserved/returned')
    check(collections.Counter(x['request_id'] for x in dr)==collections.Counter(x['request_id'] for x in dt)==collections.Counter({i:2 for i in qs}),'two D0 models per frozen request')
    first=min(datetime.datetime.fromisoformat(x['utc']).timestamp() for x in dr)
    check(max(end)<first,'all96 RAW before any D0')
    check(new==newpost==96 and returned==4800,'new generation budget')
    return dict(new_rollouts=new,new_generator_forwards=returned,new_D0_forwards=newpost,cached_RAW=96,cached_POST=96,last_RAW_unix=max(end),first_D0_unix=first,RAW_D0_gap_seconds=first-max(end))

def audit_numeric(R,reviews):
    N=rows(E/'per_output.csv');lk={(x['label'],x['request_id'],x['stage']):x for x in N}
    expected={(l,q['request_id'],s) for l in LABELS for q in R['requests'] for s in ('RAW','POST')}
    check(len(N)==len(lk)==384 and set(lk)==expected,'384 unique score identities')
    completion=read(E/'score_completion.json')
    check(completion['new_stages']==completion['reused_stages']==192 and completion['model_calls']==0,'192 new plus192 reused scores')
    oldrefs={x['path']:x['sha256'] for x in read(OLD/'evaluation/score_completion.json')['receipts']}
    for ref in completion['receipts']:
        check(sha(ref['path'])==ref['sha256'],'score receipt hash')
        s=read(ref['path']);verify(s['files'],'score artifacts')
        check(s['model_calls']==0 and len(s['boundary'])==8,'no score model calls/eight boundary settings')
        if s['label'] in LABELS[:2]:check(oldrefs[ref['path']]==ref['sha256'],'cached score identity')
        else:check(s['identity']['registration_sha256']==sha(E/'registration.json'),'new score registration')
    for x in N:
        check(sha(x['score_path'])==x['score_sha256'],'CSV score binding')
        check(x['C_preserved_exact']=='True','scored C preservation')
        check(close(x['actual_free_faces'],x['free_face_denominator']),'actual free denominator')
    for x in N:
        s=read(x['score_path']);flat={**s['metrics'],**{k:v for k,v in s['topology'].items() if isinstance(v,(int,float,bool)) or v is None},**s['distance']}
        for z in s['boundary']:flat['A_VMR__'+z['setting']]=z['VMR'];flat['PMR__'+z['setting']]=z['perfect']
        flat.update({'DESC__'+k:v for k,v in s['descriptive_boundary_distance']['metrics'].items()});flat['C_preserved_exact']=s['C_preserved_exact']
        for k,v in flat.items():
            if isinstance(v,(int,float,bool)) or v is None:check(close(x.get(k),v),'CSV value versus source receipt '+k)
    source=ast.parse((ROOT/'r2/pilots/condition_width_response_v1/evaluate.py').read_text(encoding='utf-8-sig'))
    originals=next(ast.literal_eval(x.value) for x in source.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ORIGINAL_SCALARS' for t in x.targets))
    metrics=set(originals)|{k for x in N for k in x if k.startswith('DESC__')}
    pr=rows(E/'paired_per_output.csv');groups=collections.defaultdict(list)
    exp_pairs={(after+'-minus-'+before,q['request_id'],stage,m) for before,after in R['comparisons'] for q in R['requests'] for stage in ('RAW','POST') for m in metrics}
    check(len(pr)==len(exp_pairs) and {(x['comparison'],x['request_id'],x['stage'],x['metric']) for x in pr}==exp_pairs,'all frozen paired metric cells present')
    for x in pr:
        after,before=x['comparison'].split('-minus-');m=x['metric'];aa=num(lk[before,x['request_id'],x['stage']].get(m));bb=num(lk[after,x['request_id'],x['stage']].get(m));valid=aa is not None and bb is not None
        check(x['status']==('PASS' if valid else 'NA'),'paired validity')
        check(close(x['baseline_value'],aa) and close(x['candidate_value'],bb) and close(x['delta'],bb-aa if valid else None),'independent per-output pair arithmetic')
        groups[tuple(x[k] for k in ('comparison','panel','parent_id','condition_id','condition_type','stage','metric'))].append(x)
    parents=rows(E/'paired_per_parent.csv');pg=collections.defaultdict(list)
    for x in parents:
        g=groups[tuple(x[k] for k in ('comparison','panel','parent_id','condition_id','condition_type','stage','metric'))]
        check(int(x['n_registered'])==len(g)==(8 if x['panel']=='TRAIN_FAMILIAR' else 4),'noise denominator')
        check(int(x['n_valid'])==sum(z['status']=='PASS' for z in g),'valid noise denominator')
        for k in ('baseline_value','candidate_value','delta'):check(close(x[k],mean([z[k] for z in g])),'independent parent mean')
        pg[tuple(x[k] for k in ('comparison','panel','stage','metric'))].append(x)
    for x in rows(E/'paired_summary.csv'):
        g=pg[tuple(x[k] for k in ('comparison','panel','stage','metric'))];check(int(x['n_registered'])==len(g)==4,'four parent macro denominator')
        for k in ('baseline_value','candidate_value','delta'):check(close(x[k],mean([z[k] for z in g])),'independent panel mean')
    # Only reached after complete, hash-locked anonymous review gate above.
    key=read(E/'blinding/model_key.json')['assignments']
    vl={(key[x['condition_id']][x['anonymous_label']],x['request_id'],x['stage']):x for x in reviews}
    check(len(vl)==384 and set(vl)==expected,'review/model mapping')
    pc=rows(O/'results/per_condition.csv');pan=rows(O/'results/per_panel.csv')
    check(len(pc)==64 and len(pan)==16,'condition/panel summary counts')
    for x in pc:
        rr=[z for z in N if all(x[k]==z[k] for k in ('panel','parent_id','condition_id','label','stage'))]
        check(len(rr)==int(x['n']),'condition denominator')
        for m in METRICS:check(close(x[m],mean([z[m] for z in rr])),'condition metric')
        for f in FIELDS:
            for value in ('YES','NO','UNCERTAIN'):check(int(x[f+'__'+value])==sum(vl[z['label'],z['request_id'],z['stage']][f]==value for z in rr),'condition review count')
    for x in pan:
        rr=[z for z in pc if all(x[k]==z[k] for k in ('panel','label','stage'))]
        check(len(rr)==int(x['parents'])==4 and int(x['n'])==sum(int(z['n']) for z in rr),'panel denominator')
        for m in METRICS:check(close(x[m],mean([z[m] for z in rr])),'panel metric')
        for f in FIELDS:
            for value in ('YES','NO','UNCERTAIN'):check(int(x[f+'__'+value])==sum(int(z[f+'__'+value]) for z in rr),'panel review count')
        check(int(x['valid_edge_outputs'])==sum(num(z['C_F_has_valid_edge'])==1 for z in N if all(x[k]==z[k] for k in ('panel','label','stage'))),'valid-edge output count')
    dd=rows(O/'results/paired_decisions.csv');check(len(dd)==384,'four comparisons x48 x2')
    for x in dd:
        after,before=x['comparison'].split('-minus-');av=vl[before,x['request_id'],x['stage']]['C_adoption'];bv=vl[after,x['request_id'],x['stage']]['C_adoption']
        status='UNCERTAIN' if 'UNCERTAIN' in (av,bv) else 'GAIN' if (av,bv)==('NO','YES') else 'LOSS' if (av,bv)==('YES','NO') else 'SAME'
        check(x['C_adoption_pair']==status and (x['C_adoption_before'],x['C_adoption_after'])==(av,bv),'definite versus uncertain transition')
        a,b=lk[before,x['request_id'],x['stage']],lk[after,x['request_id'],x['stage']]
        for m in METRICS:
            aa,bb=num(a[m]),num(b[m]);check(close(x[m+'__delta'],bb-aa if aa is not None and bb is not None else None),'final paired delta')
        aa,bb=num(a['C_F_has_valid_edge']),num(b['C_F_has_valid_edge'])
        check(x['local_edge_pair']==('GAIN' if bb>aa else 'LOSS' if bb<aa else 'SAME'),'local edge transition')
    ds=rows(O/'results/paired_panel_decisions.csv');check(len(ds)==16,'paired panel count')
    for x in ds:
        rr=[z for z in dd if all(x[k]==z[k] for k in ('comparison','panel','stage'))]
        check(int(x['n'])==len(rr),'paired panel denominator')
        for f in ('C_adoption_pair','local_edge_pair'):
            for v in ('GAIN','LOSS','SAME','UNCERTAIN'):check(int(x[f+'__'+v])==sum(z[f]==v for z in rr),'paired panel transitions')
    summary=read(O/'summary.json')
    check(summary['status']=='COMPLETE' and summary['actual_review_rows']==384,'complete summary')
    check(len(summary['per_panel'])==16 and len(summary['per_condition'])==64 and len(summary['style_reviews'])==32,'summary data size')
    return dict(scores=384,new_scores=192,cached_scores=192,paired_rows=len(pr),parent_rows=len(parents),per_condition=len(pc),per_panel=len(pan),paired_decisions=len(dd),independent_per_panel=pan,independent_transitions=ds)

def audit_amendment(R):
    folder=O/'engineering/001_import_path';a=read(folder/'amendment.json')
    old=a['previous_code_hashes'];new=a['current_code_hashes'];target=str(ROOT/'r2/pilots/c_reference_frame_v1/evaluate.py')
    check(a['status']=='APPLIED_ENGINEERING_ONLY','engineering amendment status')
    check(a['original_registration_sha256']==sha(E/'registration.json'),'unchanged parent evaluation registration')
    check(old==R['code_hashes'] and set(old)==set(new),'amendment exact parent code manifest')
    check([k for k in old if old[k]!=new[k]]==[target],'only evaluate.py changed')
    check(sha(a['preserved_original_path'])==a['previous_evaluate_sha256']==old[target],'preserved original evaluate bytes')
    verify(new,'amended current code')
    before=path(a['preserved_original_path']).read_text(encoding='utf-8-sig');after=path(target).read_text(encoding='utf-8-sig')
    def normalized(text,strip_package):
        tree=ast.parse(text)
        tree.body=[x for x in tree.body if not (isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name=='registration')]
        if strip_package:
            tree.body=[x for x in tree.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Attribute) and isinstance(x.value.func.value,ast.Name) and x.value.func.value.id=='C' and x.value.func.attr=='package')]
        return ast.dump(tree,include_attributes=False)
    check(normalized(before,False)==normalized(after,True),'only import ordering and amendment validation changed')
    import difflib
    diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='evaluate_before.py',tofile='evaluate.py'))
    check((folder/'code.diff').read_text(encoding='utf-8-sig')==diff,'preserved diff matches actual source')
    check(not a['training_or_sampling_definition_changed'] and a['affected_completed_outputs']==a['actual_new_D0_before_error']==a['scientific_calls_replayed']==0,'declared no scientific changes/replay')
    amend_time=datetime.datetime.fromisoformat(a['utc']).timestamp()
    raws=list((E/'raw').glob('CREF_*/*/calls.jsonl'));complete=[]
    for p in raws:
        ev=events(p);rr=[x for x in ev if x['event']=='RESERVE' and x.get('kind')=='sampling_forward'];tt=[x for x in ev if x['event']=='RETURN' and x.get('kind')=='sampling_forward']
        check(len(rr)==len(tt)==50,'original RAW ledger no replay after amendment')
        complete.extend(x['time_unix'] for x in ev if x['event']=='ROLLOUT_COMPLETE')
        check(all(x['time_unix']<amend_time for x in ev if 'time_unix' in x),'all generator scientific calls precede amendment')
    check(len(raws)==len(complete)==96,'96 completed RAW before amendment')
    de=[]
    for p in (O/'execution').rglob('*events.jsonl'):de.extend(events(p))
    check(all(datetime.datetime.fromisoformat(x['utc']).timestamp()>amend_time for x in de if x['event'].startswith('D0_FORWARD_')),'zero D0 calls before amendment')
    phase=events(O/'ledger.jsonl');failed=[x for x in phase if x['event']=='PHASE_END' and x.get('returncode')]
    check(len(failed)==1 and failed[0]['module']=='evaluate' and failed[0]['args']==['--stage','post','--arm','CREF_C0'],'only registered pre-D0 engineering phase failed')
    preserved={str(folder/'amendment.json'):sha(folder/'amendment.json'),str(folder/'code.diff'):sha(folder/'code.diff'),a['preserved_original_path']:old[target]}
    return dict(status='PASS' if not errors else 'FAIL',old_code_hashes=old,new_code_hashes=new,preserved_artifacts=preserved,scientific_replays=0,generator_calls_before_amendment=4800,D0_calls_before_amendment=0)

def final():
    ready=readiness()
    if ready['status']!='READY_FOR_FINAL_AUDIT':
        save('final_audit_pending.json',ready);return ready
    reviews,styles,reviewhashes=locked_visual_reviews()
    TR=read(O/'registration.json');R=read(E/'registration.json')
    check(tuple(R['labels'])==LABELS,'registered four models')
    check(read(O/'inputs/manifest.json')==read(OLD/'inputs/manifest.json'),'exact frozen48 request manifest')
    amendment=audit_amendment(R)
    training=audit_training(TR);outputs=audit_outputs(R);numeric=audit_numeric(R,reviews)
    protected={}
    for group in (TR['source_hashes'],R['protected'],amendment['new_code_hashes'],amendment['preserved_artifacts'],R['blind_files']):
        for p,h in group.items():
            if p in protected:check(protected[p]==h,'consistent protected digest')
            protected[p]=h
    for q in R['requests']:protected.update({q['input_path']:q['input_sha256'],q['condition_source']:q['condition_source_sha256']})
    for q in R['queries']:protected[q['path']]=q['sha256']
    for b in (R['models']['EXP67_FROZEN'],R['models']['SMALL_C200'],R['d0_checkpoint']):protected[b['path']]=b['sha256']
    verify(protected,'protected old/new frozen asset')
    images=rows(E/'gallery/images.csv');sheets=rows(E/'gallery/POST_contact_sheets.csv')
    check(len(images)==48 and len(sheets)==8,'complete image inventory')
    for x in images+sheets:check(sha(x['image_path'])==x['image_sha256'],'rendered image bytes')
    report=O/'REPORT.md';text=report.read_text(encoding='utf-8-sig')
    check(len(text)>100 and '\ufffd' not in text,'nonempty valid UTF8 report')
    report_check='FILE_EXISTENCE_UTF8_AND_HASH_ONLY; scientific narrative requires separate text review'
    out=dict(status='PASS' if not errors else 'FAIL',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      checks=dict(checks),errors=errors,training=training,inference=outputs,numeric=numeric,engineering_amendment=amendment,
      visual=dict(rows=384,style_rows=32,pages=48,sheets=8,review_hashes=reviewhashes,gate='All anonymous reviews checked before this audit opened the model key or numerical scores'),
      integrity=dict(protected_files=len(protected),unique_files_hashed=len(cache),protected_hashes=protected,report_sha256=sha(report),report_review=report_check),
      audit_execution=dict(model_calls=0,weights_deserialized=0,new_OT=0,new_noise=0,geometry_rescores=0),
      limitations=['Four parents per panel; noises are within-condition replicates.','A valid C/F edge proves a local interface, not global C adoption or a closed seam.','AI proxy visual judgments are not human ratings; prior familiarity may reveal identities.','CREF changes coordinate/prior strategy jointly; retained world Geo means no full translation-equivariance claim.','PyTorch memory counters do not establish exact process VRAM under WDDM.'])
    save('final_audit.json',out)
    lines=['# Independent final audit','','Status: '+out['status'], '',
      '- Training: 200 updates; 1600 forward and 1600 backward calls, same original SMALL initialization and exposure records.',
      '- New inference: 96 Euler50 rollouts / 4800 forwards, then 96 single D0 calls. Cached controls: 96 RAW + 96 POST.',
      '- All RAW precede first D0 by %.3f seconds.'%outputs['RAW_D0_gap_seconds'],
      '- Local noise, local Euler, world restoration, external C bytes and saved source orientation were independently checked.',
      '- 384 scored stages (192 new, 192 reused), pair arithmetic, four-parent macro means, 384 visual rows and 32 style rows checked.',
      '- Report hash/UTF8 checked; scientific prose receives a separate review.','',
      'No model, training, geometry scorer, new noise or OT was invoked by this audit.']
    (A/'final_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {k:out[k] for k in ('status','checks','errors','training','inference')}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');args=p.parse_args()
    result=final() if args.final else readiness()
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    if result['status']=='FAIL':sys.exit(1)

