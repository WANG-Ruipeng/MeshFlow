"""Independent final evidence audit: reads caches/ledgers; never imports a model.

Run after all three training arms, 48 RAW/POST cells, 96 scores and summaries
exist. It writes only audit/final_execution.json and supporting audit CSVs.
Weight files are byte-hashed, not deserialized or forwarded. Original scorers
are not called: numerical identities are reconstructed from their saved output.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import collections
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
from . import eval_common as C

ARMS=('LONG','ANTI_COVER','LOCAL_NOISE')
LABELS=('SEAM200','FULL4_C200',*ARMS)
BASE=C.ROOT/'experiments/r2_seam_slot_c200_v1'
ILLEGAL={'proper_transverse_penetration','coplanar_positive_area_overlap'}
META=('request_id','condition_id','parent_id','noise_index','N','K','split')


def need(ok,message):
    if not ok:raise AssertionError(message)


def jsonlines(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def csvrows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def scalar_equal(text,value):
    if value is None:return text==''
    if isinstance(value,bool):return text==str(value)
    if isinstance(value,(int,float)):
        try:return math.isclose(float(text),float(value),rel_tol=1.e-11,abs_tol=1.e-12)
        except ValueError:return False
    return text==str(value)


def csv_numeric(text):
    if text in ('True','False'):return float(text=='True')
    return float(text)


def normalized_rows(rows,exclude=('pid','time_unix')):
    return [{k:v for k,v in r.items() if k not in exclude} for r in rows]


class Audit:
    def __init__(self):self.hashed={};self.checks=[]
    def hash(self,path,expected=None,base=None):
        p=Path(path)
        if not p.is_absolute() and not str(path).startswith(('E:','C:','/mnt/')):
            p=(C.ROOT if base is None else base)/p
        p=C.resolve(p).resolve();key=str(p)
        if key not in self.hashed:self.hashed[key]=C.sha(p)
        if expected is not None:need(self.hashed[key]==expected,'Hash mismatch: '+key)
        return self.hashed[key]
    def mapping(self,files,base=None):
        for path,sha in files.items():self.hash(path,sha,base)
    def receipt(self,path):
        value=C.read(path);self.hash(path);self.mapping(value['files']);return value
    def record(self,name,**fields):self.checks.append(dict(check=name,status='PASS',**fields))


def training(a,root,reg):
    data=C.read(root/'data_manifest.json');schedule=C.read(root/'input_schedule.json')
    need(len(data['steps'])==800 and data['shared_samples']==6400 and data['OT_logical_calls']==6400,'Shared OT budget')
    need(len(list((root/'data/shared').glob('*.npz')))==6400 and
         len(list((root/'data/shared').glob('*.started.json')))==6400,'Actual OT sample reservations')
    need([x['source_step'] for x in data['steps']]==list(range(201,1001)),'Data stream continuation cursor')
    for entry in data['steps']:
        need(entry['samples']['LONG']==entry['samples']['ANTI_COVER']==entry['samples']['LOCAL_NOISE'],'Sample manifests not shared')
    heads=[];head_opt=[];exposures=[];head_logs=[];initial=[];rows=[]
    expected=dict(train_forward=6400,train_backward=6400,optimizer_update=800,
                  head_forward=6400,head_backward=6400,head_optimizer_update=800)
    for arm in ARMS:
        folder=root/'training'/arm;status=C.read(folder/'status.json');a.hash(folder/'status.json')
        need((status['status'],status['local_step'],status['global_step'])==('COMPLETE',800,13000),'Incomplete training '+arm)
        for key in ('checkpoint','inference_export','occupancy_export'):
            receipt=status[key];a.hash(receipt['path'],receipt['sha256'])
        need(status['checkpoint']['generator_state_sha256']==status['inference_export']['generator_state_sha256'],'Generator export identity')
        need(status['checkpoint']['head_state_sha256']==status['occupancy_export']['head_state_sha256'],'Head export identity')
        need(not status['calls']['failed'] and not status['calls']['pending_call_ids'],'Unresolved training calls')
        need(all(status['calls']['attempted'].get(k,0)==v and status['calls']['successful'].get(k,0)==v for k,v in expected.items()),'Training call budget')
        need(status['calls']['attempted'].get('preflight_forward',0)==0,'Extra preflight')
        callpath=folder/'calls.jsonl';calls=jsonlines(callpath);a.hash(callpath)
        begins={};done={}
        for r in calls:
            need(r['event'] in ('CALL_BEGIN','CALL_DONE'),'Failed/unknown training call event')
            dest=begins if r['event']=='CALL_BEGIN' else done
            need(r['call_id'] not in dest,'Duplicate call ID');dest[r['call_id']]=r
        need(set(begins)==set(done),'Unfinished training call')
        counter=collections.Counter(r['kind'] for r in begins.values());need(dict(counter)==expected,'Actual training ledger differs')
        for ci,b in begins.items():
            d=done[ci];need((b['kind'],b['context'])==(d['kind'],d['context']),'Call begin/end mismatch')
            need(d['time_unix']>=b['time_unix'],'Call chronology')
        updates=[r['context']['global_step'] for r in begins.values() if r['kind']=='optimizer_update']
        need(updates==list(range(12201,13001)),'Optimizer updates replayed/skipped')
        ini_files=sorted(folder.glob('attempts/*/initialization.json'))
        need(len(ini_files)==1,'Audit expects one successful training initialization')
        ini=C.read(ini_files[0]);a.hash(ini_files[0]);initial.append(ini)
        for field,key in (('generator_state_sha256','source_generator_sha256'),('optimizer_state_sha256','source_optimizer_sha256'),('rng_sha256','source_rng_sha256'),('head_state_sha256','source_head_sha256')):
            need(ini[field]==reg[key],'Initial state mismatch '+field)
        need(ini['global_step']==12200 and not ini['head_fresh'] and ini['exact_generator_optimizer_RNG_restoration'],'Wrong continuation policy')
        exp=jsonlines(folder/'exposure.jsonl');a.hash(folder/'exposure.jsonl');need(len(exp)==6400,'Exposure count')
        for i,r in enumerate(exp):
            step,micro=divmod(i,8);step+=1;s=data['steps'][step-1]['samples'][arm][micro]
            need((r['local_step'],r['global_step'],r['microbatch'])==(step,12200+step,micro),'Exposure cursor')
            need(C.resolve(r['path'])==C.resolve(s['path']) and r['uid']==s['uid'] and r['alpha']==s['alpha'],'Exposure sample identity')
            planned=schedule['steps'][step-1]['rows'][micro]
            need(r['uid']==planned['uid'] and r['alpha']==planned['alpha'],'Schedule mismatch')
        exposures.append(normalized_rows(exp))
        logs=jsonlines(folder/'training_log.jsonl');a.hash(folder/'training_log.jsonl');need(len(logs)==800,'Optimizer log count')
        for i,r in enumerate(logs,1):
            need((r['local_step'],r['global_step'])==(i,12200+i),'Log cursor')
            need(r['occupancy_edge_denominator']==132,'Head denominator')
            need(r['free_coordinate_denominator']==sum(x['free_coordinate_count'] for x in exp[(i-1)*8:i*8]),'FM denominator')
        if arm!='ANTI_COVER':need(all(r['anti_cover']==0 for r in logs),'Auxiliary loss leaked into control')
        hlog=jsonlines(folder/'occupancy_training.jsonl');a.hash(folder/'occupancy_training.jsonl')
        need(len(hlog)==6400,'Head log count');head_logs.append(normalized_rows(hlog))
        heads.append(status['checkpoint']['head_state_sha256']);head_opt.append(status['checkpoint']['head_optimizer_state_sha256'])
        rows.append(dict(arm=arm,updates=800,generator_forward=6400,generator_backward=6400,head_updates=800,
                         source_global_step=12200,final_global_step=13000,preflight=0,exposures=6400,
                         checkpoint_sha256=status['checkpoint']['sha256'],head_state_sha256=heads[-1]))
    need(exposures[0]==exposures[1]==exposures[2],'Exposure streams differ')
    need(head_logs[0]==head_logs[1]==head_logs[2],'Independent head numerical training differs')
    need(len(set(heads))==len(set(head_opt))==1,'Final heads/AdamW differ across arms')
    need(initial[0]==initial[1]==initial[2],'Initial continuation states differ')
    a.record('training',updates=2400,forward=19200,backward=19200,head_updates=2400,
             shared_OT=6400,identical_exposure_and_head_training=True,parameter_tensor_hash_recomputation='NA_BYTE_HASHED_EXPORTS_AND_RESTORE_RECEIPTS')
    return rows


def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def outputs(a,root,m):
    result=[];raw_last=[];d0_first=[];heads=[]
    need(sum(len(list((root/'evaluation'/arm).glob('*/raw/calls.jsonl'))) for arm in ARMS)==48,'Extra/missing RAW call ledgers')
    need(sum(len(list((root/'evaluation'/arm).glob('*/post/calls.jsonl'))) for arm in ARMS)==48,'Extra/missing D0 call ledgers')
    for arm in ARMS:
        ar=root/'evaluation'/arm
        for stage in ('raw','post'):
            comp=a.receipt(ar/(stage+'_complete.json'));need(comp['outputs']==16 and comp['status']=='COMPLETE','Stage incomplete')
        layouts=sorted((ar/'layouts').glob('*/receipt.json'));need(len(layouts)==4,'Four C-only layouts required')
        for path in layouts:
            r=a.receipt(path);need(r['occupancy_forward_calls']==1 and r['source_labels_read'] is False and r['occupancy_threshold_logit']==0,'Layout forward/oracle contract')
        for q in m['requests']:
            rid=q['request_id'];k,n=q['K'],q['N'];rd=ar/rid/'raw';pd=ar/rid/'post'
            rr=a.receipt(rd/'receipt.json');pr=a.receipt(pd/'receipt.json');prop=a.receipt(pd/'proposal_receipt.json')
            need(rr['generator_forwards']==50 and rr['new_rollouts']==1 and rr['D0_forwards']==0 and rr['source_labels_read'] is False,'RAW budget/source visibility')
            need(pr['D0_forwards']==pr['GEO_calls']==pr['orientation_solves']==1 and pr['source_labels_used'] is False,'POST budget/source visibility')
            need(prop['D0_forwards']==1 and prop['official_checkpoint_sha256']==C.read(root/'registration.json')['d0_checkpoint']['sha256'],'Official D0 identity')
            rc=jsonlines(rd/'calls.jsonl');pc=jsonlines(pd/'calls.jsonl')
            need(len(rc)==100 and [r['euler_step'] for r in rc[::2]]==list(range(50)),'Euler50 actual calls')
            need(all(rc[2*i]['event']=='FORWARD_STARTED' and rc[2*i+1]['event']=='FORWARD_RETURNED' and rc[2*i+1]['euler_step']==i for i in range(50)),'Euler call returns')
            need([r['event'] for r in pc]==['D0_FORWARD_STARTED','D0_FORWARD_RETURNED'],'Single D0 ledger')
            raw_last.append(rc[-1]['unix']);d0_first.append(pc[0]['unix'])
            raw=C.arrays(rd/'raw.npz');post=C.arrays(pd/'post.npz');pre=C.arrays(pd/'pre_geo.npz')['output']
            source=C.arrays(rd/'source_condition.npz');condition=C.arrays(q['inference_npz'])
            trajectory=C.arrays(rd/'trajectory.npz')['state'];anchor=raw['raw_anchor_ids'];fixed=anchor>=0
            need(trajectory.shape==(51,n,3,3) and exact(trajectory[-1],raw['raw']),'Trajectory terminal identity')
            vals=raw['C_vertices'][anchor[fixed]]
            need(all(exact(t[fixed],vals) for t in trajectory),'Moving C/seam anchor during rollout')
            need(exact(pre[fixed],vals),'D0 pre-GEO anchors differ')
            need(exact(raw['output'][:k],source['C_original']) and exact(post['output'][:k],source['C_original']),'Original C order/coordinates differ')
            z=C.arrays(q['noise_npz'])['z'];initial=np.zeros_like(raw['raw']);initial[k:]=z
            if arm=='LOCAL_NOISE':
                for slot in np.flatnonzero(raw['raw_seam_slot_mask']):
                    initial[slot,2]+=raw['C_vertices'][anchor[slot,:2]].mean(axis=0,dtype=np.float32)
            initial[fixed]=vals
            need(exact(initial,trajectory[0]),'Registered initial Gaussian/local mean mismatch')
            # Independent C vertex-ID and retained seam-index preservation check.
            mapping={}
            for ci,vi in zip(raw['C_faces_original'].flat,post['faces'][:k].flat):
                ci,vi=int(ci),int(vi)
                need(ci not in mapping or mapping[ci]==vi,'C source vertex split')
                mapping[ci]=vi
            need(len(mapping)==len(set(mapping.values()))==len(raw['C_vertices']),'C vertex identity collapse')
            src=post['source_face_ids'];corner=post['source_corner_ids'];transport=anchor[src[:,None],corner]
            for fi,co in np.argwhere(transport>=0):
                need(int(post['faces'][fi,co])==mapping[int(transport[fi,co])],'Retained seam does not share C index')
            need(exact(post['vertices'][post['faces']],post['output']),'POST index-coordinate mismatch')
            need(np.array_equal(raw['occupied'],raw['occupancy_logits']>=0),'Threshold differs')
            result.append(dict(arm=arm,request_id=rid,RAW_forwards=50,D0_forwards=1,C_preserved=True,
                all51_anchors_exact=True,retained_anchor_index_exact=True,noise_policy_exact=True,
                raw_sha256=a.hash(rd/'raw.npz'),post_sha256=a.hash(pd/'post.npz')))
        barriers=list(ar.glob('postprocess_attempt_*/all_RAW_barrier.json'));need(barriers,'Missing all RAW barrier')
        for bp in barriers:
            b=C.read(bp);a.hash(bp);need(b['status']=='PASS' and b['RAW']==48 and len(b['files'])==48,'Incomplete RAW barrier');a.mapping(b['files'])
    need(max(raw_last)<min(d0_first),'D0 began before final RAW completed')
    a.record('inference',new_RAW=48,generator_forwards=2400,occupancy_forwards=12,new_D0=48,
        all_RAW_before_D0=True,last_RAW_return_unix=max(raw_last),first_D0_start_unix=min(d0_first),
        original_C_order_and_anchor_identity='PASS',local_noise_actual_origin='PASS')
    return result


def numerical(a,root,m):
    completion=C.read(root/'evaluation/score_completion.json');a.hash(root/'evaluation/score_completion.json')
    need(completion['status']=='COMPLETE' and completion['cells']==96 and len(completion['receipts'])==96,'96 scores required')
    score_reg=C.read(root/'evaluation/score_registration.json')
    a.hash(root/'evaluation/score_registration.json',completion['registration_sha256'])
    a.hash(root/'eval_manifest.json',score_reg['manifest_sha256'])
    a.mapping(score_reg['source_hashes']);a.mapping(score_reg['protected'])
    need(sum(len(list((root/'evaluation/scores'/arm).glob('*/*/score.json'))) for arm in ARMS)==96,'Extra/missing new score cells')
    for rec in completion['receipts']:a.hash(rec['path'],rec['sha256'])
    csvpath=root/'evaluation/per_output.csv';actual=csvrows(csvpath);a.hash(csvpath)
    need(len(actual)==160,'Expected 5 labels x16requests x2stages')
    lookup={(r['label'],r['request_id'],r['stage']):r for r in actual};need(len(lookup)==160,'Duplicate per-output rows')
    expected={};role_totals=collections.defaultdict(collections.Counter)
    for label in LABELS:
        rroot=BASE if label in ('SEAM200','FULL4_C200') else root;arm='SEAM_SLOT_C200' if label=='SEAM200' else label
        for q in m['requests']:
            for stage in ('RAW','POST'):
                key=label,q['request_id'],stage;folder=rroot/'evaluation/scores'/arm/q['request_id']/stage
                s=a.receipt(folder/'score.json');a.hash(s['identity']['mesh_path'],s['identity']['mesh_sha256'])
                need(s['full_quality_scans']==1 and s['model_calls']==s['new_queries']==0,'Saved scorer call contract')
                row={k:s[k] for k in META};row.update(label=label,stage=stage)
                for field in ('metrics','topology','distance','seam_to_bulk'):
                    row.update({k:v for k,v in s[field].items() if not isinstance(v,(dict,list)) and k!='status'})
                row.update(seam_to_bulk_status=s['seam_to_bulk']['status'],C_preserved_exact=s['C_preserved_exact'],I_status=s['status_by_metric']['I'],
                    free_face_denominator=s['free_face_denominator'],CF_tested_pair_denominator=s['CF_tested_pair_denominator'],
                    FF_tested_pair_denominator=s['FF_tested_pair_denominator'],C_query_weight_denominator=s['C_query_weight_denominator'])
                for b in s['boundary']:row['A_VMR__'+b['setting']]=b['VMR'];row['PMR__'+b['setting']]=b['perfect']
                mesh=C.arrays(s['identity']['mesh_path']);ids=mesh['raw_slot_ids'];seam=mesh['raw_seam_slot_mask'][ids];k=q['K'];count=collections.Counter()
                pair=C.read(folder/'pairs.json')
                for kind,scan in pair.items():
                    for contact in scan['contacts']:
                        if contact['classification'] not in ILLEGAL:continue
                        i,j=map(int,contact['pair'])
                        if kind=='FF':
                            i+=k;j+=k;role='seam-seam' if seam[i] and seam[j] else 'seam-bulk' if seam[i] or seam[j] else 'bulk-bulk'
                        else:need(kind=='CF','Unexpected scan');j+=k;role='C-seam' if seam[j] else 'C-bulk'
                        count[role]+=1
                for role in ('seam-seam','seam-bulk','bulk-bulk','C-seam','C-bulk'):row['illegal_pairs_'+role]=count[role]
                need(sum(count[x] for x in ('seam-seam','seam-bulk','bulk-bulk'))==s['metrics']['FF_illegal_pair_count'],'Role FF sum')
                need(count['C-seam']+count['C-bulk']==s['metrics']['CF_illegal_pair_count'],'Role CF sum')
                for field,value in row.items():need(scalar_equal(lookup[key].get(field,''),value),'Per-output mismatch '+str(key)+' '+field)
                expected[key]=row;role_totals[label,stage].update(count)
    need([role_totals['SEAM200','POST'][r] for r in ('bulk-bulk','seam-bulk','seam-seam')]==[338,4,1],'Historical role reference differs')
    sum_path=root/'evaluation/summary.csv';summaries=csvrows(sum_path);a.hash(sum_path)
    for row in summaries:
        vals=[r[row['metric']] for key,r in expected.items() if key[0]==row['label'] and key[2]==row['stage'] and isinstance(r.get(row['metric']),(int,float,bool))]
        need(int(row['n_valid'])==len(vals),'Summary valid denominator')
        for field,func in (('mean',np.mean),('median',np.median),('min',np.min),('max',np.max),('sum',np.sum)):
            need(scalar_equal(row[field],float(func(vals)) if vals else None),'Summary statistic '+field)
    ppath=root/'evaluation/paired_per_noise.csv';paired=csvrows(ppath);a.hash(ppath)
    for row in paired:
        av=expected[row['baseline'],row['request_id'],row['stage']].get(row['metric'])
        bv=expected[row['candidate'],row['request_id'],row['stage']].get(row['metric'])
        valid=isinstance(av,(int,float,bool)) and isinstance(bv,(int,float,bool))
        need(scalar_equal(row['baseline_value'],av) and scalar_equal(row['candidate_value'],bv),'Paired source value')
        need(scalar_equal(row['delta'],float(bv)-float(av) if valid else None),'Paired delta')
    parentpath=root/'evaluation/paired_per_parent.csv';parent=csvrows(parentpath);a.hash(parentpath)
    groups=collections.defaultdict(list)
    fields=('baseline','candidate','parent_id','stage','metric')
    for row in paired:groups[tuple(row[f] for f in fields)].append(row)
    for row in parent:
        group=groups[tuple(row[f] for f in fields)];valid=[r for r in group if r['status']=='PASS']
        need(len(group)==4 and int(row['n_valid'])==len(valid),'Parent denominator')
        for field in ('baseline_value','candidate_value','delta'):
            need(scalar_equal(row[field],float(np.mean([csv_numeric(r[field]) for r in valid])) if valid else None),'Parent aggregation')
    a.record('numerical',new_scans=96,cached_reference_scores=64,per_output_rows=160,
        summary_rows=len(summaries),paired_noise_rows=len(paired),paired_parent_rows=len(parent),
        roles_verified='CF second +K; FF both +K; retained output ID to RAW slot')
    return [dict(label=key[0],stage=key[1],**count) for key,count in sorted(role_totals.items())]


def main():
    root=C.OUT;out=root/'audit';target=out/'final_execution.json'
    required=[root/'evaluation/summary_completion.json',root/'evaluation/score_completion.json',
              root/'antiflatten/FINAL_STATUS.json',root/'redirection/completion.json']
    for arm in ARMS:required.extend(root/'evaluation'/arm/(s+'_complete.json') for s in ('raw','post'))
    absent=[str(p) for p in required if not p.exists()]
    unfinished=[arm for arm in ARMS if not (root/'training'/arm/'status.json').exists() or C.read(root/'training'/arm/'status.json')['status']!='COMPLETE']
    if absent or unfinished:
        print(json.dumps(dict(status='NOT_READY',missing=absent,unfinished_training=unfinished,no_calls=True)));return
    if target.exists():
        previous=C.read(target);a=Audit();a.mapping(previous['verified_files']);a.mapping(previous['audit_artifacts'])
        print(json.dumps(dict(status='COMPLETE_VERIFIED',new_model_calls=0,new_scores=0,files=len(a.hashed))));return
    a=Audit();started=time.perf_counter();reg=C.read(root/'registration.json');tr=C.read(root/'training_registration.json')
    m=C.read(root/'eval_manifest.json');need(tuple(reg['arms'])==ARMS and tuple(m['arms'])==ARMS,'Arm registration')
    need(reg['updates_per_arm']==tr['updates_per_arm']==800 and reg['new_endpoint_rollouts']==reg['new_D0_forwards']==48,'Registered budget')
    need(len(m['requests'])==16 and len(m['conditions'])==4 and all(q['split']=='TRAIN' for q in m['requests']),'Panel scope')
    for name in ('execution_freeze.json','training_execution_freeze.json'):
        frozen=C.read(root/name);a.hash(root/name)
        a.hash(root/('training_registration.json' if name.startswith('training') else 'registration.json'),frozen['registration_sha256'])
        a.mapping(frozen['code_hashes']);a.mapping(frozen['input_hashes'],root if name.startswith('training') else None)
        if 'protected_weights' in frozen:a.mapping(frozen['protected_weights'])
    # Historical CPU check contains a documented freezer-only hash update.
    cpu=C.read(root/'training_cpu_checks.json');add=C.read(root/'training_cpu_checks_freeze_addendum.json')
    need(cpu['status']=='PASS' and add['status']=='PASS_SCIENTIFIC_IMPLEMENTATION_UNCHANGED','CPU source gate')
    allowance={x['path']:(x['checked_sha256'],x['current_sha256']) for x in add['changes']}
    for rec in cpu['code']:
        current=a.hash(rec['path'])
        need(current==rec['sha256'] or allowance.get(rec['path'])==(rec['sha256'],current),'Unrecorded CPU-gate code change')
    a.hash(root/'training_cpu_checks.json');a.hash(root/'training_cpu_checks_freeze_addendum.json')
    a.record('frozen_contract_and_source_hashes',data_source_count=4,official_D0_fixed=True,freezer_only_amendment=True)
    trainrows=training(a,root,tr);outputrows=outputs(a,root,m);roles=numerical(a,root,m)
    anti=C.read(root/'antiflatten/FINAL_STATUS.json');a.hash(root/'antiflatten/FINAL_STATUS.json')
    need(anti['status']=='COMPLETE' and anti['new_POST']==anti['original_quality_scans']==16 and anti['model_calls']==anti['D0_forwards']==0,'CPU anti-flatten completion')
    for key in ('code_hashes','protected_hashes','output_hashes'):a.mapping(anti[key])
    red=C.read(root/'redirection/completion.json');a.hash(root/'redirection/completion.json')
    need(red['status']=='COMPLETE' and red['new_model_calls']==0 and red['new_quality_scans']==16,'CPU redirection completion')
    for key in ('source_hashes','protected','artifacts'):a.mapping(red[key])
    a.record('independent_CPU_subexperiments',antiflatten_POST=16,redirection_POST=16,additional_model_calls=0,
        CPU_quality_scans=32,source_gate_edges=anti['source_edges'],source_gate_triggered=anti['source_triggered'])
    a.hash(__file__)
    need('torch' not in sys.modules,'Audit imported neural framework')
    C.csvwrite(out/'training_budget.csv',trainrows);C.csvwrite(out/'output_identity.csv',outputrows);C.csvwrite(out/'illegal_roles.csv',roles)
    artifacts={str(p):C.sha(p) for p in (out/'training_budget.csv',out/'output_identity.csv',out/'illegal_roles.csv')}
    C.write(target,dict(status='PASS',independent_auditor='seam_code_audit',new_model_calls=0,new_scores=0,
        new_geometry=0,checks=a.checks,seconds=time.perf_counter()-started,
        verified_files=a.hashed,audit_artifacts=artifacts,
        limits=['Does not deserialize weight tensors; checks file hashes and recorded exact-restoration identities.',
                'No collision detector rerun; verifies saved pair identity, role accounting, aggregation and metric denominators.',
                'Actual visual review remains a separate AI_PROXY record, never inferred by this audit.']))
    print(json.dumps(dict(status='PASS',checks=a.checks,verified_files=len(a.hashed),seconds=time.perf_counter()-started)))


if __name__=='__main__':main()
