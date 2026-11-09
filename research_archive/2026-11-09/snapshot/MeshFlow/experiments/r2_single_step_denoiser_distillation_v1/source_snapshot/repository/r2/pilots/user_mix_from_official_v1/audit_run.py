"""Independent end-of-run CPU audit. Never constructs or executes a model.

Run after training, evaluation and rendering finish. Missing stages are BLOCKED;
failures remain visible. Original inputs/results are read only. Only audit/ is
written. CUDA, sample preparation, OT and scorers are never dispatched here.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
CODE=Path(__file__).resolve().parent
DEFAULT=CODE.parents[2]/'experiments/r2_user_mix_from_official_v1'
ARM='R2_USER_MIX'
LABELS=('R2_FROZEN','R2_CONT_OLD','R2_CONT_USER',ARM)
LIMITS={'train_forward':48000,'train_backward':48000,'optimizer_update':6000}


def resolve(value):
    value=str(value)
    if os.name!='nt' and len(value)>2 and value[1]==':':
        return Path('/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/'))
    return Path(value)


def read(path):return json.loads(resolve(path).read_text(encoding='utf-8-sig'))
def jsonl(path):return [json.loads(line) for line in resolve(path).read_text(encoding='utf8').splitlines() if line.strip()]
def table(path):
    with resolve(path).open(newline='',encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))
def digest(path):
    h=hashlib.sha256()
    with resolve(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def atomic(path,value):
    path=resolve(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    os.replace(temp,path)
def require(ok,message):
    if not ok:raise AssertionError(message)
def number(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def mean(values):return math.fsum(values)/len(values) if values and all(number(v) for v in values) else None
def same(actual,expected):
    if expected is None:return actual in ('',None)
    try:return math.isclose(float(actual),float(expected),rel_tol=2e-12,abs_tol=2e-12)
    except (ValueError,TypeError):return False
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


class Audit:
    def __init__(self,out):self.out=out;self.checks=[];self.details={};self.cache={}
    def section(self,name,function):
        begin=time.perf_counter()
        try:
            details=function();self.details[name]=details
            row=dict(name=name,status='PASS',seconds=time.perf_counter()-begin,details=details)
        except FileNotFoundError as exc:
            row=dict(name=name,status='BLOCKED',reason='MISSING_ASSET: '+str(exc),seconds=time.perf_counter()-begin)
        except Exception as exc:
            row=dict(name=name,status='FAIL',reason=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-begin)
        self.checks.append(row);print(name,row['status'],flush=True)
    def hashed_files(self,values):
        failures=[]
        for path,record in values.items():
            expected=record if isinstance(record,str) else record['sha256']
            if not resolve(path).is_file():failures.append(dict(path=path,status='MISSING'))
            elif digest(path)!=expected:failures.append(dict(path=path,status='HASH_MISMATCH'))
        require(not failures,'Changed/missing files: '+json.dumps(failures[:20]))
        return len(values)
    def source(self):
        manifest=read(self.out/'code_manifest.json')
        count=self.hashed_files(manifest['source_hashes'])
        ev=read(self.out/'evaluation/registration.json')
        scorer_count=self.hashed_files(ev['scorer_sources'])
        for path,h in ev['scorer_sources'].items():
            require(digest(self.out/'source_external_snapshot'/Path(path).name)==h,'Actual scorer snapshot differs')
        return dict(frozen_sources=count,actual_scoring_sources=scorer_count,
            code_manifest_sha256=digest(self.out/'code_manifest.json'),audit_source_sha256=digest(__file__),
            audit_added_after_training_source_freeze=str(Path(__file__).resolve()) not in manifest['source_hashes'])
    def protected(self):
        rows=read(self.out/'protected_before.json')['files']
        require(len(rows)==1059,'Protected manifest count differs from registered1059')
        count=self.hashed_files(rows)
        return dict(files=count,changed=0,missing=0,total_bytes=sum(r['bytes'] for r in rows.values()))
    def train_calls(self):
        path=self.out/'training'/ARM/'calls.jsonl';rows=jsonl(path)
        begun={};closed=set();counts=Counter();steps=[];order=[]
        for row in rows:
            require(row['arm']==ARM,'Wrong ledger arm')
            kind,event,ident=row['kind'],row['event'],row['call_id']
            require(kind in LIMITS,'Extra call kind: '+kind)
            counts[event+'__'+kind]+=1
            if event=='CALL_BEGIN':
                require(ident not in begun,'Duplicate call reservation')
                require(len(begun)==len(closed),'Call overlaps an unresolved reservation')
                begun[ident]=row;order.append(kind)
                if kind=='optimizer_update':steps.append(row['context']['step'])
            elif event in ('CALL_DONE','CALL_FAILED'):
                require(ident in begun and ident not in closed,'Completion missing reservation or repeated')
                require(begun[ident]['kind']==kind and begun[ident]['context']==row['context'],'Call return identity differs')
                closed.add(ident)
            else:raise AssertionError('Unknown ledger event '+event)
        for kind,count in LIMITS.items():
            require(counts['CALL_BEGIN__'+kind]==counts['CALL_DONE__'+kind]==count,'Call budget/count differs '+kind)
            require(counts['CALL_FAILED__'+kind]==0,'Failed calls '+kind)
        require(steps==list(range(1,6001)),'Optimizer update replay or skip')
        expected_order=(['train_forward','train_backward']*8+['optimizer_update'])*6000
        require(order==expected_order,'Forward/backward/optimizer execution order differs')
        require(set(begun)==closed,'Unresolved calls')
        status=read(self.out/'training'/ARM/'completed.json')
        require(status['status']=='COMPLETE' and status['global_step']==6000,'Training not complete')
        self.cache['training_status']=status
        return dict(rows=len(rows),counts=dict(counts),optimizer_steps=6000,unresolved=0,failed=0,replayed=0)
    def exposures(self):
        registration=read(self.out/'data/input_registration.json')
        plan=read(self.out/'data/mixed_plan.json')['records'][ARM]
        inputs=read(self.out/'data'/ARM/'paired_inputs.json')['records']
        paths=sorted((self.out/'training'/ARM/'attempts').glob('*/exposure.jsonl'))
        require(paths,'No actual exposure ledger')
        rows=[r for p in paths for r in jsonl(p)]
        rows.sort(key=lambda r:(r['step'],r['sample_index']))
        require(len(rows)==len(plan)==len(inputs)==48000,'Actual/planned/materialized exposure count differs')
        require([r['sample_index'] for r in rows]==list(range(48000)),'Exposure repeated/missing')
        parents=set(registration['parents'])
        require(len(parents)==25 and Counter(r['uid'] for r in rows)=={p:1920 for p in parents},'Parent exposure imbalance')
        held=set(registration['held_condition_ids']);users=registration['tasks'][registration['condition_source_task_keys']['USER']]
        old=registration['tasks'][registration['condition_source_task_keys']['OLD']]
        require(len(users)==117 and len(old)==395,'Training condition pools differ')
        dev=set(q['parent_id'] for q in read(self.out/'evaluation/registration.json')['requests'])
        require(not parents&dev,'Train/DEV parent overlap')
        for i,(r,p,inp) in enumerate(zip(rows,plan,inputs)):
            for key in ('task_id','uid','N','K','ratio','alpha','t','source_kind','step'):
                require(r[key]==p[key],f'Actual exposure differs from plan index{i}: '+key)
            require(r['task_id'] not in held,'HOLD condition trained')
            tasks=users if r['source_kind']=='USER' else old
            require(r['task_id'] in tasks and tasks[r['task_id']]['uid']==r['uid'],'Foreign condition or parent')
            require(tasks[r['task_id']].get('split','train')=='train' and tasks[r['task_id']].get('role','train')=='train','DEV/HOLD task trained')
            require(r['free_coordinates']==9*(r['N']-r['K']),'Free coordinate denominator differs')
            for key in ('input_bytehash','label_bytehash','ot_cache_key'):
                require(r[key]==inp[key],'Exposure consumed different materialized input '+key)
        by_source=Counter(r['source_kind'] for r in rows)
        require(by_source=={'USER':36000,'OLD':12000},'6:2 mix count differs')
        for source,perparent in (('USER',1440),('OLD',480)):
            rr=[r for r in rows if r['source_kind']==source]
            require(Counter(r['uid'] for r in rr)=={p:perparent for p in parents},'Source parent balance differs')
            require(Counter(r['uid'] for r in rr if r['alpha']==1.)=={p:perparent//2 for p in parents},'Source clean balance differs')
        old_counts=Counter((r['ratio'],r['alpha']==1.) for r in rows if r['source_kind']=='OLD')
        require(old_counts=={(.2,True):3000,(.2,False):3000,(.4,True):3000,(.4,False):3000},'OLD ratio/augmentation imbalance')
        for start in range(0,48000,8):
            batch=rows[start:start+8]
            require(len({r['uid'] for r in batch})==8,'Batch duplicate parent')
            require(Counter((r['source_kind'],r['alpha']==1.) for r in batch)=={('USER',True):3,('USER',False):3,('OLD',True):1,('OLD',False):1},'Batch source/width differs')
        free={kind:sum(r['free_coordinates'] for r in rows if r['source_kind']==kind) for kind in ('USER','OLD')}
        self.cache['actual_exposures']=rows
        return dict(exposures=len(rows),parents=len(parents),per_parent=1920,by_source=dict(by_source),
            held_exposures=0,dev_exposures=0,free_coordinates=free,
            USER_free_coordinate_fraction=free['USER']/sum(free.values()),distinct_USER_conditions=len({r['task_id'] for r in rows if r['source_kind']=='USER'}),
            actual_exposure_files=[str(p) for p in paths])
    def cpu_ot(self):
        receipt=read(self.out/'data/precompute_receipt.json')
        require(receipt['status']=='PASS' and receipt['total_samples']==48000,'CPU preparation not complete')
        paths=sorted((self.out/'data'/ARM/'batches').glob('*.attempt.json'))
        require(len(paths)==6000,'OT batch attempt count differs')
        indices=[];returns=0;plansha=digest(self.out/'data/mixed_plan.json')
        for path in paths:
            r=read(path);require(r['status'] in ('COMPLETE','RECOVERED_COMPLETE'),'Unresolved OT attempt')
            require(r['plan_sha256']==plansha,'OT used different frozen plan')
            require(len(r['OT_attempts'])==8 and all(a['status']=='RETURNED' for a in r['OT_attempts']),'OT failure or incomplete return')
            batch=path.with_suffix('').with_suffix('.npz')
            require(digest(batch)==r['batch_npz_sha256'],'Materialized batch hash differs')
            indices.extend(r['sample_indices']);returns+=len(r['OT_attempts'])
        require(sorted(indices)==list(range(48000)),'OT sample repeated/missing')
        require(returns==48000,'OT return budget differs')
        require(digest(self.out/'data'/ARM/'paired_inputs.json')==receipt['arm_cache_index_sha256'][ARM],'OT input index changed')
        require(receipt['model_forwards']==receipt['model_backwards']==receipt['optimizer_updates']==receipt['sampling_calls']==0,'CPU preparation ran model calls')
        return dict(OT_attempts=48000,OT_returns=returns,batches=len(paths),failures=0,materialized_hashes_verified=len(paths),model_calls=0)
    def checkpoints(self):
        import torch
        torch.set_num_threads(2)
        trainer=module('_r2_mix_independent_checkpoint_audit',CODE/'trainer.py')
        full=self.out/'checkpoints/R2_USER_MIX_step6000.pt'
        export=self.out/'inference/R2_USER_MIX_generator_step6000.pt'
        p=torch.load(full,map_location='cpu',weights_only=True,mmap=True)
        trainer.validate_payload(p)
        require(p['global_step']==6000,'Wrong final endpoint')
        require(p['lineage']['selected_state']=='ema' and p['lineage']['fresh_AdamW'] and p['lineage']['new_optimizer_step']==0,'Nonofficial or continued optimizer initialization')
        require(p['lineage']['official_file_sha256']==trainer.base.OFFICIAL_FILE_SHA256,'Official lineage hash differs')
        require(p['lineage']['role_nonzero']==p['lineage']['geo_exit_nonzero']==0,'Wrong initialization contract')
        require(p['identities']['registration_sha256']==digest(self.out/'registration.json'),'Training registration identity differs')
        require(p['identities']['code_manifest_sha256']==digest(self.out/'code_manifest.json'),'Training code identity differs')
        groups=p['optimizer']['param_groups'];require(len(groups)==1,'Extra optimizer/head group')
        group=groups[0];names=p['optimizer_parameter_names'];require(len(names)==1,'Optimizer names groups differ')
        require(group['lr']==1e-5 and tuple(group['betas'])==(.9,.95) and group['weight_decay']==0.,'AdamW settings differ')
        ids=group['params'];require(len(ids)==len(names[0])==len(p['optimizer']['state']),'Optimizer parameter count mismatch')
        for ident,name in zip(ids,names[0]):
            state=p['optimizer']['state'][ident];require(set(state)=={'step','exp_avg','exp_avg_sq'},'AdamW state fields differ')
            require(float(state['step'])==6000.,'Perparameter AdamW step differs')
            tensor=p['generator_state'][name.removeprefix('generator.')]
            require(tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all()),'Nonfinite/nonFP32 generator tensor')
            for key in ('exp_avg','exp_avg_sq'):
                require(state[key].shape==tensor.shape and state[key].dtype==torch.float32 and bool(torch.isfinite(state[key]).all()),'AdamW moment shape/dtype/finite differs')
        v=torch.load(export,map_location='cpu',weights_only=True,mmap=True)
        require(v['schema']==trainer.INFERENCE_SCHEMA and v['global_step']==6000 and v['arm']==ARM,'Wrong generator export')
        require(not any(k in v for k in ('optimizer','rng','teacher_state','aligner_state')),'Training/teacher tensors in export')
        require(v['training_checkpoint_sha256']==digest(full),'Generator/full binding differs')
        require(trainer.state_hash(v['generator_state'])==p['generator_state_sha256']==v['generator_state_sha256'],'Export does not match trained state')
        require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization in CPU audit')
        result=dict(global_step=6000,AdamW_parameters=len(ids),all_parameter_steps=6000,full_sha256=digest(full),generator_sha256=digest(export),model_constructed=False,model_forward=0,CUDA_initialized=False)
        del p,v
        return result
    def rollouts(self):
        ev=module('_r2_mix_independent_eval_audit',CODE/'evaluation.py')
        reg=ev.load_registration(self.out);requests=reg['requests'];queries=reg['queries']
        require(len(requests)==12 and len(queries)==6,'USER evaluation budget differs')
        require(all(q['panel']=='NEWDEV' for q in requests),'Unexpected OLD panel generation')
        new=cache=0;forwards=rollouts=0;contracts=0
        for label in LABELS:
            for q in requests:
                basepath=self.out/'evaluation/outputs'/label/q['request_id']
                r=read(basepath/'receipt.json')
                expected=reg['historical_checkpoints'][label]['sha256'] if label!=ARM else read(self.out/'evaluation/checkpoints'/f'{ARM}.json')['sha256']
                _,facts=ev.verify_output(q,r,expected);require(all(facts.values()),'RAW preservation/runtime contract failed');contracts+=len(facts)
                if label==ARM:
                    require(r['new_rollouts']==1 and r['generation_status']=='NEW_EULER50','New output provenance differs');new+=1
                    rows=jsonl(basepath/'calls.jsonl');counts=Counter((a['event'],a.get('kind')) for a in rows)
                    require(counts['RESERVE','sampling_forward']==counts['RETURN','sampling_forward']==50,'Sampling forward budget differs')
                    require(counts['RESERVE','rollout_attempt']==1 and counts['ROLLOUT_COMPLETE',None]==1,'Rollout attempt/completion budget differs')
                    require(not any(a['event']=='ERROR' for a in rows),'Sampling failed')
                    require([a['ordinal'] for a in rows if a['event']=='RESERVE' and a['kind']=='sampling_forward']==list(range(1,51)),'Sampling forward sequence replayed/skipped')
                    forwards+=50;rollouts+=1
                else:
                    require(not (basepath/'calls.jsonl').exists(),'Historical output regenerated')
                    require(digest(r['historical_score_path'])==r['historical_score_sha256'],'Historical score changed')
                    require(digest(r['historical_receipt_path'])==r['historical_receipt_sha256'],'Historical receipt changed');cache+=1
        require((new,cache,forwards,rollouts)==(12,36,600,12),'Sampling total differs')
        for q in queries:ev.load_queries(q)
        self.cache['eval_registration']=reg
        return dict(new_outputs=new,historical_RAW=cache,sampling_forward_begin=forwards,sampling_forward_return=forwards,rollout_attempts=rollouts,failures=0,RAW_contract_checks=contracts,C_query_banks=6,new_Gaussian_arrays=0)
    def scores(self):
        reg=read(self.out/'evaluation/registration.json');queries={q['condition_id']:q for q in reg['queries']}
        requests={q['request_id']:q for q in reg['requests']}
        source=read(self.out/'evaluation/scores_all.json');rows=source['rows']
        require(source['status']=='COMPLETE' and len(rows)==48,'Not all48 scores complete')
        by={(r['label'],r['request_id']):r for r in rows}
        require(len(by)==48,'Duplicated score identity')
        numeric=sorted({k for r in rows for k,v in r['metrics'].items() if number(v)})
        csvrows=table(self.out/'evaluation/per_output.csv')
        require(len(csvrows)==48,'Per-output CSV missing rows')
        csvby={(r['label'],r['request_id']):r for r in csvrows};cells=0
        for label in LABELS:
            for rid,q in requests.items():
                r=by[label,rid];individual=read(self.out/'evaluation/scores'/label/(rid+'.json'))
                require(r==individual and r['status']=='PASS','Aggregate/peroutput score mismatch')
                require(r['identity']['query_sha256']==queries[q['condition_id']]['sha256'],'Scoring changed query bank')
                require(r['free_face_denominator']==q['N']-q['K'] and r['C_query_count']==2048,'Scoring denominator changed')
                for path,h in r['files'].items():require(digest(path)==h,'Score detail file hash differs')
                for key,value in r['metrics'].items():
                    if number(value):require(same(csvby[label,rid].get(key),value),'Peroutput CSV numeric mismatch '+key);cells+=1
                if label!=ARM:require(r['metrics']==read(r['historical_score_path'])['metrics'],'Historical score was redefined')
        condition_values={}
        percondition=table(self.out/'evaluation/per_condition.csv');require(len(percondition)==24,'Percondition count differs')
        condition_csv={(r['label'],r['condition_id']):r for r in percondition}
        for label in LABELS:
            for cid in queries:
                rr=[r for r in rows if r['label']==label and r['condition_id']==cid]
                require(len(rr)==2,'Two-noise condition incomplete')
                values={k:mean([r['metrics'].get(k) for r in rr]) for k in numeric}
                condition_values[label,cid]=values
                for key,value in values.items():require(same(condition_csv[label,cid].get(key),value),'Condition mean mismatch '+key);cells+=1
        summaries=(('summary_equal_parent.csv','ALL'),('summary_by_region.csv','region_type'),('summary_by_tier.csv','tier'))
        for filename,grouping in summaries:
            for row in table(self.out/'evaluation'/filename):
                relevant=[r for r in percondition if r['label']==row['label'] and (grouping=='ALL' or r[grouping]==row['group'])]
                parent_ids=sorted({r['parent_id'] for r in relevant})
                require(int(row['registered_parents'])==len(parent_ids),'Summary parent count differs')
                for key in numeric:
                    expected=mean([mean([condition_values[r['label'],r['condition_id']][key] for r in relevant if r['parent_id']==pid]) for pid in parent_ids])
                    require(same(row.get(key),expected),'Equalparent summary mismatch '+filename+' '+key);cells+=1
        paired=table(self.out/'evaluation/paired_noise.csv');require(len(paired)==36,'Paired-noise count differs')
        for row in paired:
            baseline=row['comparison'].split(' minus ')[1];rid=row['request_id']
            for key in numeric:
                a,b=by[ARM,rid]['metrics'].get(key),by[baseline,rid]['metrics'].get(key)
                require(same(row.get(key),a-b if number(a) and number(b) else None),'Paired-noise delta mismatch');cells+=1
        pairedc=table(self.out/'evaluation/paired_conditions.csv');require(len(pairedc)==18,'Pairedcondition count differs')
        for row in pairedc:
            baseline=row['comparison'].split(' minus ')[1];cid=row['condition_id']
            for key in numeric:
                a,b=condition_values[ARM,cid][key],condition_values[baseline,cid][key]
                require(same(row.get(key),a-b if number(a) and number(b) else None),'Pairedcondition delta mismatch');cells+=1
        diversity=table(self.out/'evaluation/diversity.csv')
        require(len(diversity)==24 and all(r['status']=='PASS' and int(r['noise_pairs'])==1 and math.isfinite(float(r['D_free'])) for r in diversity),'Diversity records incomplete')
        return dict(scored=48,new_scored=12,historical_reused=36,numeric_metrics=len(numeric),numeric_cells_checked=cells,per_condition=24,paired_noise=36,paired_conditions=18,diversity_records=24,rescoring_calls=0,diversity_distance_recomputed=False)
    def gallery(self):
        manifest=read(self.out/'gallery/manifest.json')
        require(manifest['status']=='COMPLETE' and manifest['PNG_count']==24 and len(manifest['pages'])==24,'Gallery incomplete')
        for page in manifest['pages']:
            require(page['status']=='COMPLETE' and digest(page['image'])==page['image_sha256'],'Gallery image changed')
            require(page['identity']['columns']==list(LABELS),'Gallery omits a comparison model')
        require(len(list((self.out/'gallery').rglob('*.png')))==24,'Extra/missing gallery PNG')
        require((self.out/'gallery/index.html').is_file(),'Missing gallery index')
        return dict(PNG_count=24,requests=12,outputs=48,full_qualitative_review='SEPARATE_REVIEW_REQUIRED; rendered images alone do not establish actually_viewed')
    def delivery(self):
        required=['registration.json','code_manifest.json','code.diff','evaluation/per_output.csv','evaluation/per_condition.csv','evaluation/paired_noise.csv','evaluation/paired_conditions.csv','evaluation/summary_equal_parent.csv','evaluation/diversity.csv','gallery/index.html']
        for relative in required:require((self.out/relative).is_file(),'Missing delivery artifact '+relative)
        roots=['source_snapshot','source_external_snapshot','evaluation','gallery','review']
        paths={self.out/r for r in required}
        for name in roots:
            path=self.out/name
            if path.is_dir():
                paths.update(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in ('.py','.json','.csv','.md','.html','.png','.diff') and p.stat().st_size<20_000_000)
        paths.update(p for p in self.out.glob('*') if p.is_file() and p.suffix.lower() in ('.md','.json','.diff'))
        target=self.out/'audit/delivery_hashes.csv';target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('w',newline='',encoding='utf8') as handle:
            writer=csv.DictWriter(handle,fieldnames=['relative_path','bytes','sha256']);writer.writeheader()
            for path in sorted(paths):writer.writerow(dict(relative_path=path.relative_to(self.out).as_posix(),bytes=path.stat().st_size,sha256=digest(path)))
        return dict(files=len(paths),hash_manifest=str(target),hash_manifest_sha256=digest(target),scope='Small review artifacts; excludes weight tensors, data/OT caches and RAW binary arrays. RAWs/weights independently checked above.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(DEFAULT));args=p.parse_args()
    out=resolve(args.out).resolve()
    require(out.name=='r2_user_mix_from_official_v1' and out.parent.name=='experiments','Unexpected output root')
    audit=Audit(out);started=time.perf_counter()
    for name,function in [('source_hashes',audit.source),('training_ledger',audit.train_calls),('actual_training_exposures',audit.exposures),('CPU_OT',audit.cpu_ot),('CPU_checkpoints',audit.checkpoints),('RAW_rollouts',audit.rollouts),('scores_and_aggregation',audit.scores),('gallery',audit.gallery),('protected_original_files',audit.protected),('delivery',audit.delivery)]:audit.section(name,function)
    statuses=Counter(r['status'] for r in audit.checks)
    result=dict(schema='r2_user_mix_independent_execution_audit_v1',status='FAIL' if statuses['FAIL'] else 'BLOCKED' if statuses['BLOCKED'] else 'PASS',
        checks=audit.checks,check_counts=dict(statuses),seconds=time.perf_counter()-started,time_unix=time.time(),
        audit_source_sha256=digest(__file__),model_constructed=False,model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,sampling_calls=0,scorer_calls=0,
        limitation='CPU artifact and accounting verification; no new geometry scores and no claim of visual usefulness or independent generalization.')
    atomic(out/'audit/execution_audit.json',result)
    print(json.dumps(dict(status=result['status'],checks=dict(statuses),seconds=result['seconds'])),flush=True)
    return 0 if result['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
