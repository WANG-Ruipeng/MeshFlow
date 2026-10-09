"""Paired condition-ratio/HYBRID recipes; old schedule40 assets are read-only.

The original 48k40% stream is authoritative. MIX replaces half of its conditions,
without changing parent/N/alpha/time/noise/permutation seeds. Only new20% OT maps
are written here. Curriculum views reorder that same frozen MIX multiset.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from . import stream as base
from . import schedule40 as previous
from .dataset import TrainingDataset
from .io import array_hash, atomic_json, digest, freeze_json, json_hash, read_json
from ..training.coupling import CONTRACT

NAMESPACE='CHAIR_CONDITION_RECIPE_FACTORIAL_V1/stage1'
RECIPES={'R1':'R1_40_H_ALL','R2':'R2_MIX_H_ALL','R3':'R3_40_H_LATE','R4':'R4_MIX_H_LATE',
         'C20_40_MIX':'C20_40_MIX','C40_20_MIX':'C40_20_MIX'}
PAIR_FIELDS=('sample_index','task_id','shared_input_sha256','input_bytehash','label_bytehash','ot_cache_key','free_coordinate_count')
HASH_FIELDS=('x0','x1','u','xt','context','t','y','known_mask','valid_mask','source_face_ids',
             'corner_permutations','face_permutation','free_gaussian_before_OT','free_epsilon',
             'free_target_before_OT','ot_face_map','ot_corner_map')


def canonical_recipe(recipe):
    if recipe in RECIPES:return RECIPES[recipe]
    if recipe in RECIPES.values():return recipe
    raise ValueError('Unknown registered recipe: '+str(recipe))


def domain_seed(namespace,domain,*identity):
    data=json.dumps([namespace,domain,*identity],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4],'little')


def source_identity():
    paths=[Path(__file__),Path(previous.__file__),Path(base.__file__),Path(__file__).with_name('dataset.py'),Path(__file__).parents[1]/'training/coupling.py']
    return {str(p.relative_to(Path(__file__).parents[1])):digest(p) for p in paths}


def eligible20(manifest,parents):
    tasks=defaultdict(list)
    for row in manifest['tasks']:
        if row['uid'] in parents and row['role']=='train' and row['status']=='READY' and float(row['ratio'])==.2:
            if row.get('split','train')!='train' or row['uid'] not in manifest['splits']['train']:
                raise ValueError('20% task outside fixed training split')
            if row['K']!=round(.2*row['N']):raise ValueError('20% face count differs')
            if sorted(row['source_face_ids']+row['free_source_ids'])!=list(range(row['N'])):
                raise ValueError('Invalid20% C/free partition')
            tasks[row['uid']].append(row)
    if set(tasks)!=set(parents) or any(len(rows)!=8 for rows in tasks.values()):
        raise ValueError('Expected eight existing20% tasks for each of26 old parents')
    return {uid:sorted(rows,key=lambda row:row['task_id']) for uid,rows in sorted(tasks.items())}


def _batch_masks(rows):
    clean=[i for i,row in enumerate(rows) if row['alpha']==1]
    augmented=[i for i,row in enumerate(rows) if row['alpha']!=1]
    if len(clean)!=4 or len(augmented)!=4 or len({r['uid'] for r in rows})!=8:
        raise ValueError('Base batch must have8 distinct parents and4clean/4aug')
    result=[]
    for a in combinations(clean,2):
        for b in combinations(augmented,2):
            mask=np.zeros(8,np.int8);mask[list(a+b)]=1;result.append(mask)
    return np.stack(result)


def choose_ratio_mask(rows,balance,namespace,pair_index):
    """Each original batch is2clean20/2clean40/2aug20/2aug40.

    A parent present in both source batches gets opposite ratios, so each merged
    pure-ratio batch has8 distinct parents. Among feasible assignments minimize
    cumulative parent imbalance, then choose a deterministic independent tie.
    """
    left,right=_batch_masks(rows[:8]),_batch_masks(rows[8:])
    choices=np.concatenate((np.repeat(left,len(right),axis=0),np.tile(right,(len(left),1))),axis=1)
    positions=defaultdict(list)
    for i,row in enumerate(rows):positions[row['uid']].append(i)
    for ids in positions.values():
        if len(ids)==2:choices=choices[choices[:,ids].sum(axis=1)==1]
        elif len(ids)!=1:raise ValueError('Parent appears more than twice in two batches')
    if not len(choices):raise ValueError('No exact balanced ratio assignment; no resampling allowed')
    after=np.stack([balance.get(uid,0)+(2*choices[:,ids]-1).sum(axis=1) for uid,ids in sorted(positions.items())],axis=1)
    peak=np.max(np.abs(after),axis=1);squares=np.sum(after.astype(np.int64)**2,axis=1)
    keep=np.flatnonzero(peak==peak.min());keep=keep[squares[keep]==squares[keep].min()]
    rng=np.random.RandomState(domain_seed(namespace,'ratio_pair_tie',pair_index))
    picked=choices[int(keep[int(rng.randint(len(keep)))])]
    for i,row in enumerate(rows):balance[row['uid']]=balance.get(row['uid'],0)+int(2*picked[i]-1)
    return picked


def make_plan(manifest,old_plan,namespace=NAMESPACE):
    if namespace!=NAMESPACE:
        raise ValueError('Independent confirmation BLOCKED: a new base random stream must be registered; changing namespace is not an independent repeat')
    parents=old_plan['eligible_parents'];old=old_plan['records'];updates=old_plan['effective_batch_count']
    if len(parents)!=26 or len(old)!=8*updates or updates%2 or not 2<=updates<=6000:
        raise ValueError('Expected an even bounded old stream with26 parents')
    options=eligible20(manifest,parents)
    if [r['sample_index'] for r in old]!=list(range(len(old))) or any(r['ratio']!=.4 for r in old):
        raise ValueError('Old40% slot identity differs')
    balance={uid:0 for uid in parents};mixed=[];pure20=[];pure40=[]
    for start in range(0,len(old),16):
        source=old[start:start+16];mask=choose_ratio_mask(source,balance,namespace,start//16)
        by_ratio={.2:[],.4:[]}
        for offset,original in enumerate(source):
            index=start+offset;record=dict(original,origin_sample_index=index)
            if mask[offset]:
                choices=options[record['uid']]
                rng=np.random.RandomState(domain_seed(namespace,'condition20',index,record['uid']))
                task=choices[int(rng.randint(len(choices)))]
                if task['N']!=record['N']:raise ValueError('20% changes parent totalN')
                record.update(task_id=task['task_id'],K=int(task['K']),ratio=.2)
            mixed.append(record);by_ratio[record['ratio']].append(index)
        for ratio,target in ((.2,pure20),(.4,pure40)):
            ids=by_ratio[ratio];selected=[mixed[index] for index in ids]
            if len(ids)!=8 or len({r['uid'] for r in selected})!=8 or sum(r['alpha']==1 for r in selected)!=4:
                raise ValueError('Two-batch pure-ratio decomposition failed')
            target.append(ids)
    # Real protocol:2000 pure20,2000 pure40,2000 unchangedMIX updates.
    # Small divisible-by6 CPU fixtures use the same proportions.
    orders={};curriculum_updates=updates//3 if updates%6==0 else None
    if curriculum_updates is not None:
        prefix_pairs=curriculum_updates
        first=[i for batch in pure20[:prefix_pairs] for i in batch]
        second=[i for batch in pure40[:prefix_pairs] for i in batch]
        tail=list(range(8*2*curriculum_updates,len(old)))
        orders={'C20_40_MIX':first+second+tail,'C40_20_MIX':second+first+tail}
        for order in orders.values():
            if sorted(order)!=list(range(len(old))) or order[16*curriculum_updates:]!=tail:
                raise ValueError('Curriculum changed fixed multiset or finalMIX tail')
    return dict(schema='meshflow_recipe_factorial_plan_v1',namespace=namespace,
        old_R1_exact_reference=True,independent_training_repeat=False,
        namespace_scope='Only new ratio-mask and20% task selection; base training random seeds are inherited unchanged',
        total_updates=updates,effective_batch_count=updates,batch_size=8,microbatch_size=1,
        sample_count=len(old),eligible_parents=parents,eligible20_tasks={uid:[r['task_id'] for r in rows] for uid,rows in options.items()},
        eligible20_task_count=sum(map(len,options.values())),eligible40_task_count=old_plan['eligible_task_count'],
        mixed_records=mixed,curriculum_orders=orders,curriculum_segment_updates=curriculum_updates,
        ratio_assignment='Adjacent original batches; exact2clean/2aug at each ratio; repeated parent assigned opposite ratios; cumulative parent balance then seeded tie',
        ratio_counts=dict(Counter(str(r['ratio']) for r in mixed)),parent20_minus40_balance=balance,
        effective_free_coordinates_all40=sum(9*(r['N']-r['K']) for r in old),
        effective_free_coordinates_MIX=sum(9*(r['N']-r['K']) for r in mixed),
        optimizer_normalization='Unchanged effective batch total valid free scalar count; no ratio reweighting',
        coupling_contract=CONTRACT,new_OT_scope='Only MIX20% records; exact old40% caches borrowed read-only')


def prepare_recipe_plan(data_root,base_stream_root,stream_root,namespace=NAMESPACE):
    data=TrainingDataset(data_root);base_root=Path(base_stream_root).resolve();out=Path(stream_root).resolve()
    old=previous.Schedule40Stream(data_root,base_root,require_cache=True)
    paired_path=base_root/'paired_inputs.json';paired=read_json(paired_path)
    if paired['plan_sha256']!=old.plan_sha256 or paired['data_manifest_sha256']!=data.manifest_sha256 or len(paired['records'])!=len(old.records):
        raise ValueError('Old paired input registration differs')
    plan=make_plan(data.manifest,old.plan,namespace)
    plan.update(data_manifest_sha256=data.manifest_sha256,base_stream_root=str(base_root),
                base_plan_sha256=old.plan_sha256,base_paired_inputs_sha256=digest(paired_path),
                implementation_sources=source_identity())
    plan_sha=freeze_json(out/'recipe_plan.json',plan)
    return dict(status='PREPARED_NO_OT',path=str(out/'recipe_plan.json'),plan_sha256=plan_sha,
                samples=plan['sample_count'],new20_OT_maps=sum(r['ratio']==.2 for r in plan['mixed_records']),
                original_R1_assets_modified=False,model_forwards=0)


def prepare(data_root,previous_run,stream_root,namespace=NAMESPACE):
    return prepare_recipe_plan(data_root,Path(previous_run)/'stream',stream_root,namespace)


def lambda_for_step(recipe,step,hybrid_schedule=None):
    recipe=canonical_recipe(recipe)
    default='late' if recipe in ('R3_40_H_LATE','R4_MIX_H_LATE') else 'all'
    schedule=default if hybrid_schedule is None else hybrid_schedule
    if schedule not in ('all','late'):raise ValueError('HYBRID schedule must be all or late')
    if recipe.startswith('R') and schedule!=default:raise ValueError('Recipe/HYBRID schedule mismatch')
    if type(step) is not int or not 1<=step<=6000:raise ValueError('Invalid optimizer step')
    return .25 if schedule=='all' or step>4000 else 0.


def set_coupling(sample,lam):
    """Only the independent pureOT extension is new; HYBRID bytes stay unchanged."""
    if lam not in (0.,.25):raise ValueError('Only registered lambdas0/.25 allowed')
    result=dict(sample)
    if lam==0.:
        n=int(sample['y']);k=int(sample['K'])
        x0=np.zeros((n,3,3),np.float32);x0[k:]=sample['free_gaussian_before_OT']
        x0=x0[sample['face_permutation']].reshape(n,9)
        result['x0']=x0;result['u']=sample['x1']-x0
        result['xt']=(np.float32(1)-sample['t'])*x0+sample['t']*sample['x1']
        result['xt'][sample['known_mask']]=sample['context'][sample['known_mask']]
        shared={key:array_hash(result[key]) for key in HASH_FIELDS}
        result['shared_input_sha256']=json_hash(dict(shared,task_id=result['task_id'],alpha=result['alpha']))
        result['coupled_input_sha256']=result['shared_input_sha256']
        result['input_bytehash']=json_hash({key:shared[key] for key in ('xt','t','y','known_mask','valid_mask')})
        result['label_bytehash']=array_hash(result['u'])
    result['lambda_value']=float(lam)
    return result


class RecipeFactorialStream(base.HybridStream):
    def __init__(self,data_root,stream_root,recipe='R2',arm=None,require_cache=True,hybrid_schedule=None):
        self.out=Path(stream_root).resolve();self.plan_path=self.out/'recipe_plan.json'
        self.plan=read_json(self.plan_path);self.plan_sha256=digest(self.plan_path)
        self.recipe=canonical_recipe(arm if arm is not None else recipe);self.arm=self.recipe
        self.data=TrainingDataset(data_root);self.namespace=self.plan['namespace']
        if self.namespace!=NAMESPACE:raise ValueError('Independent confirmation BLOCKED: unregistered base random stream')
        if self.plan['data_manifest_sha256']!=self.data.manifest_sha256 or self.plan['implementation_sources']!=source_identity():
            raise ValueError('Frozen recipe data/source identity differs')
        self._base40=previous.Schedule40Stream(data_root,self.plan['base_stream_root'],require_cache=True)
        if self._base40.plan_sha256!=self.plan['base_plan_sha256']:raise ValueError('Old plan hash differs')
        old_paired_path=Path(self.plan['base_stream_root'])/'paired_inputs.json'
        if digest(old_paired_path)!=self.plan['base_paired_inputs_sha256']:raise ValueError('Old paired hash differs')
        old_paired=read_json(old_paired_path)
        self._old_paired={row['sample_index']:row for row in old_paired['records']}
        self.total_updates=int(self.plan['total_updates']);self.batch_index=0
        self.hybrid_schedule=hybrid_schedule or ('late' if self.recipe in ('R3_40_H_LATE','R4_MIX_H_LATE') else 'all')
        lambda_for_step(self.recipe,1,self.hybrid_schedule)
        self.require_cache=bool(require_cache);self.cache_dir=self.out/'new20_ot_cache'
        self.implementation_sha256=json_hash(source_identity())
        self.counts={'actual_OT_attempts':0,'actual_OT_returns':0,'OT_cache_hits':0,'old40_cache_reads':0}
        self._base_records=self._base40.records
        if self.recipe in ('R1_40_H_ALL','R3_40_H_LATE'):
            raw=self._base_records;order=list(range(len(raw)))
        else:
            raw=self.plan['mixed_records']
            order=self.plan['curriculum_orders'].get(self.recipe,list(range(len(raw))))
            if self.recipe.startswith('C') and self.recipe not in self.plan['curriculum_orders']:
                raise ValueError('Curriculum requires total updates divisible by6')
        self.records=[dict(raw[index],origin_sample_index=index,step=pos//8+1,
                           optimizer_step=pos//8+1,effective_batch_index=pos//8,slot_index=pos%8)
                      for pos,index in enumerate(order)]
        paired_name=self.recipe+('__H_'+self.hybrid_schedule.upper() if self.recipe.startswith('C') else '')
        self.paired_path=old_paired_path if self.recipe=='R1_40_H_ALL' else self.out/'paired_inputs'/(paired_name+'.json')
        self._assert_plan()

    def _assert_plan(self):
        expected=make_plan(self.data.manifest,self._base40.plan,self.namespace)
        if any(self.plan.get(key)!=value for key,value in expected.items()):
            raise ValueError('Frozen recipe plan differs from deterministic registered construction')
        old=self._base_records;records=self.plan['mixed_records'];tasks=self.data.tasks
        if len(records)!=len(old) or len(self.records)!=len(old):raise ValueError('Record count differs')
        paired_keys=('uid','N','alpha','gaussian_seed','epsilon_seed','permutation_seed','t_seed')
        for index,record in enumerate(records):
            if record['sample_index']!=index or record['origin_sample_index']!=index:
                raise ValueError('Original slot identity differs')
            if any(record[key]!=old[index][key] for key in paired_keys):raise ValueError('Paired base random/input identity differs')
            task=tasks[record['task_id']]
            if task['uid']!=record['uid'] or task['N']!=record['N'] or task['K']!=record['K'] or task['ratio']!=record['ratio']:
                raise ValueError('Registered task differs')
            if record['ratio']==.4 and record['task_id']!=old[index]['task_id']:raise ValueError('Old40 task changed')
        if sorted(r['sample_index'] for r in self.records)!=list(range(len(old))):raise ValueError('Stream multiset differs')
        for start in range(0,len(self.records),8):
            batch=self.records[start:start+8]
            if len({r['uid'] for r in batch})!=8 or sum(r['alpha']==1 for r in batch)!=4:raise ValueError('Batch parent/augmentation contract differs')
            if self.recipe in ('R2_MIX_H_ALL','R4_MIX_H_LATE'):
                if Counter((r['ratio'],r['alpha']==1) for r in batch)!=Counter({(.2,True):2,(.2,False):2,(.4,True):2,(.4,False):2}):
                    raise ValueError('MIX ratio/augmentation contract differs')

    def state_dict(self):
        return dict(schema='recipe_factorial_stream_state_v1',recipe=self.recipe,namespace=self.namespace,
                    hybrid_schedule=self.hybrid_schedule,total_updates=self.total_updates,batch_index=self.batch_index,
                    consumed_samples=8*self.batch_index,plan_sha256=self.plan_sha256,
                    base_plan_sha256=self.plan['base_plan_sha256'],data_manifest_sha256=self.data.manifest_sha256,
                    implementation_sha256=self.implementation_sha256)

    def load_state_dict(self,state):
        expected=self.state_dict()
        if set(state)!=set(expected):raise ValueError('Stream state keys differ')
        for key,value in expected.items():
            if key not in ('batch_index','consumed_samples') and state[key]!=value:raise ValueError('Stream restore identity differs: '+key)
        cursor=state['batch_index']
        if type(cursor) is not int or not 0<=cursor<=self.total_updates or state['consumed_samples']!=8*cursor:raise ValueError('Invalid stream cursor')
        self.batch_index=cursor

    def _mapping(self,pre,noise,source_ids,corners,known_ids,record):
        if record['ratio']!=.2:raise ValueError('New OT cache only accepts20% conditions')
        if self.require_cache:
            identity=dict(contract=CONTRACT,implementation_sha256=self.implementation_sha256,
                          pre_target=array_hash(pre),raw_noise=array_hash(noise),source_ids=array_hash(source_ids),
                          corners=array_hash(corners),known_ids=array_hash(known_ids),alpha=record['alpha'],
                          task_id=record['task_id'],sample_index=record['sample_index'],
                          plan_sha256=self.plan_sha256,data_manifest_sha256=self.data.manifest_sha256)
            path=self.cache_dir/(str(record['sample_index']).zfill(4)+'_'+json_hash(identity)+'.npz')
            if not path.is_file():raise RuntimeError('New20% CPU OT must be prepared before training: '+str(path))
        return super()._mapping(pre,noise,source_ids,corners,known_ids,record)

    def sample(self,record,lambda_value=None):
        origin=record['origin_sample_index'];step=int(record.get('optimizer_step',record['step']))
        if record['ratio']==.4:
            sample=self._base40.sample(self._base_records[origin]);self.counts['old40_cache_reads']+=1
            saved=self._old_paired[origin]
            for key in PAIR_FIELDS:
                if sample[key]!=saved[key]:raise ValueError('Old40 actual sample differs from frozen paired input: '+key)
            cache_origin='OLD40_READ_ONLY'
        else:
            sample=base.HybridStream.sample(self,record);cache_origin='NEW20_SHARED'
        lam=lambda_for_step(self.recipe,step,self.hybrid_schedule) if lambda_value is None else lambda_value
        result=set_coupling(sample,lam)
        result.update(step=step,optimizer_step=step,origin_sample_index=origin,condition_policy='40' if self.recipe in ('R1_40_H_ALL','R3_40_H_LATE') else 'MIX_MULTISET',cache_origin=cache_origin)
        return result

    def samples(self,step):
        if type(step) is not int or not 1<=step<=self.total_updates:raise ValueError('Step outside registered stream')
        return [self.sample(record) for record in self.records[8*(step-1):8*step]]

    def next_effective_batch(self):
        samples=self.samples(self.batch_index+1);self.batch_index+=1
        self.last_audit=dict(batch_index=self.batch_index,
            shared_batch_sha256=json_hash([s['shared_input_sha256'] for s in samples]),
            input_batch_sha256=json_hash([s['input_bytehash'] for s in samples]),
            label_batch_sha256=json_hash([s['label_bytehash'] for s in samples]),
            effective_free_coordinates=sum(s['free_coordinate_count'] for s in samples),
            ratio20_samples=sum(s['ratio']==.2 for s in samples),ratio40_samples=sum(s['ratio']==.4 for s in samples),
            lambda_value=samples[0]['lambda_value'],origin_sample_indices=[s['origin_sample_index'] for s in samples])
        return samples


def paired_record(sample):
    return {key:sample[key] for key in PAIR_FIELDS}|dict(lambda_value=sample['lambda_value'],origin_sample_index=sample['origin_sample_index'])


def precompute(data_root,stream_root,workers=4):
    if type(workers) is not int or not 1<=workers<=8:raise ValueError('CPU OT workers must be1..8')
    out=Path(stream_root);stream=RecipeFactorialStream(data_root,out,'R2',require_cache=False)
    for path in stream.cache_dir.glob('*.attempt.json'):
        if not path.with_suffix('').with_suffix('.npz').exists():raise RuntimeError('Unresolved prior20% OT attempt; no silent retry: '+str(path))
    tick=time.perf_counter();new=0;new_hits=0;rows={RECIPES[key]:[] for key in ('R2','R3','R4')}
    def one(index):
        old_record=dict(stream._base_records[index],origin_sample_index=index,optimizer_step=index//8+1)
        old=stream.sample(old_record,lambda_value=.25)
        late=lambda_for_step('R3',index//8+1)
        r3=paired_record(set_coupling(old,late))
        mixed_record=stream.plan['mixed_records'][index]
        mix=old if mixed_record['ratio']==.4 else stream.sample(mixed_record,lambda_value=.25)
        return paired_record(mix),r3,paired_record(set_coupling(mix,late)),mix['ot_cache_status'] if mixed_record['ratio']==.2 else None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index,result in enumerate(pool.map(one,range(len(stream.records)))):
            for label,item in zip(rows,result[:3]):rows[label].append(item)
            new+=result[3]=='MISS';new_hits+=result[3]=='HIT'
            if (index+1)%256==0 or index+1==len(stream.records):
                atomic_json(out/'precompute_progress.json',dict(status='RUNNING',completed_slots=index+1,registered_slots=len(stream.records),actual_new20_OT_returns=new,existing_new20_cache_hits=new_hits,workers=workers,wall_seconds=time.perf_counter()-tick))
    course_rows={}
    for course,order in stream.plan['curriculum_orders'].items():
        for schedule,source in (('all',RECIPES['R2']),('late',RECIPES['R4'])):
            # All first4000 records remain within the first4000 positions. Thus
            # each origin keeps its lambda, while its batch/order alone changes.
            items=[rows[source][index] for index in order]
            if any(row['lambda_value']!=lambda_for_step(course,pos//8+1,schedule) for pos,row in enumerate(items)):
                raise ValueError('Course order crosses a lambda boundary; requires a separate frozen registration')
            course_rows[course+'__H_'+schedule.upper()]=items
    rows.update(course_rows)
    for label,items in rows.items():
        if sorted(r['sample_index'] for r in items)!=list(range(len(stream.records))):raise ValueError('Paired input registration incomplete')
        freeze_json(out/'paired_inputs'/(label+'.json'),dict(schema='recipe_factorial_paired_inputs_v1',recipe=label,
            plan_sha256=stream.plan_sha256,base_paired_inputs_sha256=stream.plan['base_paired_inputs_sha256'],
            data_manifest_sha256=stream.data.manifest_sha256,records=items))
    receipt=dict(status='PASS',namespace=stream.namespace,plan_sha256=stream.plan_sha256,slots=len(stream.records),
                 new20_maps_registered=sum(r['ratio']==.2 for r in stream.plan['mixed_records']),actual_new20_OT_attempts=new,
                 actual_new20_OT_returns=new,new20_cache_hits=new_hits,old40_maps_read=len(stream.records),
                 original_assets_written=0,workers=workers,executor_type='ThreadPoolExecutor',
                 accounting_authority='Main-thread ordered completed-return aggregation, not shared mutable stream counters',
                 paired_tables=list(rows),wall_seconds=time.perf_counter()-tick,model_forwards=0)
    freeze_json(out/'precompute_attempts'/(str(time.time_ns())+'.json'),receipt)
    atomic_json(out/'precompute_receipt.json',receipt)
    atomic_json(out/'precompute_progress.json',dict(receipt,status='COMPLETE',completed_slots=len(stream.records),registered_slots=len(stream.records)))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--previous-run',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--namespace',default=NAMESPACE)
    parser.add_argument('--precompute',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();print(json.dumps(prepare(args.data,args.previous_run,args.out,args.namespace)),flush=True)
    if args.precompute:print(json.dumps(precompute(args.data,args.out,args.workers)),flush=True)


if __name__=='__main__':main()