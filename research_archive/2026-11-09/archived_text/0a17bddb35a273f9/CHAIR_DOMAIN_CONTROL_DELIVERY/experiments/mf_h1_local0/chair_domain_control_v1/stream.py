"""Frozen variable-N/K, affine-target stream with original free-only OT arithmetic."""
from pathlib import Path
from collections import Counter
import hashlib
import numpy as np
from experiments.mf_h1_local0.control_mvp48.train_binding import data_api
from experiments.mf_h1_local0.control_mvp48.train_stream import FrozenStream, sample_hash
from utils.ot_utils import optimal_sum_numpy
from .common import read_json, digest
from .prepare import load_task


class ChairStream(FrozenStream):
    def __init__(self, source, out, arm):
        self.source,self.out=Path(source),Path(out)
        binding=read_json(self.source/'registration/train_data_binding.json')
        self.api=data_api(binding['loader_path'])
        self.plan_path=self.out/'train_plan.json'
        plan=read_json(self.plan_path);manifest=read_json(self.out/'task_manifest.json')
        self.records=plan['records'];self.plan_sha256=digest(self.plan_path)
        self.task_manifest_sha256=digest(self.out/'task_manifest.json')
        self.tasks={t['task_id']:t for t in manifest['tasks'] if t['role']=='train' and t['status']=='READY'}
        self.offset,self.end,self.batch_index=0,3000,0
        self.seed=plan['seed'];self.counts=dict(actual_OT_attempts=0,actual_OT_returns=0,OT_cache_hits=0)
        if len(self.records)!=24000:raise ValueError('Require exactly 3000 frozen batches of eight')
        allowed=set(binding['train_object_ids']);parents=Counter()
        for i,r in enumerate(self.records):
            task=self.tasks.get(r['task_id'])
            if task is None or r['uid'] not in allowed or task['uid']!=r['uid']:
                raise ValueError('Unregistered training parent/task')
            if (r['N'],r['K']) != (task['N'],task['K']):raise ValueError('Plan N/K differ')
            if (r['sample_index'],r['effective_batch_index'],r['slot_index']) != (i,i//8,i%8):
                raise ValueError('Frozen sample order differs')
            if r['phase_step']!=i//8+1 or r['conditional_total_step']!=1001+i//8:
                raise ValueError('Phase/total cursor differs')
            if not .9-1e-7 <= r['alpha'] <= 1.1+1e-7:raise ValueError('Alpha outside fixed range')
            if any(type(r[k]) is not int for k in ('gaussian_seed','t_seed','permutation_seed')):
                raise ValueError('Frozen per-sample seed missing')
            parents[r['uid']]+=1
        if set(parents)!=allowed or set(parents.values())!={750}:
            raise ValueError('Training parent exposure differs from 32 x 750')
        for i in range(0,len(self.records),8):
            if sum(r['alpha']==1.0 for r in self.records[i:i+8]) != 4:
                raise ValueError('Every batch must contain exactly four alpha=1 samples')

    def state_dict(self):
        return dict(seed=self.seed,batch_index=self.batch_index,sample_index=8*self.batch_index,
            RNG_mode='stateless_per_sample_frozen_seeds',plan_sha256=self.plan_sha256,
            task_manifest_sha256=self.task_manifest_sha256,offset=0,end=3000)

    def load_state_dict(self,state):
        expected=self.state_dict()
        for k in ('seed','RNG_mode','plan_sha256','task_manifest_sha256','offset','end'):
            if state.get(k)!=expected[k]:raise ValueError('Frozen stream identity differs: '+k)
        b=state['batch_index']
        if type(b) is not int or not 0<=b<=3000 or state['sample_index']!=8*b:
            raise ValueError('Invalid frozen stream cursor')
        self.batch_index=b

    def _sample(self,record):
        case=load_task(self.source,self.out,record['task_id'],alpha=record['alpha'])
        uid=record['uid'];known_ids=np.asarray(case['source_face_ids'],np.int64)
        n,k=case['N'],case['K']
        free_ids=np.asarray(case['free_source_ids'],np.int64)
        if set(known_ids)&set(free_ids) or sorted(np.concatenate((known_ids,free_ids)))!=list(range(n)):
            raise ValueError('Known/free source faces do not partition the full target')
        free_order,corners,order=self.api.permutations(record['permutation_seed'],n,k)
        pre=case['pre_ot_full'][free_ids[free_order]][np.arange(n-k)[:,None],corners[k:]]
        noise=self.api.gaussian(record['gaussian_seed'],n-k)
        self.counts['actual_OT_attempts']+=1
        paired=optimal_sum_numpy(pre,noise,optimal=True)
        self.counts['actual_OT_returns']+=1
        x0=np.zeros((n,3,3),np.float32);x1=np.zeros_like(x0)
        x0[k:]=paired.astype(np.float32);x1[k:]=(pre*2).astype(np.float32)
        context=np.zeros_like(x0)
        context[:k]=case['full_target'].reshape(n,3,3)[known_ids][np.arange(k)[:,None],corners[:k]]
        ids=np.concatenate((known_ids,free_ids[free_order]));known=np.arange(n)<k
        x0,x1,context=[a[order].reshape(n,9) for a in (x0,x1,context)]
        known,ids,corners=known[order],ids[order],corners[order]
        t=np.asarray(self.api.training_time(record['t_seed']),np.float32)
        u=x1-x0;xt=(np.float32(1)-t)*x0+t*x1;xt[known]=context[known]
        sample=dict(x0=x0,x1=x1,u=u,xt=xt,context=context,t=t,y=np.asarray(n,np.int64),
            known_mask=known,valid_mask=np.ones(n,bool),source_face_ids=ids,corner_permutations=corners,
            face_permutation=order,object_id=uid,patch_id=record['task_id'],task_id=record['task_id'],K=k,
            alpha=float(record['alpha']),ratio=float(self.tasks[record['task_id']]['ratio']),
            free_gaussian_before_OT=noise,free_noise_after_OT=paired,free_target_before_OT=pre,
            _geometry=dict(V=case['model_vertices'],F=case['source_face_vertex_ids'],bbox_L=case['bbox_diagonal']))
        sample['shared_input_sha256']=hashlib.sha256((sample_hash(sample)+repr(sample['alpha'])).encode()).hexdigest()
        self.label(sample)
        return sample

    def label(self,sample):
        # Reuse the original exact source/corner reconstruction and label contract.
        old=getattr(self,'objects',None)
        self.objects={sample['object_id']:sample['_geometry']}
        try:return super().label(sample)
        finally:self.objects=old

    def next_effective_batch(self):
        if not 0<=self.batch_index<3000:raise RuntimeError('Fixed 3000-update stream exhausted')
        i=self.batch_index*8
        samples=[self._sample(r) for r in self.records[i:i+8]]
        self.batch_index+=1
        return samples
