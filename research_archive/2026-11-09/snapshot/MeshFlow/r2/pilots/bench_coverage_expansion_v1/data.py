"""Expanded, reviewed Bench pool. Same SEAM/FM/OT/HYBRID, including no-owner C.

Only the condition pool and its parent/part exposure stream change. Closed C and
open C without a source owner use the already executed ordinary-bulk special
case. This module does not train an occupancy head or change inference layouts.
"""
from __future__ import annotations
import collections
from pathlib import Path
import numpy as np
from r2.pilots.bench_parts_convergence_v1 import data as old

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_coverage_expansion_v1'
SEED=2026102001
UPDATES=1500
resolve,read,sha,arrays,load_case,write_new=old.resolve,old.read,old.sha,old.arrays,old.load_case,old.write_new

def scaled_case(case,scale_xyz):
    scale=np.asarray(scale_xyz,np.float32)
    if scale.shape!=(3,) or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid scale')
    out={k:v.copy() for k,v in case.items()}
    for k in ('full_target','pre_ot_full','C_vertices'):out[k]*=scale
    return out

def sample_for_training(case,row):
    copy=scaled_case(case,row['scale_xyz']);n,k=int(copy['N']),int(copy['K'])
    z=np.random.RandomState(int(row['noise_seed'])).standard_normal((n-k,3,3)).astype(np.float32)
    eps=np.random.RandomState(int(row['epsilon_seed'])).standard_normal(z.shape).astype(np.float32)
    normal=float(np.random.RandomState(int(row['time_seed'])).standard_normal());t=float(1/(1+np.exp(-normal)))
    return old.supervised_sample(copy,z,eps,t,int(row['permutation_seed']),'SEAM')

def sample_for_probe(case,z,eps,t,permutation_seed):
    return old.supervised_sample(case,z,eps,t,permutation_seed,'SEAM')

def make_schedule(train):
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for index,row in enumerate(train):groups[row['parent_id']][row['part_type']].append(index)
    parents=sorted(groups)
    if len(parents)<64:raise ValueError('Must retain original64 TRAIN parents')
    rng=np.random.RandomState(SEED);aug=np.random.RandomState(SEED^0xABCDEF)
    parts={p:sorted(groups[p]) for p in parents};cursors=collections.Counter();variants=collections.Counter();exposure=collections.Counter()
    queue=[];steps=[]
    for step in range(1,UPDATES+1):
        chosen=[]
        while len(chosen)<4:
            if not queue:queue.extend(map(int,rng.permutation(len(parents))))
            at=next((i for i,p in enumerate(queue) if p not in chosen),None)
            if at is None:queue.extend(map(int,rng.permutation(len(parents))));continue
            chosen.append(queue.pop(at))
        selected=[]
        for pi in chosen:
            parent=parents[pi];part=parts[parent][cursors[parent]%len(parts[parent])];cursors[parent]+=1
            options=groups[parent][part];ci=options[variants[(parent,part)]%len(options)];variants[(parent,part)]+=1;selected.append(ci)
        batch=[]
        for augmentation in range(2):
            for ci in selected:
                src=train[ci];mi=len(batch);seed=10000000+step*1000+mi
                scale=[1.,1.,1.] if augmentation==0 else list(map(float,aug.uniform(.92,1.08,3)))
                row=dict(condition_index=ci,condition_id=src['condition_id'],parent_id=src['parent_id'],part_type=src['part_type'],sample_index=mi,
                    scale_xyz=scale,augmentation='CLEAN' if augmentation==0 else 'ANISOTROPIC',noise_seed=seed,
                    epsilon_seed=seed+100000000,time_seed=seed+200000000,permutation_seed=seed+300000000)
                batch.append(row);exposure[src['condition_id']]+=1
        steps.append(dict(step=step,samples=batch))
    return dict(schema='BENCH_COVERAGE_EXPANSION_PARENT_PART_XYZ_STREAM_V1',seed=SEED,steps=steps,batch=8,
        distinct_parents_per_update=4,record_index_domain='manifest.train',condition_exposure=dict(sorted(exposure.items())),
        policy='Same old parent-shuffled/part-variant-round-robin and xyz U[.92,1.08]; expanded input pool is the intended variable; Gaussian unchanged')
