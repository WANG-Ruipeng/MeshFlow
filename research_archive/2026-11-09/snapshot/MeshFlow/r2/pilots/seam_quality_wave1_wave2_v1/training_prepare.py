"""Freeze the first 8000 prior OT examples plus static source-face quality.

CPU only, and only after Stage 0. Never generates data, rematches, scores geometry,
constructs a neural network or reads DEV geometry/labels.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import json
import time
import os
import subprocess
import numpy as np
from . import training_data as D

OLD = D.ROOT / 'experiments/r2_seam_pool47_couple_v1'
PAIRS = D.ROOT / 'experiments/r2_target_cleanup_v1/inherited_FF_pair_survival.csv'


def prepare(root):
    root = D.resolve(root).resolve()
    D.require_cleanup(root)
    out = root / 'wave2_inputs'
    if (root / 'training_registration.json').exists():
        reg = D.read(root / 'training_registration.json')
        for path, sha in reg['protected_hashes'].items():
            if D.digest(path) != sha:
                raise ValueError('Registered preparation changed: ' + path)
        return reg
    if out.exists():
        raise FileExistsError('Partial preparation exists; explicit engineering review required')
    out.mkdir(parents=True)
    import psutil
    commands={}
    for name,args in (
        ('gpu',['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
        ('processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])):
        result=subprocess.run(args,capture_output=True,text=True)
        commands[name]=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    D.write_new(out/'resources_before_cpu.json',dict(cpu_cores=os.cpu_count(),available_ram_bytes=psutil.virtual_memory().available,
        cpu_percent=psutil.cpu_percent(interval=.1),commands=commands,CPU_workers=1,threads=1,GPU_workers=0,
        scope='Existing source-label transport only; no new geometry scoring, OT or model'))
    D.write_new(out / 'started.json', dict(status='STARTED_CPU_ONLY', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        neural_forwards=0, backwards=0, optimizer_updates=0, new_OT_calls=0))
    prior = D.read(OLD / 'training_registration.json')
    manifest = D.read(OLD / 'shared_data_manifest.json')
    if manifest['status'] != 'COMPLETE' or len(manifest['steps']) != 3000:
        raise ValueError('Original complete 3000-step stream required')
    records = prior['records']
    if len(records) != 47 or any(r['split'] != 'TRAIN' for r in records):
        raise ValueError('Exactly the original TRAIN47 required, without DEV')
    byid = {r['parent_id']: r for r in records}
    if len(byid) != 47:
        raise ValueError('Duplicate parent')
    protected = {str(p): D.digest(p) for p in (OLD/'training_registration.json', OLD/'shared_data_manifest.json', PAIRS)}
    pairs = defaultdict(list)
    risks = Counter()
    with PAIRS.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['parent_id'] not in byid:
                continue
            if row['risk'] not in ('ILLEGAL', 'UNKNOWN'):
                raise ValueError('Unrecognized inherited source risk')
            pair = tuple(map(int, json.loads(row['source_pair'])))
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError('Invalid source pair')
            pairs[row['parent_id']].append(dict(source_pair=list(pair), risk=row['risk'],
                classification=row['classification'], category=row['category']))
            risks[row['risk']] += 1
    if risks != Counter(ILLEGAL=1138, UNKNOWN=3):
        raise ValueError('Source pair totals changed: ' + str(risks))
    labels, face_rows = {}, []
    for record in records:
        uid = record['parent_id']
        for field, hashfield in (('parent_npz','parent_sha256'), ('condition_path','condition_sha256')):
            path = D.resolve(record[field])
            if D.digest(path) != record[hashfield]:
                raise ValueError('Original source geometry changed')
            protected[str(path)] = record[hashfield]
        paths = [D.resolve(record[key]) for key in ('labels_npz','metadata_npz')]
        for path in paths:
            protected[str(path)] = D.digest(path)
        with np.load(D.resolve(record['condition_path']), allow_pickle=False) as z:
            known, free = set(map(int,z['source_face_ids'])), set(map(int,z['free_source_ids']))
        with np.load(D.resolve(record['labels_npz']), allow_pickle=False) as z:
            owner = set(map(int,z['edge_owner_face'][z['occupied_labels'].astype(bool)]))
        n = record['N']
        if known & free or known | free != set(range(n)) or not owner <= free:
            raise ValueError('Source C/free/owner identity invalid')
        dirty = set()
        unique = set()
        for row in pairs[uid]:
            pair = tuple(sorted(row['source_pair']))
            if pair in unique or not set(pair) <= free:
                raise ValueError('Repeated or non-free source pair')
            unique.add(pair)
            dirty.update(pair)
        label = dict(parent_id=uid, condition_id=record['condition_id'], N=n, K=len(known),
            known_source_ids=sorted(known), free_source_ids=sorted(free), owner_source_ids=sorted(owner),
            corrupted_source_ids=sorted(dirty), dirty_bulk_source_ids=sorted(dirty-owner),
            clean_bulk_source_ids=sorted(free-owner-dirty), corrupted_owner_exempt_ids=sorted(dirty&owner),
            source_risk_pairs=pairs[uid], source_ids_scope='Original unmodified source face row, transported through cached OT and slot permutation')
        labels[uid] = label
        for fi in range(n):
            category = 'C' if fi in known else 'SEAM_OWNER' if fi in owner else 'DIRTY_BULK' if fi in dirty else 'CLEAN_BULK'
            face_rows.append(dict(parent_id=uid,source_face_id=fi,category=category,source_corrupt=fi in dirty,
                                  free_corner_weight=0 if fi in known else .2 if category=='DIRTY_BULK' else 1))
    if sum(len(x['owner_source_ids']) for x in labels.values()) != 360:
        raise ValueError('Original unique seam owner count changed')
    D.write_new(out/'source_labels.json', dict(status='FROZEN', schema='source_quality_labels_v1',
        source_pairs_path=str(PAIRS),source_pairs_sha256=protected[str(PAIRS)],
        labels_by_parent=labels, pair_counts=dict(risks), parents_with_illegal=sum(any(r['risk']=='ILLEGAL' for r in pairs[uid]) for uid in labels),
        surviving_cleanup_column_ignored=True, labels_do_not_enter_forward=True, DEV_labels_read=False))
    with (out/'source_face_labels.csv').open('x',encoding='utf-8',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(face_rows[0]));writer.writeheader();writer.writerows(face_rows)
    chosen = manifest['steps'][:D.UPDATES]
    if [s['step'] for s in chosen] != list(range(1,D.UPDATES+1)) or any(len(s['samples']) != 8 for s in chosen):
        raise ValueError('Original first1000x8 required')
    exposures = Counter(); checks = []; weighted = []; owners_seen=set()
    stream_digest=__import__('hashlib').sha256()
    for row in chosen:
        samples=[]
        for micro, record in enumerate(row['samples']):
            sample=D.load_sample(record,root,D.ARMS[0],row['step'],micro,labels)
            # Independent source target/OT face and corner correspondence check.
            source=D.source_arrays(record['parent_npz'],record['parent_sha256'])
            full=source['full_target'].reshape(sample['N'],3,3).copy()
            full[:,:,0]*=np.float32(sample['alpha'])
            target=full[sample['source_face_ids']]
            permutations=sample['source_corner_permutations']
            target=np.take_along_axis(target,np.asarray(permutations)[...,None],axis=1)
            if not np.array_equal(target,sample['x1']):
                raise ValueError('Source labels no longer track target OT/corner identity')
            samples.append(sample);exposures[sample['uid']]+=1;owners_seen.add(sample['uid'])
            stream_digest.update(json.dumps(record,sort_keys=True,separators=(',',':')).encode())
        plain,weight=D.batch_denominators(samples);weighted.append(plain/weight)
        checks.append(dict(local_step=row['step'],plain_scalar_denominator=plain,weighted_scalar_denominator=weight,
                           effective_unit_seam_coefficient_ratio_vs_plain=plain/weight))
        if row['step']%100==0:print(json.dumps(dict(stage='STATIC_LABEL_TRANSPORT',step=row['step'],models=0)),flush=True)
    if sum(exposures.values()) != 8000 or len(owners_seen) != 47:
        raise ValueError('Training exposure coverage changed')
    D.write_new(out/'shared_data_manifest.json',dict(schema='quality_shared_existing_8000_v1',status='COMPLETE',
        source_manifest=str(OLD/'shared_data_manifest.json'),source_manifest_sha256=protected[str(OLD/'shared_data_manifest.json')],
        steps=chosen,samples=8000,new_OT_calls=0,original_examples_exact=True,record_sequence_sha256=stream_digest.hexdigest(),
        parent_exposures=dict(sorted(exposures.items()))))
    D.write_new(out/'transport_audit.json',dict(status='PASS',samples=8000,parents=47,unique_owners=360,
        pair_counts=dict(risks),neural_forwards=0,backwards=0,new_OT_calls=0,
        source_face_bijection=True,source_corner_transport_exact=True,C_and_anchor_weight_zero=True,owner_third_weight_one=True,
        denominator_scope='sum of weighted free scalar coordinates over all eight microbatches',
        unit_seam_coefficient_ratio_range=[min(weighted),max(weighted)],
        relative_seam_warning='Unit seam weight does not keep its absolute normalized gradient unchanged; denominator shrinks.',
        PLAIN_clean_dirty_diagnostics='NA_NOT_RECORDED_IN_EXISTING_PLAIN_FORWARD_LOGS',batch_denominators=checks))
    for path in (out/'source_labels.json',out/'shared_data_manifest.json',out/'transport_audit.json'):
        protected[str(path)]=D.digest(path)
    code=[Path(__file__),Path(D.__file__),Path(__file__).with_name('training.py')]
    for path in code:protected[str(path)]=D.digest(path)
    reg={k:v for k,v in prior.items() if k.startswith('source_') and k not in ('source_records',)}
    reg.update(schema='r2_seam_quality_training_registration_v1',status='FROZEN',arms=list(D.ARMS),
        source_global_step=13000,final_global_step=14000,source_seam_step=1000,final_seam_step=2000,
        updates_per_arm=1000,batch=8,microbatch=1,records=records,
        shared_data_manifest=str(out/'shared_data_manifest.json'),source_labels=str(out/'source_labels.json'),
        protected_hashes=protected,diagnostic_time_bin_edges=[0,.2,.4,.6,.8,1],
        dirty_bulk_weight=.2,owner_third_weight=1,weighted_denominator='WHOLE_EFFECTIVE_BATCH_WEIGHTED_FREE_SCALARS',
        frozen_layer_prefix='backbone.layers.',frozen_layer_indices=list(range(8)),
        optimizer_restoration='EXACT_SOURCE_PER_PARAMETER_MOMENTS_STEP_13000; SURGICAL removes frozen parameters only',
        optimizer=dict(lr=1e-5,betas=[.9,.95],eps=1e-8,weight_decay=0,clip=1),
        head_policy='FROZEN_NOT_LOADED_OR_CALLED',schedule='EXACT_FIRST_1000_STEPS_OF_EXISTING_POOL47_CONTINUATION',
        code_hashes={str(p):D.digest(p) for p in code},no_automatic_retry=True)
    D.write_new(root/'training_registration.json',reg)
    return reg


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='experiments/r2_seam_quality_wave1_wave2_v1')
    args=parser.parse_args()
    r=prepare(args.root)
    print(json.dumps(dict(status=r['status'],arms=r['arms'],updates=r['updates_per_arm']),indent=2))
