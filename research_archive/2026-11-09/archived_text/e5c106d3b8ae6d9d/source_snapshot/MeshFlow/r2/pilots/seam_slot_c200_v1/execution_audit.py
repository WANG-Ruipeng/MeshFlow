"""Independent CPU receipt, identity and free-only graph audit.

This module imports no model framework, runs no scorer and changes no experiment
asset. It audits completed artifacts only. Legal/illegal edge classifications
remain those of the frozen scorer; their connected-component aggregation is
recomputed independently with union-find, without traversing C. Tensor state
restoration is verified against saved initialization receipts and file hashes,
not by instantiating or forwarding another generator.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

VERSION = 'seam_c200_independent_execution_audit_v1'
REPOSITORY = Path(__file__).resolve().parents[3]
ARMS = ('FULL4_C200', 'SEAM_SLOT_C200')
INTERNAL = ('retained_seam_faces', 'retained_bulk_faces', 'retained_seam_area',
    'retained_bulk_area', 'legal_CF_seed_seam_faces', 'legal_CF_seed_seam_area',
    'free_only_reached_seam_faces', 'free_only_reached_bulk_faces',
    'free_only_reached_bulk_area', 'free_only_reached_bulk_area_fraction',
    'legal_seam_bulk_edge_count', 'legal_seam_bulk_edge_length',
    'seed_connected_legal_seam_bulk_edge_count',
    'seed_connected_legal_seam_bulk_edge_length', 'bulk_reach_ge_0_30_descriptive',
    'deleted_seam_faces')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    p = str(value)
    if os.name == 'nt' and p.startswith('/mnt/') and len(p) > 7:
        p = p[5].upper() + ':/' + p[7:]
    elif os.name != 'nt' and len(p) > 2 and p[1] == ':':
        p = '/mnt/' + p[0].lower() + '/' + p[3:].replace('\\', '/')
    v = Path(p)
    return v if v.is_absolute() else REPOSITORY / v


def read(p):
    return json.loads(path(p).read_text(encoding='utf-8-sig'))


def lines(p):
    return [json.loads(s) for s in path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def table(p):
    with path(p).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def arrays(p):
    with np.load(path(p), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def boolean(x):
    require(x in (True, False, 'True', 'False', 'true', 'false', '1', '0'), 'Invalid boolean')
    return x in (True, 'True', 'true', '1')


def close(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-10)


class Auditor:
    def __init__(self, root):
        self.root = path(root).resolve()
        require(self.root == REPOSITORY / 'experiments/r2_seam_slot_c200_v1', 'Wrong audit root')
        self.hashes, self.checks = {}, []

    def sha(self, p):
        p = path(p).resolve()
        stat = p.stat(); key = (str(p), stat.st_size, stat.st_mtime_ns)
        if key not in self.hashes:
            h = hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    h.update(block)
            self.hashes[key] = h.hexdigest()
        return self.hashes[key]

    def verify(self, records):
        require(isinstance(records, dict), 'Hash record must be a path/digest mapping')
        for p, digest in records.items():
            require(self.sha(p) == digest, 'File hash mismatch: ' + str(p))

    def receipt(self, p):
        r = read(p)
        require(r.get('status') in ('COMPLETE', 'PASS'), 'Incomplete receipt: ' + str(p))
        self.verify(r.get('files', {}))
        return r

    def check(self, name, fn):
        tick = time.perf_counter()
        try:
            data = fn()
            item = dict(name=name, status='PASS', evidence=data)
        except FileNotFoundError as exc:
            item = dict(name=name, status='NOT_READY', error=str(exc))
        except Exception as exc:
            item = dict(name=name, status='FAIL', error=repr(exc))
        item['seconds'] = time.perf_counter() - tick
        self.checks.append(item)
        return item

    def frozen(self):
        reg, freeze = read(self.root/'registration.json'), read(self.root/'execution_freeze.json')
        require(freeze['registration_sha256'] == self.sha(self.root/'registration.json'), 'Registration changed')
        for k in ('code_hashes', 'input_hashes'):
            require(bool(freeze[k]), 'Empty execution freeze ' + k)
            self.verify(freeze[k])
        self.verify(reg['protected_hashes'])
        self.verify({reg['source_checkpoint']: reg['source_checkpoint_sha256'],
                     reg['d0_checkpoint']['path']: reg['d0_checkpoint']['sha256']})
        manifest = read(self.root/'data_manifest.json')
        entries = [s for step in manifest['steps'] for arm in ARMS for s in step['samples'][arm]]
        require(len(entries) == 3200, 'Expected 3200 cached supervised samples')
        self.verify({s['path']: s['sha256'] for s in entries})
        return dict(frozen_code=len(freeze['code_hashes']), frozen_inputs=len(freeze['input_hashes']),
                    protected=len(reg['protected_hashes']), cached_samples=3200,
                    source_generator_and_D0_file_hashes_verified=True)

    def training(self, arm):
        reg = read(self.root/'registration.json'); folder = self.root/'training'/arm
        status = read(folder/'status.json')
        require((status['status'], status['local_step'], status['global_step']) == ('COMPLETE',200,12200),
                arm + ' has not completed fixed step 200')
        require(status['registration_sha256'] == self.sha(self.root/'registration.json'), 'Training registration mismatch')
        require(status['data_manifest_sha256'] == self.sha(self.root/'data_manifest.json'), 'Training data manifest mismatch')
        init = [read(p) for p in sorted((folder/'attempts').glob('*/initialization.json'))]
        original = [r for r in init if r.get('global_step') == 12000]
        require(len(original) == 1, 'Exactly one original initializer required')
        r = original[0]
        for key, source in [('generator_state_sha256','source_generator_sha256'),
                            ('optimizer_state_sha256','source_optimizer_sha256'),('rng_sha256','source_rng_sha256')]:
            require(r[key] == reg[source], 'Initial identity mismatch ' + key)
        require(r['EMA'] is False and r['exact_generator_optimizer_RNG_restoration'], 'Wrong recovery contract')
        require(r['head_fresh'] == (arm == ARMS[1]), 'Wrong head optimizer initialization')
        require(r['head_parameters'] == (833 if arm == ARMS[1] else 0), 'Wrong head size')
        require(r['separate_head_optimizer'], 'Optimizer ownership is not independent')
        require(all(h['lr']==1e-5 and h['betas']==[.9,.95] and h['weight_decay']==0 for h in r['generator_hyperparameters']), 'Generator optimizer recipe differs')
        if arm==ARMS[1]:
            require(r['occupancy_class_weights']==[8.25,66/124], 'Occupancy class weights differ')
        events = lines(folder/'calls.jsonl')
        begun = [r for r in events if r['event'] == 'CALL_BEGIN']
        done = [r for r in events if r['event'] == 'CALL_DONE']
        require(len(events) == len(begun)+len(done), 'Unresolved/failed scientific call')
        ids = lambda rr: [(r['attempt_id'],r['call_id']) for r in rr]
        require(len(set(ids(begun))) == len(begun) and set(ids(begun)) == set(ids(done)), 'Unpaired or duplicated call')
        by_id = {i:r for i,r in zip(ids(done),done)}
        for i,r in zip(ids(begun),begun):
            d = by_id[i]
            require(d['kind'] == r['kind'] and d['context'] == r['context'] and r['arm'] == d['arm'] == arm, 'Call identity changed')
        counts = dict(collections.Counter(r['kind'] for r in begun))
        expected = dict(train_forward=1600,train_backward=1600,optimizer_update=200)
        if arm == ARMS[1]:
            expected.update(head_forward=1600,head_backward=1600,head_optimizer_update=200,preflight_forward=2)
            gate = read(folder/'all_null_gate.json')
            require(gate['status']=='PASS' and gate['actual_forward_calls']==2 and gate['bitwise_equal'] and gate['max_abs']==0,
                    'Actual all-Null equivalence gate did not pass')
        require(counts == expected, 'Training call budget differs: '+str(counts))
        require([r['context']['global_step'] for r in begun if r['kind']=='optimizer_update'] == list(range(12001,12201)),
                'Generator optimizer cursor differs')
        exposures = lines(folder/'exposure.jsonl'); logs = lines(folder/'training_log.jsonl')
        schedule = read(self.root/'input_schedule.json')['steps']
        wanted = {(s['step'],i):v for s in schedule for i,v in enumerate(s['rows'])}
        got = {(r['local_step'],r['microbatch']):r for r in exposures}
        require(len(exposures)==len(got)==1600 and set(got)==set(wanted), 'Exposure coverage differs')
        for key, r in got.items():
            w = wanted[key]
            require(all(r[k]==w[k] for k in ('uid','sample_index','alpha')) and r['global_step']==12000+key[0], 'Input stream differs')
        parent_counts = dict(collections.Counter(r['uid'] for r in exposures))
        require(parent_counts == {r['parent_id']:400 for r in reg['records']}, 'Unequal parent exposures')
        for parent in parent_counts:
            require(sum(r['uid']==parent and r['alpha']==1 for r in exposures)==200, 'Clean/width exposure split differs')
        require([r['local_step'] for r in logs]==list(range(1,201)), 'Training log update sequence differs')
        for r in logs:
            rows = [got[(r['local_step'],i)] for i in range(8)]
            require(r['free_coordinate_denominator']==sum(x['free_coordinate_count'] for x in rows), 'Actual FM denominator mismatch')
            if arm == ARMS[1]:
                require(r['occupancy_edge_denominator']==132, 'BCE denominator is not full batch 132')
        diagnostics = lines(folder/'geometry_diagnostics.jsonl')
        require(len(diagnostics)==1600, 'Missing same-forward diagnostic records')
        for name in ('checkpoint','inference_export') + (('occupancy_export',) if arm==ARMS[1] else ()):
            rr = status[name]
            require((rr['arm'],rr['local_step'],rr['global_step'])==(arm,200,12200), 'Export is not fixed endpoint')
            require(self.sha(rr['path'])==rr['sha256'], 'Endpoint file hash differs')
        gate = read(folder/'first_formal_backward.json')
        require(gate['status']=='PASS' and gate['no_extra_forward'], 'First formal backward check failed')
        return dict(calls=counts,exposures=len(exposures),per_parent=parent_counts,step=200,
                    initial_generator_sha=original[0]['generator_state_sha256'],
                    initial_AdamW_sha=original[0]['optimizer_state_sha256'],initial_RNG_sha=original[0]['rng_sha256'])

    def paired_stream(self):
        records=[lines(self.root/'training'/arm/'exposure.jsonl') for arm in ARMS]
        require(all(len(r)==1600 for r in records), 'Paired stream is incomplete')
        keys=('local_step','global_step','microbatch','uid','sample_index','alpha','t','N','K')
        require([[tuple(x[k] for k in keys) for x in rows] for rows in records][0]
                ==[tuple(x[k] for k in keys) for x in records[1]], 'Arms did not replay the same parent/geometry/time stream')
        return dict(paired_instances=1600,shared_fields=keys,free_coordinate_denominators='Arm specific by corner mask')

    def inference(self):
        manifest=read(self.root/'eval_manifest.json'); requests=manifest['requests']
        require(len(requests)==16 and len({q['request_id'] for q in requests})==16, 'Wrong paired request count')
        require(len(manifest['conditions'])==4, 'Wrong independent C count')
        totals=collections.Counter(); latest_raw=0.; earliest_D0=math.inf
        for arm in ARMS:
            base=self.root/'evaluation'/arm
            for name in ('raw_complete.json','post_complete.json'):
                require(self.receipt(base/name)['outputs']==16, 'Missing arm output')
            layouts={}
            for rp in sorted((base/'layouts').glob('*/receipt.json')):
                r=self.receipt(rp); layouts[r['condition_id']]=arrays(rp.parent/'layout.npz')
                require(r['source_labels_read'] is False and r['predicted_layout_shared_by_noises']==4, 'C-only layout scope differs')
                require(r['occupancy_forward_calls']==int(arm==ARMS[1]), 'Unexpected occupancy inference call')
                totals['occupancy_inference']+=r['occupancy_forward_calls']
            require(len(layouts)==4, 'Missing or duplicate C layout')
            barriers=list(base.glob('postprocess_attempt_*/all_RAW_barrier.json'))
            require(bool(barriers), 'Missing all-RAW-first barrier')
            for p in barriers:
                b=self.receipt(p); require(b['RAW']==32 and len(b['files'])==32, 'Incomplete RAW barrier')
            for q in requests:
                dest=base/q['request_id']; raw=arrays(dest/'raw/raw.npz'); post=arrays(dest/'post/post.npz')
                for stage in ('raw','post'):
                    r=self.receipt(dest/stage/'receipt.json')
                    require((r['arm'],r['request_id'],r['condition_id'])==(arm,q['request_id'],q['condition_id']), 'Metadata identity differs')
                    require(r.get('source_labels_read',r.get('source_labels_used')) is False, 'Y entered inference')
                raw_r=read(dest/'raw/receipt.json'); post_r=read(dest/'post/receipt.json')
                endpoint=read(self.root/'training'/arm/'status.json')['inference_export']
                require(raw_r['checkpoint']['sha256']==endpoint['sha256'] and raw_r['checkpoint']['generator_state_sha256']==endpoint['generator_state_sha256'], 'Generation used a different endpoint')
                require(raw_r['generator_forwards']==50 and raw_r['D0_forwards']==0, 'RAW budget differs')
                require(post_r['D0_forwards']==post_r['GEO_calls']==post_r['orientation_solves']==1, 'POST budget differs')
                reg=read(self.root/'registration.json')
                require(post_r['official_D0_sha256']==reg['d0_checkpoint']['sha256'], 'Different D0 checkpoint used')
                require(raw_r['noise_sha256']==self.sha(q['noise_npz']) and raw_r['input_bundle_sha256']==self.sha(q['inference_npz']), 'Input/noise identity differs')
                calls=lines(dest/'raw/calls.jsonl')
                for event in ('FORWARD_STARTED','FORWARD_RETURNED'):
                    rr=[x for x in calls if x['event']==event]
                    require([x['euler_step'] for x in rr]==list(range(50)), 'Euler call sequence differs')
                require(len(calls)==100, 'Extra/failed Euler call')
                latest_raw=max(latest_raw,max(x['unix'] for x in calls))
                dd=lines(dest/'post/calls.jsonl')
                require([x['event'] for x in dd]==['D0_FORWARD_STARTED','D0_FORWARD_RETURNED'], 'Extra/failed D0 call')
                earliest_D0=min(earliest_D0,dd[0]['unix'])
                totals['Euler_forward']+=50; totals['D0_forward']+=1; totals['RAW']+=1; totals['POST']+=1
                self.identity(q, raw, post, layouts[q['condition_id']],dest)
        require(latest_raw <= earliest_D0, 'D0 started before all RAW finished')
        require(dict(totals)==dict(occupancy_inference=4,Euler_forward=1600,D0_forward=32,RAW=32,POST=32), 'Total inference scope differs')
        return dict(totals=totals,all_RAW_before_D0=True,independent_C=4,index_identity_and_frozen_51_state_anchors=True)

    def identity(self,q,raw,post,layout,dest):
        inp=arrays(q['inference_npz']); meta=arrays(q['metadata_npz']); k,n=int(q['K']),int(q['N'])
        cv,cf=inp['C_vertices'],inp['C_faces_original']; original=cv[cf]
        require(equal(raw['C_original_source_vertex_ids'],meta['C_source_vertex_ids'][cf]), 'C source vertex identity differs')
        meta_permutation=meta.get('source_C_corner_permutations',meta.get('C_source_corner_permutations'))
        require(meta_permutation is not None and equal(raw['C_source_corner_permutations'],meta_permutation), 'C source corner map differs')
        require(equal(raw['raw_anchor_ids'],layout['anchor_ids']), 'RAW anchor layout differs')
        require(equal(raw['occupied'],layout['occupied']), 'Noise changed occupied layout')
        source=arrays(dest/'raw/source_condition.npz')
        face_ids=meta.get('source_C_face_ids',meta.get('C_original_source_face_ids'))
        require(face_ids is not None and equal(source['C_original_source_face_ids'],face_ids), 'Source C face identities differ')
        traj=arrays(dest/'raw/trajectory.npz')['state']; fixed=raw['raw_anchor_ids']>=0
        require(traj.shape==(51,n,3,3) and equal(traj[-1],raw['raw']), 'Rollout trajectory/end differs')
        expected=cv[raw['raw_anchor_ids'][fixed]]
        require(all(equal(state[fixed],expected) for state in traj), 'C or seam anchors moved during rollout')
        pre=arrays(dest/'post/pre_geo.npz')['output']
        require(equal(pre[fixed],expected), 'D0 anchors were not exactly restored before GEO')
        require(len(raw['output'])==n, 'RAW N changed')
        for mesh in (raw,post):
            require(equal(mesh['output'],mesh['vertices'][mesh['faces']]) and equal(mesh['output'][:k],original), 'Indexed geometry or source C differs')
            ids=np.asarray(mesh['source_face_ids'],int); corners=np.asarray(mesh['source_corner_ids'],int)
            require(len(set(ids.tolist()))==len(ids) and np.array_equal(ids[:k],np.arange(k)) and ((ids>=0)&(ids<n)).all(), 'Invalid RAW slot identity')
            require(corners.shape==(len(ids),3) and np.all(np.sort(corners,axis=1)==np.arange(3)), 'Invalid corner permutation')
            mapping={}
            for c,v in zip(cf.ravel(),mesh['faces'][:k].ravel()):
                c,v=int(c),int(v)
                require(c not in mapping or mapping[c]==v, 'C vertex identity split')
                mapping[c]=v
            require(len(mapping)==len(cv) and len(set(mapping.values()))==len(cv), 'C identities merged or lost')
            transported=raw['raw_anchor_ids'][ids[:,None],corners]
            for face,corner in np.argwhere(transported>=0):
                require(int(mesh['faces'][face,corner])==mapping[int(transported[face,corner])], 'Retained anchor is not shared C vertex index')
            require(np.array_equal(mesh['retained_seam_mask'],raw['raw_seam_slot_mask'][ids]), 'Seam/RAW ID transport differs')
        anchors=read(dest/'post/anchor_identity.json')
        retained=int(post['retained_seam_mask'].sum()); deleted=int(raw['raw_seam_slot_mask'].sum())-retained
        require(anchors['retained_seam_faces']==retained and anchors['deleted_seam_faces']==deleted, 'Deletion accounting mismatch')
        require(len(anchors['deleted_seam_records'])==deleted, 'Missing deleted seam reasons')

    def graphs(self):
        manifest=read(self.root/'eval_manifest.json'); outputs=[]; full_NA=0
        completion=self.receipt(self.root/'evaluation/score_completion.json')
        require(completion['cells']==64 and len(completion['receipts'])==64, 'Missing score cells')
        for row in completion['receipts']:
            require(self.sha(row['path'])==row['sha256'], 'Score completion identity mismatch')
        for arm in ARMS:
            for q in manifest['requests']:
                for stage in ('RAW','POST'):
                    folder=self.root/'evaluation/scores'/arm/q['request_id']/stage
                    score=self.receipt(folder/'score.json'); saved=read(folder/'seam_to_bulk.json')
                    require((score['label'],score['request_id'],score['stage'])==(arm,q['request_id'],stage), 'Score identity mismatch')
                    require(score['model_calls']==0 and score['new_queries']==0, 'Scoring exceeded scope')
                    meshpath=self.root/'evaluation'/arm/q['request_id']/stage.lower()/(stage.lower()+'.npz')
                    require(score['identity']['mesh_sha256']==self.sha(meshpath), 'Score mesh hash differs')
                    if arm==ARMS[0]:
                        require(saved['status']=='NA_NOT_APPLICABLE_FULL_ARM' and all(saved[k] is None for k in INTERNAL), 'FULL internal metrics must be NA')
                        require(table(folder/'seam_bulk_faces.csv')==[], 'FULL internal face rows should be empty')
                        full_NA+=1; continue
                    a=arrays(meshpath); k=int(q['K']); tri=a['output'].astype(np.float64)
                    area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
                    faces=table(folder/'seam_bulk_faces.csv'); edges=table(folder/'valid_edges.csv')
                    require(len(faces)==len(tri)-k, 'Incomplete free face area ledger')
                    parsed=[]
                    for e in edges:
                        e=dict(e); e['valid_edge']=boolean(e['valid_edge'])
                        e['incident_face_ids']=json.loads(e['incident_face_ids']);e['length_model']=float(e['length_model'])
                        parsed.append(e)
                    found,seen,seeds=components(k,area,a['retained_seam_mask'],a['raw_seam_slot_mask'],parsed)
                    require(set(found)==set(saved), 'Internal summary schema differs')
                    for name in found:
                        require(close(found[name],saved[name]), 'Independent graph differs '+str((q['request_id'],stage,name,found[name],saved[name])))
                        require(close(found[name],score['seam_to_bulk'][name]), 'Score embeds stale graph')
                    by_id={int(r['output_face_id']):r for r in faces}
                    require(set(by_id)==set(range(k,len(tri))), 'Free face ledger identities differ')
                    for i,r in by_id.items():
                        require(int(r['raw_slot_id'])==int(a['raw_slot_ids'][i]), 'Face area RAW identity differs')
                        require(r['role']==('SEAM' if a['retained_seam_mask'][i] else 'BULK'), 'Face area role differs')
                        require(close(float(r['area_model2']),area[i]), 'Saved face area differs')
                        require(boolean(r['reached_free_only'])==(i in seen) and boolean(r['legal_CF_seed'])==(i in seeds), 'Saved face reach flags differ')
                    outputs.append(dict(arm=arm,request_id=q['request_id'],stage=stage,**found))
        return dict(candidate_cells=len(outputs),FULL_NA_cells=full_NA,rows=outputs,
                    algorithm='Independent union-find over saved legal F/F edges; C excluded',
                    inherited_evidence='Saved legal edge classifications; no new intersection or quality scans')

    def occupancy(self):
        m=read(self.root/'eval_manifest.json'); summary=read(self.root/'evaluation/occupancy_summary.json')
        require(summary['independent_C']==4, 'Occupancy counted repeated noises as independent C')
        counts={a:collections.Counter() for a in (ARMS[1],'ALL_POSITIVE_REFERENCE')}
        predicted_edges=[]
        for c in m['conditions']:
            truth=arrays(c['labels_npz'])['occupied_labels'].astype(bool)
            qs=[q for q in m['requests'] if q['condition_id']==c['condition_id']]
            require(len(qs)==4, 'Wrong noises per C')
            aa=[arrays(self.root/'evaluation'/ARMS[1]/q['request_id']/'raw/raw.npz') for q in qs]
            logits=aa[0]['occupancy_logits'];pred=aa[0]['occupied'].astype(bool)
            require(np.isfinite(logits).all() and np.array_equal(pred,logits>=0), 'Occupancy threshold differs')
            require(all(equal(a['occupied'],aa[0]['occupied']) and equal(a['occupancy_logits'],logits) for a in aa), 'C prediction differs across noises')
            for label,p in ((ARMS[1],pred),('ALL_POSITIVE_REFERENCE',np.ones_like(truth))):
                counts[label].update(dict(edge_count=len(truth),positive_labels=int(truth.sum()),Null_labels=int((~truth).sum()),
                    TP=int((truth&p).sum()),TN=int((~truth&~p).sum()),FP=int((~truth&p).sum()),FN=int((truth&~p).sum()),
                    exact_layout=int(np.array_equal(truth,p)),conditions=1))
            predicted_edges.extend((c['condition_id'],i,bool(t),bool(p),float(l)) for i,(t,p,l) in enumerate(zip(truth,pred,logits)))
        for row in summary['rows']:
            c=counts[row['label']]
            require(all(row[k]==v for k,v in c.items()), 'Occupancy aggregate counts differ')
            require(close(row['accuracy'],(c['TP']+c['TN'])/c['edge_count']) and close(row['positive_recall'],c['TP']/c['positive_labels'])
                    and close(row['Null_recall'],c['TN']/c['Null_labels']), 'Occupancy aggregate rate differs')
        baseline=counts['ALL_POSITIVE_REFERENCE']
        require((baseline['edge_count'],baseline['positive_labels'],baseline['Null_labels'],baseline['exact_layout'])==(66,62,4,2), 'Source occupancy domain differs')
        saved=table(self.root/'evaluation/occupancy_edges.csv')
        got={(r['condition_id'],int(r['edge_id'])):(boolean(r['source_occupied']),boolean(r['predicted_occupied']),float(r['logit'])) for r in saved}
        require(len(saved)==len(got)==66, 'Occupancy edge table duplicates or omissions')
        for c,i,t,p,l in predicted_edges:
            require(got[(c,i)]==(t,p,l), 'Occupancy edge evidence differs')
        return dict(independent_C=4,edges=66,noise_duplicates_excluded=12,counts=counts)


def components(k,area,seam_mask,raw_seam_mask,edges):
    """Independent aggregator: union-find cannot create a path through C."""
    n=len(area);seam={int(i) for i in np.flatnonzero(seam_mask)};bulk=set(range(k,n))-seam
    require(not any(i<k for i in seam), 'C cannot be a seam seed')
    parent={i:i for i in range(k,n)}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    seeds=set();cross=[]
    for e in edges:
        if not e['valid_edge']:continue
        require(len(e['incident_face_ids'])==2, 'Legal edge must have two incident faces')
        i,j=map(int,e['incident_face_ids'])
        require(0<=i<n and 0<=j<n and i!=j, 'Invalid legal edge faces')
        if i>=k and j>=k:
            parent[find(i)]=find(j)
            if (i in seam)!=(j in seam):cross.append((i,j,float(e['length_model'])))
        elif min(i,j)<k<=max(i,j) and max(i,j) in seam:
            seeds.add(max(i,j))
    roots={find(i) for i in seeds};seen={i for i in range(k,n) if find(i) in roots}
    reached=seen&bulk;den=float(sum(area[i] for i in sorted(bulk)));num=float(sum(area[i] for i in sorted(reached)))
    fraction=num/den if den>0 else None;connected=[e for e in cross if e[0] in seen and e[1] in seen]
    summary=dict(status='PASS' if den>0 else 'NA_EMPTY_OR_ZERO_AREA_BULK',retained_seam_faces=len(seam),retained_bulk_faces=len(bulk),
        retained_seam_area=float(sum(area[i] for i in sorted(seam))),retained_bulk_area=den,
        legal_CF_seed_seam_faces=len(seeds),legal_CF_seed_seam_area=float(sum(area[i] for i in sorted(seeds))),
        free_only_reached_seam_faces=len(seen&seam),free_only_reached_bulk_faces=len(reached),free_only_reached_bulk_area=num,
        free_only_reached_bulk_area_fraction=fraction,legal_seam_bulk_edge_count=len(cross),legal_seam_bulk_edge_length=sum(e[2] for e in cross),
        seed_connected_legal_seam_bulk_edge_count=len(connected),seed_connected_legal_seam_bulk_edge_length=sum(e[2] for e in connected),
        bulk_reach_ge_0_30_descriptive=None if fraction is None else int(fraction>=.30),deleted_seam_faces=int(np.asarray(raw_seam_mask,bool).sum())-len(seam))
    return summary,seen,seeds


def selftest():
    edge=lambda i,j,valid=True:dict(incident_face_ids=[i,j],valid_edge=valid,length_model=1.)
    area=np.asarray([1.,1.,2.,3.]);seam=np.asarray([False,True,False,False])
    result,_,_=components(1,area,seam,seam,[edge(0,1),edge(0,2),edge(2,3)])
    require(result['free_only_reached_bulk_area']==0, 'Audit accidentally traverses C')
    result,_,_=components(1,area,seam,seam,[edge(0,1),edge(1,2),edge(2,3)])
    require(result['free_only_reached_bulk_area_fraction']==1 and result['retained_bulk_area']==5, 'Bulk denominator or free reach differs')
    result,_,_=components(1,area,seam,seam,[edge(0,1),edge(1,2,False),edge(2,3)])
    require(result['free_only_reached_bulk_area']==0, 'Illegal FF edge leaked')
    result,_,_=components(1,area[:2],seam[:2],seam[:2],[edge(0,1)])
    require(result['free_only_reached_bulk_area_fraction'] is None, 'Empty bulk is not NA')
    require('torch' not in sys.modules, 'Model framework unexpectedly imported')
    return dict(status='PASS',synthetic_graph_cases=4,generator_calls=0,D0_calls=0,CUDA_calls=0,model_framework_imported=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(REPOSITORY/'experiments/r2_seam_slot_c200_v1'))
    parser.add_argument('--selftest',action='store_true')
    args=parser.parse_args()
    if args.selftest:
        print(json.dumps(selftest()));return 0
    a=Auditor(args.root);started=time.perf_counter();utc=dt.datetime.now(dt.timezone.utc)
    a.check('frozen_and_protected_hashes',a.frozen)
    for arm in ARMS:a.check('training_'+arm,lambda arm=arm:a.training(arm))
    a.check('paired_parent_geometry_time_stream',a.paired_stream)
    a.check('inference_scope_and_identity',a.inference)
    a.check('independent_free_only_graphs',a.graphs)
    a.check('four_condition_occupancy',a.occupancy)
    statuses={r['status'] for r in a.checks}
    status='FAIL' if 'FAIL' in statuses else 'NOT_READY' if 'NOT_READY' in statuses else 'PASS'
    require('torch' not in sys.modules,'Audit imported a model framework')
    actual=collections.Counter()
    for row in a.checks:
        if row['status']=='PASS' and row['name'].startswith('training_'):
            actual.update(row['evidence']['calls'])
        elif row['status']=='PASS' and row['name']=='inference_scope_and_identity':
            actual.update(row['evidence']['totals'])
    report=dict(version=VERSION,status=status,utc=utc.isoformat(),audit_source_sha256=a.sha(__file__),checks=a.checks,
        actual_completed_call_totals=actual,
        seconds=time.perf_counter()-started,hashed_files=len(a.hashes),
        call_scope=dict(generator_calls=0,D0_calls=0,CUDA_calls=0,new_quality_scans=0,new_sampling=0),
        boundary='Computational/identity audit only. Saved legal edge classifications inherited; semantic use and visual quality require separate actual image review.')
    out=a.root/'audits'/('execution_final_'+utc.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=status,path=str(out),checks=[{k:r[k] for k in ('name','status')} for r in a.checks])))
    return 0 if status=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
