"""Freeze the authorized 47 SEAT + 31 BACK inputs and 1500-update stream.

CPU only. Reuses cached source geometry classifications; does not rescore meshes,
repair targets, train a head, or invoke any generator/denoiser.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import collections
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
from . import data as D

ROOT, OUT = D.ROOT, D.OUT
TRAIN = ROOT / 'experiments/r2_user_data_scaling_32_67_v1/data/train_manifest.json'
SEAT = ROOT / 'experiments/r2_seam_abc_v1/B_DATA_REGISTRATION.json'
ELIGIBLE = ROOT / 'experiments/r2_chair_back_transfer_v1/eligibility.json'
BACK_EVAL = ROOT / 'experiments/r2_chair_back_transfer_v1/eval_manifest.json'


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **value)


def csvwrite(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def cpu_checks(records, schedule):
    import torch
    from . import head as H
    torch.set_num_threads(1)
    checks = []; ot_calls = 0
    for i, record in enumerate(records):
        case = D.load_case(record)
        fixture = dict(condition_id=record['condition_id'], part_type=record['part_type'], alpha=1.,
                       noise_seed=710000+i, epsilon_seed=810000+i, time_seed=910000+i, permutation_seed=1010000+i)
        sample = D.sample_for_training(case, fixture); ot_calls += 1
        n, k = int(case['N']), int(case['K']); active = int(case['occupied_labels'].sum())
        assert int(sample['free_coordinate_count']) == 9 * (n-k) - 6 * active
        assert np.array_equal(np.sort(sample['source_face_ids']), np.arange(n))
        cp = sample['source_corner_permutations']; ids = sample['source_face_ids']
        assert np.array_equal(sample['x1'], case['full_target'][ids[:, None], cp])
        anchored = sample['anchor_ids'] >= 0
        assert np.all(sample['u'][anchored] == 0)
        assert np.array_equal(sample['xt'][anchored], case['C_vertices'][sample['anchor_ids'][anchored]])
        assert int(sample['free_corner_mask'].sum()) * 3 == int(sample['free_coordinate_count'])
        assert sample['head_features'].shape == (len(case['occupied_labels']), 28)
        assert np.isfinite(sample['head_features']).all()
        checks.append(dict(condition_id=record['condition_id'], N=n, K=k, active=active,
                           free_coordinates=int(sample['free_coordinate_count']), source_identity='PASS', anchors='PASS'))
    # Representative positive-edge data output must literally equal the old owner extension.
    from r2.pilots.seam_abc_v1.slot_data_extension import supervised_sample
    case = D.load_case(records[0]); row = schedule['steps'][0]['samples'][0]
    ci = row['condition_index']; case = D.load_case(records[ci])
    a = D.sample_for_training(case, row); ot_calls += 1
    c = {k: np.asarray(v).copy() for k, v in case.items()}
    for key in ('full_target', 'pre_ot_full', 'C_vertices'): c[key][..., 0] *= np.float32(row['alpha'])
    count = int(c['N']) - int(c['K'])
    z = np.random.RandomState(row['noise_seed']).randn(count,3,3).astype(np.float32)
    ep = np.random.RandomState(row['epsilon_seed']).randn(count,3,3).astype(np.float32)
    t = float(1/(1+np.exp(-np.random.RandomState(row['time_seed']).randn())))
    b = supervised_sample(c,z,ep,t,row['permutation_seed']); ot_calls += 1
    for key in b:
        assert np.array_equal(a[key],b[key]), key
    before = torch.random.get_rng_state().clone(); head = H.new_head(2026101909)
    assert torch.equal(before, torch.random.get_rng_state())
    head.eval(); x = torch.from_numpy(a['head_features']); permutation = torch.arange(len(x)-1,-1,-1)
    with torch.no_grad():
        y = head(x); yp = head(x[permutation]); head_forwards = 2
        assert torch.allclose(yp, y[permutation], atol=2e-6, rtol=2e-6)
        padded = torch.zeros(1,len(x)+3,28); padded[0,:len(x)] = x
        mask = torch.zeros(1,len(x)+3,dtype=torch.bool); mask[0,:len(x)] = True
        ym = head(padded,mask); head_forwards += 1
        assert torch.allclose(ym[0,:len(x)],y,atol=2e-6,rtol=2e-6)
    # One finite gradient fixture, zero optimizer updates. Discard the fixture head.
    head.train(); output = head(x); head_forwards += 1
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output,torch.from_numpy(a['occupied_labels']).float(),reduction='sum')/len(x)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    return dict(status='PASS', packing_checks=checks, OT_calls=ot_calls,
                head_forwards=head_forwards, head_backwards=1, head_optimizer_updates=0,
                generator_forwards=0, denoiser_forwards=0, GPU_jobs=0,
                head_parameter_count=sum(p.numel() for p in head.parameters()),
                prior_owner_sample_bitwise_identical=True, head_permutation_equivariance=True,
                head_padding_invariance=True, isolated_head_initialization_preserves_CPU_RNG=True)


def run():
    if (OUT/'completion.json').exists():
        result = D.read(OUT/'completion.json')
        for p,h in result['files'].items(): assert D.sha(p)==h,p
        return dict(status='VERIFIED_EXISTING', summary=D.read(OUT/'manifest.json')['summary'])
    if (OUT/'attempt.json').exists():
        raise RuntimeError('Prior preparation attempt incomplete; explicit engineering review required')
    OUT.mkdir(parents=True, exist_ok=True)
    started=time.perf_counter()
    from r2.pilots.bench_parts_convergence_v1.data import resources
    write(OUT/'resources.json',resources())
    write(OUT/'attempt.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),CPU_only=True,
                                 planned_packing_OT_calls=80,planned_head_forwards=4,planned_head_backwards=1,head_updates=0))
    train, seat, eligibility, beval = map(D.read,(TRAIN,SEAT,ELIGIBLE,BACK_EVAL))
    protected={str(p):D.sha(p) for p in (TRAIN,SEAT,ELIGIBLE,BACK_EVAL)}
    eligible={r['condition_id'] for r in eligibility['rows'] if r['group']=='FAMILIAR' and r['status']=='ELIGIBLE'}
    seats=sorted(seat['records'],key=lambda r:r['condition_id'])
    backs=sorted((r for r in train['tasks'].values() if r['condition_id'] in eligible),key=lambda r:r['condition_id'])
    assert len(seats)==47 and len(backs)==31
    train_parents={r['parent_id'] for r in seats}
    assert {r['parent_id'] for r in backs} <= train_parents
    eval_dev={r['parent_id'] for r in seat['DEV_records']} | {r['parent_id'] for r in beval['conditions'] if r['split']=='DEV'}
    assert not train_parents & eval_dev
    records=[]; audits=[]; case_checks=[]
    for part, selected in [('SEAT',seats),('BACK',backs)]:
        for r in selected:
            cid,pid=r['condition_id'],r['parent_id']
            task=train['tasks'][cid]
            pn=r['parent_npz'] if part=='SEAT' else train['parents'][pid]['npz']
            cn=r['condition_path'] if part=='SEAT' else r['npz']
            qn=task['quality_path']
            for path,expected in ((pn,r['parent_sha256'] if part=='SEAT' else train['parents'][pid]['sha256']),
                                  (cn,r['condition_sha256'] if part=='SEAT' else r['sha256']),
                                  (qn,task['quality_sha256'])):
                assert D.sha(path)==expected,path
                protected[path]=expected
            parent,condition,quality=D.arrays(pn),D.arrays(cn),D.read(qn)
            assert quality['status']=='COMPLETE' and quality['independent_verification']
            assert quality['input_npz_sha256']==D.sha(pn)
            case,audit=D.source_case(parent,condition,quality)
            if part=='SEAT':
                old=D.arrays(r['labels_npz']);protected[r['labels_npz']]=D.sha(r['labels_npz'])
                for key in old: assert np.array_equal(old[key],case[key]),(cid,key)
                case_checks.append(dict(condition_id=cid,existing_all_arrays_equal=True))
            n,k=int(case['N']),int(case['K'])
            path=OUT/'cases'/(cid+'.npz');save(path,case)
            bundle={key:case[key] for key in ('C_vertices','C_faces_original','C_faces_model')}
            bundle['total_N']=np.asarray(n,np.int64)
            ip=OUT/'C_only'/(cid+'.npz');save(ip,bundle)
            for on in (np.zeros(len(case['occupied_labels']),bool),case['occupied_labels'],np.ones(len(case['occupied_labels']),bool)):
                layout=D.inference_layout(**bundle,occupied=on)
                assert len(layout['seam_slots'])+len(layout['bulk_slots'])==n-k
            feature=__import__(__package__+'.head',fromlist=['features']).features(case['C_vertices'],case['C_faces_original'],n)
            assert feature.shape==(len(case['occupied_labels']),28) and np.isfinite(feature).all()
            edge=case['edge_records'];vm=case['C_vertices']
            for row in audit:
                ei=row['edge_id'];mid=(vm[edge[ei,1]]+vm[edge[ei,2]])/2
                audits.append(dict(condition_id=cid,parent_id=pid,part_type=part,edge_midpoint_world_y=float(mid[1]),**row))
            rec=dict(condition_id=cid,parent_id=pid,uid=pid,part_type=part,condition_type=part+'_FULL',group='TRAIN',split='TRAIN',
                     case_path=str(path),case_sha256=D.sha(path),inference_npz=str(ip),inference_npz_sha256=D.sha(ip),
                     parent_npz=pn,parent_sha256=D.sha(pn),condition_path=cn,condition_sha256=D.sha(cn),
                     quality_path=qn,quality_sha256=D.sha(qn),N=n,K=k,boundary_edges=len(edge),
                     owner_count=int(case['occupied_labels'].sum()),alias_count=int(case['source_alias_labels'].sum()),
                     null_unsupported_count=int((~case['source_supported_labels']).sum()),
                     third_in_C_owner_count=int((case['source_third_in_C']&case['occupied_labels']).sum()),
                     conditional_pretrain_seen=task.get('conditional_pretrain_seen','UNKNOWN'))
            records.append(rec)
    schedule=D.make_schedule(records);write(OUT/'schedule.json',schedule)
    bypart={part:dict(conditions=sum(r['part_type']==part for r in records),
               **{key:sum(r[key] for r in records if r['part_type']==part) for key in ('boundary_edges','owner_count','alias_count','null_unsupported_count','third_in_C_owner_count')}) for part in ('SEAT','BACK')}
    fam=[r['condition_id'] for r in beval['conditions'] if r['split']=='FAMILIAR']
    assert set(fam)<=eligible
    summary=dict(conditions=78,train_parents=47,parts=bypart,updates=1500,samples=12000,
                 SEAT_exposures=6000,BACK_exposures=6000,DEV_parent_leakage=0,
                 familiar_BACK_eval_conditions_in_this_training=fam,
                 familiar_label='Previously seen TRAIN47 parent and now explicitly trained BACK C; not unseen-C generalization')
    manifest=dict(schema='CHAIR_MULTIPART_SEAM_DATA_V1',status='FROZEN',rows=records,train=records,dev=[],summary=summary,
                  schedule_path=str(OUT/'schedule.json'),schedule_sha256=D.sha(OUT/'schedule.json'),protected_inputs=protected,
                  label_policy='One minimum open-edge owner per verified legal source Y face; aliases and unsupported edges inactive. No source owner at inference; source completion is not unique task truth.',
                  inference_input_keys=['C_vertices','C_faces_original','C_faces_model','total_N'],
                  height_axis='Y=coordinate1, existing world/model coordinates; no z-up assumption',
                  height_evidence='Existing Chair source/data render and region classification use y as vertical; no axis conversion added',
                  head_features='Existing24 + midpoint relative to C bbox center/span3 + midpoint world y/L1 (last explicitly duplicates base y channel)',
                  excluded_back_rows=[r for r in eligibility['rows'] if r['group']=='FAMILIAR' and r['status']!='ELIGIBLE'])
    write(OUT/'manifest.json',manifest);csvwrite(OUT/'source_roles.csv',audits);csvwrite(OUT/'conditions.csv',records)
    write(OUT/'seat_legacy_identity.json',dict(conditions=47,checks=case_checks))
    checks=cpu_checks(records,schedule);write(OUT/'cpu_checks.json',checks)
    for p,h in protected.items():assert D.sha(p)==h,p
    code=[Path(__file__),Path(__file__).with_name('data.py'),Path(__file__).with_name('head.py'),
          ROOT/'r2/pilots/seam_abc_v1/slot_data_extension.py',ROOT/'r2/src/meshflow_control/seam/slot_data.py']
    write(OUT/'source_provenance.json',dict(files={str(p):D.sha(p) for p in code},
          closest_prior='R31 owner-only SEAM plus owner_layout_v1 context classifier; joint Seat/Back data and requested context features/dropout replace the old per-edge head; no old head pretraining reused',
          source_geometry_scans=0,source_pair_labels='Existing independently verified quality caches'))
    files={str(p):D.sha(p) for p in OUT.rglob('*') if p.is_file()}
    write(OUT/'completion.json',dict(status='COMPLETE',files=files,wall_seconds=time.perf_counter()-started,
          summary=summary,OT_calls=checks['OT_calls'],head_forwards=checks['head_forwards'],head_backwards=1,
          training_updates=0,generator_forwards=0,denoiser_forwards=0,GPU_jobs=0,protected_sources_unchanged=True))
    return dict(status='COMPLETE',summary=summary,CPU_checks=checks['status'])


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
