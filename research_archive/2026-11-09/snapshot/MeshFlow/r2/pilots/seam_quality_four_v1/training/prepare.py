"""CPU-only 800-step continuation stream; each OT result is shared by all arms."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import time
import numpy as np
from . import support as S
from r2.prototypes.seam_slot_c200_v1.run import make_case
from meshflow_control.seam.slot_data import supervised_sample

PRIOR = S.ROOT/'experiments/r2_seam_slot_c200_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)


def arrays(path):
    with np.load(S.resolve(path), allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}


def schedule(records):
    rng = np.random.RandomState(2026101607)
    old, new = [], []
    for source_step in range(1, 1001):
        rows = []
        order = rng.permutation(4)
        for aug in range(2):
            for ci in order:
                alpha = 1. if aug == 0 else float(rng.uniform(.9, 1.1))
                seed = 10000000+source_step*100+int(ci)*10+aug
                rows.append(dict(condition_index=int(ci), uid=records[ci]['parent_id'],
                    sample_index=len(rows), alpha=alpha, noise_seed=seed,
                    epsilon_seed=seed+100000000, time_seed=seed+200000000,
                    permutation_seed=seed+300000000))
        if source_step <= 200: old.append(dict(step=source_step, rows=rows))
        else: new.append(dict(step=source_step-200, source_step=source_step, rows=rows))
    expected = S.read(PRIOR/'input_schedule.json')['steps']
    if old != expected: raise ValueError('Replay of independent old data RNG differs')
    return dict(schema='seam_quality_continued_data_schedule_v1', seed=2026101607,
                replay_verified_steps=200, independent_from_checkpoint_torch_RNG=True,
                steps=new, ending_MT_state=dict(key=rng.get_state()[1].tolist(), position=rng.get_state()[2]))


def freeze(root):
    path = root/'training_registration.json'
    if path.exists(): return S.read(path)
    reg0 = S.read(PRIOR/'registration.json')
    checkpoint = PRIOR/'checkpoints/SEAM_SLOT_C200_step200.pt'
    receipt = S.read(checkpoint.with_suffix('.json'))
    if sha(checkpoint) != receipt['sha256']: raise ValueError('Source SEAM bytes changed')
    data_schedule = schedule(reg0['records'])
    write(root/'input_schedule.json', data_schedule)
    reg = dict(schema='r2_seam_quality_four_training_registration_v1', status='FROZEN',
        arms=list(S.ARMS), updates_per_arm=800, batch=8, microbatch=1,
        source_global_step=12200, source_head_step=200, final_global_step=13000,
        source_checkpoint=str(checkpoint), source_checkpoint_sha256=receipt['sha256'],
        source_generator_sha256=receipt['generator_state_sha256'],
        source_optimizer_sha256=receipt['optimizer_state_sha256'], source_rng_sha256=receipt['rng_sha256'],
        source_head_sha256=receipt['head_state_sha256'], source_head_optimizer_sha256=receipt['head_optimizer_state_sha256'],
        head_seed=20261016, head_lr=.001, head_class_weights=[8.25,66/124], head_threshold_logit=0.,
        records=reg0['records'], input_schedule_sha256=sha(root/'input_schedule.json'),
        diagnostic_time_bin_edges=[0,.2,.4,.6,.8,1],
        source_occupancy_training_only=True, inference_occupancy='predicted C-only logit>=0',
        local_noise='only seam tip mixed Gaussian origin += edge midpoint; variance sigma=1; OT/bulk unchanged',
        anti_cover=dict(version='finite_C_anti_cover_centroid_tip_v1', lambda_value=1., tau_L=.02,
            time_min=.5, denominator='all effective-batch occupied seams times two probes, including inactive gates',
            source_gate='antiflatten/source_gate.json', nonzero_coplanar_side='negative C normal'),
        scientific_budget=dict(OT=6400, samples_shared=6400, generator_forward=19200,
            generator_backward=19200, generator_updates=2400, head_forward=19200,
            head_backward=19200, head_updates=2400, extra_preflight_forward=0),
        stopping='Fixed additional800; one final endpoint; no checkpoint selection or silent retries')
    write(path, reg)
    return reg


def worker(item):
    root_text, entry, records = item
    root = Path(root_text); step = entry['step']; out = []; calls = 0
    for row in entry['rows']:
        path = root/'data/shared'/f's{entry["source_step"]:04d}_m{row["sample_index"]}.npz'
        receipt = path.with_suffix('.json')
        if path.exists():
            if not receipt.exists() or sha(path) != S.read(receipt)['sha256']:
                raise RuntimeError('Incomplete existing data cell: '+str(path))
        else:
            # Reserve before doing the single authorized OT. Interrupted cells
            # remain explicit and cannot be silently recomputed on resume.
            path.parent.mkdir(parents=True, exist_ok=True)
            started = path.with_suffix('.started.json')
            write(started, dict(step=step, source_step=entry['source_step'], row=row, OT_calls_reserved=1))
            record = records[row['condition_index']]
            case = make_case(record, row['alpha'])
            parent = arrays(record['parent_npz'])
            n, k = int(case['N']), int(case['K'])
            z = np.random.RandomState(row['noise_seed']).randn(n-k,3,3)
            eps = np.random.RandomState(row['epsilon_seed']).randn(n-k,3,3)
            t = np.float32(1/(1+np.exp(-np.random.RandomState(row['time_seed']).randn())))
            sample = supervised_sample(case, z, eps, t, row['permutation_seed'], 'SEAM_SLOT_C200'); calls += 1
            source_third = np.zeros((n,3), bool); source_owner = np.zeros(n,bool)
            for ei in np.flatnonzero(case['occupied_labels']):
                fi = int(case['edge_owner_face'][ei]); third = case['edge_source_triple'][ei,2]
                source_third[fi] = case['source_face_vertex_ids'][fi] == third; source_owner[fi] = True
            ids, cp = sample['source_face_ids'], sample['source_corner_permutations']
            current = case['source_face_vertex_ids'][ids[:,None],cp]
            original = parent['source_faces'][ids]
            winding = np.array([[int(np.flatnonzero(cur == v).item()) for v in ori]
                                for cur,ori in zip(current,original)], np.int64)
            initial_slots = k+np.flatnonzero(case['occupied_labels'])
            final_slots = np.argsort(sample['face_permutation'])[initial_slots]
            sample.update(seam_third_corner_mask=source_third[ids[:,None],cp],
                source_interface_face_mask=source_owner[ids], source_winding_permutation=winding,
                uid=np.asarray(row['uid']), alpha=np.asarray(row['alpha']), sample_index=np.asarray(row['sample_index']),
                source_stream_step=np.asarray(entry['source_step']), C_faces_original=case['C_faces_original'],
                anti_cover_seam_slots=final_slots,
                anti_cover_owner_faces=np.asarray(sample['edge_features'][:0], np.int64))
            # Boundary records are C-only topology; owner face is the C face
            # incident to the original open edge, not hidden free completion.
            from meshflow_control.seam.slot_data import boundary_from_C
            edge = boundary_from_C(case['C_vertices'],case['C_faces_original'])
            sample['anti_cover_owner_faces'] = edge[case['occupied_labels'],0]
            assert np.array_equal(np.take_along_axis(current,winding,1),original)
            assert np.array_equal(sample['anchor_ids'][final_slots,:2], edge[case['occupied_labels'],1:][:,::-1])
            temp = path.with_suffix('.partial.npz'); np.savez_compressed(temp, **sample); os.replace(temp,path)
            write(receipt, dict(sha256=sha(path), step=step, source_step=entry['source_step'], OT_calls=1))
        out.append(dict(path=str(path), sha256=sha(path), uid=row['uid'], alpha=row['alpha'], sample_index=row['sample_index']))
    return dict(step=step, source_step=entry['source_step'], samples={arm:out for arm in S.ARMS}, new_OT_calls=calls)


def data(root, workers=2):
    reg = freeze(root)
    target = root/'data_manifest.json'
    if target.exists(): return S.read(target)
    if workers not in (1,2): raise ValueError('Only resource-approved one/two CPU OT workers')
    entries = S.read(root/'input_schedule.json')['steps']; tick = time.perf_counter(); rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(worker,(str(root),entry,reg['records'])) for entry in entries]
        for result in concurrent.futures.as_completed(jobs):
            rows.append(result.result())
            if len(rows)%50 == 0: print(json.dumps(dict(completed_steps=len(rows),seconds=time.perf_counter()-tick)),flush=True)
    rows.sort(key=lambda x:x['step'])
    result = dict(status='COMPLETE',steps=rows,shared_samples=6400,OT_logical_calls=6400,
        OT_calls_this_invocation=sum(r['new_OT_calls'] for r in rows), CPU_workers=workers,seconds=time.perf_counter()-tick)
    write(target,result); return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='experiments/r2_seam_quality_four_v1')
    parser.add_argument('--stage',choices=['freeze','data'],default='freeze');parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args(); root=S.resolve(args.root).resolve()
    if not root.is_relative_to((S.ROOT/'experiments').resolve()): raise ValueError('Isolated experiment output required')
    print(json.dumps(freeze(root) if args.stage=='freeze' else data(root,args.workers),ensure_ascii=False)[:2000])

if __name__ == '__main__': main()
