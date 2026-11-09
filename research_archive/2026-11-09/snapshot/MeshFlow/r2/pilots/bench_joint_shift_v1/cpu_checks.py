"""Finite CPU-only affine/SEAM packing acceptance, no neural code or CUDA calls."""
from __future__ import annotations
import collections
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from . import data as D


def eq(a, b, name):
    if not np.array_equal(a, b):
        raise AssertionError(name)


def digest_array(a):
    a = np.asarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def geometry(triangles):
    t = np.asarray(triangles, np.float64)
    edges = np.stack((t[:,1]-t[:,0], t[:,2]-t[:,1], t[:,0]-t[:,2]), 1)
    lengths = np.linalg.norm(edges, axis=2)
    cross = np.cross(t[:,1]-t[:,0], t[:,2]-t[:,0])
    area2 = np.linalg.norm(cross, axis=1)
    normals = cross/np.maximum(area2[:,None], np.finfo(float).tiny)
    angle = []
    for k in range(3):
        u, v = t[:,(k+1)%3]-t[:,k], t[:,(k+2)%3]-t[:,k]
        angle.append(np.degrees(np.arccos(np.clip(np.sum(u*v,axis=1)/
            np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),np.finfo(float).tiny),-1,1))))
    return dict(length=lengths, area2=area2, normals=normals, angles=np.stack(angle,1))


def model_coordinate_checks(original, transformed, name):
    n, k = int(transformed['N']), int(transformed['K'])
    v, full, pre = transformed['model_vertices'], transformed['full_target'], transformed['pre_ot_full']
    known, free = transformed['known_ids'], transformed['free_ids']
    for key in ('source_vertices','normalized_vertices','transform_json','target_version_id',
            'source_faces','source_face_vertex_ids','source_corner_permutations','known_ids','free_ids',
            'source_face_ids','free_source_face_ids','C_faces_original','C_faces_model',
            'C_source_vertex_ids','C_source_corner_permutations','C_source_faces_original',
            'N','K','occupied_labels','edge_owner_face','edge_source_triple','source_alias_labels',
            'source_supported_labels','edge_records','source_third_in_C'):
        if key in original:
            eq(original[key], transformed[key], name+': identity/provenance '+key)
    eq(np.sort(np.r_[known,free]), np.arange(n), name+': C/Y partition')
    eq(v[transformed['source_face_vertex_ids']], full, name+': canonical vertex reconstruction')
    eq(pre*np.float32(2), full, name+': exact preOT half scale')
    eq(transformed['C'], full[known], name+': C subset')
    eq(transformed['Y'], full[free], name+': Y subset')
    eq(transformed['C_pre_ot'], pre[known], name+': C_preOT')
    eq(transformed['Y_pre_ot'], pre[free], name+': Y_preOT')
    eq(transformed['C_vertices'], v[transformed['C_source_vertex_ids']], name+': C IDs')
    eq(transformed['C_vertices'][transformed['C_faces_model']], full[known], name+': C model winding')
    eq(transformed['C_vertices'][transformed['C_faces_original']],
       v[transformed['source_faces'][known]], name+': original C winding')
    restored = np.take_along_axis(full, np.argsort(transformed['source_corner_permutations'],1)[:,:,None], 1)
    eq(restored, v[transformed['source_faces']], name+': full source winding inverse')
    for key in ('model_vertices','full_target','pre_ot_full','C_vertices','C','Y','C_pre_ot','Y_pre_ot'):
        if transformed[key].dtype != np.float32 or not np.isfinite(transformed[key]).all():
            raise AssertionError(name+': finite FP32 '+key)
    edges, features = D.reference.old.edge_features(transformed['C_vertices'], transformed['C_faces_original'], n)
    eq(edges, transformed['edge_records'], name+': C boundary identity')
    if not np.isfinite(features).all():
        raise AssertionError(name+': C edge features finite')
    layout = D.reference.old.inference_layout(transformed['C_vertices'], transformed['C_faces_original'],
        transformed['C_faces_model'], n, transformed['occupied_labels'])
    fixed = layout['anchor_ids'] >= 0
    if int(layout['free_coordinate_count']) != 9*(n-k)-6*int(transformed['occupied_labels'].sum()):
        raise AssertionError(name+': unchanged FM denominator')
    # Every owner base refers to the same transformed indexed C endpoints.
    sv = transformed['source_face_vertex_ids']
    for ei in np.flatnonzero(transformed['occupied_labels']):
        fi = int(transformed['edge_owner_face'][ei]); ids = transformed['edge_source_triple'][ei]
        cp = np.asarray([np.flatnonzero(sv[fi] == vi).item() for vi in ids])
        eq(full[fi,cp][:2], transformed['C_vertices'][layout['anchor_ids'][k+ei,:2]], name+': owner anchors')
    return layout, features


def sample_checks(sample, case, t, label):
    n, k = int(case['N']), int(case['K'])
    fixed = sample['anchor_ids'] >= 0
    eq(np.sort(sample['source_face_ids']), np.arange(n), label+': each source face exactly once')
    eq(case['full_target'][sample['source_face_ids'][:,None],sample['source_corner_permutations']], sample['x1'], label+': x1 exact source')
    eq(sample['x0'][fixed], case['C_vertices'][sample['anchor_ids'][fixed]], label+': exact C anchors')
    eq(sample['x1'][fixed], sample['x0'][fixed], label+': target anchors')
    eq(sample['xt'][fixed], sample['x0'][fixed], label+': state anchors')
    eq(sample['u'][fixed], np.zeros_like(sample['u'][fixed]), label+': zero anchor velocity')
    eq(sample['free_corner_mask'], ~fixed, label+': free mask')
    eq(sample['u'], sample['x1']-sample['x0'], label+': exact FM target')
    expected = (np.float32(1)-np.float32(t))*sample['x0'] + np.float32(t)*sample['x1']
    expected[fixed] = sample['x0'][fixed]
    eq(sample['xt'], expected, label+': interpolation')
    if t == 0: eq(sample['xt'], sample['x0'], label+': t0')
    if t == 1: eq(sample['xt'], sample['x1'], label+': t1')
    if int(sample['free_coordinate_count']) != 9*(n-k)-6*int(case['occupied_labels'].sum()):
        raise AssertionError(label+': scalar denominator')
    for key in ('x0','x1','xt','u','C_vertices','edge_features'):
        if not np.isfinite(sample[key]).all(): raise AssertionError(label+': finite '+key)


def run(manifest, schedule):
    started = time.perf_counter()
    check_stream = D.validate_schedule(schedule['steps'])
    first = {}
    for step in schedule['steps']:
        for row in step['samples']:
            first.setdefault((row['condition_id'], row['augmentation']), row)
    cases = {}; summaries = []; representatives = {}
    max_length_error = max_angle_error = max_area_error = 0.
    classes = collections.Counter()
    for row in manifest['train']:
        cid = row['condition_id']; case = D.load_case(row); cases[cid] = case
        before = {k:digest_array(v) for k,v in case.items()}
        occupancy = case['occupied_labels']
        kind = 'CLOSED' if len(occupancy)==0 else ('OWNER' if occupancy.any() else 'OPEN_NO_OWNER')
        classes[kind] += 1; representatives.setdefault(kind, cid)
        item = dict(condition_id=cid, case_sha256=D.sha(row['case_path']), kind=kind, variants=[])
        for variant in ('CLEAN','ANISOTROPIC'):
            stream = first[(cid,variant)]
            clean_scaled = D.transformed_case(case, stream['scale_xyz'], [0.,0.,0.])
            moved = D.transformed_case(case, stream['scale_xyz'], stream['translation_xyz'])
            layout0, features0 = model_coordinate_checks(case, clean_scaled, cid+':scale')
            layout, features = model_coordinate_checks(case, moved, cid+':'+variant)
            old_inputs = D.reference.scaled_case(case, stream['scale_xyz'])
            for key in ('full_target','pre_ot_full','C_vertices'):
                # Production fields are unchanged for zero translation, including signed zeros.
                if digest_array(old_inputs[key]) != digest_array(clean_scaled[key]):
                    raise AssertionError(cid+': zero-shift production array bytes '+key)
            for key in ('anchor_ids','known_mask','valid_mask','free_corner_mask','seam_slots','bulk_slots'):
                eq(layout0[key], layout[key], cid+': mask/topology '+key)
            g0 = geometry(clean_scaled['full_target']); g = geometry(moved['full_target'])
            if np.any(g['area2'] <= 0) or np.any(g0['area2'] <= 0):
                raise AssertionError(cid+': new/existing exact zero-area model face')
            length_error = float(np.max(np.abs(g['length']-g0['length'])))
            angle_error = float(np.max(np.abs(g['angles']-g0['angles'])))
            area_error = float(np.max(np.abs(g['area2']-g0['area2'])))
            # FP32 addition can round coordinates; tolerate arithmetic noise only.
            if length_error > 3e-6 or angle_error > .03 or area_error > 3e-5:
                raise AssertionError(f'{cid}: translation geometry drift {length_error}/{angle_error}/{area_error}')
            if np.min(np.sum(g0['normals']*g['normals'],axis=1)) < .99999:
                raise AssertionError(cid+': source normal changed beyond FP32 noise')
            # Absolute edge midpoint/C center translate; all other edge24 fields remain invariant.
            if len(features):
                shift = np.asarray(stream['translation_xyz'],np.float32)/np.float32(D.L)
                np.testing.assert_allclose(features[:,:3]-features0[:,:3], np.broadcast_to(shift,features[:,:3].shape), atol=1e-7,rtol=1e-5)
                np.testing.assert_allclose(features[:,15:18]-features0[:,15:18],np.broadcast_to(shift,features[:,15:18].shape),atol=1e-7,rtol=1e-5)
                other = list(range(3,15))+list(range(18,24))
                np.testing.assert_allclose(features[:,other],features0[:,other],atol=1e-5,rtol=2e-4)
            max_length_error=max(max_length_error,length_error);max_angle_error=max(max_angle_error,angle_error);max_area_error=max(max_area_error,area_error)
            item['variants'].append(dict(augmentation=variant,translation_xyz=stream['translation_xyz'],
                max_length_rounding_error=length_error,max_angle_rounding_error_degrees=angle_error,
                model_min=moved['model_vertices'].min(0).tolist(),model_max=moved['model_vertices'].max(0).tolist(),
                free_coordinate_count=int(layout['free_coordinate_count']),status='PASS'))
        if before != {k:digest_array(v) for k,v in case.items()}:
            raise AssertionError(cid+': original case mutated')
        summaries.append(item)
    if set(representatives) != {'OWNER','CLOSED','OPEN_NO_OWNER'}:
        raise AssertionError('Representative packing cases missing')
    sample_results=[]; ot_calls=0
    for kind,cid in representatives.items():
        case=cases[cid]
        for variant in ('CLEAN','ANISOTROPIC'):
            row=first[(cid,variant)];n,k=int(case['N']),int(case['K'])
            z=np.random.RandomState(row['noise_seed']).standard_normal((n-k,3,3)).astype(np.float32)
            eps=np.random.RandomState(row['epsilon_seed']).standard_normal(z.shape).astype(np.float32)
            hashes=(digest_array(z),digest_array(eps))
            baseline_case=D.reference.scaled_case(case,row['scale_xyz'])
            baseline=D.sample_for_probe(baseline_case,z,eps,.5,row['permutation_seed']);ot_calls+=int(baseline['production_OT_calls'])
            transformed=D.transformed_case(case,row['scale_xyz'],row['translation_xyz'])
            for t in (0.,.5,1.):
                sample=D.sample_for_probe(transformed,z,eps,t,row['permutation_seed']);ot_calls+=int(sample['production_OT_calls'])
                sample_checks(sample,transformed,t,cid+':'+variant)
                for key in ('anchor_ids','known_mask','valid_mask','free_corner_mask','face_permutation','occupied_labels','free_coordinate_count'):
                    eq(sample[key],baseline[key],cid+': original packing masks '+key)
                free=sample['free_corner_mask']
                eq(sample['x0'][free],baseline['x0'][free],cid+': Gaussian/HYBRID free origin unchanged')
                if variant=='CLEAN' and t==.5:
                    for key in baseline:
                        eq(sample[key],baseline[key],cid+': identity full packing '+key)
                sample_results.append(dict(condition_id=cid,kind=kind,augmentation=variant,t=t,
                    changed_OT_face_entries=int(np.count_nonzero(sample['ot_face_map']!=baseline['ot_face_map'])),
                    production_OT_recomputed=True,source_partition_exact=True,fixed_velocity_zero=True,status='PASS'))
            if hashes!=(digest_array(z),digest_array(eps)):
                raise AssertionError('Input noise modified')
    shifts=np.asarray([r['translation_xyz'] for e in schedule['steps'] for r in e['samples'] if r['augmentation']=='ANISOTROPIC'])
    code=[Path(__file__),Path(D.__file__),Path(D.reference.__file__),Path(D.reference.old.__file__),
          D.ROOT/'r2/pilots/seam_abc_v1/slot_data_extension.py',
          D.ROOT/'r2/src/meshflow_control/seam/slot_data.py',D.ROOT/'r2/src/meshflow_control/training/coupling.py']
    return dict(status='PASS',schema='BENCH_JOINT_SHIFT_CPU_ACCEPTANCE_V1',
        neural_forwards=0,GPU_calls=0,new_OT_calls=ot_calls,
        preprocessing_checks=182,production_packing_checks=len(sample_results),baseline_OT_comparisons=6,
        exact_current_pool=classes,stream=check_stream,condition_results=summaries,sample_results=sample_results,
        translation_stats=dict(fixed_L=D.L,per_axis_max=D.TRANSLATION_AXIS_MAX,actual_axis_min=shifts.min(0).tolist(),
            actual_axis_max=shifts.max(0).tolist(),max_vector_norm=float(np.linalg.norm(shifts,axis=1).max()),
            mean_vector_norm=float(np.linalg.norm(shifts,axis=1).mean())),
        FP32_translation_rounding=dict(max_edge_length_error=max_length_error,max_angle_error_degrees=max_angle_error,max_doubled_area_error=max_area_error),
        code_hashes={str(p):D.sha(p) for p in code},elapsed_seconds=time.perf_counter()-started,
        OT_note='Original scipy Euclidean cost rerun; target-only translation need not preserve Hungarian face assignment; no loss/OT definition changed')


if __name__ == '__main__':
    result=run(D.read(D.OUT/'data/manifest.json'),D.read(D.OUT/'data/schedule.json'))
    D.write_new(D.OUT/'data/translation_cpu_checks.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('condition_results','sample_results','code_hashes')},ensure_ascii=False))
