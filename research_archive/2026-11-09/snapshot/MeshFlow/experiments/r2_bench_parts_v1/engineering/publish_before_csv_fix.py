"""Publish reviewed Bench source-part data; no models, RNG, repair or source writes.

All candidates, reviews and audits remain immutable. Combinations are new
source-face unions with their own audit/review; they do not inherit a PASS.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, collections, csv, io, json
from pathlib import Path
import numpy as np
from .construct import ROOT, OLD, partition, geom, topology, write, sha
from .render import resolve

OUT = ROOT / 'experiments/r2_bench_parts_v1'


def read(p):
    return json.loads(resolve(p).read_text('utf-8-sig'))


def rows(p):
    v = read(p)
    return v if isinstance(v, list) else v['rows']


def immutable_bytes(p, data):
    p = resolve(p)
    assert p.resolve().is_relative_to(OUT.resolve())
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if p.read_bytes() != data:
            raise FileExistsError('Refusing changed output: ' + str(p))
    else:
        p.write_bytes(data)


def csv_file(p, records, fields):
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fields, extrasaction='ignore'); w.writeheader()
    for record in records:
        w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in record.items() if k in fields})
    immutable_bytes(p, ('\ufeff' + s.getvalue()).encode('utf-8'))


def save_npz(p, arrays):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            assert set(z.files) == set(arrays)
            assert all(z[k].dtype == x.dtype and z[k].shape == x.shape and
                       z[k].tobytes() == x.tobytes() for k, x in arrays.items())
    else:
        with p.open('xb') as f: np.savez_compressed(f, **arrays)


def collect(bundles):
    all_rows, active, exclusion, references = [], [], [], {}
    seen_ids, seen_faces = set(), set()
    for bundle in bundles:
        manifest = resolve(bundle['manifest']); qpath = resolve(bundle['quality'])
        q = read(qpath)
        assert q['manifest_sha256'] == sha(manifest)
        qq = {r['candidate_id']: r for r in q['rows']}
        reviews = {}
        for path in bundle['reviews']:
            references[str(resolve(path))] = sha(resolve(path))
            for r in rows(path):
                cid = r['candidate_id']
                if cid in reviews: raise ValueError('Duplicate semantic review: ' + cid)
                assert sha(resolve(r['image_path'])) == r['image_sha256']
                assert r['reviewer'].startswith('AI_PROXY')
                reviews[cid] = r
        rr = rows(manifest)
        assert set(reviews) == {r['candidate_id'] for r in rr}, 'Every candidate must have an actual review'
        assert set(qq) == set(reviews)
        references[str(manifest)] = sha(manifest); references[str(qpath)] = sha(qpath)
        for r0 in rr:
            cid = r0['candidate_id']; assert cid not in seen_ids; seen_ids.add(cid)
            assert sha(resolve(r0['case_path'])) == r0['case_sha256']
            assert sha(resolve(r0['source_case_path'])) == r0['source_case_sha256']
            qr = qq[cid]; quality_file = qpath.parent / qr['quality_file']
            assert sha(quality_file) == qr['quality_sha256']
            detail = read(quality_file)
            r = dict(r0, condition_id=cid, review=reviews[cid],
                     part_type=reviews[cid]['actual_part'], extent_description=reviews[cid]['extent_description'],
                     quality_file=str(quality_file), quality_sha256=sha(quality_file),
                     supervision_file=str(qpath.parent / qr['supervision_file']),
                     supervision_sha256=qr['supervision_sha256'],
                     geometry_status=qr['geometry_status'], seam_applicability=qr['seam_applicability'],
                     seam_reasons=qr['seam_reasons'],
                     current_bench_entry_consumable=detail['current_bench_entry_consumable'],
                     current_bench_entry_limitations=detail['current_bench_entry_limitations'],
                     source_owner_count=detail['source_owner_count'],
                     source_alias_count=detail['source_alias_count'],
                     source_null_unsupported_count=detail['source_null_unsupported_count'],
                     source_target_risks=detail['source_target_risks'])
            reason = []
            if qr['geometry_status'] != 'PASS_PENDING_SEMANTIC_REVIEW': reason += qr['reasons']
            if reviews[cid]['decision'] != 'ACCEPT': reason.append('SEMANTIC_' + reviews[cid]['decision'])
            if reviews[cid]['human_plausible_input'] != 'YES': reason.append('INPUT_PLAUSIBILITY_' + reviews[cid]['human_plausible_input'])
            facekey = (r['parent_id'], tuple(sorted(r['source_face_ids'])))
            if not reason and facekey in seen_faces: reason.append('DUPLICATE_SAME_PARENT_FACE_SET')
            if reason:
                r['status'] = 'EXCLUDED'; r['exclusion_reasons'] = reason; exclusion.append(r)
            else:
                seen_faces.add(facekey); r['status'] = 'ACTIVE_REVIEWED_SOURCE_CONDITION'; active.append(r)
            all_rows.append(r)
    return all_rows, active, exclusion, references


def combinations(bundles):
    all_rows, active, excluded, refs = collect(bundles)
    write(OUT / 'base_reviewed.json', dict(rows=all_rows, active_count=len(active), excluded_count=len(excluded), references=refs))
    groups = collections.defaultdict(list)
    for r in active: groups[r['parent_id']].append(r)
    out = OUT / 'combinations'; candidates, skipped = [], []
    for parent, rr in sorted(groups.items()):
        rr.sort(key=lambda r: r['condition_id'])
        seats = [r for r in rr if r['part_type'] == 'SEAT_SURFACE']
        legs = [r for r in rr if r['part_type'] == 'LEG_SUPPORT']
        backs = [r for r in rr if r['part_type'] == 'BACK_SURFACE']
        proposals = []
        if seats and legs: proposals.append(('SEAT_SUPPORT', [seats[0], legs[0]]))
        if seats and backs: proposals.append(('SEAT_BACK', [seats[0], backs[0]]))
        # The third proposal is a larger same-parent assembly, never random faces.
        full = seats[:1] + legs[:2] + backs[:1]
        if len(full) >= 2: proposals.append(('PART_ASSEMBLY', full))
        elif len(legs) >= 2: proposals.append(('PART_ASSEMBLY', legs[:2]))
        seen = {tuple(sorted(r['source_face_ids'])) for r in rr}
        for kind, parts in proposals[:3]:
            ids = sorted(set(i for r in parts for i in r['source_face_ids']))
            key = tuple(ids)
            if key in seen or len(ids) >= parts[0]['N']:
                skipped.append(dict(parent_id=parent, part_type=kind, reason='DUPLICATE_FACE_SET_OR_EMPTY_FREE', component_ids=[r['condition_id'] for r in parts])); continue
            seen.add(key)
            src = parts[0]; path = resolve(src['source_case_path'])
            with np.load(path, allow_pickle=False) as z: a = {k:z[k] for k in z.files}
            cid = parent + '__COMBO_' + kind
            dest = out / 'cases' / (cid + '.npz'); save_npz(dest, partition(a, ids))
            v,f,t,area,n,cent = geom(a); pts=t[ids].reshape(-1,3)
            candidates.append(dict(candidate_id=cid, parent_id=parent, group=src['group'], official_split=src['official_split'],
                source_case_path=str(path), source_case_sha256=src['source_case_sha256'], source_face_ids=ids,
                proposed_part_type=kind, construction_evidence=dict(method='union_of_reviewed_same_parent_original_face_sets',
                    component_ids=[r['condition_id'] for r in parts],
                    component_extents=[r['extent_description'] for r in parts]),
                uncertainty=['COMBINATION_REQUIRES_OWN_GEOMETRY_AND_ACTUAL_VISUAL_REVIEW'],
                status='PENDING_COMBINATION_REVIEW', labels_fed_to_model=False,
                semantic_authority='COMPOSITION_OF_AI_PROXY_REVIEWED_PARTS_PENDING_COMPOSITION_REVIEW',
                case_path=str(dest), case_sha256=sha(dest), N=len(f), K=len(ids), face_fraction=len(ids)/len(f),
                area=float(area[ids].sum()), area_fraction=float(area[ids].sum()/area.sum()),
                area_units='official_normalized_object_units_squared', area_denominator='sum_all_original_source_faces_including_duplicates',
                bbox_normalized=[pts.min(0).tolist(),pts.max(0).tolist()], span_normalized=np.ptp(pts,axis=0).tolist(),
                region_normal_not_single_plane=bool(np.linalg.norm(np.std(n[ids],axis=0))>.1),
                geometry_unchanged=True, source_partition_bitwise=True, **topology(a,ids)))
    write(out/'manifest.json', dict(schema='bench_reviewed_part_union_candidates_v1', rows=candidates,
        skipped=skipped, references=refs, model_calls=0, source_modifications=0))
    print(json.dumps(dict(active_base=len(active), excluded_base=len(excluded), combination_candidates=len(candidates))))


def finalize(bundles):
    all_rows, active, excluded, refs = collect(bundles)
    active.sort(key=lambda r: (r['group']!='DEV',r['parent_id'],r['part_type'],r['condition_id']))
    export = OUT / 'data'; verification=[]
    for r in active:
        cid = r['condition_id']
        with np.load(resolve(r['source_case_path']),allow_pickle=False) as z: a={k:z[k] for k in z.files}
        paired=partition(a,r['source_face_ids']); dest=export/'paired'/(cid+'.npz'); save_npz(dest,paired)
        inference_keys=('C','C_vertices','C_faces_original','C_faces_model','C_source_corner_permutations','N','K')
        inp={k:paired[k].copy() for k in inference_keys}
        infer=export/'inputs'/(cid+'.npz');save_npz(infer,inp)
        obj=export/'inputs'/(cid+'.obj')
        text='# Original source winding; model coordinates; only supplied C\n'
        text+=''.join('v '+' '.join(format(float(x),'.9g') for x in v)+'\n' for v in inp['C_vertices'])
        text+=''.join('f '+' '.join(str(int(x)+1) for x in f)+'\n' for f in inp['C_faces_original'])
        immutable_bytes(obj,text.encode('ascii'))
        r.update(paired_path=str(dest),paired_sha256=sha(dest),input_path=str(infer),input_sha256=sha(infer),
                 input_obj_path=str(obj),input_obj_sha256=sha(obj),labels_fed_to_model=False)
        verification.append(dict(condition_id=cid,C_plus_Y='BITWISE_EXACT',inference_unknown_Y_keys=0,
                                 C_indexed_roundtrip='BITWISE_EXACT',min_K_pass=r['K']>=2,free_nonempty=r['K']<r['N']))
    old=read(OLD); protect={str(OLD):sha(OLD)}
    for s in old['train']+old['dev']:
        path=resolve(s['case_path']);assert sha(path)==s['case_sha256'];protect[str(path)]=sha(path)
    groups=collections.defaultdict(list)
    for r in active:groups[r['parent_id']].append(r)
    assert not ({r['parent_id'] for r in active if r['group']=='TRAIN'} & {r['parent_id'] for r in active if r['group']=='DEV'})
    coverage=[]
    for s in old['train']+old['dev']:
        rr=groups[s['parent_id']]
        cc=collections.Counter(r['part_type'] for r in rr)
        missing=[k for k in ('SEAT_SURFACE','LEG_SUPPORT','BACK_SURFACE') if not cc[k]]
        coverage.append(dict(parent_id=s['parent_id'],group=s['group'],official_split=s['official_split'],
            active_conditions=len(rr),part_counts=dict(cc),missing_pure_part_types=missing,
            missing_status='NA_OR_NOT_CONFIDENTLY_ISOLATED; consult candidate/review ledger',
            condition_ids=[r['condition_id'] for r in rr]))
    summary=dict(active_conditions=len(active),excluded_candidates=len(excluded),all_candidates=len(all_rows),
        parents=len(groups),train_parents=len({r['parent_id'] for r in active if r['group']=='TRAIN'}),
        dev_parents=len({r['parent_id'] for r in active if r['group']=='DEV'}),
        split_counts=dict(collections.Counter(r['group'] for r in active)),
        part_counts=dict(collections.Counter(r['part_type'] for r in active)),
        part_split_counts={k:dict(collections.Counter(r['part_type'] for r in active if r['group']==k)) for k in ('TRAIN','DEV')},
        nonplanar_conditions=sum(r['region_normal_not_single_plane'] for r in active),
        multi_component_conditions=sum(r['source_edge_components']>1 for r in active),
        closed_C_conditions=sum(r['boundary_edges']==0 for r in active),
        current_bench_entry_consumable=sum(r['current_bench_entry_consumable'] for r in active),
        seam_applicability=dict(collections.Counter(r['seam_applicability'] for r in active)),
        K_stats=dict(min=min(r['K'] for r in active),median=float(np.median([r['K'] for r in active])),max=max(r['K'] for r in active)),
        face_fraction_stats=dict(min=min(r['face_fraction'] for r in active),median=float(np.median([r['face_fraction'] for r in active])),max=max(r['face_fraction'] for r in active)),
        area_fraction_stats=dict(min=min(r['area_fraction'] for r in active),median=float(np.median([r['area_fraction'] for r in active])),max=max(r['area_fraction'] for r in active)),
        model_calls=0,training_updates=0,new_generated_meshes=0,new_noise_arrays=0,source_geometry_changes=0)
    manifest=dict(schema='BENCH_MULTI_PART_REVIEWED_DATA_V1',rows=active,summary=summary,references=refs,
        protocol='AI_PROXY visual part review + independent C geometry audit; original C/Y source-ID partition',
        source_split_preserved=True, unique_source_style_is_not_generation_success_criterion=True,
        sampling_recommendation='Uniform parent, then explicitly chosen part type/variant. Never sample flat rows implicitly.',
        model_input_policy='Use only data/inputs NPZ. Part names, source owners/aliases, Y and IDs are external bookkeeping/supervision.',
        training_status='DATA_READY; NOT_TRAINED; current entry/SEAM eligibility recorded per condition; no automatic training')
    write(export/'manifest.json',manifest)
    for group in ('TRAIN','DEV'):write(export/(group.lower()+'.json'),dict(rows=[r for r in active if r['group']==group]))
    write(export/'current_seam_compatible.json',dict(
        policy='Data contract eligibility only, not a trained-model performance claim',
        rows=[r for r in active if r['current_bench_entry_consumable'] and r['seam_applicability']=='SUPPORTED_CURRENT_SINGLE_EDGE_SLOT']))
    write(export/'native_requires_entry_extension.json',dict(
        policy='Valid reviewed C; current Bench entry or current single-edge SEAM layout needs an explicit extension before training/sampling',
        rows=[r for r in active if not r['current_bench_entry_consumable'] or r['seam_applicability']!='SUPPORTED_CURRENT_SINGLE_EDGE_SLOT']))
    write(export/'candidate_disposition.json',dict(rows=all_rows))
    write(export/'parent_coverage.json',dict(rows=coverage))
    write(export/'verification.json',dict(status='PASS',rows=verification,source_hashes_unchanged=protect,summary=summary))
    fields=['condition_id','parent_id','group','official_split','part_type','extent_description','N','K','face_fraction','area_fraction','source_edge_components','boundary_edges','region_normal_not_single_plane','source_owner_count','source_alias_count','source_null_unsupported_count','seam_applicability','seam_reasons','current_bench_entry_consumable','current_bench_entry_limitations','input_path','paired_path','source_face_ids','quality_file']
    csv_file(export/'conditions.csv',active,fields)
    csv_file(export/'parent_coverage.csv',coverage,['parent_id','group','official_split','active_conditions','part_counts','missing_pure_part_types','missing_status','condition_ids'])
    csv_file(export/'excluded.csv',excluded,['condition_id','parent_id','group','proposed_part_type','part_type','exclusion_reasons','review','geometry_status'])
    csv_file(export/'semantic_review.csv',[dict(candidate_id=r['candidate_id'],parent_id=r['parent_id'],group=r['group'],**r['review']) for r in all_rows],['candidate_id','parent_id','group','decision','actual_part','extent_description','human_plausible_input','notes','reviewer','image_path','image_sha256'])
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['combine','finalize']);p.add_argument('--bundles',required=True);args=p.parse_args()
    bundles=read(args.bundles)['bundles']
    if args.stage=='combine':combinations(bundles)
    else:finalize(bundles)

if __name__=='__main__':main()
