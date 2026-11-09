"""Close and subsequently verify the completed CPU experiment; never rescore."""
from pathlib import Path
import collections
import csv
import json
import numpy as np
from r2.pilots.seam_slot_c200_v1 import common as C

OUT=C.ROOT/'experiments/r2_seam_quality_four_v1/redirection'
CAP=C.ROOT/'experiments/r2_seam_geometry_guard_v1/capacity'


def verify(mapping):
    for path, expected in mapping.items():
        if C.sha(C.resolve(path)) != expected:
            raise ValueError('Frozen file changed: '+path)


def main():
    completion=OUT/'completion.json'
    if completion.exists():
        done=C.read(completion)
        for key in ('protected','source_hashes','artifacts'): verify(done[key])
        print(json.dumps(dict(status='COMPLETE_VERIFIED',model_calls=0,scoring_calls=0,
                             artifacts=len(done['artifacts']))))
        return
    reg=C.read(OUT/'registration.json')
    verify(reg['protected']);verify(reg['source_hashes'])
    manifest=C.read(OUT/'eval_manifest.json')
    geo=C.read(OUT/'geometry_completion.json')['rows']
    scores=C.read(OUT/'score_completion.json')
    with (OUT/'per_output.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    bundle={(r['label'],r['request_id']):r for r in rows}
    identity=[]
    for q in manifest['requests']:
        rid=q['request_id'];rel=Path('evaluation/SEAM_SLOT_C200')/rid/'post/post.npz'
        old=C.arrays(CAP/rel);new=C.arrays(OUT/rel)
        same=all(np.array_equal(old[k],new[k]) for k in ('vertices','faces','source_face_ids','output'))
        identity.append(dict(request_id=rid,mesh_arrays_identical_to_capacity=same,
            file_bytes_identical_to_capacity=C.sha(CAP/rel)==C.sha(OUT/rel),
            changed=not same))
    assert sum(r['changed'] for r in identity)==1
    C.csvwrite(OUT/'capacity_geometry_identity.csv',identity)
    # These are actual assistant observations from two opened composite images.
    # No unviewed request is silently given a fresh visual-success label.
    review=[]
    gallery={r['request_id']:r for r in C.read(OUT/'gallery/manifest.json')['rows']}
    for rid in ('73b369b9_n3','3f741759_n2'):
        for method in ('ORIGINAL_GEO','CAPACITY_GUARD','TIP_REDIRECT'):
            if rid.startswith('73'):
                note=('Recognizable broad back/seat and lower continuous pedestal-like panel. '
                      'The side cut retains a gray lower layer separated below blue C; '
                      'no clear whole-shape or gap improvement distinguishable in these views. '
                      'The one new shared edge is a numeric topology finding, not a visually certified repair.')
            else:
                note=('Recognizable four-leg chair with slatted back and blue C at seat height. '
                      'Whole views and central cuts do not reveal the conflict-covered area reliably. '
                      'No visible improvement across columns; overlap tail remains a scorer finding, '
                      'not a new visual label.')
            review.append(dict(request_id=rid,method=method,reviewer='AI_PROXY',blind=False,
                status='ACTUALLY_VIEWED',image=gallery[rid]['image_path'],
                image_sha256=gallery[rid]['image_sha256'],views='two orthographic + two finite central cuts',
                whole_shape_change='NO_CLEAR_CHANGE',C_use_change='NO_CLEAR_CHANGE',
                connectivity_visual_certificate='UNKNOWN',observation=note))
    C.csvwrite(OUT/'gallery/actual_visual_review.csv',review)
    coverage=[]
    for row in identity:
        coverage.append(dict(**row,review_status='ACTUALLY_VIEWED_COMPOSITE' if row['request_id'] in gallery else
            'NO_NEW_VISUAL_JUDGMENT_IDENTICAL_CAPACITY_GEOMETRY',reviewer='AI_PROXY' if row['request_id'] in gallery else 'NA'))
    C.csvwrite(OUT/'gallery/review_coverage.csv',coverage)
    group=[]
    for label in ('ORIGINAL_GEO','CAPACITY_GUARD','TIP_REDIRECT'):
        for parent in sorted({r['parent_id'] for r in rows}):
            rr=[r for r in rows if r['label']==label and r['parent_id']==parent]
            rec=dict(label=label,parent_id=parent,n=len(rr))
            for field in ('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction',
                          'ConflictCov_C','free_only_reached_bulk_area_fraction','C_F_valid_edge_count',
                          'legal_seam_bulk_edge_count','deleted_free_faces'):
                rec[field+'_mean']=sum(float(r[field]) for r in rr)/len(rr)
            rec['bulk_nonzero']=sum(float(r['free_only_reached_bulk_area_fraction'])>0 for r in rr)
            rec['bulk_at_least_30pct']=sum(float(r['free_only_reached_bulk_area_fraction'])>=.3 for r in rr)
            group.append(rec)
    C.csvwrite(OUT/'per_parent.csv',group)
    counts={}
    for label in ('ORIGINAL_GEO','CAPACITY_GUARD','TIP_REDIRECT'):
        rr=[r for r in rows if r['label']==label]
        counts[label]=dict(bulk_nonzero=sum(float(r['free_only_reached_bulk_area_fraction'])>0 for r in rr),
            bulk_at_least_30pct=sum(float(r['free_only_reached_bulk_area_fraction'])>=.3 for r in rr),
            legal_CF_edges=sum(int(r['C_F_valid_edge_count']) for r in rr))
    timing=dict(geometry_sum_seconds=sum(r['seconds'] for r in geo),
        scoring_wall_seconds=scores['seconds'],geometry_workers=1,scoring_workers=2,threads_per_worker=1,
        concurrent_environment='Other parent experiment CPU work may overlap; not an isolated benchmark',
        rendering_tool_wall_seconds=1.92,model_calls=0,training_updates=0,new_noises=0,
        new_POST=16,new_full_quality_scans=16,
        initial_launch='WSL E_ACCESSDENIED before Python; elevated authorized invocation succeeded',
        retries_of_scientific_work=0,synthetic_checks=2,
        capacity_rejection_events=sum(r['capacity_rejections'] for r in geo),
        candidate_attempts=sum(r['candidate_attempts'] for r in geo),
        accepted=sum(r['accepted'] for r in geo),
        actual_guard_pair_tests=sum(r['guard_face_pair_tests'] for r in geo))
    C.write(OUT/'execution_ledger.json',timing)
    C.write(OUT/'coverage_counts.json',counts)
    report='''# Scheme 4: original-radius seam-tip redirection

Status: COMPLETE. This bounded CPU change did not improve overall quality. It added one legal seam–bulk shared edge on one of 16 requests, with no new illegal CF/FF pairs, deleted faces or nonmanifold edges relative to the capacity-guard reference. It did not improve CF, FF or I, and did not add a request with nonzero or at least 30% reachable bulk area.

## Frozen scope and actual method

The inputs are the existing 16 C-only PREDICTED SEAM_SLOT_C200 outputs, saved single-D0 PRE_GEO, and original capacity-guard rejection/mapping records. There were zero generator/D0 calls, zero updates, no new noise/query, no changed occupancy and no radius sweep. The original radius is 0.04197095609837992 model units.

Only a retained, non-hard bulk corner rejected by capacity protection can nominate the already active owner seam tip of that blocked C edge. The final actual capacity cluster, including all its input corners, is assigned to the existing tip representative. Every moved corner must lie within the original radius measured from its own PRE_GEO position. C and seam anchors and the destination cluster cannot move. This freezes the final capacity map; it is not a rerun or reordering of the original greedy GEO_C clustering.

All affected source faces are checked for new degeneracy, duplicates, deleted/revived identities and new/worse nonmanifold edges. Every affected retained face is compared with every retained face before/after using the original verified triangle classifier. New illegal CF/FF or unresolved classifications reject the edit. Exact shared-index decode and original source-winding export follow. The unchanged full scorer ran on all 16 endpoints.

This differs from R10 arbitrary terminal partner attraction and the partial-edge 1/2/3.5%L test: the destination is an existing occupied owner tip, the displacement remains within the original radius, and actual clusters/identities are preserved. These restrictions test one local decoder operation. They neither infer bulk intent nor establish a physical upper bound on other decoders.

## Paired results

| Metric, macro mean unless count | Original GEO | Capacity guard | Tip redirect |
|---|---:|---:|---:|
| CF affected free faces | 1.70357% | 1.66101% | 1.66101% |
| FF affected free faces | 14.56293% | 15.34847% | 15.34847% |
| I, frozen C-query conflict coverage | 3.30200% | 3.29895% | 3.29895% |
| Small-angle free faces | 18.09988% | 18.12708% | 18.12708% |
| Legal C/F shared edges, total | 175 | 192 | 192 |
| C/F nonmanifold edges, total | 18 | 0 | 0 |
| Free-only reachable bulk area | 49.24826% | 48.89648% | 48.90113% |
| Deleted free faces | 32.2500 | 31.9375 | 31.9375 |

The reachable-bulk metric uses the existing legal free-only graph seeded by seam faces attached legally to C. It excludes propagation through C. It is distinct from direction-consistent reachable total free area and does not certify a complete seam.

The sole accepted request is 73b369b9/n3: input corner 262 of source slot 87 is assigned to owner seam slot 41 tip corner 125. The actual PRE_GEO displacement is 0.041214196654182016, below the fixed radius. One affected face and 282 before/after pair tests passed. A second proposed corner of the same face would collapse it and was rejected.

On that request, legal seam–bulk edges increase 3→4; their total length increases 2.87828→6.53101 model units. Free-only reached bulk faces increase 19→20 and bulk reachable area 28.87880%→28.95322%, still below 30%. The macro increase is only 0.004651 percentage points. The overall direction-consistent free-area fraction changes 30.00431%→30.07654%; these are different denominators and must not be substituted for bulk reach. No new legal C/F edge is created.

Relative to the original GEO, the capacity guard's FF cost remains: +0.78554 percentage points. The redirection did not repair that cost. 3f/n2 retains its capacity I tail of 42.48047%; no route-wide success should be inferred from the one new edge.

## Rejection accounting

| Candidate disposition | Count |
|---|---:|
| Cluster exceeds original radius | 20 |
| New exact degeneracy | 2 |
| Accepted | 1 |
| Source face deleted by capacity | 1 |
| No occupied owner | 1 |
| Source is not non-hard bulk | 1 |

These are all logged attempts, including early ineligible events, not 26 independent requests. The full per-request candidate files retain actual displacement, cluster members, owner identity and every rejection. No candidate reached a new-CF/FF rejection after the earlier filters; this does not show that such conflicts are generally absent.

## Actual visual inspection and limits

The assistant opened the 73/n3 changed-request image and 3f/n2 tail-control image, each with three methods, two same-scale whole views and two finite central cuts. This is nonblind AI_PROXY inspection, not human review. No clear whole-shape or gap improvement is discernible. In 73/n3 a lower gray layer remains separated below blue C in the side cut. The 3f/n2 chair silhouette looks coherent, but these views do not reliably reveal its measured coplanar conflict. Neither appearance certifies watertightness or no overlap.

All other 15 tip-redirect endpoint arrays equal the capacity endpoint exactly; their byte/array evidence is in capacity_geometry_identity.csv. Only the two opened composites receive new visual observations. Unviewed identical cases have no fabricated fresh rating.

## Interpretation and recovery

Retain this as a safe but very limited local engineering result. It does not support adopting tip redirection as a quality upgrade or increasing its radius. The current constraints leave most destinations out of reach; the experiment does not demonstrate that bulk intended to connect to those tips. Keep the existing capacity and original GEO comparisons visible.

Outputs: per_output.csv (48 rows), paired_per_noise.csv, per_parent.csv, both source-identity edge/pair ledgers, per-request candidate ledgers, 16 indexed POSTs, original full scores, gallery/actual_visual_review.csv, execution_ledger.json and all input/source hashes in registration.json. The two rendered images are linked from gallery/index.html.

Recovery of this completed branch is verification-only:

```text
python -B -m r2.pilots.seam_quality_four_v1.redirection.closeout
```

It checks frozen sources, protected inputs and completed artifact hashes, then exits without re-scoring or model work. The original run --stage all entry is retained as execution provenance; do not use it to overwrite a completed score-timing receipt. No default, commit or push is changed by this branch.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    # Protect every completed artifact, including the reviewed image hashes.
    artifacts={str(p):C.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=completion}
    sources=dict(reg['source_hashes']);sources[str(Path(__file__))]=C.sha(Path(__file__))
    done=dict(status='COMPLETE',reviewer='AI_PROXY',actually_viewed_requests=2,
        mesh_identity_verified_requests=16,new_model_calls=0,new_quality_scans=16,
        protected=reg['protected'],source_hashes=sources,artifacts=artifacts,
        counts=counts,accounting=timing)
    C.write(completion,done)
    print(json.dumps(dict(status='COMPLETE',accounting=timing,counts=counts,artifacts=len(artifacts))))


if __name__=='__main__': main()
