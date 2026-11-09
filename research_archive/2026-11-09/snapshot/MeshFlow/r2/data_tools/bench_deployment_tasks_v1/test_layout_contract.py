"""CPU contract checks and exact production-version table selection; no OT/model."""
import collections
import csv
import json
import numpy as np
from . import layout_contract as A


def run():
    rows=list(csv.DictReader((A.OUT/'per_condition_layout.csv').open(encoding='utf-8-sig')))
    q={r['condition_id']:r for r in A.read(A.PANEL)['requests']}
    production=[r for r in rows if r['split']=='FIXED_EVAL' and r['cache_sha256']==q[r['condition_id']]['old_layout_sha256']]
    assert len(production)==24 and len({r['condition_id'] for r in production})==24
    A.write_csv(A.OUT/'actual_production_panel.csv',production)
    manifest=A.read(A.MANIFEST);train=manifest['train'];checks=[]
    # Turning off a true owner must return that same face to bulk, without
    # changing N or deleting the difficult interface target.
    case=A.arrays(train[0]['case_path']);own=case['occupied_labels'].copy()
    ei=int(np.flatnonzero(own)[0]);fi=int(case['edge_owner_face'][ei]);own[ei]=False
    pack,detail=A.compatible_endpoint(case,own)
    assert detail['status']=='EXACT_TARGET_BIJECTION_PASS'
    lay=A.layout(case,own);assert fi in pack['source_face_ids'][lay['bulk_slots']]
    checks.append(dict(test='INACTIVE_OWNER_RETURNS_ORIGINAL_FACE_TO_BULK',status='PASS'))
    # An alias may replace its owner's edge, but both cannot instantiate one Y face.
    aliasrow=next(r for r in train if A.arrays(r['case_path'])['source_alias_labels'].any())
    case=A.arrays(aliasrow['case_path']);alias=int(np.flatnonzero(case['source_alias_labels'])[0])
    owner=int(np.flatnonzero(case['occupied_labels'] & (case['edge_owner_face']==case['edge_owner_face'][alias]))[0])
    mask=case['occupied_labels'].copy();mask[owner]=False;mask[alias]=True
    _,detail=A.compatible_endpoint(case,mask);assert detail['status']=='EXACT_TARGET_BIJECTION_PASS'
    cmp,_,_=A.compare(case,mask);assert cmp['layout_comparison']=='ROLE_ANCHOR_DIFFERENCE'
    mask[owner]=True
    _,detail=A.compatible_endpoint(case,mask);assert detail['duplicated_source_face_edges']
    checks.append(dict(test='ALIAS_REPLACEMENT_PASS_BUT_DOUBLE_INSTANTIATION_REJECT',status='PASS',condition_id=aliasrow['condition_id']))
    nullrow=next(r for r in train if (~A.arrays(r['case_path'])['source_supported_labels']).any())
    case=A.arrays(nullrow['case_path']);mask=case['occupied_labels'].copy()
    mask[int(np.flatnonzero(~case['source_supported_labels'])[0])]=True
    _,detail=A.compatible_endpoint(case,mask);assert detail['active_unsupported_edges']
    checks.append(dict(test='ACTIVE_NULL_REJECTS_LEGAL_OWNER_TARGET_PACKING',status='PASS',condition_id=nullrow['condition_id']))
    # Independent combinatorial check: can any original Y face (not just the
    # previously legal owner labels) supply both requested source endpoint IDs?
    # This distinguishes a stricter legal-source policy failure from literal
    # fixed-anchor/source-face representability.
    exact=[]
    for r in rows:
        if r['deployment_endpoint_status']!='BLOCKED_NONREPRESENTABLE_LEGAL_SOURCE_LAYOUT':continue
        case=A.arrays(r['case_path']);mask=A.arrays(r['cache_path'])['occupied']
        candidates={}
        for e in np.flatnonzero(mask):
            u,v=case['edge_records'][e,1:];su,sv=case['C_source_vertex_ids'][[u,v]]
            candidates[int(e)]=[int(fi) for fi in case['free_ids'] if su in case['source_face_vertex_ids'][fi] and sv in case['source_face_vertex_ids'][fi]]
        match={}
        def visit(e,seen):
            for fi in candidates[e]:
                if fi in seen:continue
                seen.add(fi)
                if fi not in match or visit(match[fi],seen):match[fi]=e;return True
            return False
        for e in sorted(candidates):visit(e,set())
        exact.append(dict(split=r['split'],condition_id=r['condition_id'],variant=r['variant'],
            active_edges=len(candidates),matched_unique_source_faces=len(match),
            zero_candidate_edges=';'.join(str(e) for e in candidates if not candidates[e]),
            exact_source_identity_endpoint_possible=len(match)==len(candidates)))
    A.write_csv(A.OUT/'blocked_exact_source_incidence.csv',exact)
    report=dict(status='PASS',checks=checks,production_conditions=24,
        production_layout_counts=dict(collections.Counter(r['layout_comparison'] for r in production)),
        production_endpoint_counts=dict(collections.Counter(r['deployment_endpoint_status'] for r in production)),
        production_group_counts={g:dict(collections.Counter(r['deployment_endpoint_status'] for r in production if q[r['condition_id']]['group']==g)) for g in ('TRAIN','DEV')},
        production_edge_totals={k:sum(int(r[k]) for r in production) for k in ('boundary_edges','source_owners','predicted_active_count','changed_edges','active_null_edges')},
        broader_identity_matching_counts=dict(collections.Counter(str(r['exact_source_identity_endpoint_possible']) for r in exact)),
        neural_forwards=0,new_OT=0,GPU_calls=0,
        source_hashes={str(A.MANIFEST):A.sha(A.MANIFEST),str(A.PANEL):A.sha(A.PANEL),
                      str(A.OUT/'per_condition_layout.csv'):A.sha(A.OUT/'per_condition_layout.csv'),
                      str(A.Path(__file__)):A.sha(__file__),str(A.Path(A.__file__)):A.sha(A.__file__)})
    A.write_json(A.OUT/'contract_checks.json',report)
    return report


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
