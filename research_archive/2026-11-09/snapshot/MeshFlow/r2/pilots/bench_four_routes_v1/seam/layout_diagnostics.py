"""Read cached production layouts; source labels are retrospective diagnostics only."""
import csv
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import OUT, read, sha, resolve, write_new


def run():
    dest = OUT / 'layout_diagnostics'
    done = dest / 'complete.json'
    if done.exists():
        result = read(done)
        for name, digest in result['inputs'].items():
            assert sha(name) == digest, name
        for name, digest in result['files'].items():
            assert sha(name) == digest, name
        return result
    manifest = OUT / 'data/eval_manifest.json'
    inputs = {str(manifest): sha(manifest), str(Path(__file__)): sha(__file__)}
    unique = OrderedDict()
    for q in read(manifest)['requests']:
        unique.setdefault(q['condition_id'], q)
    assert len(unique) == 23
    rows, edges = [], []
    for cid, q in unique.items():
        folder = OUT / 'evaluation/S/layouts' / cid
        receipt_path = folder / 'receipt.json'
        receipt = read(receipt_path)
        assert receipt['status'] == 'COMPLETE'
        assert receipt['source_labels_read'] is False
        assert receipt['head']['oracle'] is False
        assert receipt['head']['threshold_logit'] == 0
        inputs[str(receipt_path)] = sha(receipt_path)
        for name, digest in receipt['files'].items():
            assert sha(name) == digest, name
            inputs[name] = digest
        case_path = resolve(q['case_path'])
        assert sha(case_path) == q['case_sha256']
        inputs[str(case_path)] = q['case_sha256']
        with np.load(folder/'layout.npz', allow_pickle=False) as z:
            active = z['occupied'].astype(bool)
            logits = z['occupancy_logits'].copy()
            records = z['edge_records'].copy()
            seam_slots = z['seam_slots'].copy()
            assert len(active) == receipt['boundary_edges']
            assert int(active.sum()) == receipt['active'] == len(seam_slots)
            assert np.array_equal(active, logits >= 0)
        with np.load(case_path, allow_pickle=False) as z:
            owner = z['occupied_labels'].astype(bool)
            alias = z['source_alias_labels'].astype(bool)
            supported = z['source_supported_labels'].astype(bool)
            assert np.array_equal(records, z['edge_records']), cid
        assert owner.shape == alias.shape == supported.shape == active.shape
        assert not np.any(owner & alias)
        assert np.array_equal(supported, owner | alias)
        null = ~supported
        closed = len(active) == 0
        assert receipt['closed_C_skipped'] == closed
        assert receipt['occupancy_forward_calls'] == int(not closed)
        row = dict(condition_id=cid,parent_id=q['parent_id'],report_group=q['report_group'],
                   N=q['N'],K=q['K'],boundary_edges=len(active),closed_C=closed,
                   active_slots=int(active.sum()),inactive_edges=int((~active).sum()),
                   source_owners=int(owner.sum()),source_aliases=int(alias.sum()),
                   source_nulls=int(null.sum()),active_source_owners=int((active&owner).sum()),
                   inactive_source_owners=int((~active&owner).sum()),
                   active_source_aliases=int((active&alias).sum()),active_source_nulls=int((active&null).sum()),
                   owner_recall=None if not owner.any() else float((active&owner).sum()/owner.sum()),
                   source_owner_applicability='NA_CLOSED_C' if closed else 'SOURCE_RECONSTRUCTION_DIAGNOSTIC',
                   head_forwards=receipt['occupancy_forward_calls'],
                   inference_uses_source_labels=False,source_diagnostic_uses_cached_labels=True)
        rows.append(row)
        for i, record in enumerate(records):
            edges.append(dict(condition_id=cid,parent_id=q['parent_id'],edge_index=i,
                              C_face_id=int(record[0]),C_vertex0=int(record[1]),C_vertex1=int(record[2]),
                              logit=float(logits[i]),active=bool(active[i]),
                              cached_source_role='OWNER' if owner[i] else 'ALIAS' if alias[i] else 'NULL',
                              source_label_used_by_production=False))
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, data in [('per_condition.csv', rows), ('per_edge.csv', edges)]:
        p = dest/name
        with p.open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
        files[str(p)] = sha(p)
    result = dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),
                  conditions=len(rows),closed_conditions=sum(r['closed_C'] for r in rows),
                  sums={k:sum(r[k] for r in rows) for k in ('boundary_edges','active_slots','source_owners',
                        'source_aliases','source_nulls','active_source_owners','inactive_source_owners',
                        'active_source_aliases','active_source_nulls','head_forwards')},
                  interpretation='Source-owner agreement is retrospective reconstruction metadata, not an open-completion correctness label. No source label was supplied to production inference; no head/model was called by this diagnostic.',
                  new_model_calls=0,new_scoring_calls=0,inputs=inputs,files=files)
    write_new(done,result)
    return result


if __name__ == '__main__':
    r=run()
    print(json.dumps({k:r[k] for k in ('status','conditions','closed_conditions','sums')},indent=2))
