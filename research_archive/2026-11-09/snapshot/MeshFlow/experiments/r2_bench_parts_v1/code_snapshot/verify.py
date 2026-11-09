"""Finite CPU regression of the new quality contract against old saved inputs."""
import json
from pathlib import Path
import time
import numpy as np
from .quality import SourceAuditor, ROOT, local_path, sha, write_new, topology


def main():
    started = time.perf_counter()
    manifest_path = ROOT / 'experiments/r2_bench_control_transfer_v1/data/manifest.json'
    data = json.loads(manifest_path.read_text('utf-8'))
    records = data['train'][:4] + data['dev']
    results = []
    for row in records:
        path = local_path(row['case_path'])
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
        audit = SourceAuditor(arrays, sha(path))
        report, sup = audit.candidate(arrays['known_ids'], dict(candidate_id=row['condition_id'], **{k: row[k] for k in ('parent_id', 'group')}))
        for field in ('occupied_labels', 'edge_owner_face', 'edge_source_triple', 'source_alias_labels',
                      'source_supported_labels', 'C_vertices', 'C_faces_original', 'C_faces_model', 'known_ids', 'free_ids'):
            if not np.array_equal(sup[field], arrays[field]):
                raise AssertionError((row['parent_id'], field))
        assert report['geometry_status'] == 'PASS_PENDING_SEMANTIC_REVIEW', (row['parent_id'], report['reasons'])
        assert report['source_owner_count'] == row['source_owner_count']
        results.append(dict(parent_id=row['parent_id'], group=row['group'], K=row['K'],
            source_sha256=sha(path), owner=report['source_owner_count'], alias=report['source_alias_count'],
            status='PASS_EXACT_ARRAY_AND_LABEL_REGRESSION', calls=dict(audit.calls)))
    # Topological unit checks are analytical and do not generate experiment meshes.
    tetra = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    t = topology(tetra)
    assert not t['open_edges'] and not t['nonmanifold_vertices'] and not t['same_winding_double_edges']
    bowtie = topology(np.asarray([[0, 1, 2], [0, 3, 4]]))
    assert any(row['vertex_id'] == 0 for row in bowtie['nonmanifold_vertices'])
    assert len(bowtie['edge_connected_components']) == 2
    out = ROOT / 'experiments/r2_bench_parts_v1/quality_validation/receipt.json'
    write_new(out, dict(status='PASS', original_manifest_sha256=sha(manifest_path),
        implementation_sha256=sha(Path(__file__).with_name('quality.py')), cases=results,
        analytical_topology_checks=['Closed tetrahedron accepted without open boundary', 'Vertex-only bowtie identified'],
        model_calls=0, GPU_calls=0, source_modifications=0, threads=1,
        invocation_note='An initial inline python command failed at shell quoting before module import; no scan/model call occurred; replaced with this file entry.',
        wall_seconds=time.perf_counter() - started))
    print(json.dumps(dict(status='PASS', cases=len(results), receipt=str(out))))


if __name__ == '__main__':
    main()
