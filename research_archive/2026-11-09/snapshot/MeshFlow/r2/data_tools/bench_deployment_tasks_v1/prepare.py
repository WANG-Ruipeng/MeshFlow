"""Record resources and the finite local clean-source supply for this data task."""
import json
from pathlib import Path
import importlib.metadata
from r2.data_tools.bench_user_v2 import build as b
from r2.pilots.bench_adaptation_strength_v1.common import resources

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_deployment_task_data_v1'
OLD = ROOT / 'experiments/r2_bench_user_data_v2_expansion_v1'


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    current = b.read(ROOT/'experiments/r2_bench_user_relation_rebuild_v1/proposed_merged_manifest.json')
    train = current['train']
    used = {r['parent_id'] for r in train}
    rows = [r for r in b.read(OLD/'high_source/audited.json')['rows'] if r['source_finite_pass'] and r['parent_id'] not in used]
    dependency_versions = {}
    for name in ('numpy','scipy','trimesh','shapely','fast-simplification','mapbox-earcut'):
        try: dependency_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: dependency_versions[name] = 'NOT_IN_MAIN_ENV'
    protected = {}
    for r in rows:
        for field in ('source_case_path','clean_case_path','source_geometry_path','source_lineage_path'):
            path = b.resolve(r[field]); protected[str(path)] = b.sha(path)
    value = dict(status='CPU_DATA_PREPARATION',resources=resources(),
        execution_scope='Data, source geometry and C-only contract checks only; no neural forward, training, D0 or new generated model outputs.',
        dispatch='Three independent read-only audit agents; later geometry workers bounded and recorded separately. GPU workers=0.',
        current_parents=len(used),current_conditions=len(train),clean_high_source_candidates=rows,
        dependency_versions=dependency_versions,protected_source_hashes=protected,
        raw_assets_readonly=True,old_DEV_regression_readonly=True)
    b.save(OUT/'preparation.json',value)
    print(json.dumps(dict(status=value['status'],current_parents=len(used),current_conditions=len(train),
        candidate_parents=len(rows),dependency_versions=dependency_versions,resources=value['resources']),ensure_ascii=False))
    return value


if __name__=='__main__':run()
