"""Display-only revision. Reads frozen outputs; no model, OT or scoring calls."""
from pathlib import Path
import argparse
import time
import matplotlib
matplotlib.use('Agg')
from experiments.mf_h1_local0.chair_domain_control_v1 import fixed_figures as ff
from experiments.mf_h1_local0.chair_domain_control_v1.common import read_json, digest, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    assert read_json(out/'run_state.json')['status'] == 'DONE'
    dest = out/'figures/layout_v2'
    manifest_path = dest/'manifest.json'
    if manifest_path.exists():
        previous = read_json(manifest_path)
        for row in previous['figures']:
            assert digest(out/row['path']) == row['sha256']
        print('Verified cached display revision', len(previous['figures']))
        return
    panels = read_json(out/'panels.json')
    original = read_json(out/'figures/fixed_manifest.json')
    source = {row['case_ids'][0]: row for row in original['figures'] if row['table'] in ('A','C')}
    index = {}
    for arm in ff.ARMS:
        ledger = read_json(out/'evaluation/phase3000'/arm/'ledger.json')
        assert ledger['status'] == 'PASS'
        for row in ledger['rows']:
            index[(arm, row['case_id'])] = row
    save_original = ff._save
    def save_layout(fig, path):
        if fig.axes:
            fig.subplots_adjust(top=.88)
        save_original(fig, path)
    ff._save = save_layout
    rows = []
    start = time.perf_counter()
    for table in ('A','C'):
        for record in panels[table]:
            if record['noise_index'] != 0:
                continue
            prior = source[record['case_id']]
            assert digest(out/prior['path']) == prior['sha256']
            path = dest/table/(record['case_id']+'.png')
            details = ff._context_figure(ff._load_case(out, record, index), path)
            for key, value in details.items():
                assert prior[key] == value, (record['case_id'], key)
            rows.append(dict(table=table, path=str(path.relative_to(out)), sha256=digest(path),
                             original_path=prior['path'], original_sha256=prior['sha256'], **details))
    assert len(rows) == 48
    atomic_json(manifest_path, dict(status='PASS', description='A/C title clearance only: subplot top .94 to .88',
        elapsed_seconds=time.perf_counter()-start, script_sha256=digest(Path(__file__)),
        original_manifest_sha256=digest(out/'figures/fixed_manifest.json'),
        camera_extent_and_dependencies_identical=True, model_forwards=0, optimizer_updates=0,
        OT_calls=0, independent_geometry_score_calls=0, original_figures_unchanged=True, figures=rows))
    print('Completed display-only revision:', len(rows))


if __name__ == '__main__':
    main()
