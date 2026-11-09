"""Two CPU-only source identity pages, using the unchanged prior raster/cuts.

No model, GEO, scoring or automatic visual labels. The five-stage source specs
and output root are supplied to the existing renderer in memory; no old file is
modified. A new render manifest hashes this wrapper and all source arrays.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
from pathlib import Path
from .identity_prepare import ROOT, OUT, DEST, OLD, read, resolve, sha, write_new, verify


def specs(task):
    row = task['identity_row']
    directory = resolve(row['output_dir'])
    return [('SOURCE_C_PLUS_Y', row['raw_path']),
            ('D0_FULL_PROPOSAL', str(directory/'proposal.npz')),
            ('RESTORED_C_PRE_GEO', str(directory/'pre_geo.npz')),
            ('SOURCE_GEO_REFERENCE', row['source_GEO_path']),
            ('SOURCE_D0_GEO', str(directory/'post.npz'))]


def run():
    done = DEST/'gallery/COMPLETE.json'
    if done.exists():
        result = read(done)
        verify(result['files'])
        return result
    status = read(DEST/'COMPLETE.json')
    assert status['status'] == 'COMPLETE', 'Identity proposals/geometry must finish first'
    registration = read(DEST/'registration.json')
    previous = read(OLD/'manifest.json')
    templates = {x['condition_id']: x for x in previous['tasks'] if x['kind']=='SOURCE_GEO'}
    tasks = []
    for row in registration['rows']:
        receipt = read(resolve(row['output_dir'])/'complete.json')
        verify(receipt['files'])
        template = templates[row['condition_id']]
        tasks.append(dict(template, job_id=row['condition_id']+'__SOURCE_IDENTITY',
                          label='SOURCE IDENTITY (not generated output)', identity_row=row))
    assert len(tasks) == 2
    paths = {str(resolve(p)): sha(p) for t in tasks for _,p in specs(t)}
    record = dict(schema='BENCH_D0_SOURCE_IDENTITY_RENDER_V1', tasks=tasks,
                  identity_registration_sha256=sha(DEST/'registration.json'),
                  identity_complete_sha256=sha(DEST/'COMPLETE.json'),
                  wrapper_sha256=sha(__file__), sources=paths,
                  stages=[s for s,_ in specs(tasks[0])],
                  protocol='Same original two views and two central sections; D0 proposal blue triangles are moved proposed C, not preserved input. Source is a compatibility reference, not unique generated target style.',
                  model_calls=0, GEO_calls=0, scoring_calls=0)
    manifest = DEST/'render_manifest.json'
    if manifest.exists():
        assert read(manifest) == record, 'Render registration changed'
    else:
        write_new(manifest, record)
    # These two renderer globals are explicit dependency injection, not edits to
    # sealed source. The original rendering and finite-plane math remain intact.
    from r2.pilots.bench_backend_counterfactual_v1 import render as renderer
    renderer.OUT = DEST
    renderer.stage_specs = specs
    tick = time.perf_counter()
    receipts = [renderer.one(t, sha(manifest)) for t in tasks]
    html = '<meta charset="utf-8"><h1>Two source D0 identity checks</h1><p>Source / proposed C+F / restored C / source GEO / D0 GEO. Blue in proposal is moved C; all pages use the inherited source frame and cuts. No automatic ratings.</p>'
    for receipt in receipts:
        rel = Path(receipt['path']).relative_to(DEST/'gallery').as_posix()
        html += '<h2>'+receipt['job_id']+'</h2><img style="width:100%" src="'+rel+'">'
    target = DEST/'gallery/index.html'
    with target.open('x',encoding='utf8') as f:
        f.write(html)
    result = dict(status='RENDERED_PENDING_ACTUAL_REVIEW', pages=2, cells=10,
                  images=receipts, seconds=time.perf_counter()-tick,
                  model_calls=0, GEO_calls=0, scoring_calls=0,
                  files={str(p):sha(p) for p in (DEST/'gallery').rglob('*') if p.is_file()})
    write_new(done, result)
    return result


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    result=run()
    print(json.dumps({k:v for k,v in result.items() if k not in ('images','files')}))
