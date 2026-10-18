"""File-only publication checks and support manifest; never runs geometry/models.

Run after the two explicit portable replays. Input receipts and meshes remain
local; only small validation receipts enter the archive. Historical source and
evidence files are not modified. Existing unequal receipt copies are refused.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def freeze_support():
    support = [p for p in HERE.iterdir() if p.is_file() and p.name not in
               ('support_manifest.json', 'source_manifest.json', 'evidence_manifest.json')]
    support += list((HERE / 'validation').rglob('*'))
    files = [dict(path=p.relative_to(HERE).as_posix(), sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(support) if p.is_file()]
    write(HERE / 'support_manifest.json', dict(status='FINAL', files=files,
        scope='Publication adapters, docs, build/check scripts and migration receipts; original science/evidence have separate manifests'))
    return files


def main():
    validation = HERE / 'validation'
    manifest = read(HERE / 'source_manifest.json')
    source = {r['archive_relative_path']: r['sha256'] for r in manifest['source_files']}
    cases = []
    for rid in ['608b46b0_n2', '73b369b9_n0']:
        original = ROOT / 'experiments/r2_hinge_v2_publication_20261018/replays' / rid / 'receipt.json'
        target = validation / (rid + '_replay_receipt.json')
        if not original.is_file():
            raise FileNotFoundError('Explicit migration receipt missing: ' + str(original))
        data = original.read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Refuse to overwrite different migration receipt')
        target.write_bytes(data)
        result = read(target)
        assert result['status'] == 'PASS' and result['reference_pairs_equal'] is True
        assert result['source_manifest_sha256'] == sha(HERE / 'source_manifest.json')
        assert all(x['byte_equal'] for x in result['reference_array_comparison'])
        assert all(source[x['path']] == x['sha256'] for x in result['scientific_modules'].values())
        timing = (validation / (rid + '_persistent_process_time.txt')).read_text()
        peak = int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', timing)[1])
        wall = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([^\n]+)', timing)[1].strip()
        cases.append(dict(request_id=rid, arrays_compared=len(result['reference_array_comparison']),
                          all_array_bytes_equal=True, pair_cache_equal=True,
                          process_peak_RSS_KiB=peak, process_wall_time=wall,
                          geometry_seconds=result['elapsed_seconds'],
                          science_modules=len(result['scientific_modules']),
                          original_inputs_unchanged=result['original_input_hashes_unchanged']))

    broken = []
    docs = [HERE / 'README.md', ROOT / 'r2/docs/denoising.md']
    for doc in docs:
        for link in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            path = link.split('#')[0].strip('<>')
            if not path or '://' in path or path.startswith('mailto:'):
                continue
            target = (doc.parent / unquote(path)).resolve()
            if target == validation / 'publication_checks.json':
                continue  # This receipt is written below in this same call.
            if not target.exists():
                broken.append(dict(document=str(doc), link=link))
    gallery = HERE / 'evidence/gallery/index.html'
    for link in re.findall(r'(?:src|href)=[\"\']([^\"\']+)', gallery.read_text(encoding='utf-8')):
        if '://' not in link and not link.startswith(('#', 'data:')) and not (gallery.parent / unquote(link)).exists():
            broken.append(dict(document=str(gallery), link=link))
    assert not broken, broken

    resources = read(ROOT / 'experiments/r2_hinge_v2_comparison_20261018/publication_resources_20261018.json')
    public_resources = {k: resources[k] for k in ('timestamp_utc', 'stage', 'gpu', 'logical_cores',
                        'cpu_load_percent', 'total_ram_KiB', 'available_ram_KiB', 'plan')}
    public_resources.update(active_GPU_process_count=len(resources['gpu_processes']),
                            process_detail='Recorded locally; Windows reports names/memory unavailable for some graphics processes',
                            measured_recheck_before_second_case='RTX5080: total16303MiB used2934MiB free13046MiB utilization4%',
                            measured_recheck_before_persistent_pair=dict(GPU_total_MiB=16303, GPU_used_MiB=2983,
                                GPU_free_MiB=12997, GPU_utilization_percent=19, CPU_load_percent=10, available_RAM_KiB=70829592),
                            initial_actual_workers=1, persistent_replay_workers=2, CPU_threads_per_worker=1, GPU_workers=0,
                            scheduling_reason='Initial cases were serial while measuring peak (~102MiB). WSL /tmp receipts did not persist; exactly the same two cases were replayed to the project drive with two independent one-thread workers after resource recheck, without new scientific samples.')
    write(validation / 'resource_dispatch.json', public_resources)
    write(validation / 'publication_checks.json', dict(status='PASS', cases=cases,
        environment=dict(python=sys.version, platform=platform.platform(),
                         packages={n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'trimesh', 'matplotlib', 'Pillow')}),
        markdown_and_gallery_links='PASS', model_calls=0, new_generation=0,
        tests_scope='Two fixed migration cases, each replayed twice because initial WSL temporary receipts did not persist; no expanded sample panel, metric rerun or new scientific result',
        source_manifest_sha256=sha(HERE / 'source_manifest.json'),
        evidence_manifest_sha256=sha(HERE / 'evidence_manifest.json')))

    sys.path.insert(0, str(HERE))
    import verify_archive
    freeze_support()
    result = verify_archive.run(check_import=True)
    assert result['status'] == 'PASS_ARCHIVE_HASHES', result
    write(validation / 'archive_hash_and_import.json', result)
    files = freeze_support()
    print(json.dumps(dict(status='PASS_PUBLICATION_FINALIZED', cases=cases, support_files=len(files)), indent=2))


if __name__ == '__main__':
    main()
