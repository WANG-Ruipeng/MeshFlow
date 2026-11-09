"""Inventory final local deliverables without invoking experiment code."""
import difflib
import hashlib
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[2]
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

assert read(OUT/'RUN_COMPLETE.json')['status'] == 'DONE'
assert read(OUT/'independent_audit/audit_result.json')['status'] == 'PASS'
assert read(OUT/'independent_audit/results_arithmetic_audit.json')['status'] == 'PASS'
for p in (OUT/'controller').glob('*.json'):
    assert read(p)['status'] == 'DONE', p
frozen = read(OUT/'registration/implementation_hashes.json')
for rel, value in frozen.items():
    assert sha(ROOT/rel) == value, rel
for target in re.findall(r'\]\(([^)]+)\)', (OUT/'DELIVERY_INDEX.md').read_text(encoding='utf-8')):
    assert (OUT/target).is_file(), target

scripts = sorted((OUT/'independent_audit').glob('*.py')) + sorted((OUT/'reporting_corrections').glob('*.py'))
diff = []
for p in scripts:
    diff.extend(difflib.unified_diff([], p.read_text(encoding='utf-8').splitlines(True),
                 fromfile='/dev/null', tofile='b/'+p.relative_to(ROOT).as_posix()))
(OUT/'supplemental_code.diff').write_text(''.join(diff), encoding='utf-8')

selected = set(p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.md','.json','.diff','.sha256'))
selected.discard(OUT/'delivery_manifest.json')
for folder in ('registration','cpu_tests','tests','paired_results','diagnostic_results','independent_audit','reporting_corrections','resources','stages','controller'):
    selected.update(p for p in (OUT/folder).glob('*') if p.is_file() and
                    p.suffix in ('.md','.json','.csv','.diff','.py','.png','.sha256'))
selected.update(p for p in (OUT/'training').glob('*') if p.suffix in ('.csv','.json'))
selected.update(p for p in (OUT/'checkpoints').glob('*.json'))
manifest = dict(status='COMPLETE', model_calls_in_packaging=0,
    final_report_sha256=sha(OUT/'CHAIR_HYBRID_COUPLING_REPORT.md'),
    frozen_implementation_verified=len(frozen), frozen_implementation=frozen,
    supplemental_source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in scripts},
    artifacts=[dict(path=p.relative_to(OUT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(selected)],
    checkpoint_manifest=read(OUT/'checkpoint_manifest.json'),
    exclusion_note='Large PT/raw/OT caches are not duplicated; actual PT hashes were independently checked. Raw/query/score hashes remain in per-output and generation manifests. Audit prior versions are retained in their revision directories.',
    no_extra_archive=True, no_running_experiment=True)
(OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(dict(status='COMPLETE', artifacts=len(manifest['artifacts']),
                     frozen_code_files=len(frozen), supplemental_scripts=len(scripts),
                     report_sha256=manifest['final_report_sha256'])))
