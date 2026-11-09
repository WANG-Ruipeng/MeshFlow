"""Repair audit text encoding and record the explicit BACK-resume retention decision.

Metadata only: no checkpoint loads, model calls, deletions or scientific changes.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
path = OUT / 'chair_audit.json'
old = path.read_bytes()
d = json.loads(old.decode('utf-8-sig'))
before_rows = json.loads(json.dumps(d['rows']))
keep = 'E:/MeshFlow/experiments/r2_chair_multipart_seam_v1/training/checkpoint_step1500.pt'
found = [r for r in d['rows'] if r['path'] == keep]
assert len(found) == 1
row = found[0]
old_action = row['action']
assert old_action == 'RETIRE_OBSOLETE_WEIGHT'
backup = OUT / 'chair_audit.before_BACK_resume_retention.json'
with backup.open('xb') as f:
    f.write(old)
row['action'] = 'KEEP_DISTINCT_POSITIVE_RESUME'
row['reason'] = 'Root requested retaining full restore for the distinct BACK-positive MULTIPART_S1500 branch; preserve matching optimizer and RNG for an explicitly authorized future continuation.'

summary = {}
for action in sorted({r['action'] for r in d['rows']}):
    rows = [r for r in d['rows'] if r['action'] == action]
    inode_bytes = {}
    for r in rows:
        inode_bytes[(r['device'], r['inode'])] = r['bytes']
    summary[action] = dict(files=len(rows), logical_bytes=sum(r['bytes'] for r in rows),
        unique_listed_inode_bytes=sum(inode_bytes.values()),
        caveat='Not a guaranteed disk-free delta; root must inspect external hardlinks before deleting')
d['summary'] = summary
d['audit_revision'] = dict(reason='Explicit root retention decision and human-readable text encoding repair',
    prior_json_sha256=hashlib.sha256(old).hexdigest(), prior_json=backup.name,
    changed_recommendation=dict(path=keep, before=old_action, after=row['action']),
    scientific_data_changed=False)
d['maintained_entrypoint_dependency_review'] = {
    'native_t1_default': 'native_t1/__main__.py defaults to chair-hybrid; native_t1/working_model.py DEFAULT_WORKING_CHECKPOINT is CHAIR_START_CHECKPOINT.',
    'start_loader': 'native_t1/chair_checkpoint.py loads the complete START state directly. Ancestral source_checkpoint_sha256 is identity metadata, not a file dependency; no pretrained download is required.',
    'legacy_n112': 'fm-geo and edge5 are explicit legacy profiles. Their old weights are proposed for retirement, including fm_cumulative3000.pt and jedge5_cumulative3000.pt. WORKING_CHECKPOINT still aliases EDGE5 for compatibility, but is not the maintained default.',
    'r2_baseline': 'r2/src/meshflow_control/r2_cli.py sample requires explicit --checkpoint. Original R2 main/confirmation weights are proposed for retirement; source, recipe and results remain.',
    's2000': 'r2/pilots/chair_back_transfer_v1/worker.py loads only PLAIN_S2000_generator.pt, LONG_occupancy.pt and official chair_denoiser_0030000.pt. Generator construction is independent of prior model files.',
    'official': 'Official Chair generator and D0 are protected. All official/public pretrained assets outside this Chair audit also remain protected.',
    'remaining_small_files': 'Retain YAML/config, JSON identity sidecars, condition metadata and cached LONG layouts; cleanup must not retire these as if they were model checkpoints.',
    'checks_performed': 'Read source text and inventory only. No neural imports, checkpoint deserialization, model forward or GPU operation.'
}
for rel in ('native_t1/__main__.py', 'native_t1/working_model.py', 'native_t1/chair_checkpoint.py',
            'r2/src/meshflow_control/models/native.py', 'r2/src/meshflow_control/r2_cli.py',
            'r2/pilots/chair_back_transfer_v1/worker.py'):
    p = ROOT / rel
    d['source_documents'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
for a, b in zip(before_rows, d['rows']):
    for k in a:
        if a['path'] == keep and k in ('action', 'reason'):
            continue
        assert a[k] == b[k], (a['path'], k)
path.write_text(json.dumps(d, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')

lines = [
    '# Chair checkpoint retention audit', '',
    'Read-only asset audit, 2026-10-25. No model loads, GPU work, checkpoint deletions or scientific-data changes were performed. Root owns all cleanup mutations. Per-file identities and decisions are in `chair_audit.json`.', '',
    'Selected pipeline: **PLAIN_S2000 -> official Chair D0 -> GEO_C / source orientation -> HINGE_OPP015 -> optional ALPHA / HOLE**. The CPU geometry stages require no additional neural weights.', '',
    '## Required retained files', ''
]
for r in d['rows']:
    if r['action'] == 'KEEP_REQUIRED':
        lines += [f"- `{r['path']}` ({r['bytes']/2**30:.4f} GiB): {r['reason']}"]
lines += ['', '## Distinct BACK-positive branch', '',
    'MULTIPART_S1500 has a separate bounded BACK result; it is not a uniform replacement for S2000. Root requested retaining both its inference assets and full optimizer/RNG restore. This changes one retention recommendation, not any experiment result.', '']
for r in d['rows']:
    if r['action'].startswith('KEEP_DISTINCT_POSITIVE'):
        lines += [f"- `{r['path']}` ({r['bytes']/2**30:.4f} GiB): {r['reason']}"]
lines += ['', '## Obsolete checkpoint candidates', '',
    'The candidate set includes old SEAM200/FULL4, LONG generator, ABC, nonselected COUPLE/PLAIN endpoints, CLEAN/SMOOTH/SURGICAL/WISE, DEDUP, CREF/SMALL, USER/EXP67, original R2 main/confirmation, FULL1000/3000, R29/R30 final adapters and old Q/G or N112 weights. Preserve their source, data, RAW/POST outputs, metrics, images, reviews, reports and provenance. Retirement ends exact model replay or optimizer continuation; do not silently download or regenerate retired checkpoints.', '',
    'The 2.105 GB checkpoint in `.worktrees/a-layout-value-20260924/.../step00020000.pt` belongs to an abandoned fresh-random CLOSED16/N512 run. Its plan was 50,000 updates; the log reached about 23,000 and the saved checkpoint is at 20,000. It is **not an official pretrained model**, despite `official` appearing in the run directory name. Preserve the protocol and failure records.', '',
    '## Current dependency review', '']
for key, value in d['maintained_entrypoint_dependency_review'].items():
    lines += [f'- **{key}**: {value}']
lines += ['', '## Counts and storage accounting', '',
    '| Decision | Files | Logical GiB | Unique listed inode GiB |',
    '| --- | ---: | ---: | ---: |']
for action, s in summary.items():
    lines += [f"| {action} | {s['files']} | {s['logical_bytes']/2**30:.4f} | {s['unique_listed_inode_bytes']/2**30:.4f} |"]
lines += ['',
    'START canonical/original names are hardlinks to one allocation; retain both names. The historical R2 `latest.pt` and step6000 checkpoint are also hardlinks. Unique-inode totals only deduplicate the listed paths; actual free-space change depends on any additional hardlinks and filesystem accounting.', '',
    'The old N112 profiles and original R2 checkpoints are included among retirement candidates. They are explicit historical routes, not maintained default dependencies. Cleanup documentation must mark their local weight replay as retired. No default selection was changed by this audit.', '',
    'The original machine-readable audit is preserved as `chair_audit.before_BACK_resume_retention.json`. All size, identity and scientific evidence fields remain unchanged. The revised Markdown is ASCII-compatible UTF-8 to avoid terminal code-page corruption.', '']
text = '\n'.join(lines)
assert text.isascii()
(OUT / 'chair_audit.md').write_text(text, encoding='utf-8')
assert '???' not in (OUT / 'chair_audit.md').read_text(encoding='utf-8')
print(json.dumps({'status':'PASS', 'recommendations':summary,
    'json_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'markdown_sha256':hashlib.sha256((OUT/'chair_audit.md').read_bytes()).hexdigest()}))
