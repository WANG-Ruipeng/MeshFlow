"""CPU-only final evidence archive; no model/scorer imports or scientific edits.

Default --status is read-only. --build refuses a COMPLETE archive until the actual
6000-update run, all six frozen-model panels, final analyses, report, galleries,
and 192 main-panel AI_PROXY reviews are complete. Large assets remain local.
"""
from __future__ import annotations
import argparse
import csv
import difflib
import hashlib
import html
import io
import os
import json
from pathlib import Path
import re
import subprocess
import zipfile
from . import common as C

VERSION = 'full_seat_learning_curve_artifact_archive_v2_audited_cache_inventory'
MAIN = tuple(C.EVALUATION_LABELS)
REFERENCES = ('SMALL_C200', 'CREF_C200')
LABELS = MAIN + REFERENCES
EV = C.OUT / 'evaluation'
DEST = C.OUT / 'archive'
REVIEW = EV / 'review/AI_PROXY_reviews.csv'
MAX_BYTES = 10_000_000
BOUNDARY_COMPONENTS = 'evaluation/boundary_components_v1'
LATEST_PACKAGED = ('training_analysis', 'evaluation/comparison', BOUNDARY_COMPONENTS, 'evaluation/result_plots')
SOURCE_SUFFIXES = {'.py', '.ps1', '.md', '.toml', '.yaml', '.yml', '.json'}
HEAVY_SUFFIXES = {'.pt', '.pth', '.ckpt', '.bin', '.safetensors'}


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def _csv_bytes(rows, fallback=('status',)):
    rows = list(rows)
    keys = list(dict.fromkeys(k for row in rows for k in row)) or list(fallback)
    f = io.StringIO(newline='')
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
    return f.getvalue().encode('utf-8-sig')


def _rows(path):
    with C.resolve(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _canonical(value):
    # Lexical absolute paths avoid repeated cross-filesystem lstat; file bytes are
    # still hashed, and archive names are checked separately against traversal.
    return Path(os.path.abspath(str(C.resolve(value))))


def _path(value):
    try:
        return _canonical(value)
    except (OSError, ValueError, TypeError):
        return None


def _latest(folder):
    pointer = C.OUT / folder / 'latest.json'
    if not pointer.is_file():
        return None, None
    p = C.read(pointer)
    target = C.resolve(p['path'])
    if not target.is_file() or C.sha(target) != p['sha256']:
        raise ValueError('Latest snapshot hash differs: ' + str(pointer))
    return target, C.read(target)


def readiness():
    """Read-only evidence gates; no hashes of big weights and no filesystem writes."""
    missing, details = [], {}
    regpath = C.OUT / 'registration.json'
    if not regpath.is_file():
        return dict(status='PREPARED',missing=['registration.json'],version=VERSION)
    reg = C.read(regpath)
    requests = [q['request_id'] for q in reg['requests']]
    if len(requests) != 24 or len(set(requests)) != 24 or tuple(reg['evaluation_labels']) != MAIN:
        raise ValueError('Archive scope differs from the fixed 24-request/four-checkpoint panel')
    expected = {(label, request, stage) for label in MAIN for request in requests for stage in ('RAW', 'POST')}
    train = C.OUT / 'training/FULL_SEAT/status.json'
    training = C.read(train) if train.is_file() else {}
    details['training'] = {k: training.get(k) for k in ('status','local_step','global_step')}
    if training.get('status') != 'COMPLETE' or training.get('local_step') != 6000 or training.get('global_step') != 12000:
        missing.append('TRAINING_COMPLETE_LOCAL6000_GLOBAL12000')
    for step in (1000,3000,6000):
        weight = C.OUT / 'checkpoints' / ('FULL_SEAT_step' + str(step) + '.pt')
        receipt = weight.with_suffix('.json')
        if not weight.is_file() or not receipt.is_file(): missing.append('RETAINED_FULL_CHECKPOINT_' + str(step))
    for label in MAIN:
        binding = EV / 'bindings' / (label + '.json')
        value = C.read(binding) if binding.is_file() else {}
        if not value.get('path') or not C.resolve(value['path']).is_file(): missing.append(label + ':RETAINED_BOUND_GENERATOR')
    for label in LABELS:
        target = {(request, stage) for request in requests for stage in ('RAW','POST')}
        table = EV / 'tables' / label / 'per_output.csv'
        rows = _rows(table) if table.is_file() else []
        ids = [(r['request_id'], r['stage']) for r in rows]
        if len(rows) != 48 or len(set(ids)) != 48 or set(ids) != target or any(r['label'] != label for r in rows):
            missing.append(label + ':48_PER_OUTPUT_CELLS')
        for name in ('score_completion', 'boundary_completion'):
            p = EV / name / (label + '.json')
            x = C.read(p) if p.is_file() else {}
            if x.get('status') != 'COMPLETE' or len(x.get('receipts', [])) != 48:
                missing.append(label + ':' + name)
        for stage in ('raw','post'):
            p = EV / 'indices' / label / (stage + '_index.json')
            x = C.read(p) if p.is_file() else {}
            identities = [r.get('request_id') for r in x.get('rows', [])]
            if x.get('status') != 'COMPLETE' or len(identities) != 24 or set(identities) != set(requests):
                missing.append(label + ':' + stage + '_index')
    audit_path = C.OUT / 'audits/final/final_execution_audit.json'
    audit = C.read(audit_path) if audit_path.is_file() else {}
    if not str(audit.get('status','')).startswith('PASS') or audit.get('blocking_issues') != []:
        missing.append('FINAL_INDEPENDENT_EXECUTION_AUDIT_PASS')
    details['final_execution_audit'] = dict(path=str(audit_path), status=audit.get('status'))
    review_summary_path, review_summary = _latest('evaluation/review/summary')
    if review_summary is None or review_summary.get('status') != 'COMPLETE':
        missing.append('COMPLETE_ACTUAL_REVIEW_SUMMARY')
    report = C.OUT / 'REPORT.md'
    if not report.is_file() or len(report.read_text(encoding='utf-8-sig').strip()) < 100:
        missing.append('ACTUAL_REPORT_MD')
    rows = _rows(REVIEW) if REVIEW.is_file() else []
    seen, invalid = set(), []
    for i, row in enumerate(rows, 2):
        key = (row.get('label'), row.get('request_id'), row.get('stage'))
        why = []
        if key in seen: why.append('duplicate_identity')
        seen.add(key)
        if key not in expected: why.append('unexpected_identity')
        kind = row.get('reviewer_kind', row.get('review_kind', row.get('reviewer','')))
        if kind != 'AI_PROXY': why.append('not_AI_PROXY')
        if row.get('review_status') not in ('REVIEWED','COMPLETE'): why.append('not_reviewed')
        if row.get('C_adoption') not in ('YES','NO','UNCERTAIN'): why.append('no_actual_C_adoption')
        if not str(row.get('notes','')).strip(): why.append('no_observation_notes')
        if not str(row.get('reviewer','')).strip(): why.append('reviewer_missing')
        if str(row.get('blinded','')).lower() not in ('false','0'): why.append('blinding_identity_missing')
        for prefix in ('image', 'section_image'):
            p = C.resolve(row.get(prefix + '_path','__MISSING__'))
            h = row.get(prefix + '_sha256', row.get(prefix + '_hash',''))
            if not p.is_file() or not re.fullmatch(r'[0-9a-fA-F]{64}', h):
                why.append(prefix + '_evidence_missing')
        if why: invalid.append(dict(csv_line=i,reasons=why))
    details['review'] = dict(rows=len(rows),expected=192,unique_identities=len(seen),invalid=invalid)
    if len(rows) != 192 or seen != expected or invalid: missing.append('192_ACTUAL_MAIN_AI_PROXY_REVIEWS')
    for folder in ('training_analysis','evaluation/comparison','evaluation/result_plots'):
        p, value = _latest(folder)
        if value is None or value.get('status') != 'COMPLETE': missing.append(folder + ':COMPLETE_LATEST')
        details[folder] = dict(path=str(p) if p else None,status=value.get('status') if value else None)
    component_path, component = _latest(BOUNDARY_COMPONENTS)
    component_detail = dict(path=str(component_path) if component_path else None,
                            status=component.get('status') if component else None,
                            required_main_cells=192, verified_main_cells=0)
    if component is None or component.get('status') != 'COMPLETE':
        missing.append(BOUNDARY_COMPONENTS + ':COMPLETE_MAIN192_LATEST')
    else:
        allowed_labels = set(MAIN + REFERENCES)
        component_labels = set(component.get('labels', []))
        if not set(MAIN).issubset(component_labels) or not component_labels.issubset(allowed_labels):
            raise ValueError('Boundary-component snapshot labels differ from registered model scope')
        for path, digest in component['files'].items():
            if C.sha(path) != digest: raise ValueError('Boundary-component table hash differs: ' + path)
        if C.sha(component['definition_path']) != component['definition_sha256']:
            raise ValueError('Boundary-component derived definition hash differs')
        expected_registration = component['input_hashes'].get(str(C.OUT / 'registration.json'))
        if expected_registration != C.sha(regpath):
            raise ValueError('Boundary-component snapshot belongs to a different registration')
        table_paths = [p for p in component['files'] if Path(p).name == 'per_output.csv']
        if len(table_paths) != 1: raise ValueError('Exactly one complete boundary-component per_output table required')
        component_rows = _rows(table_paths[0])
        main_rows = [row for row in component_rows if row.get('label') in MAIN]
        main_ids = [(row.get('label'), row.get('request_id'), row.get('stage')) for row in main_rows]
        if (len(main_rows) != 192 or len(set(main_ids)) != 192 or set(main_ids) != expected
                or any(row.get('status') != 'AVAILABLE' for row in main_rows)
                or component.get('available_cells') != component.get('expected_cells')
                or component.get('pending_cells') != 0 or len(component_rows) != component.get('expected_cells')):
            missing.append(BOUNDARY_COMPONENTS + ':COMPLETE_MAIN192_IDENTITIES')
        else:
            component_detail['verified_main_cells'] = 192
        component_detail.update(expected_cells=component.get('expected_cells'),
                                available_cells=component.get('available_cells'),
                                labels=component.get('labels'))
    details[BOUNDARY_COMPONENTS] = component_detail
    for name, status in (('gallery','COMPLETE'),('sections_gallery','COMPLETE_RENDERING_NOT_REVIEW')):
        p = EV / name / 'manifest.json'
        x = C.read(p) if p.is_file() else {}
        ids = [v['request_id'] for v in x.get('images',[])]
        if x.get('status') != status or len(ids) != 24 or set(ids) != set(requests): missing.append(name + ':24_FIXED_REQUESTS')
    return dict(status='READY_TO_ARCHIVE' if not missing else 'PREPARED',version=VERSION,missing=missing,details=details,
                registration_sha256=C.sha(regpath),model_calls=0,additional_updates=0,new_image_reviews=0)


def _git(*args):
    x = subprocess.run(['git', '-C', str(C.ROOT), *args],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
    if x.returncode: raise RuntimeError('Read-only git command failed: ' + x.stderr)
    return x.stdout


def _code_diff(codefiles):
    """Tracked HEAD diff plus actual additions for untracked new experiment files."""
    relative = C.CODE.relative_to(C.ROOT).as_posix()
    tracked = set(_git('ls-files','--',relative).splitlines())
    result = _git('diff','--no-ext-diff','--binary','HEAD','--',relative)
    for p in codefiles:
        rel = p.relative_to(C.ROOT).as_posix()
        if rel not in tracked:
            lines = p.read_text(encoding='utf-8-sig').splitlines(keepends=True)
            result += 'diff --git a/' + rel + ' b/' + rel + '\nnew file mode 100644\n'
            result += ''.join(difflib.unified_diff([],lines,fromfile='/dev/null',tofile='b/'+rel))
            if lines and not lines[-1].endswith('\n'): result += '\n\\ No newline at end of file\n'
    return result.encode('utf-8'), dict(head=_git('rev-parse','HEAD').strip(),
        status=_git('status','--porcelain=v1','--untracked-files=all','--',relative),
        scope=relative,includes_actual_untracked_contents=True,no_commit_or_push=True)


def build():
    ready = readiness()
    if ready['status'] != 'READY_TO_ARCHIVE':
        # A premature invocation remains a plan, never a purported finished package.
        C.write(DEST/'prepared.json',ready,replace=True)
        return ready
    DEST.mkdir(parents=True, exist_ok=True)
    entries, origins, known, source_paths = {}, {}, {}, set()
    declared_hashes = []
    authoritative = {}
    reg = C.read(C.OUT / 'registration.json')
    for key in ('training_code_hashes', 'evaluation_code_hashes'):
        authoritative.update({str(_canonical(p)): h for p, h in reg[key].items()})
    for label in LABELS:
        value = C.read(EV / 'score_completion' / (label + '.json'))
        authoritative.update({str(_canonical(p)): h for p, h in value['definition_hashes'].items()})
        for receipt in C.read(EV / 'boundary_completion' / (label + '.json'))['receipts']:
            authoritative.update({str(_canonical(p)): h for p, h in receipt['identity']['definition_hashes'].items()})
    metadata = []
    def add_bytes(name, payload, origin=None):
        if name.startswith('/') or '..' in Path(name).parts: raise ValueError('Unsafe archive name')
        if name in entries and entries[name] != payload: raise ValueError('Archive-name collision: ' + name)
        entries[name] = payload
        if origin: origins[str(_canonical(origin))] = name
    def add(path, name=None):
        p = _canonical(path)
        if not p.is_file(): raise FileNotFoundError(p)
        if name is None: name = 'evidence/' + p.relative_to(C.OUT).as_posix()
        add_bytes(name,p.read_bytes(),p)
    def remember(p,h,receipt):
        p = _path(p)
        if p is None or not isinstance(h,str) or not re.fullmatch(r'[0-9a-fA-F]{64}',h): return
        old = known.get(str(p))
        declared_hashes.append(dict(path=str(p), sha256=h.lower(), receipt=str(receipt)))
        if old and old['sha256'] != h.lower():
            if p.suffix.lower() in HEAVY_SUFFIXES: raise ValueError('Conflicting retained weight file hashes: ' + str(p))
            # Mutable logs and preserved pre-fix definitions may have several real
            # historical versions. Preserve every declaration; verify the final
            # authoritative definition separately instead of rewriting history.
            old['multiple_versions'] = True
        elif old is None:
            known[str(p)] = dict(sha256=h.lower(),receipt=str(receipt),multiple_versions=False)
        if p.suffix.lower() in SOURCE_SUFFIXES and ('/r2/' in p.as_posix() or '/src/' in p.as_posix() or '/tools/' in p.as_posix()):
            if p.suffix.lower() in {'.py','.ps1'}: source_paths.add(p)
    def collect(value, receipt):
        if isinstance(value, dict):
            for k,v in value.items():
                if isinstance(k,str) and ('/' in k or '\\' in k) and isinstance(v,str): remember(k,v,receipt)
                if isinstance(v,str) and (k.endswith('_path') or k in ('path','source_checkpoint','source_data','condition_source','summary_path','vectors_path')):
                    stem = k[:-5] if k.endswith('_path') else k
                    for hk in (stem+'_sha256',k+'_sha256','sha256' if k=='path' else '__none__'):
                        if hk in value: remember(v,value[hk],receipt)
                collect(v,receipt)
        elif isinstance(value,list):
            for v in value: collect(v,receipt)
    # Examine only this experiment's small receipts; do not scan old experiment trees.
    files = []
    for directory, subdirs, names in os.walk(C.OUT):
        base = Path(directory)
        subdirs[:] = sorted(d for d in subdirs if d != '__pycache__' and not (base == C.OUT and d in ('archive','ot','plans')))
        files.extend(base / n for n in names if not n.endswith(('.tmp','.partial')))
    files.sort()
    print(json.dumps(dict(archive_stage='COLLECT_SMALL_RECEIPTS',files=len(files),OT_and_plans='Indexed from existing audit receipts, not rescanned')),flush=True)
    for p in files:
        if p.suffix == '.json' and p.stat().st_size <= 16_000_000 and not p.is_relative_to(C.OUT/'plans'):
            value = C.read(p); collect(value,p); metadata.append((p,value))
    for row in _rows(C.OUT/'cleanup/retained_verified.csv'):
        remember(row['path'],row['sha256'],C.OUT/'cleanup/retained_verified.csv')
    audit_binding_path = C.OUT / 'audits/final/binding.json'
    audit_source = C.OUT / 'audits/final/final_execution_audit.py'
    audit_hash = C.read(audit_binding_path)['audit_source_sha256']
    if C.sha(audit_source) != audit_hash:
        raise ValueError('Actually executed final auditor differs from its binding')
    remember(audit_source, audit_hash, audit_binding_path)
    authoritative[str(audit_source)] = audit_hash
    codefiles = sorted(p for p in C.CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in SOURCE_SUFFIXES)
    source_paths.update(codefiles)
    source_manifest = []
    for p in sorted(source_paths):
        if not p.is_file(): raise FileNotFoundError('Referenced source is missing: ' + str(p))
        actual = C.sha(p)
        if str(p) in authoritative and actual != authoritative[str(p)]: raise ValueError('Referenced scientific source changed: ' + str(p))
        name = 'source/repository/' + p.relative_to(C.ROOT).as_posix() if p.is_relative_to(C.ROOT) else 'source/external/' + hashlib.sha256(str(p).encode()).hexdigest()[:12] + '/' + p.name
        add(p,name)
        source_manifest.append(dict(path=str(p),sha256=actual,bytes=p.stat().st_size,archive_path=name,registered_sha256=authoritative.get(str(p))))
    diff, git = _code_diff(codefiles)
    add_bytes('code.diff',diff); add_bytes('git_state.json',_json_bytes(git))
    add_bytes('source_hashes.csv',_csv_bytes(source_manifest))
    # Include complete per-output/paired/parent tables, not merely summary figures.
    top_names = {'registration.json','ledger.jsonl','REPORT.md','resources_start.json','resources_windows_start.txt'}
    folders = ('cleanup','audits','inputs','execution','resources','orchestration')
    for p in files:
        rel = p.relative_to(C.OUT); parts=rel.parts
        wanted = (len(parts)==1 and (p.name in top_names or p.suffix=='.md')) or parts[0] in folders
        if wanted and p.suffix.lower() in {'.json','.jsonl','.csv','.md','.txt','.html','.npz','.py'}:
            # Masks are small derived labels; frozen user input/raw arrays are not packed.
            if p.suffix == '.npz' and 'interface_masks' not in parts: continue
            if p.stat().st_size > 4_000_000: continue
            add(p)
        if parts[:2] == ('evaluation','tables') and p.suffix=='.csv': add(p)
        if len(parts)>1 and parts[0]=='evaluation' and parts[1] in ('indices','bindings','score_completion','boundary_completion','review','reviews') and p.suffix in {'.csv','.json','.md'}: add(p)
        if parts[:2]==('training','FULL_SEAT') and p.name in ('status.json','initialization.json','failure.json','training_log.jsonl','events.jsonl') and p.stat().st_size < 8_000_000: add(p)
        if parts[0] in ('checkpoints','inference') and p.suffix=='.json': add(p)
    for folder in LATEST_PACKAGED:
        target,value = _latest(folder)
        add(C.OUT/folder/'latest.json')
        for p in sorted(target.parent.iterdir()):
            if p.is_file() and p.suffix in {'.csv','.json','.png','.svg'}: add(p)
    add(EV / 'figure_visual_review.json')
    # Cached and new boundary summaries are both copied, preserving their actual bytes.
    for label in LABELS:
        block=C.read(EV/'boundary_completion'/f'{label}.json')
        for rec in block['receipts']:
            identity=rec['identity']; p=C.resolve(rec['summary_path'])
            expected=rec['files'].get(str(p),rec['files'].get(rec['summary_path']))
            if expected != C.sha(p): raise ValueError('Boundary summary changed')
            name='evidence/boundary_summaries/'+label+'/'+identity['request_id']+'/'+identity['stage']+'.json'
            add(p,name)
    # Review evidence is checked, but no labels or observations are fabricated here.
    for row in _rows(REVIEW):
        for prefix in ('image','section_image'):
            p=C.resolve(row[prefix+'_path']); h=row.get(prefix+'_sha256',row.get(prefix+'_hash'))
            if C.sha(p)!=h: raise ValueError('Actual reviewed image changed: '+str(p))
    # Evidence files and declared OT/plan paths are indexed; big checkpoint hashes come from frozen save
    # receipts to avoid another multi-GB read. OT/plan caches use existing independent audit records. This provenance is explicit, not claimed
    # as a new rehash. Smaller files and all included sources are actually rehashed.
    inventory=[]
    all_paths=set(files)|{Path(p) for p in known}
    print(json.dumps(dict(archive_stage='INVENTORY',known_paths=len(all_paths))),flush=True)
    for p in sorted(all_paths):
        p=_canonical(p); declared=known.get(str(p),{})
        if p.is_relative_to(C.OUT/'ot') or p.is_relative_to(C.OUT/'plans'):
            inventory.append(dict(path=str(p),exists='NOT_RECHECKED_AUDIT_RECORDED',bytes=None,sha256=declared.get('sha256'),
                checksum_method='RECORDED_AUDIT_SHA256_NOT_REHASHED',checksum_receipt=declared.get('receipt'),
                multiple_declared_versions=declared.get('multiple_versions',False),archive_path=None,in_zip=False,
                reason='LOCAL_ONLY_OT_OR_PLAN_ALREADY_CHECKED_BY_INDEPENDENT_EXECUTION_AUDIT'))
            continue
        st=p.stat() if p.is_file() else None
        heavy=p.suffix.lower() in HEAVY_SUFFIXES or (st is not None and st.st_size>64_000_000)
        if st is not None and heavy and declared:
            digest=declared['sha256']; method='RECORDED_FILE_SHA256_FROM_RECEIPT_NOT_REHASHED'
        elif st is not None:
            digest=C.sha(p); method='ACTUAL_ARCHIVE_TIME_SHA256'
            if declared and not declared.get('multiple_versions') and digest!=declared['sha256']:
                # Training logs may advance beyond an earlier captured analysis;
                # their old hashes remain recorded, never mislabeled current.
                if not p.is_relative_to(C.OUT/'training'):
                    raise ValueError('Protected file changed: '+str(p))
        else:
            digest=declared.get('sha256'); method='DECLARED_HASH_MISSING_LOCAL_FILE'
        inventory.append(dict(path=str(p),exists=st is not None,bytes=st.st_size if st else None,sha256=digest,
            checksum_method=method,checksum_receipt=declared.get('receipt'),multiple_declared_versions=declared.get('multiple_versions',False),archive_path=origins.get(str(p)),
            in_zip=str(p) in origins,reason='INCLUDED' if str(p) in origins else 'LOCAL_ONLY_LARGE_WEIGHT_RAW_OT_OR_FULL_TRACE' if st else 'MISSING_DECLARED_REFERENCE'))
    add_bytes('artifact_inventory.csv',_csv_bytes(inventory))
    grouped_declarations = {}
    for row in declared_hashes:
        key = (row['path'], row['sha256'])
        grouped_declarations.setdefault(key,set()).add(row['receipt'])
    unique_declarations = [dict(path=p,sha256=h,receipts_json=json.dumps(sorted(receipts))) for (p,h),receipts in sorted(grouped_declarations.items())]
    add_bytes('declared_path_hashes.csv',_csv_bytes(unique_declarations))
    add_bytes('archive_readiness.json',_json_bytes(ready))
    add_bytes('RESUME_AND_CONTENTS.md',('''# Full-seat learning-curve evidence package

This archive preserves actual code, unchanged score definitions, the complete per-output/parent/paired CSVs, learning-curve records, the independent boundary normal/tangential component summaries, 192 actual AI_PROXY judgments, and the final report. Main checkpoints are the preregistered 0/1000/3000/6000 points; SMALL_C200 and CREF_C200 remain separate frozen context references.

`source_hashes.csv` maps source bytes to original paths. Text members use ZIP-LZMA; declared_path_hashes.csv consolidates identical path/hash declarations while preserving every receipt in receipts_json. `code.diff` contains new untracked source file contents as well as tracked changes under this experiment directory. `artifact_inventory.csv` indexes local weights, raw meshes, OT caches and full logs, including recorded file hashes; its checksum_method explicitly distinguishes new hashing from an existing save/audit receipt. OT and plan caches are indexed from existing audit receipts without another filesystem scan or rehash. Weights, all RAW arrays, large OT caches and full per-boundary-point traces stay local and are not in this small ZIP. Missing historical paths are not fabricated. Deleted weight identities and receipts remain in cleanup CSVs.

The images in `preview/` are uniformly resized JPEG previews of all 24 requests for both saved-mesh views and finite sections, without result-dependent selection. Original reviewed images and their hashes remain in the review CSV and gallery manifests. Preview compression adds no new visual judgments.

Local workspace: E:/MeshFlow. Read `REPORT.md` and the frozen registration before using the preserved `source/repository/r2/pilots/full_seat_learning_curve_v1/resume.ps1`, `run.py`, or `orchestrate.py`. These are the actual recovery entrypoints, not authorization for new training, generation, changed budgets, a default-model switch, commit or push. Final CPU summaries can be refreshed with these commands, using only completed saved logs, geometry-query arrays and actual reviews:

```bash
python -B -m r2.pilots.full_seat_learning_curve_v1.analysis
python -B -m r2.pilots.full_seat_learning_curve_v1.comparison
python -B -m r2.pilots.full_seat_learning_curve_v1.boundary_components
python -B -m r2.pilots.full_seat_learning_curve_v1.review_summary
python -B -m r2.pilots.full_seat_learning_curve_v1.artifact_archive --status
python -B -m r2.pilots.full_seat_learning_curve_v1.artifact_archive --build
```

Only the `boundary_components_v1/latest.json` snapshot is packaged; earlier partial snapshots and its per-point NPZ arrays remain local and indexed. Its final manifest must cover all 192 registered main model/request/stage cells. Derived normal/tangential components retain their independent definition and NA values, without changing the original boundary scorer.  Package completion is an evidence-completeness statement, not a favorable scientific outcome.
''').encode('utf-8'))
    galleries=[]
    for name in ('gallery','sections_gallery'):
        p=EV/name/'manifest.json'; manifest=C.read(p); add(p)
        for image in manifest['images']:
            source=C.resolve(image['path'])
            if C.sha(source)!=image['sha256']: raise ValueError('Gallery image changed')
            galleries.append((name,source,image))
    # Compression settings apply uniformly to every request; no cherry-picked omission.
    from PIL import Image
    chosen=None
    print(json.dumps(dict(archive_stage="COMPRESS",base_entries=len(entries))),flush=True)
    for width,quality in ((1600,78),(1400,70),(1200,62),(1000,55),(850,48)):
        stage=dict(entries); previews=[]
        for name,p,rec in galleries:
            with Image.open(p) as im:
                im=im.convert('RGB'); original=list(im.size)
                if im.width>width: im=im.resize((width,round(im.height*width/im.width)),Image.Resampling.LANCZOS)
                buffer=io.BytesIO(); im.save(buffer,format='JPEG',quality=quality,optimize=True)
                target='preview/'+name+'/'+p.stem+'.jpg'; stage[target]=buffer.getvalue()
                previews.append(dict(request_id=rec['request_id'],source_path=str(p),source_sha256=rec['sha256'],archive_path=target,preview_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),original_pixels=original,preview_pixels=list(im.size),quality=quality))
        stage['preview/manifest.json']=_json_bytes(dict(status='DERIVATIVE_PREVIEWS_NOT_NEW_REVIEWS',uniform_max_width=width,uniform_quality=quality,images=previews))
        page='<meta charset="utf-8"><h1>All fixed requests: lightweight previews</h1>'
        page+=''.join('<h2>'+html.escape(x['archive_path'])+'</h2><img style="max-width:100%" src="'+html.escape(x['archive_path'].removeprefix('preview/'))+'">' for x in previews)
        stage['preview/index.html']=page.encode()
        manifest=dict(status='COMPLETE',version=VERSION,registration_sha256=ready['registration_sha256'],actual_AI_PROXY_main_cells=192,
            fixed_main_labels=list(MAIN),separate_reference_labels=list(REFERENCES),complete_main_stage_outputs=192,
            uniform_preview_max_width=width,uniform_preview_quality=quality,weights_in_zip=False,raw_mesh_arrays_in_zip=False,
            text_compression='ZIP_LZMA',path_hash_declarations='ONE_ROW_PER_PATH_SHA_WITH_ALL_RECEIPTS',
            file_hashes={name:hashlib.sha256(payload).hexdigest() for name,payload in sorted(stage.items())},
            no_model_calls=True,no_new_training=True,no_new_scoring=True,no_new_reviews=True,git=git)
        stage['MANIFEST.json']=_json_bytes(manifest)
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for name,payload in sorted(stage.items()):
                info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0))
                kind = zipfile.ZIP_LZMA if Path(name).suffix in {'.csv','.json','.jsonl','.py','.ps1','.md','.txt','.diff','.html','.svg'} else zipfile.ZIP_DEFLATED
                z.writestr(info,payload,compress_type=kind,compresslevel=9 if kind==zipfile.ZIP_DEFLATED else None)
        payload=buffer.getvalue()
        print(json.dumps(dict(archive_stage='PREVIEW_CANDIDATE',width=width,quality=quality,bytes=len(payload))),flush=True)
        if len(payload)<MAX_BYTES:
            chosen=(stage,manifest,payload);break
    if chosen is None:
        failure=dict(status='PREPARED',missing=['SMALL_PACKAGE_BUDGET'],compressed_bytes=len(payload),limit_bytes=MAX_BYTES,
                     reason='All mandatory evidence retained; no incomplete archive marked COMPLETE. Reduce redundant metadata explicitly before retry.')
        cache=DEST/'prepared_entries.zip'
        with zipfile.ZipFile(cache,'w',compression=zipfile.ZIP_LZMA) as z:
            for n,b in sorted(stage.items()): z.writestr(n,b)
        failure['prepared_entries_path']=str(cache)
        C.write(DEST/'prepared.json',failure,replace=True);return failure
    stage,manifest,payload=chosen
    # Ensure mutable pointers/required results did not become a different snapshot.
    again=readiness()
    if again!=ready: raise ValueError('Evidence changed while archiving; preserve local assets and rerun finalization')
    tag=hashlib.sha256(payload).hexdigest()[:16]; out=DEST/('snapshot_'+tag)
    zpath=out/'R2_FULL_SEAT_LEARNING_CURVE_V1_REVIEW.zip'
    C._write_bytes(zpath,payload)
    C._write_bytes(out/'artifact_inventory.csv',entries['artifact_inventory.csv'])
    C._write_bytes(out/'source_hashes.csv',entries['source_hashes.csv'])
    C._write_bytes(out/'code.diff',entries['code.diff'])
    C.write(out/'manifest.json',manifest)
    result=dict(status='COMPLETE',version=VERSION,zip_path=str(zpath),zip_sha256=C.sha(zpath),zip_bytes=len(payload),max_bytes=MAX_BYTES,
                entries=len(stage),actual_AI_PROXY_main_cells=192,manifest_path=str(out/'manifest.json'),model_calls=0,training_updates=0,new_reviews=0)
    C.write(out/'completion.json',result)
    C.write(DEST/'latest.json',dict(path=str(out/'completion.json'),sha256=C.sha(out/'completion.json'),**result),replace=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true',help='Build only after every evidence gate passes; otherwise PREPARED')
    parser.add_argument('--status',action='store_true',help='Read-only readiness check (default)')
    a=parser.parse_args()
    if a.build and a.status: parser.error('Choose --status or --build')
    print(json.dumps(build() if a.build else readiness(),ensure_ascii=False,indent=2))


if __name__=='__main__': main()