"""Prepare the user-requested research handoff, without running any experiment."""
import collections
import csv
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from inventory import ROOTS, SKIP, files

HERE = Path(__file__).resolve().parent
OUT = HERE/'publish/research_archive/2026-11-09'
CODE = {'.py', '.ps1', '.sh', '.bat', '.cmd', '.cpp', '.h', '.hpp', '.c', '.cu', '.cuh', '.js', '.css', '.diff', '.patch'}
CONFIG = {'.yaml', '.yml', '.toml', '.ini', '.cfg'}
DOC = {'.md', '.rst', '.txt'}
META = re.compile(r'protocol|config|contract|budget|prereg|frozen|freeze|manifest|registry|route|handoff|dataset_ready|complete|completion|closeout|summary|retained|evaluation_plan|sampling_plan|sample_plan|seeds|parameters|hyperparams|condition_list|request_list|active_conditions|fixed_inputs|metrics_schema', re.I)
TABLE = re.compile(r'per_output|per_condition|per_parent|aggregate|summary|paired|comparison|ablation|results?|metrics|scores?|observations?|judgments?|ratings?|review|panel|conditions?|requests?|manifest|registry|inventory|rollup|gate|exposure_summary|training_curve|loss_curve|routes?|coverage', re.I)
DETAIL = re.compile(r'per_face|per_triangle|per_vertex|face_pairs|pair_contacts|all_saved_pair|edge_pairs|edge_ledger|contact_pairs|per_step|step_trace|train_trace|gradient_trace|calls|events|heartbeats?|stdout|stderr|deletion_ledger|protected_before|inventory_before', re.I)
SECRET = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{32,}|github_pat_[A-Za-z0-9_]{40,}|(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])|sk-proj-[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{30,}')
NAMES = {'Dockerfile', 'Makefile', '.gitignore', '.gitattributes', 'LICENSE', 'NOTICE', 'CMakeLists.txt', '.gitmodules'}
MAX_STRUCTURED = 4*1024*1024


def category(rel):
    p = PurePosixPath(str(rel).replace('\\', '/'))
    name = p.name
    suffix = p.suffix.lower()
    if any(x in SKIP or x in {'.vscode', '.idea'} for x in p.parts):
        return None
    if any(x.lower().endswith('venv') or x.lower() in {'site-packages', 'dist-packages'} for x in p.parts[:-1]):
        return None
    if name.lower().startswith('.env') or name.lower() in {'credentials', 'credentials.json', '.pypirc', '.netrc', '.npmrc'}:
        return None
    if suffix in CODE:
        return 'source'
    if suffix in CONFIG or name in NAMES:
        return 'configuration'
    if suffix in {'.md', '.rst'}:
        return 'document'
    if DETAIL.search(name):
        return None
    if suffix in DOC:
        return 'document'
    # Individual geometry jobs and transient per-sample/step records are outside
    # the user's research-handoff scope. Their aggregate result tables remain.
    if any(re.search(r'[a-f0-9]{32}|__n\d+|^(?:sample|case|slot|batch|round)_\d+$', part, re.I) for part in p.parts[:-1]):
        return None
    if suffix in {'.csv', '.tsv'} and TABLE.search(name):
        return 'result_table'
    if suffix in {'.json', '.jsonl'}:
        if META.search(name):
            return 'protocol_metadata'
        if any(x in {'configs', 'config', 'production_contract', 'data_tools'} for x in p.parts):
            return 'configuration'
        # Root experiment records and final result/analysis tables, not output slots.
        for marker in ('experiments', 'runs', 'followups'):
            if marker in p.parts:
                tail = p.parts[p.parts.index(marker)+1:]
                if len(tail) == 2:
                    return 'experiment_record'
                if len(tail) == 3 and tail[1] in {'results', 'analysis', 'data', 'evaluation', 'protocol', 'registration'}:
                    return 'experiment_record'
        if len(p.parts) <= 2 and suffix == '.json':
            return 'experiment_record'
    return None


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        raise RuntimeError('Archive output already exists; do not overwrite a prepared snapshot.')
    OUT.mkdir(parents=True)
    counts = collections.Counter()
    omitted = collections.Counter()
    manifests = []
    reports = []
    containers = []
    failures = []
    assets = []
    seen = set()

    def accept(data, dest, origin, kind, container=None):
        if SECRET.search(data):
            failures.append({'source': origin, 'reason': 'credential_pattern_requires_review'})
            return
        if b'\x00' in data[:8192] and not data.startswith((b'\xff\xfe', b'\xfe\xff')):
            omitted['binary_content_in_text_extension'] += 1
            return
        sha = hashlib.sha256(data).hexdigest()
        target = OUT/dest
        if str(dest).casefold() in seen:
            if target.read_bytes() != data:
                failures.append({'source': origin, 'reason': 'path_collision'})
            return
        seen.add(str(dest).casefold())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        row = {'path': dest.as_posix(), 'source': origin, 'bytes': len(data), 'sha256': sha, 'category': kind}
        if container:
            row['container'] = container
        manifests.append(row)
        counts[kind] += 1
        if kind == 'document' and dest.suffix.lower() == '.md':
            first = data.decode('utf-8-sig', errors='replace').splitlines()
            title = next((x.lstrip('# ').strip() for x in first if x.startswith('#')), dest.name)
            reports.append({'path': dest.as_posix(), 'title': title, 'source': origin, 'sha256': sha})

    for root in ROOTS:
        scanned = selected = 0
        for p in files(root):
            scanned += 1
            rel = p.relative_to(root)
            kind = category(rel)
            if kind:
                size = p.stat().st_size
                if kind not in {'source', 'document', 'configuration'} and size > MAX_STRUCTURED:
                    omitted['structured_file_over_4_MiB'] += 1
                    continue
                before = p.stat()
                data = p.read_bytes()
                after = p.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    failures.append({'source': str(p), 'reason': 'changed_during_read'})
                    continue
                accept(data, Path('snapshot')/root.name/rel, str(p), kind)
                selected += 1
                if selected % 5000 == 0:
                    print(json.dumps({'root': root.name, 'selected': selected, 'scanned': scanned}), flush=True)
            elif p.suffix.lower() in {'.pt', '.pth', '.ckpt', '.safetensors'}:
                assets.append({'source': str(p), 'bytes': p.stat().st_size, 'backup_status': 'NOT_INCLUDED_TEXT_ONLY'})
            else:
                omitted['excluded_' + (p.suffix.lower() or 'no_extension')] += 1
        print(json.dumps({'root': root.name, 'selected': selected, 'scanned': scanned, 'done': True}), flush=True)

    inv = json.loads((HERE/'inventory.json').read_text(encoding='utf-8'))
    for root_record in inv:
        root = Path(root_record['root'])
        for entry in root_record['archives']:
            p = root/entry['path']
            if p.suffix.lower() not in {'.zip', '.tar'}:
                continue  # Official dataset tar.gz files and retired detailed inventories.
            aid = hashlib.sha256(str(p).encode()).hexdigest()[:16]
            extracted = 0
            try:
                if p.suffix.lower() == '.zip':
                    archive = zipfile.ZipFile(p)
                    members = [(x.filename, x.file_size, x) for x in archive.infolist() if not x.is_dir()]
                    read = lambda obj: archive.read(obj)
                else:
                    archive = tarfile.open(p)
                    members = [(x.name, x.size, x) for x in archive.getmembers() if x.isfile()]
                    read = lambda obj: archive.extractfile(obj).read()
                with archive:
                    for name, size, obj in members:
                        rel = PurePosixPath(name.replace('\\', '/'))
                        if rel.is_absolute() or '..' in rel.parts or any(':' in part for part in rel.parts):
                            failures.append({'source': str(p)+'!'+name, 'reason': 'unsafe_archive_member_path'})
                            continue
                        kind = category(rel)
                        if not kind or (kind not in {'source', 'document', 'configuration'} and size > MAX_STRUCTURED):
                            continue
                        accept(read(obj), Path('archived_text')/aid/Path(*rel.parts), name, kind, str(p))
                        extracted += 1
                containers.append({'archive_id': aid, 'source_container': str(p), 'text_members': extracted})
            except (OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
                failures.append({'source': str(p), 'reason': type(exc).__name__})

    manifests.sort(key=lambda x:x['path'])
    with (OUT/'MANIFEST.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for row in manifests:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    write_csv(OUT/'DOCUMENT_INDEX.csv', sorted(reports, key=lambda x:x['path']), ['path', 'title', 'source', 'sha256'])
    write_csv(OUT/'ARCHIVE_CONTAINERS.csv', containers, ['archive_id', 'source_container', 'text_members'])
    write_csv(OUT/'WEIGHTS_NOT_BACKED_UP.csv', assets, ['source', 'bytes', 'backup_status'])
    summary = {'archive_date': '2026-11-09', 'scope': 'Research handoff: documents, source, protocols, condition/seed manifests, key result tables; no assets or low-level diagnostics.', 'files': len(manifests), 'bytes': sum(x['bytes'] for x in manifests), 'categories': dict(counts), 'document_index_rows': len(reports), 'archive_containers_scanned': len(containers), 'omitted_counts': dict(sorted(omitted.items())), 'failures': failures, 'roots': [str(x) for x in ROOTS], 'structured_size_cap_bytes': MAX_STRUCTURED, 'source_repo_head': '7af33f892b8fb6194fbc54748b6010cf54507169', 'source_modified_files': ['r2/README.md'], 'not_a_training_resume_or_full_replay_backup': True}
    (OUT/'SCOPE.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (OUT/'.gitattributes').write_text('* -text\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    if failures:
        raise SystemExit('Preparation requires review; no upload is permitted yet.')


if __name__ == '__main__':
    main()
