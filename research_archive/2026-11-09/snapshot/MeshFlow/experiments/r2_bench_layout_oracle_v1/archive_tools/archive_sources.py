"""Freeze small source files only; no imports of experiment/model code."""
from __future__ import annotations
import ast
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT / 'experiments/r2_bench_layout_oracle_v1'
CODE = ROOT / 'r2/pilots/bench_layout_oracle_v1'
ARCHIVE = OUT / 'source_archive'
HASH = re.compile(r'^[a-fA-F0-9]{64}$')
SOURCE_SUFFIXES = {'.py', '.ps1', '.cpp', '.h', '.cu', '.cuh'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def resolve(value):
    s = str(value).replace('\\', '/')
    if re.match(r'^[A-Za-z]:/', s):
        return Path('/mnt') / s[0].lower() / s[3:]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def archive_rel(p):
    if p.is_relative_to(ROOT):
        return p.relative_to(ROOT)
    parts = p.parts
    if len(parts) > 3 and parts[1] == 'mnt':
        return Path('_external') / parts[2].upper() / Path(*parts[3:])
    raise ValueError('Unexpected external source root: ' + str(p))


def source_maps(value, source_registration, expected, roles):
    if isinstance(value, dict):
        for key, val in value.items():
            if Path(str(key)).suffix.lower() in SOURCE_SUFFIXES and isinstance(val, str) and HASH.fullmatch(val):
                p = resolve(key)
                if p in expected and expected[p] != val.lower():
                    raise ValueError('Conflicting registered hash: ' + str(p))
                expected[p] = val.lower()
                roles.setdefault(p, set()).add('registered:' + str(source_registration.relative_to(ROOT)))
            else:
                source_maps(val, source_registration, expected, roles)
    elif isinstance(value, list):
        for val in value:
            source_maps(val, source_registration, expected, roles)


def direct_imports(path):
    """Resolve workspace Python imports without importing or executing them."""
    found = set()
    try:
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    except (UnicodeError, SyntaxError):
        return found
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
            names.extend(node.module + '.' + a.name for a in node.names if a.name != '*')
        elif isinstance(node, ast.ImportFrom) and node.level:
            base = path.parent
            for _ in range(node.level - 1): base = base.parent
            if node.module: base = base / Path(*node.module.split('.'))
            candidates = [base]
            candidates.extend(base / a.name for a in node.names if a.name != '*')
            for candidate in candidates:
                for p in (candidate.with_suffix('.py'), candidate / '__init__.py'):
                    if p.is_file() and p.is_relative_to(ROOT): found.add(p)
    for name in names:
        if name.startswith('r2.'):
            module = ROOT / Path(*name.split('.'))
        elif name.startswith('meshflow_control.'):
            module = ROOT / 'r2/src' / Path(*name.split('.'))
        else:
            continue
        for p in (module.with_suffix('.py'), module / '__init__.py'):
            if p.is_file(): found.add(p)
    return found


def run():
    if ARCHIVE.exists() or (OUT / 'source_hashes.json').exists() or (OUT / 'code.diff').exists():
        raise FileExistsError('Source snapshot is immutable; create a supplement instead of replacing it')
    registrations = [OUT / 'registrations/PLAIN.json', OUT / 'registrations/LAYOUT.json',
        OUT / 'evaluation/score_registration.json']
    prior = ROOT / 'experiments/r2_bench_four_routes_v1/exposure/evaluation/inference_registrations'
    registrations += [prior / ('EXPOSURE__' + k + '.json') for k in ('RAW', 'D0')]
    expected, roles = {}, {}
    for p in registrations:
        source_maps(json.loads(p.read_text(encoding='utf-8-sig')), p, expected, roles)
    own = set(CODE.glob('*.py'))
    own.update((OUT / 'archive_tools').glob('*.py'))
    for p in own: roles.setdefault(p, set()).add('new_isolated_pilot_or_archive_tool')
    # Snapshot the frozen registered closure and static local dependencies. This
    # is intentionally conservative; it is not a dynamic per-function trace.
    pending = list(set(expected) | own)
    visited = set()
    while pending:
        p = pending.pop()
        if p in visited: continue
        visited.add(p)
        if not p.is_file(): raise FileNotFoundError(p)
        if p.suffix == '.py':
            for imported in direct_imports(p):
                roles.setdefault(imported, set()).add('static_workspace_import_dependency')
                if imported not in visited: pending.append(imported)
    for p in visited:
        if p.stat().st_size > 2_000_000:
            raise ValueError('Unexpected large source asset: ' + str(p))
        if p in expected and sha(p) != expected[p]:
            raise ValueError('Registered source changed: ' + str(p))
    rows = []
    for p in sorted(visited, key=str):
        target = ARCHIVE / archive_rel(p)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = p.read_bytes()
        with target.open('xb') as stream: stream.write(body)
        digest = sha(p)
        if sha(target) != digest: raise ValueError('Copy hash mismatch')
        rows.append(dict(source_path=str(p), archive_path=str(target),
            relative_archive_path=str(target.relative_to(ARCHIVE)), sha256=digest,
            bytes=len(body), roles=sorted(roles.get(p, {'static_workspace_import_dependency'})),
            registered_expected_sha256=expected.get(p),
            unchanged_against_frozen_registration=(digest == expected[p]) if p in expected else None))
    chunks = []
    for p in sorted(own, key=str):
        relative = str(p.relative_to(ROOT))
        lines = p.read_text(encoding='utf-8-sig').splitlines(keepends=True)
        if lines and not lines[-1].endswith('\n'): lines[-1] += '\n'
        chunks.append('diff --git a/' + relative + ' b/' + relative + '\nnew file mode 100644\n')
        chunks.extend(difflib.unified_diff([], lines, fromfile='/dev/null', tofile='b/' + relative))
    with (OUT / 'code.diff').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(''.join(chunks))
    receipt = dict(status='COMPLETE', version='layout_oracle_source_archive_v1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Frozen registration closure plus static local Python import closure and isolated pilot source; no dynamic call coverage claim',
        git_head_read_only='7af33f892b8fb6194fbc54748b6010cf54507169',
        git_command_policy='git -c safe.directory=E:/MeshFlow; no git config, index, commit or push changes',
        diff_scope='New isolated pilot/archive files as additions from /dev/null; not the entire dirty-worktree diff',
        inherited_source_policy='Frozen registered original sources verified byte-identical; other import dependencies snapshot-only',
        files=rows, file_count=len(rows), source_bytes=sum(r['bytes'] for r in rows),
        new_code_files=len(own), registered_source_files=len(expected),
        all_registered_sources_unchanged=True,
        copied_weights=0, copied_datasets=0, model_calls=0,
        code_diff=dict(path=str(OUT/'code.diff'),sha256=sha(OUT/'code.diff'),bytes=(OUT/'code.diff').stat().st_size),
        registrations={str(p):sha(p) for p in registrations})
    with (OUT / 'source_hashes.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({k:receipt[k] for k in ('status','file_count','source_bytes','new_code_files','registered_source_files','all_registered_sources_unchanged')}))


if __name__ == '__main__': run()
