"""Read-only, bounded Bench data inventory; no removal or model/weight loading."""
from __future__ import annotations
import collections
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SCOPES = {
    'r2_bench_control_transfer_v1': ['data', 'evaluation_inputs'],
    'r2_bench_parts_v1': ['data', 'candidates', 'probe', 'combinations'],
    'r2_bench_parts_convergence_v1': ['data', 'fixed_loss_probe', 'evaluation_inputs'],
    'r2_bench_clean_multipart_seam_v1': ['data', 'base_pair_caches', 'evaluation_inputs'],
    'r2_bench_coverage_expansion_v1': ['data', 'candidates', 'preparation', 'base_pair_caches'],
}
EXCLUDED = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin'}

def rel(p):
    return str(p.relative_to(ROOT)).replace('\\', '/')

def normalized(s):
    s = str(s).replace('\\', '/')
    for prefix in ('/mnt/e/MeshFlow/', 'E:/MeshFlow/', 'e:/MeshFlow/'):
        if s.startswith(prefix):
            return s[len(prefix):]
    if s.startswith(('experiments/', 'r2/', 'downloaded_data/')):
        return s
    return None

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def refs(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from refs(k)
            yield from refs(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from refs(v)
    elif isinstance(obj, str):
        r = normalized(obj)
        if r:
            yield r


def main():
    tick = time.perf_counter()
    active_manifest = ROOT / 'experiments/r2_bench_coverage_expansion_v1/data/manifest.json'
    active = read(active_manifest)
    direct = set(refs(active))
    protected = set(direct)
    provenance = {}
    metadata_paths = [active_manifest,
        ROOT / 'experiments/r2_bench_coverage_expansion_v1/registration.json',
        ROOT / 'experiments/r2_bench_coverage_expansion_v1/cpu_gate.json',
        ROOT / 'experiments/r2_bench_coverage_expansion_v1/results/completion.json',
        ROOT / 'experiments/r2_bench_coverage_expansion_v1/gallery/definition.json',
        ROOT / 'experiments/r2_bench_coverage_expansion_v1/final_verify_latest.json']
    for optional in [ROOT / 'experiments/r2_bench_coverage_expansion_v1/cpu_gate.json',
                     ROOT / 'experiments/r2_bench_coverage_expansion_v1/final_verify_latest.json']:
        if optional.exists():
            obj = read(optional)
            for key in ('input_hashes', 'receipt'):
                if key in obj:
                    pp = ROOT / normalized(obj[key])
                    metadata_paths.append(pp)
                    if key == 'receipt' and pp.exists():
                        receipt = read(pp)
                        if 'input_hashes' in receipt:
                            metadata_paths.append(ROOT / normalized(receipt['input_hashes']))
    for p in dict.fromkeys(metadata_paths):
        if p.exists():
            values = set(refs(read(p)))
            protected.update(values)
            provenance[rel(p)] = dict(sha256=sha(p), referenced_paths=len(values))

    files = []
    roots = []
    excluded = []
    for experiment, names in SCOPES.items():
        for name in names:
            folder = ROOT / 'experiments' / experiment / name
            if not folder.exists():
                continue
            roots.append(folder)
            for current, dirs, fnames in os.walk(folder, followlinks=False):
                dirs[:] = [d for d in dirs if not Path(current, d).is_symlink()]
                for filename in fnames:
                    p = Path(current, filename)
                    if p.is_symlink() or p.suffix.lower() in EXCLUDED:
                        excluded.append(rel(p))
                        continue
                    rr = rel(p)
                    files.append(dict(path=rr, bytes=p.stat().st_size, root=rel(folder),
                                      current_manifest_reference=rr in direct,
                                      current_frozen_evidence_reference=rr in protected))
    bysize = collections.defaultdict(list)
    for row in files:
        bysize[row['bytes']].append(row)
    duplicates = []
    for size, group in bysize.items():
        if len(group) < 2 or size == 0:
            continue
        byhash = collections.defaultdict(list)
        for row in group:
            byhash[sha(ROOT / row['path'])].append(row)
        for digest, equal in byhash.items():
            if len(equal) < 2:
                continue
            duplicates.append(dict(sha256=digest, bytes_each=size, files=[r['path'] for r in equal],
                                   redundant_bytes_arithmetic=(len(equal)-1)*size,
                                   current_manifest_referenced_files=sum(r['current_manifest_reference'] for r in equal),
                                   current_frozen_evidence_referenced_files=sum(r['current_frozen_evidence_reference'] for r in equal),
                                   deletion_status='NOT_AUTHORIZED; IDENTICAL_BYTES_DO_NOT_REMOVE_PATH_DEPENDENCY_OR_EVIDENCE_VALUE'))
    directories = []
    for root in roots:
        prefix = rel(root)
        members = [f for f in files if f['root'] == prefix]
        directories.append(dict(path=prefix, files=len(members), bytes=sum(f['bytes'] for f in members),
                                current_manifest_referenced_files=sum(f['current_manifest_reference'] for f in members),
                                current_manifest_referenced_bytes=sum(f['bytes'] for f in members if f['current_manifest_reference']),
                                current_frozen_evidence_referenced_files=sum(f['current_frozen_evidence_reference'] for f in members),
                                current_frozen_evidence_referenced_bytes=sum(f['bytes'] for f in members if f['current_frozen_evidence_reference']),
                                classification='PROTECTED_DATA_OR_REPRODUCIBLE_EXPERIMENT_EVIDENCE; NOT_AUTOMATICALLY_TEMPORARY'))
    pycache = [r for r in files if '__pycache__' in Path(r['path']).parts and r['path'].endswith('.pyc')
               and not r['current_frozen_evidence_reference']]
    potential = []
    for item in pycache:
        potential.append(dict(**item, reason='Reconstructible Python bytecode, if not retained in historical receipts',
                              deletion_status='CANDIDATE_ONLY_NOT_DELETED; HISTORICAL_RECEIPTS_NOT_EXHAUSTIVELY_SEARCHED'))
    result = dict(status='READ_ONLY_INVENTORY_COMPLETE', schema='BENCH_BOUNDED_DATA_STORAGE_INVENTORY_V1',
                  active_training_parents=len({r['parent_id'] for r in active['train']}),
                  active_training_conditions=len(active['train']),
                  bounded_scopes=SCOPES, directories=directories, total_files=len(files),
                  total_bytes=sum(r['bytes'] for r in files),
                  source_metadata=provenance, protected_referenced_path_count=len(protected),
                  identical_byte_groups=sorted(duplicates, key=lambda x:-x['redundant_bytes_arithmetic']),
                  identical_redundant_bytes_arithmetic=sum(x['redundant_bytes_arithmetic'] for x in duplicates),
                  obvious_reconstructible_cache_candidates=potential,
                  obvious_cache_candidate_bytes=sum(r['bytes'] for r in potential),
                  safe_delete_recommendation='NONE_IDENTIFIED' if not potential else 'BYTECODE_ONLY_PENDING_DEPENDENCY_POLICY',
                  files=files, excluded_weight_or_symlink_paths=excluded,
                  interpretation=['Original data, RAW, reports and scientific evidence remain protected.',
                                  'Directories called cache/probe/candidates may be frozen inputs or provenance, not trash.',
                                  'Byte-identical duplicates are not safe deletions when immutable manifests reference separate paths.',
                                  'Unreferenced here means absent from selected current metadata, not proof of no historical dependency.',
                                  'No full-disk/weight scan; no geometry reconstruction, intersection score or model calls.'],
                  deleted_files=0, model_calls=0, GPU_calls=0, seconds=time.perf_counter()-tick,
                  script_sha256=sha(Path(__file__)))
    target = OUT / 'storage_inventory.json'
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k:result[k] for k in ('status','total_files','total_bytes','identical_redundant_bytes_arithmetic','obvious_cache_candidate_bytes','safe_delete_recommendation','seconds')},indent=2))
    print(str(target))


if __name__ == '__main__':
    main()
