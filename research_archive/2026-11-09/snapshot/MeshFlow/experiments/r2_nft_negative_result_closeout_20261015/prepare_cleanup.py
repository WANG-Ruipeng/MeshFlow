"""Read-only checkpoint inventory and evidence snapshot; never loads a model."""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'experiments/r2_terminal_reward_nft_v1'
FROZEN = ROOT / 'experiments/r2_nft_training_c_frozen_v1'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def info(p, hash_file=True):
    s = p.stat()
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, sha256=sha(p) if hash_file else None)

def main():
    dest = OUT / 'cleanup_manifest.json'
    if dest.exists():
        raise SystemExit('Manifest already frozen; do not overwrite.')
    docs = ['WORKSPACE.md', 'AGENTS.md', 'r2/docs/research_route_memory.md',
            'r2/docs/research_route_registry.json', 'r2/docs/research_history_sources.json',
            'r2/docs/research_history_sources.md']
    (OUT / 'before_documents').mkdir(exist_ok=True)
    for rel in docs:
        p = ROOT / rel
        backup = OUT / 'before_documents' / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, backup)
    candidates, retained = [], []
    for arm in ['NFT', 'SELF_FM']:
        for rnd in range(16):
            p = RUN / 'checkpoints' / arm / f'round{rnd:02d}.pt'
            assert p.is_file() and not p.is_symlink()
            assert p.resolve().is_relative_to((RUN / 'checkpoints').resolve())
            row = info(p)
            row.update(arm=arm, round=rnd)
            (retained if rnd == 15 else candidates).append(row)
    assert len(candidates) == 30 and len(retained) == 2
    all_weights = sorted(p for p in RUN.rglob('*') if p.is_file() and p.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors'})
    assert set(p.relative_to(ROOT).as_posix() for p in all_weights) == set(r['path'] for r in candidates + retained)
    base = ROOT / 'experiments/r2_full_seat_learning_curve_v1/inference/FULL_S6000_generator.pt'
    retained.append(info(base))
    den = ROOT / 'experiments/r2_postprocess_avmr_pmr_v1/assets/chair_denoiser_0030000.pt'
    official = info(den, False)
    official['previously_recorded_sha256'] = '6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36'
    official['verification_scope'] = 'size/mtime only this maintenance; not rehashed'
    assert official['bytes'] == 7289979546
    assert retained[0]['sha256'] == 'facd79ff479196f16cd66fd51214ed3aa1066788145579f1abd5fd36b050741c'
    assert retained[1]['sha256'] == '36d2f070e741fc668732ef9b31ca73ddccf219e9a02dfbd8bb57801e8bc65405'
    assert retained[2]['sha256'] == '86b38d35c44d3a6fd83a5b8144b57813f5238efab557e0cf46a7872bbde5c5cc'
    evidence = []
    for run in [RUN, FROZEN]:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p not in all_weights:
                evidence.append(info(p, False))
    critical = []
    for run in [RUN, FROZEN]:
        for name in ['REPORT.md', 'completion.json', 'report_decision.json']:
            critical.append(info(run / name))
        for p in sorted((run / 'evidence_pack').glob('*.zip')):
            critical.append(info(p))
    for rel in ['r2/pilots/terminal_reward_nft_v1','r2/pilots/nft_training_c_frozen_v1']:
        for p in sorted((ROOT / rel).glob('*')):
            if p.is_file():
                critical.append(info(p))
    data = dict(schema='nft_closeout_cleanup_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                authorization='User requested NFT negative-result closeout and redundant weight cleanup',
                deletion_scope='Only NFT/SELF_FM round00..14 .pt; no other files',
                candidates=candidates, logical_delete_bytes=sum(r['bytes'] for r in candidates),
                retained_hash_checked=retained, official_denoiser_protected=official,
                critical_files_hash_checked=critical, original_nonweight_inventory=evidence,
                document_before=[info(ROOT / p) for p in docs],
                requires='Independent checkpoint_review.json PASS; exact path/hash checks before removal',
                training_replay_after_cleanup='Historical round-by-round weight replay unavailable; final endpoints and all evidence retained')
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(manifest=str(dest), candidates=len(candidates), bytes=data['logical_delete_bytes'], retained=len(retained), evidence_files=len(evidence), critical=len(critical))))

if __name__ == '__main__':
    main()
