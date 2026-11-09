"""Recover/verify the bounded CPU data task; never dispatches models or training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import importlib
import json
from pathlib import Path
from .prepare import OUT, ROOT
from r2.data_tools.bench_user_v2 import build as b

STAGES = (
    ('preparation', 'preparation.json'),
    ('source_inventory', 'asset_inventory/COMPLETE.json'),
    ('layout_contract', 'layout_contract/HANDOFF.json'),
    ('split_inventory', 'split_audit/verification.json'),
    ('target_proposals', 'rebuild_summary.json'),
    ('independent_target_check', 'rebuild_verification/summary.json'),
    ('staging_split', 'split_audit/new_data_split.json'),
    ('source_tasks', 'dataset/source_tasks_pending_adapter/COMPLETE.json'),
    ('delivery', 'COMPLETE.json'),
)


def verify():
    protected = {}
    if (OUT/'preparation.json').exists():
        protected.update(b.read(OUT/'preparation.json')['protected_source_hashes'])
    if (OUT/'COMPLETE.json').exists():
        complete = b.read(OUT/'COMPLETE.json')
        protected.update(complete.get('protected_source_hashes', {}))
        protected.update(complete.get('delivery_hashes', {}))
    for path, expected in protected.items():
        if b.sha(path) != expected:
            raise RuntimeError('HASH_MISMATCH: '+str(path))
    result = dict(stages={name: ('PRESENT' if (OUT/path).exists() else 'PENDING') for name,path in STAGES},
        verified_hashes=len(protected),model_calls=0,training_updates=0,
        scientific_readiness='See DATASET_STATUS.json; file existence never certifies train readiness.')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def resume():
    # Completed stages are not rerun with new timestamps or changed proposals.
    # Manual/source visual review cannot be regenerated as an automatic PASS.
    verify()
    if (OUT/'COMPLETE.json').exists():
        print('COMPLETE: verified only; no new data, audits, sampling or training.'); return
    if not (OUT/'preparation.json').exists():
        importlib.import_module(__package__+'.prepare').run()
    if not (OUT/'layout_contract/HANDOFF.json').exists():
        importlib.import_module(__package__+'.layout_contract').main()
    if not (OUT/'rebuild_summary.json').exists():
        from . import planar_rebuild as p
        if (OUT/'REBUILD_PROTOCOL.json').exists():
            protocol=b.read(OUT/'REBUILD_PROTOCOL.json')
            if b.sha(p.__file__)!=protocol['code_sha256']:
                raise RuntimeError('Frozen construction code changed; explicit new version required.')
            # Sequential recovery of only missing parents, all original definitions retained.
            rows=[p.one(row) for row in protocol['sources']]
            b.save(OUT/'rebuild_summary.json',dict(status='TARGET_PROPOSALS_COMPLETE',rows=rows,
                parents=len(rows),geometric_pass=sum(not r['reasons'] for r in rows),
                recovery='Verified existing parents, computed missing only; no algorithm retries',
                new_model_calls=0,new_noise=0))
        else:
            p.main()
    verify()
    print('No automatic review decisions, training launch, cache-head forward or fallback simplifier.')


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--verify',action='store_true')
    args=ap.parse_args()
    resume() if args.resume else verify()


if __name__=='__main__': main()
