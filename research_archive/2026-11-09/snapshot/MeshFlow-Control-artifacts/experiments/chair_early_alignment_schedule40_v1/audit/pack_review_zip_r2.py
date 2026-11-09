#!/usr/bin/env python3
"""Final small-receipt membership extension; delegates unchanged R1 ZIP validation."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json


def main():
    r1=Path(__file__).with_name('pack_review_zip_r1.py')
    spec=importlib.util.spec_from_file_location('schedule40_review_packer_r1',r1)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--out',default=module.DEFAULT)
    args,_=parser.parse_known_args();root=Path(args.out).resolve()
    record=dict(schema='schedule40_review_packer_identity_r2',source='audit/pack_review_zip_r2.py',
                sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                delegate_source='audit/pack_review_zip_r1.py',delegate_sha256=hashlib.sha256(r1.read_bytes()).hexdigest(),
                added_members=['audit/final_idle_check.json'],scope='Add final idle receipt only; preserve existing R1 validation/compression and original sources')
    identity=root/'audit/review_zip_packer_identity_r2.json'
    if identity.exists() and json.loads(identity.read_text())!=record:raise ValueError('Registered R2 identity differs')
    if not identity.exists():identity.write_bytes(module.json_bytes(record))
    module.STATIC.extend(['audit/final_idle_check.json','audit/pack_review_zip_r2.py','audit/review_zip_packer_identity_r2.json'])
    module.main()


if __name__=='__main__':main()