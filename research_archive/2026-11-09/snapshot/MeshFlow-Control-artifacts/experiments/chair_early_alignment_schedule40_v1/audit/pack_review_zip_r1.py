#!/usr/bin/env python3
"""CPU-only lossless fixed-panel review packer. Default is inventory, never final ZIP.

All membership is explicit. No model imports, scoring, image edits or source edits.
Use --finalize only after the report owner explicitly declares the report ready.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
from html.parser import HTMLParser
import io
from itertools import combinations, product
import json
from pathlib import Path, PurePosixPath
import posixpath
import time
from urllib.parse import unquote, urlsplit
import zipfile
import zlib

DEFAULT='/mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1'
ZIP_NAME='CHAIR_EARLY_ALIGNMENT_SCHEDULE40_REVIEW.zip'
LIMIT=10_000_000
STATIC='''EXPERIMENT_REPORT.md
METHODS_AND_AUDIT_NOTES.md
RUN_AND_RESUME.md
SOURCE_INDEX.md
OPEN_GALLERY.html
protocol.json
code.diff
evaluation/per_output.csv
evaluation/paired_parent_deltas.csv
evaluation/summary.csv
evaluation/per_output_conflict.csv
evaluation/conflict_paired_parent_deltas.csv
evaluation/conflict_summary.csv
evaluation/visual_review_per_output.csv
evaluation/per_output.json
evaluation/per_output_conflict.json
evaluation/requests.json
evaluation/condition_construction.json
evaluation/conflict_registration.json
evaluation/conflict_scoring_receipt.json
evaluation/scoring_receipt.json
gallery/manifest.json
gallery/camera_manifest.json
gallery/method_mapping.json
audit/source_hashes_before.json
audit/new_source_hashes.json
audit/postfreeze_analysis_hashes.json
audit/delivery_source_hashes.json
audit/source_delivery_verification.json
audit/final_checkpoint_verification.json
audit/final_accounting.json
audit/input_pairing.json
audit/resource_timing_audit.json
audit/perparent_actual_exposure.csv
audit/dispatch_state.json
audit/training_dispatch_receipt.json
audit/sampling_dispatch_receipt.json
audit/environment.json
audit/conflict_cpu_validation.json
audit/cpu_validation.json
audit/analyze_actual_resources_preparation.json
audit/summarize_fixed_panel_preparation.json
audit/verify_final_checkpoints_execution.json
audit/engineering/code_diff_before_delivery.diff
audit/fixed_panel_summary/summary_manifest.json
audit/fixed_panel_summary/receipt.json
audit/fixed_panel_summary/input_diagnostics.json
audit/pack_review_zip.py
audit/review_zip_packer_identity.json
audit/pack_review_zip_r1.py
audit/review_zip_packer_identity_r1.json
audit/merge_visual_review.py
audit/visual_review_manifest.json'''.splitlines()
PATTERNS=['gallery/*/*.png','source_code/**/*','audit/fixed_panel_summary/*.csv',
          'audit/actual_final_cpu_analysis/*','audit/dispatch_r1/**/*.json',
          'audit/dispatch_r1/**/*.py','audit/dispatch_r1/*.diff',
          'audit/serial_tail_guard/*.json','audit/serial_tail_guard/*.jsonl']
EXPECTED_CSV_ROWS={
 'evaluation/per_output.csv':48,'evaluation/per_output_conflict.csv':48,
 'evaluation/visual_review_per_output.csv':48,'evaluation/paired_parent_deltas.csv':48,
 'evaluation/conflict_paired_parent_deltas.csv':120,'evaluation/summary.csv':6,
 'evaluation/conflict_summary.csv':6,'audit/fixed_panel_summary/per_model_by_stratum.csv':720,
 'audit/fixed_panel_summary/per_output_registered_metrics.csv':3840,
 'audit/fixed_panel_summary/paired_parent_differences.csv':9600,
 'audit/fixed_panel_summary/paired_summary_by_stratum.csv':1800}


def sha(data):return hashlib.sha256(data).hexdigest()
def json_bytes(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')
def parse_json(data):return json.loads(data.decode('utf-8-sig'))
def write_result(path,value):Path(path).write_bytes(json_bytes(value))


def sources(root):
    names=set(STATIC)
    for pattern in PATTERNS:
        names.update(p.relative_to(root).as_posix() for p in root.glob(pattern) if p.is_file())
    for name in names:
        parts=PurePosixPath(name).parts
        if '..' in parts or name.startswith('/') or ':' in name or Path(name).suffix.lower() in ('.pt','.pth','.npz','.npy','.safetensors','.ply','.obj'):
            raise ValueError('Forbidden package member: '+name)
        if not (root/name).resolve().is_relative_to(root):raise ValueError('Member leaves fixed run: '+name)
    missing=sorted(name for name in names if not (root/name).is_file())
    payload={name:(root/name).read_bytes() for name in sorted(names) if (root/name).is_file()}
    return payload,missing


class Links(HTMLParser):
    def __init__(self):super().__init__();self.targets=[]
    def handle_starttag(self,tag,attrs):
        for name,value in attrs:
            if name in ('href','src') and value:self.targets.append(value)


def validate(payload,missing,final):
    if final and missing:raise ValueError('Required delivery files missing: '+repr(missing))
    if set(missing)-{'EXPERIMENT_REPORT.md','evaluation/visual_review_per_output.csv'}:
        raise ValueError('Required existing artifacts missing: '+repr(missing))
    requests=parse_json(payload['evaluation/requests.json'])
    labels=requests['labels'];parents=[row['parent_id'] for row in requests['rows']]
    if len(labels)!=6 or len(set(labels))!=6 or len(parents)!=8 or len(set(parents))!=8:raise ValueError('Panel identity changed')
    expected_outputs=set(product(labels,parents))
    csv_rows={}
    counts={}
    for name,expected in EXPECTED_CSV_ROWS.items():
        if name not in payload:
            continue
        rows=list(csv.DictReader(io.StringIO(payload[name].decode('utf-8-sig'))))
        csv_rows[name]=rows;counts[name]=len(rows)
        if len(rows)!=expected:raise ValueError(f'Wrong row count: {name}: {len(rows)} != {expected}')
    for name in ('evaluation/per_output.csv','evaluation/per_output_conflict.csv','evaluation/visual_review_per_output.csv'):
        if name in csv_rows and {(row['label'],row['parent_id']) for row in csv_rows[name]}!=expected_outputs:
            raise ValueError('Incomplete or duplicate 48-output identity: '+name)
    frozen=parse_json(payload['audit/fixed_panel_summary/summary_manifest.json'])
    metrics=[item['name'] for item in frozen['numeric_fields']]
    strata=frozen['strata']
    if len(metrics)!=40 or len(set(metrics))!=40 or {k:len(v) for k,v in strata.items()}!={'overall8':8,'existing_A40_6':6,'expanded_C_2':2}:
        raise ValueError('Metric or stratum registry changed')
    pairset={tuple(sorted(pair)) for pair in combinations(labels,2)}
    checks=[('audit/fixed_panel_summary/per_model_by_stratum.csv',
             {(g,label,metric) for g in strata for label in labels for metric in metrics},
             lambda row:(row['stratum'],row['label'],row['metric'])),
            ('audit/fixed_panel_summary/per_output_registered_metrics.csv',
             {(g,label,parent,metric) for g,ps in strata.items() for label in labels for parent in ps for metric in metrics},
             lambda row:(row['stratum'],row['label'],row['parent_id'],row['metric'])),
            ('audit/fixed_panel_summary/paired_summary_by_stratum.csv',
             {(g,pair,metric) for g in strata for pair in pairset for metric in metrics},
             lambda row:(row['stratum'],tuple(sorted([row['first'],row['second']])),row['metric'])),
            ('audit/fixed_panel_summary/paired_parent_differences.csv',
             {(g,parent,pair,metric) for g,ps in strata.items() for parent in ps for pair in pairset for metric in metrics},
             lambda row:(row['stratum'],row['parent_id'],tuple(sorted([row['first'],row['second']])),row['metric']))]
    for name,expected,key in checks:
        if {key(row) for row in csv_rows[name]}!=expected:raise ValueError('Incomplete fixed paired/stratum membership: '+name)
    if {(row['parent_id'],tuple(sorted([row['first'],row['second']]))) for row in csv_rows['evaluation/conflict_paired_parent_deltas.csv']}!={(parent,pair) for parent in parents for pair in pairset}:
        raise ValueError('Incomplete I all-pair membership')
    source_manifest=parse_json(payload['audit/delivery_source_hashes.json'])
    copied_sources={name[len('source_code/'):]:data for name,data in payload.items() if name.startswith('source_code/')}
    if set(copied_sources)!=set(source_manifest['sources']):raise ValueError('Copied source inventory differs from delivery manifest')
    for name,data in copied_sources.items():
        if sha(data)!=source_manifest['sources'][name]['sha256']:raise ValueError('Copied source hash mismatch: '+name)
    expected_images={f'gallery/{mode}/{row["request_id"]}.png' for row in requests['rows'] for mode in ('neutral','colored')}
    images={name for name in payload if name.endswith('.png')}
    if images!=expected_images or len(images)!=16:raise ValueError('Fixed16 original image membership differs')
    parser=Links();parser.feed(payload['OPEN_GALLERY.html'].decode('utf-8-sig'))
    links=[]
    for target in parser.targets:
        url=urlsplit(target)
        if url.scheme or url.netloc or url.path.startswith(('/', '\\')):raise ValueError('Nonrelative gallery resource: '+target)
        if not url.path:continue
        normalized=posixpath.normpath(unquote(url.path))
        if normalized.startswith('../') or normalized not in payload:raise ValueError('Unpackaged gallery resource: '+target)
        links.append(normalized)
    if not expected_images.issubset(links):raise ValueError('Gallery does not link all16 original images')
    statuses={name:parse_json(payload[name]).get('status') for name in (
        'audit/final_accounting.json','audit/input_pairing.json','audit/final_checkpoint_verification.json',
        'audit/resource_timing_audit.json','evaluation/scoring_receipt.json','evaluation/conflict_scoring_receipt.json')}
    expected_statuses=['COMPLETE','PASS','PASS','CLOSED_OBSERVED_INTERVALS','COMPLETE','COMPLETE']
    if final and list(statuses.values())!=expected_statuses:raise ValueError('Final audit statuses not complete: '+repr(statuses))
    return dict(csv_rows=counts,fixed_outputs=48,models=6,parents=8,metrics=40,strata={k:len(v) for k,v in strata.items()},
                source_files=len(copied_sources),original_images=16,relative_gallery_links=links,
                final_audit_statuses=statuses,all_registered_pair_and_stratum_keys_verified=True,
                source_snapshot_hashes_verified=True,visual_review_validation='Identity columns and row count only; observations/mapping not interpreted by this packer')


def prepare(root,final):
    identity=dict(schema='schedule40_review_packer_identity_r1',source='audit/pack_review_zip_r1.py',
                  sha256=sha(Path(__file__).read_bytes()),scope='Independent CPU lossless archive and verification; no scientific source changes')
    identity_path=root/'audit/review_zip_packer_identity_r1.json'
    if identity_path.exists() and parse_json(identity_path.read_bytes())!=identity:raise ValueError('Packer identity changed after registration')
    if not identity_path.exists():identity_path.write_bytes(json_bytes(identity))
    payload,missing=sources(root)
    checks=validate(payload,missing,final)
    manifest=dict(schema='schedule40_review_zip_manifest_v1',byte_limit_exclusive=LIMIT,
        compression='ZIP_DEFLATED level9; original member bytes unchanged',
        status='COMPLETE_REVIEW_MEMBERSHIP' if not missing else 'PENDING_REPORT',
        excluded=['model weights and checkpoints','RAW meshes and trajectories','OT caches and giant input plans',
                  'full training call/exposure/log files and derived training_curves.csv',
                  'redundant intermediate report/diff copies not linked from SOURCE_INDEX'],
        omitted_large_evidence_note='Original complete local run retains exclusions. Final optimizer/input/checkpoint audits, complete48 outcomes, all paired/stratified tables and16 original images are included.',
        audit_precedence='resource_timing_audit.json provides corrected training/sampling phase timing; original final_accounting.json is retained unchanged.',
        manifest_self_hash_note='This member hashes payload members; the external verification separately hashes every ZIP member including this manifest.',
        checks=checks,missing=missing,
        members=[dict(path=name,bytes=len(data),sha256=sha(data)) for name,data in payload.items()])
    payload['REVIEW_MANIFEST.json']=json_bytes(manifest)
    compressed={name:len(zlib.compress(data,9))-6 for name,data in payload.items()}
    estimated=sum(compressed.values())+sum(76+2*len(name.encode('utf-8')) for name in payload)+22
    inventory=dict(schema='schedule40_review_dry_inventory_v1',mode='DRY_INVENTORY',zip_written=False,
        estimated_zip_bytes=estimated,limit_exclusive=LIMIT,estimated_margin_bytes=LIMIT-estimated,
        pending_missing=missing,payload_file_count=len(payload),uncompressed_bytes=sum(map(len,payload.values())),
        checks=checks,largest_compressed_members=sorted([dict(path=k,compressed_bytes=v) for k,v in compressed.items()],key=lambda x:-x['compressed_bytes'])[:25],
        packer_source_sha256=identity['sha256'])
    return payload,manifest,inventory


def verify_zip(data,payload,checks):
    members=[]
    with zipfile.ZipFile(io.BytesIO(data),'r') as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(payload):raise ValueError('ZIP member set differs')
        if archive.testzip() is not None:raise ValueError('ZIP CRC failed')
        for name in names:
            actual=archive.read(name)
            if actual!=payload[name]:raise ValueError('ZIP member bytes differ: '+name)
            members.append(dict(path=name,sha256=sha(actual),bytes=len(actual),compressed_bytes=archive.getinfo(name).compress_size))
        # Re-run all core membership/HTML/source checks on extracted ZIP bytes.
        validate({name:archive.read(name) for name in names if name!='REVIEW_MANIFEST.json'},[],True)
    if len(data)>=LIMIT:raise ValueError(f'ZIP exceeds exclusive limit: {len(data)} >= {LIMIT}')
    return dict(status='PASS',zip_bytes=len(data),byte_limit_exclusive=LIMIT,margin_bytes=LIMIT-len(data),
                zip_sha256=sha(data),member_count=len(members),all_member_hashes_verified=True,
                crc_verified=True,core_membership_revalidated_from_zip=True,checks=checks,members=members)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(DEFAULT))
    parser.add_argument('--finalize',action='store_true',help='Explicitly approved final archive; default writes dry inventory only')
    args=parser.parse_args();root=args.out.resolve();tick=time.perf_counter()
    payload,manifest,inventory=prepare(root,args.finalize)
    write_result(root/'audit/review_zip_dry_inventory.json',inventory)
    if not args.finalize:
        print(json.dumps({k:v for k,v in inventory.items() if k not in ('checks','largest_compressed_members')},sort_keys=True));return
    if inventory['estimated_zip_bytes']>=LIMIT:raise RuntimeError('Estimated package exceeds10,000,000 bytes; no ZIP written, no image or panel omitted')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9,allowZip64=False) as archive:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,8,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.create_system=3;info.external_attr=0o100644 << 16
            archive.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    zipped=buffer.getvalue();verification=verify_zip(zipped,payload,inventory['checks'])
    if len(zipped)!=inventory['estimated_zip_bytes']:raise ValueError('ZIP byte estimate mismatch; review packer metadata assumptions')
    # Verify every file remained byte-identical while building the archive.
    for name,data in payload.items():
        if name!='REVIEW_MANIFEST.json' and (root/name).read_bytes()!=data:raise RuntimeError('Source changed during packaging: '+name)
    target=root/ZIP_NAME
    if target.exists():
        if target.read_bytes()!=zipped:raise RuntimeError('Existing review ZIP differs; refusing overwrite')
    else:
        with target.open('xb') as handle:handle.write(zipped)
    with zipfile.ZipFile(target,'r') as archive:
        if archive.testzip() is not None:raise ValueError('Written ZIP CRC failure')
    verification.update(path=str(target),wall_seconds=time.perf_counter()-tick,source_files_unchanged_during_pack=True,
                        model_forwards=0,scorer_calls=0,images_transformed=0)
    write_result(root/'audit/review_zip_manifest.json',manifest)
    write_result(root/'audit/review_zip_verification.json',verification)
    print(json.dumps({k:v for k,v in verification.items() if k not in ('checks','members')},sort_keys=True))


if __name__=='__main__':main()