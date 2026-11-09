"""Compact a verified review archive; preserve complete local evidence."""
from pathlib import Path
import zipfile,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[3]/'experiments/r2_seam_slot_c200_v1'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 original=ROOT/'SEAM_SLOT_C200_REVIEW.zip'
 kept=ROOT/'SEAM_SLOT_C200_REVIEW_FULL_LOCAL.zip'
 assert original.resolve().parent==ROOT.resolve() and not kept.exists()
 original.rename(kept)
 old_readme=(ROOT/'REVIEW_README.md').read_bytes()
 (ROOT/'audits/REVIEW_README_full_archive.md').write_bytes(old_readme)
 message='''# Compact review evidence (below 10 MB)

Read REPORT.md first. Both 200-update arms, 32 RAW + 32 POST, 64 scores and all 64 nonblind AI_PROXY image reviews were actually completed.

Includes complete per-output/per-parent/per-noise score tables, all score.json files and legal-edge CSV, POST exact pair records, occupancy by edge/condition, all 16 full-resolution paired image pages, training per-update/per-micro CSV and learning curves, exposure/call logs, original scorer and executed source copies/hashes, code.diff, protocol/manifests, independent audits and actual weight identities.

Omitted ONLY FROM THIS ZIP: large weights, training OT cache, mesh/trajectory/query-vector NPZ, large RAW pair lists, per-face shape JSON, and geometry_diagnostics.jsonl that is also represented in training_per_micro.csv. Original records remain in the experiment directory. The first full local review archive is also preserved as SEAM_SLOT_C200_REVIEW_FULL_LOCAL.zip. Nothing was regenerated or deleted to make this small archive.

This is evidence for aggregate recalculation and source review, not a standalone geometry rerun installation. WEIGHTS.json points to the actual saved checkpoints. Full meshes and all unabridged scorer outputs remain locally available.

AI_PROXY is nonblind and not an expert human rating. Contact appearance does not certify shared indices. FULL's candidate-specific seam-to-bulk values are NA, not zero. See PROTOCOL.md for exact comparison boundaries.

Recovery: r2/pilots/seam_slot_c200_v1/resume.ps1 defaults to read-only status; completed stages verify receipts without extra model calls.
'''
 (ROOT/'REVIEW_README.md').write_text(message,encoding='utf-8')
 payload={};omitted=[]
 with zipfile.ZipFile(kept) as z:
  assert z.testzip() is None
  for i in z.infolist():
   name=i.filename
   if name=='PACKAGE_MANIFEST.json':continue
   if (name.endswith('/shape_per_face.json') or (name.endswith('/pairs.json') and '/RAW/' in name)
       or (name.startswith('training/') and name.endswith('/geometry_diagnostics.jsonl'))):
    omitted.append(dict(path=name,bytes=i.file_size,compressed_bytes=i.compress_size));continue
   payload[name]=z.read(name)
 payload['REVIEW_README.md']=message.encode()
 payload['packaging/compact_review.py']=Path(__file__).read_bytes()
 audit=dict(stage='archive_size_only',initial_bytes=kept.stat().st_size,limit_bytes=10_000_000,
  preserved_full_archive=str(kept),omitted_from_compact_only=omitted,
  model_calls=0,new_scores=0,geometry_edits=0)
 audit_bytes=json.dumps(audit,indent=2).encode()
 (ROOT/'audits/compact_archive_size_record.json').write_bytes(audit_bytes)
 payload['audits/compact_archive_size_record.json']=audit_bytes
 entries=[dict(path=k,bytes=len(v),sha256=sha(v)) for k,v in sorted(payload.items())]
 manifest=json.dumps(dict(version='compact_v2',files=entries,omitted_from_zip_only=omitted),indent=2).encode()
 (ROOT/'COMPACT_PACKAGE_MANIFEST.json').write_bytes(manifest)
 payload['COMPACT_PACKAGE_MANIFEST.json']=manifest
 with zipfile.ZipFile(original,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for k,v in sorted(payload.items()):z.writestr(k,v)
 with zipfile.ZipFile(original) as z:
  assert z.testzip() is None
  for e in entries:assert sha(z.read(e['path']))==e['sha256']
 assert original.stat().st_size<10_000_000
 receipt=dict(status='PASS',archive=str(original),bytes=original.stat().st_size,sha256=sha(original.read_bytes()),
  files=len(payload),full_local_archive=str(kept),all_zip_member_hashes_verified=True,source_manifest='COMPACT_PACKAGE_MANIFEST.json')
 with (ROOT/'package_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
 print(json.dumps(receipt))
if __name__=='__main__':main()
