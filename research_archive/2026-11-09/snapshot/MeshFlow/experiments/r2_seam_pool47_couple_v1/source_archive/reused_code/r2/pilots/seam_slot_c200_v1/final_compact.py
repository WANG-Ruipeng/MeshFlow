"""Final size-bounded evidence archive from the preserved full local archive."""
from pathlib import Path
import zipfile,json,hashlib
ROOT=Path(__file__).resolve().parents[3]/'experiments/r2_seam_slot_c200_v1'
def sha(x):return hashlib.sha256(x).hexdigest()
def main():
 dest=ROOT/'SEAM_SLOT_C200_REVIEW.zip';attempt=ROOT/'SEAM_SLOT_C200_REVIEW_COMPACT_ATTEMPT_LOCAL.zip'
 assert dest.resolve().parent==ROOT.resolve() and not attempt.exists()
 dest.rename(attempt)
 source=ROOT/'SEAM_SLOT_C200_REVIEW_FULL_LOCAL.zip'
 payload={};omitted=[]
 with zipfile.ZipFile(source) as z:
  assert z.testzip() is None
  for i in z.infolist():
   n=i.filename
   if n=='PACKAGE_MANIFEST.json':continue
   skip=n.endswith('/shape_per_face.json') or (n.endswith('/pairs.json') and not n.endswith('SEAM_SLOT_C200/3f741759_n2/POST/pairs.json')) or (n.startswith('training/') and n.endswith('/geometry_diagnostics.jsonl'))
   if skip:omitted.append(dict(path=n,bytes=i.file_size,compressed_bytes=i.compress_size));continue
   payload[n]=z.read(n)
 message='''# Compact completed-experiment review

REPORT.md contains the result and limitations. The paired 200-update training, 32 RAW + 32 POST, 64 scores and 64 actual nonblind AI_PROXY reviews are complete.

Included: ALL per-output/per-condition/per-noise metrics, ALL score.json and legal-edge CSV, seam-bulk face/area results, occupancy labels/predictions, all16 full-resolution paired gallery pages and reviews, per-update/per-micro training CSV, exposure/call logs, curves, frozen manifests, independent audits, actual original scoring/executed source copies and hashes, code.diff, real weight identities. The 3f/n2 bad-tail pair file is also included along with its cached-evidence audit.

Omitted from this compact zip only: big weights, training OT cache, mesh/trajectory/query-vector NPZ, full pair-candidate lists (except the named bad-tail POST), per-face shape JSON, and geometry_diagnostics.jsonl already represented in training_per_micro.csv. These complete original records are preserved locally. SEAM_SLOT_C200_REVIEW_FULL_LOCAL.zip retains the original larger review package. No evidence was deleted or regenerated.

This package supports aggregate recalculation, actual image/source review; it is not a standalone full geometry rerun environment. WEIGHTS.json points to retained actual checkpoints. Source JSON/CSV file paths refer to their original workspace locations.

Reviewer labels are AI_PROXY, nonblind. Visual contact does not certify topology. FULL has NA for candidate-specific seam roles. Recovery entry r2/pilots/seam_slot_c200_v1/resume.ps1 defaults to read-only status and does not start a new experiment.
'''
 (ROOT/'REVIEW_README.md').write_text(message,encoding='utf-8');payload['REVIEW_README.md']=message.encode()
 payload['packaging/final_compact.py']=Path(__file__).read_bytes()
 record=dict(status='SIZE_SELECTION_ONLY',preserved_full_local_bytes=source.stat().st_size,first_compact_bytes=attempt.stat().st_size,
  omitted_from_compact_only=omitted,model_calls=0,new_quality_scans=0,deleted_local_evidence=0)
 (ROOT/'audits/final_compact_selection.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
 payload['audits/final_compact_selection.json']=json.dumps(record,indent=2).encode()
 entries=[dict(path=k,bytes=len(v),sha256=sha(v)) for k,v in sorted(payload.items())]
 manifest=json.dumps(dict(version='compact_final_v3',files=entries,omitted_from_zip_only=omitted),indent=2).encode()
 (ROOT/'COMPACT_PACKAGE_MANIFEST_FINAL.json').write_bytes(manifest);payload['COMPACT_PACKAGE_MANIFEST_FINAL.json']=manifest
 with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for k,v in sorted(payload.items()):z.writestr(k,v)
 with zipfile.ZipFile(dest) as z:
  assert z.testzip() is None
  for e in entries:assert sha(z.read(e['path']))==e['sha256']
 assert dest.stat().st_size<10_000_000
 receipt=dict(status='PASS',archive=str(dest),bytes=dest.stat().st_size,sha256=sha(dest.read_bytes()),files=len(payload),
  all_zip_member_hashes_verified=True,manifest='COMPACT_PACKAGE_MANIFEST_FINAL.json')
 with (ROOT/'package_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
 print(json.dumps(receipt))
if __name__=='__main__':main()
