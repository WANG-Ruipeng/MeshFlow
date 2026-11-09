"""Retained CPU-only source review receipt generator; never imports the model."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
CODE=ROOT/'experiments/mf_h1_local0/chair_control_feedback_auto_v1'
OUT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

files={name:CODE/name for name in ('checkpoint.py','training.py','worker.py','common.py','replay.py','evaluation.py')}
text={name:p.read_text(encoding='utf-8') for name,p in files.items()}
for source in text.values():ast.parse(source)
cp,tr=text['checkpoint.py'],text['training.py']
assert "feedback_pool_sha256=digest(out/'feedback/pool_v1/manifest.json')" in cp
assert "assert meta['protocol_sha256']==json_hash(protocol)" in tr
assert "meta['feedback_pool_sha256']==digest(out/'feedback/pool_v1/manifest.json')" in tr
assert tr.index("assert prepared['status']=='PASS'") < tr.index('while step<until:')

findings=[
    dict(id='R1',status='RESOLVED',file='checkpoint.py',
         before='Saved feedback_pool_sha256 read a nonexistent feedback/pool.json and became None.',
         after='Saves actual feedback/pool_v1/manifest.json hash; auxiliary resume compares that version.'),
    dict(id='R2',status='RESOLVED',file='training.py',
         before='Restore checked arm/phase but not the currently supplied protocol.',
         after='Restore requires checkpoint protocol hash to equal current protocol JSON hash.'),
    dict(id='R3',status='RESOLVED',file='training.py',
         before='A BLOCKED feedback prepare result could be ignored until after primary forward/backward calls.',
         after='Requires prepare_feedback status PASS before entering the first training update.'),
    dict(id='R4',status='RESOLVED_BY_SCORING_OWNER',file='evaluation.py',
         before='Export contract booleans were read from different nesting than worker writes.',
         after='Scoring owner confirmed top-level facts plus actual NPZ/PLY readback are now consumed.'),
]
accepted=[
    'START load delegates to original strict checkpoint/AdamW restoration and verifies total5000/OT_HYBRID; historical stream state is discarded for new phase cursor0.',
    'Restored continuation binds model/optimizer/RNG/stream hashes, AdamW step5000+s, original optimizer hyperparameters and new phase/arm identity.',
    'Primary masked FM uses sum of all eight valid-free scalar coordinates as common denominator for every microbatch; no per-sample reweighting introduced.',
    'Additional feedback loss is .1 times one valid-free mean velocity squared error; NumPy fixed target is supervision, differentiable model velocity is not detached.',
    'Primary and auxiliary loss.backward run inside native_execution; gradients accumulate and clip once before the same restored optimizer step.',
    'Saved DEV evaluation uses model.set_trainable(False).eval and clears gradients; successful return restores captured RNG, set_trainable(True).eval and clears gradients.',
    'Saved rollout retains all51 states plus50 velocities, original noise and C; binary PLY/NPZ export re-reads exact geometry and known/source associations.',
]
receipt=dict(status='STATIC_REVIEW_PASS',scope='Only source review. No model load/forward/backward/update, CUDA, noise draw, OT or metric computation.',
    source_files={str(p):sha(p) for p in files.values()},resolved_findings=findings,accepted_contracts=accepted,
    recovery_note='If checkpoint500 exists but DEV500 failed, controller must finish that frozen label through sample mode before claiming the 500-step panel complete; train resume starts at501.',
    runtime_claim='This receipt does not substitute for actual training, gradient or recovery evidence.',
    reviewer_source=str(Path(__file__)),reviewer_source_sha256=sha(__file__),model_calls=0,OT_calls=0)
(OUT/'training_source_review.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=receipt['status'],resolved_findings=len(findings),receipt=str(OUT/'training_source_review.json'))))
