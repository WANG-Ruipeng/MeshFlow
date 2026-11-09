from pathlib import Path
import json
p=Path('/mnt/e/MeshFlow-Control/README.md')
s=p.read_text()
old='The prepared training data contain 32 parents and 453 original READY conditions. The registered stream has 8,000 shared CPU OT maps. Successful CPU import, stream preparation and synthetic checks have external receipts; they are not evidence of completed GPU training or improved generation. Actual completion requires the external training status, call ledger, checkpoints, per-output evaluation and completed visual review. A rendered gallery alone is not a visual review.'
new='The fixed study completed all three 1,000-update arms, all six step-500/1,000 checkpoints and 272 new Euler50 outputs plus 16 verified START cached outputs. All 288 results were scored and passed exact C/trajectory/NPZ/PLY checks. The independent installed package passed 28 CPU tests; real GPU execution is recorded separately. Training used 32 parents, 453 READY conditions and 8,000 byte-paired CPU OT inputs per arm. See [actual results and limits](docs/results.md); external artifacts retain the full per-output tables, gallery, actual AI review, call ledger and EXPERIMENT_REPORT.md.'
assert old in s;s=s.replace(old,new)
s=s.replace('Source-target error and reference-conditioned S/I/E are useful diagnostics with stated uncertainty','Source-target error and reference-conditioned S/E, together with source-independent C/F conflict I, are useful diagnostics with stated uncertainty')
p.write_text(s)
t=json.loads(Path('/mnt/e/MeshFlow-Control-artifacts/analysis/report_tables.json').read_text())
result='''# Actual Direct Context Readout V1 results

The fixed local study completed A/B/C at 1,000 updates each, with secondary checkpoints at 500. Each arm generated 32 P1 outputs at 500 and 48 P1/P2 outputs at 1,000: 80 full Euler50 outputs per arm. START supplied 32 new outputs and 16 verified historical cached outputs. All 288 scored outputs passed C byte/order/multiplicity/winding, all 51 trajectory states, actual Gaussian, finite N/K and lossless NPZ/PLY checks. Hard retention does not establish interface welding or semantic compatibility.

P1 uses eight previously observed DEV parents, four noises each, aggregated within parent before equally weighting parents. Primary 1,000-update results are below. Percentage rows multiply a fraction by 100; distance MSE/L² rows and diversity multiply by 1,000. Source S/E/distance/coverage are reference-completion diagnostics, not unique-GT acceptance criteria.

'''+t['p1_1000']+'''

B and C did not show a consistent advantage over equal-budget A. B-A and C-A mean Dsym changed by -0.554% and -1.575%, respectively, but each had four improving and four worsening parent means; all eight parents had mixed noise directions. C-B reduced FF affected-face fraction in seven of eight parents, while CF/I increased and reference coverage/p95 worsened. C-B's mean Dsym advantage reverses after omitting f596 in a descriptive sensitivity check; no parent was removed from the actual results.

All 500-update observations remain in the external report. The primary endpoint was not changed retrospectively. No model was promoted and START remains the default.

P2 has four fixed controls and four candidates per control, with no hidden target. The two seat tessellations are not independent shape evidence. Actual multi-view inspection shows recognizable chair variants, but also seat overhang/misalignment, duplicate or separate seat layers, generated back panels opposite the supplied back, and local spikes. B/C did not consistently resolve these. Sparse controls are out-of-training-distribution pressure tests, not an estimate of all-context failure. Human usability is NOT_RUN.

The readouts learned nonzero updates; the very small training-FM changes and attention entropy are not quality evidence. Residual/hidden norms were measured before the BF16 addition, so effective post-addition signal or quantization loss remains unmeasured. One continuation stream does not establish across-training-seed robustness or unseen-parent generalization.

Actual work: 24,000 training forwards and backwards, 3,000 optimizer steps; 47/11 one-step preflight F/B; 278 completed Euler50 attempts including six initialization replays; 13,900 sampling forwards; 92.26 minutes cumulative GPU lock ownership. No model failure or training/sample retry occurred. Engineering adapter/import-check failures and fixes were retained separately. B/C F+B cost was approximately 11% above A on the measured RTX 5080; overlapping CPU audit work prevents treating this as an isolated benchmark.

The source package works without historical Python imports or the old workspace on PYTHONPATH. It needs separately supplied START or new candidate weights and prepared arrays. GPU orchestration currently uses Linux/WSL POSIX locks. See [reproduction](reproduction.md), [metric definitions](metrics.md), [source map](source_map.md), [integration notes](integration.md), and [license exceptions](../THIRD_PARTY_NOTICES.md).

The full local record is stored outside the source checkout in the sibling MeshFlow-Control-artifacts directory: EXPERIMENT_REPORT.md, METHOD.md, checkpoints/, outputs/all_completed.json, evaluation/per_output_metrics.csv, analysis/comparisons/, analysis/visual_review_all.csv, gallery/INDEX.html and audit/. These are external delivery artifacts, not runtime imports. The installed package does not require that sibling name; CLI/config paths select assets. This source checkout contains no model weights or private data.

The fixed study is finished. No additional training, loss changes, postprocessing, architecture search, commit/push or default-model change was performed.
'''
Path('/mnt/e/MeshFlow-Control/docs/results.md').write_text(result,encoding='utf-8')
print('README actual-results section and docs/results.md written')
