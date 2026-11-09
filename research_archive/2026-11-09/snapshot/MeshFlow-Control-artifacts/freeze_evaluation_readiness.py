from pathlib import Path
import hashlib,json,datetime
root=Path('/mnt/e/MeshFlow-Control');art=Path('/mnt/e/MeshFlow-Control-artifacts')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=list((root/'src/meshflow_control/evaluation').glob('*.py'))+[root/'tests/test_evaluation.py',root/'docs/metrics.md',root/'docs/evaluation_source_map.json']
v=dict(status='READY_FROZEN',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='CPU evaluation/render modules only; no model, optimizer, CUDA, or generated mesh execution',
    sources={str(p.relative_to(root)):sha(p) for p in files},
    panels=dict(requests=str(art/'data/evaluation/requests.json'),P1=32,P2=16,historical_START=16,
                request_manifest_sha256=sha(art/'data/evaluation/requests.json')),
    cpu_tests=dict(passed=7,failed=0,final_command='python -B -m pytest -q tests/test_evaluation.py',actual_seconds=2.03),
    scalar_regression=json.loads((art/'tests/evaluation_migration/regression.json').read_text()),
    repack_test=json.loads((art/'tests/evaluation_repack/receipt.json').read_text()),
    render_review=dict(kind='AI_VISUAL_REVIEW of synthetic CPU fixture only',status='PASS',
        path=str(art/'tests/evaluation_cpu_final_ready/test_render_complete_guard_and0/rendered/blind/synthetic__n0.png'),
        observations='Four views and wire present; consistent scale; C/free colors; title/row labels readable after spacing fix; no model output quality inference'),
    engineering_failures=['Extraction initially omitted math and Counter imports; fixed before freeze',
        'Initial pytest invocation before local installation lacked package path; temporary new-src-only test path used',
        'Matplotlib empty facecolors incompatible for wire; replaced by explicit Line3DCollection before freeze',
        'Static audit found local fallback historical import; replaced by migrated identical verifier before freeze'],
    original_repository_writes=0,model_calls=0)
p=art/'audit/evaluation_readiness.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'receipt':str(p),'evaluate_sha256':v['sources']['src/meshflow_control/evaluation/evaluate.py'],'source_map_sha256':v['sources']['docs/evaluation_source_map.json']}))
