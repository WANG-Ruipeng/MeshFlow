import hashlib,json,pathlib,xml.etree.ElementTree as ET
root=pathlib.Path('.')
run=root/'output/a_taskaligned_20260924_01'
tests=ET.parse(run/'PYTEST_SEAL.xml').getroot().findall('testsuite')
count=sum(int(t.attrib['tests']) for t in tests)
assert count>=22
assert all(int(t.attrib[k])==0 for t in tests for k in ('failures','errors','skipped'))
path=run/'CHECKS.json';checks=json.loads(path.read_text())
checks['new_pytest']={'passed':count,'failed':0,'skipped':0,'scope':'all layout_student/test_*.py including serialized and cross-run identity regressions','command':'python -B -m pytest -q -p no:cacheprovider experiments/a_radial/layout_student --junitxml=output/a_taskaligned_20260924_01/PYTEST_SEAL.xml','junit_evidence':'PYTEST_SEAL.xml'}
checks['prior_baseline_replay']['note']='Vertices and connectivity values match the previous run exactly. Logical mesh hashes use the original int32 template connectivity; this does not claim compressed NPZ files or saved face dtypes are byte-identical.'
checks['mesh_integrity']={'evidence':'MESH_INTEGRITY.json','original_artifacts_preserved':True,'logical_face_dtype':'int32','storage_face_dtype':'int64 except 16 teacher rows int32','expected_rows':128,'expected_dtype_only_differences':112,'correction_provenance':'code_corrections/README_zh.md','full_seal_evidence':'FINAL_AUDIT.json'}
checks['tested_final_sources_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'experiments/a_radial/layout_student').glob('*.py'))}
path.write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'tests_passed':count,'checks_file':str(path)}))