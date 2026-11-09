import ast,json,hashlib,pathlib
root=pathlib.Path('.')
run=root/'output/a_taskaligned_20260924_01'
for rel in ['experiments/a_radial/layout_student/evaluate.py','experiments/a_radial/layout_student/summarize.py']:
    ast.parse((root/rel).read_text())
old=run/'code_corrections/evaluate_before_hash_clarification.py'
h=hashlib.sha256(old.read_bytes()).hexdigest()
inv=[json.loads(p.read_text()) for p in sorted((run/'eval_invocations').glob('*.json'))]
checks=[x['source_hashes']['layout_student/evaluate.py']==h for x in inv]
assert inv and all(checks)
record={'syntax':'PASS','original_evaluation_source_sha256':h,'original_source_matches_all_invocations':True,'invocations_checked':len(inv),'new_evaluation_source_sha256':hashlib.sha256((root/'experiments/a_radial/layout_student/evaluate.py').read_bytes()).hexdigest(),'note':'Future ledger recording clarified after experiment. Original NPZ, ledger, model, metrics and deployment unchanged.'}
path=run/'code_corrections/SOURCE_RECORD.json'
with path.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))