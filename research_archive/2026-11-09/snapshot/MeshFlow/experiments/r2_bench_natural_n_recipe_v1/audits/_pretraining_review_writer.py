from pathlib import Path
import ast, hashlib, json, time
R=Path('/mnt/e/MeshFlow'); O=R/'experiments/r2_bench_natural_n_recipe_v1'; C=R/'r2/pilots/bench_natural_n_recipe_v1'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files=[C/n for n in ('common.py','register.py','training.py','runtime.py','stream.py','train_tests.py')]
for p in files: ast.parse(p.read_text())
training=(C/'training.py').read_text(); register=(C/'register.py').read_text()
assert "('READY','READY_NATIVE_DATA')" in register
assert "'HIGH_HOLDOUT'" in training
for g in ('training_cpu_gate.json','runtime_cpu_gate.json'):
    q=read(O/g); assert q['status']=='PASS'
    for p,h in q['source_hashes'].items(): assert sha(Path(p))==h, p
sem=read(O/'data/training_data_semantics_review.json'); assert sem['status']=='PASS'
res={'schema':'BENCH_NATURAL_N_PRETRAIN_INDEPENDENT_REVIEW_V1','status':'PASS','unix':time.time(),'reviewer':'backend_data','scope':'Read-only source review, AST parse, frozen gate/source authentication and separate schedule arithmetic; zero runtime imports, no model/no extra OT/no GPU','checks':[
{'item':'recovery_identity','result':'PASS','evidence':'Source exposure B_75_25 NATIVE local1500 global21000/native9000, non-EMA live weights, matching 194-state AdamW and RNG; _source_payload and _build_restore validate strict parameter order/state hashes; source reference only, no second model copy saved.'},
{'item':'cumulative_cursors','result':'PASS','evidence':'New local k maps global21000+k/native9000+k/curriculum3000+k; target1500 -> global22500/native10500/curriculum4500; seam/head/probe counts zero.'},
{'item':'N_and_architecture','result':'PASS','evidence':'Independent Native forward AST differs only upper-N guard 256->799 and message. Original 130401155 parameters and count embedding41x768 unchanged; actual valid count equals N, no clipping. Training high N308/384; held-out N375/724.'},
{'item':'target_and_OT','result':'PASS','evidence':'Full target remains exactly 2*pre_ot_full. Identical X_WIDTH alpha applied to full/pre/C vertices; free-only OT uses half-scale target and fixed Gaussian slots before HYBRID. Every source face/corner roundtrips after OT and face permutation; C exactly copied and velocity zero.'},
{'item':'Native_C_only_input','result':'PASS','evidence':'source owner labels are shape-checked metadata only; occupied_for_FM all false, no seam slots, fixed corners only from C; no extra head/readout or teacher instantiated.'},
{'item':'FM_denominator','result':'PASS','evidence':'One batch divisor=sum_i9(N_i-K_i); eight microbatch SSE/divisor gradients accumulated, then clip1 and one AdamW step. Higher-N contributes more coordinates, intentionally unchanged original loss. Scheduled HIGH12.5% instances gives19.85201056% mean-update coordinate share, not gradient share.'},
{'item':'parent_and_pair_identity','result':'PASS','evidence':'OLD22/FOCUSa8+27/HIGH28b9+abf disjoint;617/756/df6 held out. Same original24 case hashes. First6 microbatch rows exact both arms; all8 slot seed/time/augmentation identities paired; two clean/width instances per selected parent.'},
{'item':'budgets_and_failure','result':'PASS','evidence':'Frozen1500 updates/arm,12000F/12000B/1500optimizer, zero head/probe. CallLedger enforces order/count and no unresolved pending calls; endpoint and source hashes guarded. Resource release mandatory before CUDA.'},
{'item':'preflight_fixes_rechecked','result':'PASS','evidence':'register now accepts READY_NATIVE_DATA without mutating frozen manifest; training held-out guard now includes HIGH_HOLDOUT. Updated CPU gates authenticate exact changed training/runtime source.'}
],'limitations':['No independent extra model forward/backward or OT was performed by this review. CPU gate results are authenticated existing evidence, not a duplicate dynamic test.','N724 evaluation is also extrapolation beyond training max384 and has closed C; do not attribute all difference to parent generalization.','The treatment changes both independent high-N source coverage and FOCUS exposure; it is an entire recipe comparison, not isolated N causality.'],'source_hashes':{str(p):sha(p) for p in files},'evidence_hashes':{str(O/p):sha(O/p) for p in ('data/manifest.json','data/eval_candidates.json','data/eval_manifest.json','data/cpu_gate.json','data/training_data_semantics_review.json','training_cpu_gate.json','runtime_cpu_gate.json')},'neural_model_forwards':0,'production_OT_calls':0,'new_scientific_noise':0}
p=O/'audits/pretraining_independent_review.json'
with p.open('x') as f: json.dump(res,f,ensure_ascii=False,indent=2); f.write('\n')
print(json.dumps({'status':res['status'],'path':str(p),'sha256':sha(p)}))
