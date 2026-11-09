"""Small CPU interface/readiness/pixel-copy checks; zero actual rendering."""
from pathlib import Path
import ast
from collections import defaultdict
import hashlib
import json
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
sys.path.insert(0,str(ROOT))
from experiments.mf_h1_local0.chair_capped500_half_v1 import visualization as v
from experiments.mf_h1_local0.chair_capped500_half_v1 import inspect_gallery as inspector

assert v.FIXED_LABELS==('START','HIST_CAPPED500','HALF_CAPPED500')
assert tuple(v.ev.LABELS)==v.FIXED_LABELS
assert v.renderer._renderer_module is None
assert [x[0] for x in v.VIEWS]==list(inspector.VIEWS)
pixel=inspector.self_test()
tree=ast.parse(Path(v.__file__).read_text(encoding='utf-8'))
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_require_complete')
rows=[dict(request_id=f'p{p}_n{n}',uid=f'p{p}',candidate=n,split='OLD' if n<2 else 'FRESH')
      for p in range(8) for n in range(4)]
fixture=SimpleNamespace(rows=rows,missing=False,bad_hash=False,calls=0)
def raw(out,label,q):
    fixture.calls+=1
    return None if fixture.missing and q['request_id']=='p7_n3' and label=='HALF_CAPPED500' else 'immutable_raw'
def metric(out,label,q):
    return dict(metrics=dict(raw_sha256='bad' if fixture.bad_hash else 'expected'))
env=dict(defaultdict=defaultdict,FIXED_LABELS=v.FIXED_LABELS,_raw=raw,_metric_source=metric,
    ev=SimpleNamespace(LABELS=v.FIXED_LABELS,manifest=lambda out,panel:fixture.rows,digest=lambda path:'expected'))
exec(compile(ast.Module(body=[function],type_ignores=[]),'actual_readiness_function','exec'),env)
check=env['_require_complete'];check(None)
assert fixture.calls==96
cases=[]
for name,change,error in [('missing_cell',lambda:setattr(fixture,'missing',True),RuntimeError),
                          ('wrong_raw_identity',lambda:setattr(fixture,'bad_hash',True),ValueError),
                          ('missing_request',lambda:setattr(fixture,'rows',rows[:-1]),ValueError)]:
    fixture.rows=rows;fixture.missing=False;fixture.bad_hash=False
    change()
    try:check(None)
    except error:cases.append(dict(case=name,status='PASS'))
    else:raise AssertionError('Failed to reject '+name)
assert v.renderer._renderer_module is None
result=dict(status='PASS',labels=list(v.FIXED_LABELS),pixel_copy=pixel,
    complete_96_cell_guard='PASS',rejection_cases=cases,renderer_not_initialized=True,
    source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (v.__file__,inspector.__file__)},
    model_calls=0,scoring_calls=0,render_calls=0)
(HERE/'cpu_contract_receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
