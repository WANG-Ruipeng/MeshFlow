import ast,json
from pathlib import Path
paths=list(Path(__file__).parent.glob('*.py'))
for p in paths:ast.parse(p.read_text(encoding='utf8'),filename=str(p))
print(json.dumps(dict(status='AST_PASS',files=len(paths),model_calls=0)))