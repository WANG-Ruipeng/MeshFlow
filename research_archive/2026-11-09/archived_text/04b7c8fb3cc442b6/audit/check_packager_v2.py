import ast
from pathlib import Path
p=Path('/mnt/e/MeshFlow-Control/tools/package_recipe_review_v2.py')
ast.parse(p.read_text(encoding='utf8'))
print('packager_v2 AST PASS; no package built')
