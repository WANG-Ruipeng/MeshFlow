"""Copy the explicit R2 dependency closure without changing historical sources."""
from pathlib import Path
import ast, hashlib, json, shutil
SRC=Path('/mnt/e/MeshFlow-Control')
DST=Path('/mnt/e/MeshFlow/r2')
OUT=Path('/mnt/e/MeshFlow/native_t1/maintenance/r2_port_20261009')
OUT.mkdir(parents=True,exist_ok=True)
if DST.exists(): raise RuntimeError('r2 destination already exists; do not overwrite')
package=SRC/'src/meshflow_control'
mods={}
for p in package.rglob('*.py'):
    bits=p.relative_to(SRC/'src').with_suffix('').parts
    if bits[-1]=='__init__':bits=bits[:-1]
    mods['.'.join(bits)]=p
roots=['training.recipe_factorial','training.recipe_confirmation','data.recipe_factorial',
       'data.recipe_confirmation','sampling','mesh_io','runtime','accounting','precision',
       'data.contextual_stream','data.prepare']
todo=['meshflow_control.'+x for x in roots]; seen=set()
while todo:
    mod=todo.pop()
    if mod in seen or mod not in mods:continue
    seen.add(mod)
    p=mods[mod]
    pkg=mod if p.name=='__init__.py' else mod.rsplit('.',1)[0]
    for i in range(1,len(mod.split('.'))):todo.append('.'.join(mod.split('.')[:i]))
    for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
        if isinstance(node,ast.Import):
            todo.extend(n.name for n in node.names if n.name.startswith('meshflow_control'))
        elif isinstance(node,ast.ImportFrom):
            if node.level:
                parts=pkg.split('.')
                prefix='.'.join(parts[:len(parts)-node.level+1])
                base=prefix+('.'+node.module if node.module else '')
            else:base=node.module or ''
            if base.startswith('meshflow_control'):
                todo.append(base);todo.extend(base+'.'+n.name for n in node.names)
files=[mods[x] for x in sorted(seen)]
files += [SRC/'tools/train_recipe_factorial.py',SRC/'tools/train_recipe_confirmation.py',
          SRC/'LICENSE',SRC/'THIRD_PARTY_NOTICES.md']
tests=['test_recipe_factorial_train.py','test_recipe_factorial_data.py',
       'test_recipe_confirmation_train.py','test_recipe_confirmation_data.py',
       'test_training_data.py','test_precision.py']
files += [SRC/'tests'/t for t in tests]
rows=[]
for p in files:
    rel=p.relative_to(SRC)
    dest=DST/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(p,dest)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(p)==sha(dest)
    rows.append(dict(path=str(rel),source=str(p),sha256=sha(p),bytes=p.stat().st_size))
doc=dict(schema='r2_source_port_v1',source_root=str(SRC),destination=str(DST),
         source_files_copied_byte_exact=rows,source_mutations=0,
         historical_adapters='Byte-exact dependency closure preserves checkpoint, stream and input hashes. Maintained new entry exposes only R2.',
         upstream_remote='https://github.com/WANG-Ruipeng/MeshFlow.git',
         verified_remote_master='8961b7a41fc153b5b45f0c8903370c6bfd1cf04f')
(DST/'docs').mkdir(exist_ok=True)
(DST/'docs/source_manifest.json').write_text(json.dumps(doc,indent=2)+'\n')
(OUT/'copy_receipt.json').write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps(dict(copied=len(rows),bytes=sum(r['bytes'] for r in rows),modules=sorted(seen)),indent=2))
