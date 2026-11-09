from pathlib import Path
import json,re,sys,datetime
root=Path('experiments/research_history_audit_20261014')
rows=json.loads((root/'source_inventory.json').read_text())
ids=[int(v) for v in sys.argv[1].split(',')]
for r in rows:
 if r['id'] not in ids:continue
 lines=Path(r['path']).read_text(encoding='utf-8-sig').splitlines();fence=False;body=[]
 for n,s in enumerate(lines,1):
  if s.strip().startswith('```'):fence=not fence;continue
  if not fence and s.strip():body.append((n,s))
 if len(sys.argv)>2 and sys.argv[2]=='heads':body=[(n,s) for n,s in body if s.startswith('#')]
 elif len(sys.argv)>2 and sys.argv[2]=='brief':
  body=[(n,s) for n,s in body if n<=22 or n>len(lines)-28 or (not s.startswith('|') and re.search(r'\u7ed3\u8bba|\u664b\u7ea7|\u5931\u8d25|\u4e0d\u652f\u6301|\u4e0d\u5efa\u8bae|\u672a\u5efa\u7acb|\u963b\u585e|\u5426\u51b3|\u65e0\u6cd5|\u4f18\u5148|\u672a\u901a\u8fc7|\u4e0b\u4e00\u6b65|\u4e0d\u7a33\u5b9a|\u6ca1\u6709\u8bc1\u636e|\u4ee3\u4ef7',s))]
 print('\nSOURCE',r['id'],r['relative_path'],len(lines),'lines')
 for n,s in body:print(str(n)+': '+s)
 with (root/'read_ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'id':r['id'],'mode':sys.argv[2] if len(sys.argv)>2 else 'outside_fences','line_numbers':[n for n,s in body],'sha256':r['sha256'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n')