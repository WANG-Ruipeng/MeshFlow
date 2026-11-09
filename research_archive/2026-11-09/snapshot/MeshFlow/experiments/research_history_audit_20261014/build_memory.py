from pathlib import Path
import json,re,hashlib,datetime
root=Path('/mnt/e/MeshFlow');out=root/'experiments/research_history_audit_20261014';docs=root/'r2/docs'
rows=json.loads((out/'source_inventory.json').read_text())
def win(s):
 return re.sub(r'^/mnt/([a-z])/',lambda m:m[1].upper()+':/',s)
def target(r):
 p=Path(r['path'])
 try:return '../../'+str(p.relative_to(root))
 except ValueError:return win(str(p))
lookup={r['id']:r for r in rows}
p=docs/'research_route_memory.md';raw=p.read_text(encoding='utf-8');routes=[]
for line in raw.splitlines():
 m=re.match(r'\| \*\*((?:R|B)\d\d) (.*?)\*\* \| (.*?) \| (.*?) \|$',line)
 if m:
  ids=[int(a or b) for a,b in re.findall(r'(?:\[\[S(\d+)\]\]|\[S(\d+)\]\()',m[3])]
  routes.append(dict(id=m[1],name=m[2],status='NOT_STABLE_UPGRADE' if m[1][0]=='R' else 'PRETRAIN_GATE_BLOCKED',evidence=re.sub(r'\[\[S\d+\]\]','',m[3]).strip(),retry_boundary=m[4],source_ids=ids))
assert {r['id'] for r in routes if r['id'][0]=='R'}=={f'R{i:02d}' for i in range(1,29)}
assert {r['id'] for r in routes if r['id'][0]=='B'}=={'B01','B02'}
raw=re.sub(r'\[\[S(\d+)\]\]',lambda m:'[S'+m[1]+'](<'+target(lookup[int(m[1])])+'>)',raw)
p.write_text(raw,encoding='utf-8')
reg=dict(version='research_route_memory_20261014_v1',scope='mechanism families; historical reports, not new experiments',counts=dict(not_stable_upgrade=28,pretrain_gate_blocked=2,primary_reports=len(rows)),routes=sorted(routes,key=lambda x:x['id']))
(docs/'research_route_registry.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest=[];md=['# Local experiment source index','', '2026-10-14. Primary reports: 83. This index records existing reports, not 83 independent methods or training replications. SHA256 is for the current readable report, including disclosed document repairs. Historical drafts, packaged copies, cleanup notes and nested audits are not counted as extra primary experiments.','', 'Companion: [research route memory](research_route_memory.md), [machine-readable route register](research_route_registry.json).','', '| ID | Existing report | Lines | Bytes | SHA256 |','|---|---|---:|---:|---|']
for r in rows:
 b=Path(r['path']).read_bytes();actual=hashlib.sha256(b).hexdigest();assert actual==r['sha256'],r['path']
 x=dict(id=r['id'],path=win(r['path']),workspace_relative_path=str(Path(r['path']).relative_to(root)) if str(r['path']).startswith(str(root)+'/') else None,bytes=len(b),lines=len(b.decode('utf-8-sig').splitlines()),sha256=actual)
 manifest.append(x);label=x['workspace_relative_path'] or x['path'];md.append('| S'+str(r['id'])+' | ['+label+'](<'+target(r)+'>) | '+str(x['lines'])+' | '+str(x['bytes'])+' | `'+actual+'` |')
(docs/'research_history_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(docs/'research_history_sources.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(json.dumps({'reports_unchanged':len(rows),'routes':len(routes),'report_bytes':p.stat().st_size,'torch_imported':False}))