from pathlib import Path
import json,re,hashlib,difflib,datetime
root=Path('/mnt/e/MeshFlow');out=root/'experiments/research_history_audit_20261014';docs=root/'r2/docs'
sha=lambda b:hashlib.sha256(b).hexdigest()
rows=json.loads((out/'source_inventory.json').read_text());unchanged=[]
for r in rows:
 assert sha(Path(r['path']).read_bytes())==r['sha256'],r['path']
 unchanged.append(r['id'])
reg=json.loads((docs/'research_route_registry.json').read_text())
assert len(reg['routes'])==30
assert {r['id'] for r in reg['routes'] if r['id'].startswith('R')}=={f'R{i:02d}' for i in range(1,29)}
assert all(r['source_ids'] for r in reg['routes'])
checks=[];links=[]
files=['r2/docs/research_route_memory.md','r2/docs/research_history_sources.md','r2/docs/research_route_registry.json','r2/docs/research_history_sources.json','AGENTS.md','WORKSPACE.md','r2/README.md']
for rel in files:
 p=root/rel;b=p.read_bytes();s=b.decode('utf-8-sig');assert '\ufffd' not in s
 if rel in files[:2]:
  assert '[[S' not in s
  for t in re.findall(r'\]\(([^\n]+?)\)',s):
   t=t.strip('<>');t=t.split('#')[0]
   if not t or '://' in t:continue
   t=re.sub(r'^([A-Za-z]):/',lambda m:'/mnt/'+m[1].lower()+'/',t)
   q=Path(t) if t.startswith('/') else p.parent/t
   assert q.is_file(),(rel,t)
   links.append(dict(document=rel,target=str(q.resolve())))
 checks.append(dict(path=rel,bytes=len(b),sha256=sha(b)))
diffs=[];edits=[]
for rel in ['AGENTS.md','WORKSPACE.md','r2/README.md']:
 before=(out/(rel.replace('/','__')+'.before')).read_bytes();after=(root/rel).read_bytes()
 diffs.extend(difflib.unified_diff(before.decode('utf-8-sig').splitlines(True),after.decode('utf-8-sig').splitlines(True),fromfile=rel+'.before',tofile=rel))
 edits.append(dict(path=rel,before_sha256=sha(before),after_sha256=sha(after)))
(out/'documentation.diff').write_text(''.join(diffs),encoding='utf-8')
seen={json.loads(s)['id'] for s in (out/'read_ledger.jsonl').read_text().splitlines()}
receipt=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),primary_reports=83,primary_report_hashes_unchanged=len(unchanged),extracted_report_ids=len(seen),routes=reg['counts'],valid_report_links=len(links),documents=checks,existing_file_edits=edits,scientific_activity=dict(training_updates=0,model_loads=0,model_forwards=0,new_generation=0,new_mesh_scoring=0,new_visual_review=0,weight_cleanup=0),limitations=['Report narrative and tables reviewed; no new raw-array numerical replication.','Tool extraction ledger is not a certification of word-by-word review of all embedded JSON.','Retired model checkpoint availability was not rehashed or restored.'],commit=False,push=False)
(out/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ['status','primary_report_hashes_unchanged','extracted_report_ids','routes','valid_report_links']}))