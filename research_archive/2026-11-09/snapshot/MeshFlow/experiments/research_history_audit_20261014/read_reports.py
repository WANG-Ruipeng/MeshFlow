from pathlib import Path
import json,sys,hashlib,datetime
out=Path(__file__).resolve().parent
rows=json.loads((out/'source_inventory.json').read_text())
ids=[int(i) for i in sys.argv[1].split(',')]; start=int(sys.argv[2]) if len(sys.argv)>2 else 1; limit=int(sys.argv[3]) if len(sys.argv)>3 else 10000
for i in ids:
 r=next(r for r in rows if r['id']==i);p=Path(r['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'];lines=raw.decode('utf-8-sig').splitlines();end=min(len(lines),start+limit-1)
 print('SOURCE',i,r['relative_path'],'LINES',start,end,'TOTAL',len(lines))
 for n in range(start-1,end):print(str(n+1)+': '+lines[n])
 with (out/'read_ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(id=i,start=start,end=end,sha256=r['sha256'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
