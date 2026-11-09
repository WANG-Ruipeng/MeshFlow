"""Snapshot source and audit all local review links. No experiments."""
from pathlib import Path
from html.parser import HTMLParser
from collections import Counter
import csv,json,difflib
from prepare import DEFAULT_OUT,read,write,sha,event

class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.articles=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='article':self.articles.append(a)
  for key in ('href','src'):
   if key in a:self.links.append(a[key])

def main(out=DEFAULT_OUT):
 out=Path(out);code=Path(__file__).parent
 parser=Links();parser.feed((out/'gallery/index.html').read_text(encoding='utf-8'))
 assert len(parser.articles)==129
 assert Counter(r['data-usage'] for r in parser.articles)=={'active':123,'held':6}
 missing=[p for p in parser.links if not (out/'gallery'/p).resolve().exists()]
 assert not missing,missing
 with (out/'tables/active_conditions.csv').open(encoding='utf-8',newline='') as f:
  table=list(csv.DictReader(f))
 assert len(table)==123 and Counter(r['split'] for r in table)=={'train':117,'dev':6}
 snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
 hashes={};diff=[]
 for p in sorted(code.iterdir()):
  if p.suffix not in ('.py','.md'):continue
  target=snap/p.name
  if target.exists():assert target.read_bytes()==p.read_bytes()
  else:target.write_bytes(p.read_bytes())
  hashes[str(p)]=dict(sha256=sha(p),snapshot=str(target),bytes=p.stat().st_size)
  diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(keepends=True),
   fromfile='/dev/null',tofile='b/r2/data_tools/user_conditions_v1/'+p.name))
 score_dir=out/'scoring_source';score_dir.mkdir(exist_ok=True)
 frozen=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_user_condition_data_audit_v1/geometry/source_snapshot/src/meshflow_control/evaluation')
 for name in ('intersections.py','pair_audit.py','seams.py'):
  p=frozen/name;target=score_dir/name
  if target.exists():assert target.read_bytes()==p.read_bytes()
  else:target.write_bytes(p.read_bytes())
  hashes[str(p)]=dict(sha256=sha(p),snapshot=str(target),bytes=p.stat().st_size)
 write(out/'source_hashes.json',hashes)
 delta=''.join(diff);dpath=out/'code.diff'
 if dpath.exists():assert dpath.read_text(encoding='utf-8')==delta
 else:dpath.write_text(delta,encoding='utf-8')
 receipt=dict(status='PASS',candidate_images=129,active_gallery_cards=123,held_cards=6,
  checked_local_link_occurrences=len(parser.links),broken_links=0,active_csv_rows=123,
  source_files_snapshotted=len(hashes),source_hashes_sha256=sha(out/'source_hashes.json'),
  original_scorers_preserved=True,model_calls=0,training_updates=0,OT_calls=0)
 write(out/'package_validation.json',receipt)
 event(out,'DATA_PREPARATION_COMPLETED',active_train=117,active_dev=6,old_requests=16,model_calls=0,training_updates=0)
 # Exclude changing execution ledger and this file's own manifest to avoid circular hashes.
 records=[]
 for p in sorted(out.rglob('*')):
  if p.is_file() and p.name not in ('files_manifest.json','ledger.jsonl'):
   records.append(dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size))
 write(out/'files_manifest.json',dict(rows=records,total_bytes=sum(r['bytes'] for r in records)))
 print(json.dumps(dict(receipt,files=len(records),total_bytes=sum(r['bytes'] for r in records))))
if __name__=='__main__':main()
