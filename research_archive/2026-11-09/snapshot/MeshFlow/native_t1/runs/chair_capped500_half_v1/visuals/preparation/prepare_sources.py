"""Bounded gallery-controller port; old source is read-only, no rendering."""
from pathlib import Path
import difflib
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
OLD=ROOT/'experiments/mf_h1_local0/chair_capped500_gradbudget_v1'
NEW=ROOT/'experiments/mf_h1_local0/chair_capped500_half_v1'

def replace_once(text, old, new):
    if text.count(old)!=1:raise AssertionError('Expected exactly one source match: '+old)
    return text.replace(old,new)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

source=(OLD/'visualization.py').read_text(encoding='utf-8')
v=source.replace('all 288 raw outputs','all 96 raw outputs').replace(
    'Earlier gallery snapshots remain distinct and are explicitly provisional.',
    'Only complete three-label galleries are permitted for this fixed experiment.')
v=v.replace("VERSION='chair_capped500_gradbudget_gallery_v1'","VERSION='chair_capped500_half_gallery_v1'")
v=v.replace('All nine registered labels','All three registered labels').replace('available==288','available==96').replace('expected_outputs=288','expected_outputs=96')
v=v.replace('Gradient budget meshes','Half-weight control meshes')
v=v.replace("    os.environ.setdefault(_key,'2')","    os.environ[_key]='2'").replace(
    "os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')","os.environ['LIBGL_ALWAYS_SOFTWARE']='1'").replace(
    "os.environ.setdefault('GALLIUM_DRIVER','llvmpipe')","os.environ['GALLIUM_DRIVER']='llvmpipe'\nos.environ['CUDA_VISIBLE_DEVICES']=''")
v=replace_once(v,'VIEWS=renderer.VIEWS',"VIEWS=renderer.VIEWS\nFIXED_LABELS=('START','HIST_CAPPED500','HALF_CAPPED500')")
v=replace_once(v,"pointer=request.get('historical_cache',{}).get(label) if request['split']=='OLD' else None", "pointer=request.get('historical_cache',{}).get(label)")
guard='''def _require_complete(out):
    """Refuse any render dispatch before all fixed raws and scores exist."""
    if tuple(ev.LABELS)!=FIXED_LABELS:raise ValueError('Half-weight label identity differs')
    requests=ev.manifest(out,'ALL')
    byparent=defaultdict(list);missing=[]
    if len(requests)!=32 or len({q['request_id'] for q in requests})!=32:
        raise ValueError('Expected all 32 unique registered requests')
    for q in requests:
        byparent[q['uid']].append(int(q['candidate']))
        if q['split'] not in ('OLD','FRESH') or int(q['candidate']) not in ((0,1) if q['split']=='OLD' else (2,3)):
            raise ValueError('Panel/noise registration differs')
        for label in FIXED_LABELS:
            raw=_raw(out,label,q)
            source=_metric_source(out,label,q)
            if raw is None or source is None:missing.append((q['request_id'],label))
            elif source['metrics'].get('raw_sha256')!=ev.digest(raw):
                raise ValueError('Scored vectors refer to a different raw')
    if len(byparent)!=8 or any(sorted(v)!=[0,1,2,3] for v in byparent.values()):
        raise ValueError('Expected eight parents with all four frozen noises')
    if missing:raise RuntimeError('Rendering waits for all 96 raw/score cells: '+str(missing))


'''
v=replace_once(v,'def _geometry(path,condition):',guard+'def _geometry(path,condition):')
v=replace_once(v,"out=ev.assert_out(out);requests=ev.manifest(out,'ALL');data={};sources={};available=0",
    "out=ev.assert_out(out);_require_complete(out)\n    requests=ev.manifest(out,'ALL');data={};sources={};available=0")
v=replace_once(v,"if not labels or any(label not in ev.LABELS for label in labels):raise ValueError('Unknown gallery label')",
    "if tuple(labels)!=FIXED_LABELS:raise ValueError('All three fixed labels are required in registered order')")
v=v.replace('Common cameras also include other-panel and other-method available meshes.',
    'Each parent camera includes all three labels and all four OLD/FRESH noises.')

i=(OLD/'inspect_gallery.py').read_text(encoding='utf-8')
i=i.replace('chair_capped500_gradbudget_v1','chair_capped500_half_v1')
i=i.replace('len(labels) != 9 or len(set(labels)) != 9','len(labels) != 3 or len(set(labels)) != 3').replace('all nine labels','all three labels').replace("get('available_outputs') != 288","get('available_outputs') != 96")
start=i.index('GROUPS = {');end=i.index('\nVIEWS = ',start)
i=i[:start]+"GROUPS = {'comparison': ('Input C', 'START', 'HIST_CAPPED500', 'HALF_CAPPED500', 'GT C+free')}\n"+i[end:]
i=i.replace('def inspect(out, panel, uid, candidate, group, overlay=False):',"def inspect(out, panel, uid, candidate, group='comparison', overlay=False):")
i=replace_once(i,"    camera_path = resolve(manifest['camera_manifest'])", "    if labels != ['START','HIST_CAPPED500','HALF_CAPPED500']:\n        raise ValueError('Unexpected half-weight label identity or order')\n    camera_path = resolve(manifest['camera_manifest'])")
i=i.replace("GROUPS['history']","GROUPS['comparison']").replace("parser.add_argument('--group', choices=tuple(GROUPS))","parser.add_argument('--group', choices=tuple(GROUPS), default='comparison')")

NEW.mkdir(parents=True,exist_ok=True)
rows=[]
for name,text in [('visualization.py',v),('inspect_gallery.py',i)]:
    old=OLD/name;new=NEW/name
    compile(text,str(new),'exec')
    new.write_text(text,encoding='utf-8')
    delta=''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(keepends=True),text.splitlines(keepends=True),fromfile=str(old),tofile=str(new)))
    (HERE/(name+'.diff')).write_text(delta,encoding='utf-8')
    rows.append(dict(source=str(old),source_sha256=digest(old),new=str(new),new_sha256=digest(new)))
receipt=dict(status='PREPARED_NOT_RENDERED',files=rows,render_calls=0,model_calls=0,scoring_calls=0,
    inherited=['Orthographic all-vertex frame math','C blue/free orange flat-triangle rendering','Four views and wire','Saved S/I/E vector projection','JSON tuple/list canonicalization'],
    changes=['Three fixed label identities and 96 outputs','Strict all-raw/all-score readiness','Two-thread software OpenGL','Five-column pixel-exact inspector'],
    actual_visual_review='NOT_RUN_WAITING_FOR_FINAL_GALLERIES')
(HERE/'source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
