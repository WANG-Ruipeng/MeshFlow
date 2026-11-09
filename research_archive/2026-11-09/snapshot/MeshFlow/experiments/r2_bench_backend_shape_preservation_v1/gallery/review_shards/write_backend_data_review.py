"""Transcription of actual backend_data view_image inspection; no metric input."""
import datetime, hashlib, json
from pathlib import Path

root = Path('/mnt/e/MeshFlow/experiments/r2_bench_backend_shape_preservation_v1')
old = Path('/mnt/e/MeshFlow/experiments/r2_bench_geo_only_acceptance_v1/manifest.json')
tasks = json.loads(old.read_text())['tasks'][10:20]
axes = ['overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold']
R = ['REASONABLE','PLAUSIBLE','PRESENT','NO_OBVIOUS','NO_OBVIOUS']
records = []

def a(shape='REASONABLE', use='PLAUSIBLE', support='PRESENT', gap='NO_OBVIOUS', layer='NO_OBVIOUS'):
    return [shape,use,support,gap,layer]

for i,t in enumerate(tasks):
    req=t['request_id']; receipt=json.loads((root/'gallery'/req/'complete.json').read_text())
    if i == 0:
        vals=[a(gap='YES') for _ in range(5)]
        notes=[
          '蓝色顶面被箱式主体采用，长侧板及端板仍在；背向视角左端可见细裂口，未见明显双层座面。',
          '箱式整体和支撑保留，但背向视角左端出现清楚的小白色孔口。',
          '蓝色顶面及完整箱身保留，左端有细缝；未见独立悬片或明显重层。',
          '箱身总体保留，左端白色缺口仍明显，未修复为连续外壳。',
          '最终整理后箱身和顶面用途不变，左端缺口仍在。']
    elif i == 1:
        vals=[a() for _ in range(5)]
        notes=['两视角均为完整箱式长凳，蓝C作为顶面，端板和侧板保留；局部三角分片不据此判作裂孔或非法折层。']*5
    elif i in (2,3):
        vals=[a(gap='UNCERTAIN',layer='UNCERTAIN') for _ in range(5)]
        notes=['完整背板、座面和两端支撑仍可辨，蓝C位于合理座面位置。侧截面有零碎内层线段，但两视角不足以确定它们是悬片、薄壁厚度还是非法重层，故两项记不确定。']*5
    elif i in (4,5):
        vals=[a(),a(shape='INCOMPLETE',support='MISSING_OR_DAMAGED',gap='YES'),a(),
              a(shape='INCOMPLETE',support='MISSING_OR_DAMAGED',gap='YES'),
              a(shape='INCOMPLETE',support='MISSING_OR_DAMAGED',gap='YES')]
        notes=[
          '三段蓝色弯曲座面与两端较完整的灰色支撑/端框形成长凳；两视角没有明显漂浮碎片。',
          '两端支撑上缘和竖向部分明显缺损，留下破碎窄条；蓝色座面位置仍合理，不能以座面保留代替主体完整。',
          '两端支撑和座面下方材料仍较完整，未见官方D0列那种大片端框缺失。',
          '相对官方D0保留了更多端部材料，但端框仍存在明显孔洞、破边及缺段，未恢复RAW的支撑外形。',
          '最终整理后端框破损仍在；蓝座面用途保留，但整体支撑不完整。']
    elif i in (6,7):
        vals=[a(use='REPLACED_OR_IGNORED',layer='YES') for _ in range(5)]
        notes=['长凳主体、靠背和端腿可辨且保留；前/侧截面共同显示生成的灰座面位于蓝C上方，蓝C未成为唯一有效座面。五列均未消除此错层。']*5
    elif i == 8:
        vals=[a(),a(shape='INCOMPLETE',support='MISSING_OR_DAMAGED',gap='YES'),a(),
              a(shape='UNCERTAIN',support='MISSING_OR_DAMAGED',gap='YES'),
              a(shape='UNCERTAIN',support='MISSING_OR_DAMAGED',gap='YES')]
        notes=[
          '蓝C作为可见座面，宽背板和两端矩形腿框构成长凳；没有把截面未切到端腿误判为缺腿。',
          '座面仍在但两端支撑基本消失，背部只剩少量窄条及游离线段，主体明显不完整。',
          '背板和端腿框总体保留，蓝座面关系合理。',
          '比官方D0多保留两端腿框与背条，但腿框收缩成很细的线状结构，局部缺段；能辨长凳但支撑完整性不足，整体记不确定。',
          '最终整理保留相同细线腿框，未恢复RAW的完整支撑；可见缺段仍在。']
    else:
        vals=[a(use='REPLACED_OR_IGNORED',layer='YES'),
              a(shape='INCOMPLETE',use='REPLACED_OR_IGNORED',support='MISSING_OR_DAMAGED',gap='YES',layer='YES'),
              a(use='REPLACED_OR_IGNORED',layer='YES'),
              a(shape='UNCERTAIN',use='REPLACED_OR_IGNORED',support='MISSING_OR_DAMAGED',gap='YES',layer='YES'),
              a(shape='UNCERTAIN',use='REPLACED_OR_IGNORED',support='MISSING_OR_DAMAGED',gap='YES',layer='YES')]
        notes=[
          '灰色平台与四腿形成完整长凳/桌状主体，但平台明显高于蓝C；两个截面均显示蓝C下沉在灰座面以下。',
          '多条腿缺失或成为游离细线，上方灰平台变稀疏；蓝C仍在下方，错层未消除。',
          '灰平台和腿总体保留，蓝C与灰座面之间的大间距仍在。',
          '比官方D0保留更多腿，但细线化与缺口仍明显，灰平台仍架在蓝C上方；未形成正确采用。',
          '最终整理后的腿与灰平台残片仍不完整，C下沉错层与BASE相同。']
    path=Path(receipt['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['image_sha256']
    for cell,value,note in zip(receipt['cells'],vals,notes):
        row={k:receipt[k] for k in ['request_id','parent_id','condition_id','noise_index','eval_group']}
        row.update({k:cell[k] for k in ['cell_id','column','label','stage']})
        row.update(image_path=receipt['path'],image_sha256=receipt['image_sha256'],reviewer='backend_data',
          reviewer_kind='AI_PROXY',blind=False,actually_viewed=True,
          actual_view_source='functions.view_image original; two surface views and two fixed central sections',
          viewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),note=note)
        row.update(dict(zip(axes,value)));records.append(row)
assert len(records)==50 and len({r['cell_id'] for r in records})==50
dest=root/'gallery/review_shards/backend_data.json'
with dest.open('x',encoding='utf-8') as f:json.dump(dict(rows=records),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(rows=len(records),pages=10,path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())))
