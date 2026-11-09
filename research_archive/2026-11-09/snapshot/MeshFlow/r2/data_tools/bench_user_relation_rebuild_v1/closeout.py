"""Seal explicitly reviewed data, reuse production CPU acceptance, and verify delivery."""
from __future__ import annotations
from .run import *
import html

# Recorded after viewing the exact packaged two-view M/C/Y/C+Y pages.
NOTES={
 ('a8e987','SLATTED_TOP_FULL'):('HOLD','完整木条组与端框上表面，视觉范围成立；原拓扑在顶点处出现断开的link，几何资格不通过，隔离。'),
 ('a8e987','SINGLE_SLAT_TOP_FULL'):('ACCEPT','完整中央木条的矩形上表面，两三角形；不是随机裁断条，不是闭合木条实体。长宽比约23，保留其真实细长形状。'),
 ('a8e987','END_FRAME_FULL'):('ACCEPT','完整横向端架：两条腿与顶横梁一起保留，自然交界处分区，没有中段截腿。'),
 ('7b58a3','SINGLE_SLAT_TOP_FULL'):('ACCEPT','极窄但完整的中央木条上表面，两三角形；长宽比约84，普通等尺度图近似一条线，补充顶视图核对。'),
 ('7b58a3','END_WALL_FULL'):('ACCEPT','完整弯折端墙的内外与弧形表面；独立木条端封片不属于端墙，未混入本条C。'),
 ('9e429b','CURVED_THREE_BAY_SEAT_SURFACE'):('ACCEPT','连续三湾上座面覆盖三个实际座位与其自然隆起分隔，不包含座底和端部扶手外曲面。'),
 ('9e429b','COMPLETE_LEFT_END_OUTER_SURFACE'):('ACCEPT','完整端部外曲面，从底脚绕过顶部到座面接点；是开放表面，不是完整闭合端块。'),
 ('9e429b','CURVED_SEAT_PLUS_LEFT_END'):('ACCEPT','三湾座面与一侧完整端部外曲面连续组合，保留宽度方向的真实轮廓。'),
 ('943dde','TRUE_CURVED_SEAT_UPPER_SURFACE'):('ACCEPT','真正弯曲的座面上表面，覆盖完整宽度与深度，不是以前的平面座底。'),
 ('943dde','COMPLETE_INNER_CURVED_BACK'):('ACCEPT','完整内曲背表面含下缘向座面的自然过渡，不是平外背。'),
 ('943dde','CURVED_INNER_SEAT_BACK'):('ACCEPT','真实上座面和内曲背组成连续座背曲面，使用同一目标的原接缝。'),
}
def details():
    from PIL import Image,ImageDraw
    from r2.data_tools.bench_user_v2.render import font
    m=read(OUT/'candidate_manifest.json');rs=[r for r in m['conditions'] if r['region_name']=='SINGLE_SLAT_TOP_FULL']
    im=Image.new('RGB',(1400,500),'white');d=ImageDraw.Draw(im)
    for i,r in enumerate(rs):
        a=arrays(r['case_path']);v=a['C_vertices'].astype(float);f=a['C_faces_original'];xz=v[:,[0,2]]
        scale=1250/np.ptp(xz,axis=0).max();xy=(xz-xz.mean(0))*scale;xy[:,1]*=-1;xy+=np.array([700,150+235*i])
        d.text((15,15+235*i),r['parent_id'][:6]+' | C top view, uniform scale; full rectangular surface, original 2 triangles',font=font(20),fill='black')
        span=np.ptp(xz,axis=0);d.text((15,53+235*i),'X/Z extent (model units): '+str(span.round(6).tolist())+' | aspect='+str(round(float(span.max()/span.min()),2)),font=font(18),fill='black')
        for face in f:d.polygon([tuple(xy[j]) for j in face],fill='#178abc',outline='black',width=1)
        for j,p in enumerate(xy):
            d.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill='red')
    b=io.BytesIO();im.save(b,format='PNG');bytes_new(OUT/'gallery/slat_top_details.png',b.getvalue())
def admit():
    verify();m=read(OUT/'candidate_manifest.json');gm={r['condition_id']:r for r in read(OUT/'gallery_manifest.json')}
    reviews=[];rows=[]
    for r in m['conditions']:
        key=(r['parent_id'][:6],r['region_name']);decision,note=NOTES[key];e=gm[r['condition_id']]
        assert sha(e['path'])==e['sha256']
        if decision=='ACCEPT':assert r['adapter_compatible'] and not r['duplicate_old_conditions']
        q=read(r['quality_file']);a=arrays(r['case_path']);v=a['C_vertices'].astype(float)
        vv=v-v.mean(0);_,_,basis=np.linalg.svd(vv,full_matrices=False);rms=float(np.sqrt(np.mean((vv@basis[-1])**2)))
        rev=dict(condition_id=r['condition_id'],parent_id=r['parent_id'],decision=decision,
          reviewer='AI_PROXY_primary_agent',actually_viewed=True,image=e,notes=note,
          scope='Exact packaged source geometry multiview, no model output review; not a human user survey')
        if r['region_name']=='SINGLE_SLAT_TOP_FULL':rev['supplemental_image']=dict(path=str(OUT/'gallery/slat_top_details.png'),sha256=sha(OUT/'gallery/slat_top_details.png'))
        reviews.append(rev)
        parent=next(p for p in m['parents'] if p['parent_id']==r['parent_id'])
        rows.append(dict(r,review=rev,review_status=decision,eligibility='ACCEPTED_PENDING_CPU_PACKING' if decision=='ACCEPT' else 'HELD_GEOMETRY',
          source_face_id_domain='Frozen versioned target_case.source_faces row indices; not pre-QEM/raw face indices',
          source_lineage_path=parent.get('lineage_path'),source_lineage_sha256=sha(parent['lineage_path']) if parent.get('lineage_path') else None,
          C_vertex_plane_rms_over_L=rms/Q.L,plane_descriptor_scope='unweighted unique C vertices; descriptive only',
          C_bad_vertex_count=len(q['topology']['nonmanifold_vertices']),C_illegal_pair_count=len(q['illegal_C_pairs'])))
    write(OUT/'visual_review.json',reviews)
    new=[r for r in rows if r['review_status']=='ACCEPT'];assert len(new)==10
    m.update(conditions=rows,train=new,status='VISUALLY_ACCEPTED_PENDING_CPU_PACKING',
      visual_review_path=str(OUT/'visual_review.json'),visual_review_sha256=sha(OUT/'visual_review.json'),training_registered=False)
    write(OUT/'manifest.json',m);csvout(OUT/'conditions.csv',rows)
    event('ACTUAL_VISUAL_REVIEW_AND_ADMISSION',accepted=len(new),held=len(rows)-len(new),reviewer='AI_PROXY_primary_agent')
    print(json.dumps(dict(status='FROZEN_FOR_CPU_ACCEPTANCE',accepted=len(new),held=len(rows)-len(new))))
def finalize():
    verify();m=read(OUT/'manifest.json');latest=read(OUT/'validation/latest.json');receipt=read(latest['receipt'])
    assert latest['status']=='PASS' and sha(latest['receipt'])==latest['receipt_sha256'] and latest['manifest_sha256']==sha(OUT/'manifest.json')
    assert receipt['passed']==10 and receipt['new_CPU_OT_completed']==10 and receipt['new_CPU_OT_attempts']==10
    old=read(OLD);new=m['train'];oldparents={r['parent_id']:r for r in old['parents']};newparents={r['parent_id']:r for r in m['parents']}
    oldready=OLD.parent/'DATASET_READY.json'
    combined=dict(schema=VERSION+'_COMBINED_PROPOSAL',status='CPU_VALIDATED_PROPOSAL_NOT_ACTIVATED',
      conditions=old['conditions']+new,train=old['train']+new,parents=list((oldparents|newparents).values()),
      proposed_extension_manifest=str(OUT/'manifest.json'),proposed_extension_sha256=sha(OUT/'manifest.json'),
      protected_source_hashes=m['protected_source_hashes'],training_registered=False,training_schedule=None,
      dev=old['dev'],dev_source_manifest_path=old['dev_source_manifest_path'],dev_source_manifest_sha256=old['dev_source_manifest_sha256'],
      old_manifest_path=str(OLD),old_manifest_sha256=sha(OLD),
      validation_lineage=[dict(scope='reused_old81',path=str(oldready),sha256=sha(oldready)),dict(scope='new10',**latest)])
    assert combined['dev']==old['dev'] and len(combined['parents'])==26 and len(combined['train'])==91
    assert len({r['condition_id'] for r in combined['train']})==91
    assert len({(r['parent_id'],tuple(sorted(r['target_face_ids']))) for r in combined['train']})==91
    write(OUT/'proposed_merged_manifest.json',combined)
    summary=dict(schema=VERSION,status='DATA_READY_NOT_ACTIVATED',candidate_conditions=len(m['conditions']),accepted_new_conditions=len(new),
      held_conditions=1,old_conditions=81,proposed_combined_conditions=91,old_parents=25,proposed_combined_parents=26,
      new_source_parent_ids=sorted(set(newparents)-set(oldparents)),condition_counts_by_parent=dict(Counter(r['parent_id'] for r in new)),
      condition_counts_by_type=dict(Counter(r['part_type'] for r in new)),K_min=min(r['K'] for r in new),K_max=max(r['K'] for r in new),
      unchanged_dev_conditions=len(old['dev']),unchanged_dev_parents=len({r['parent_id'] for r in old['dev']}),
      exact_target_geometry_modified=False,training_updates=0,model_forwards=0,GPU_compute_calls=0,new_CPU_OT_calls=10,new_CPU_validation_noise_arrays=20,
      GPU_peak='NA_NO_GPU_COMPUTE',CPU_design_workers=2,CPU_pack_and_validation_workers=1,default_pool_changed=False,
      production_CPU_receipt=latest,manifest_path=str(OUT/'manifest.json'),manifest_sha256=sha(OUT/'manifest.json'),
      proposed_merged_manifest=str(OUT/'proposed_merged_manifest.json'),proposed_merged_sha256=sha(OUT/'proposed_merged_manifest.json'),
      limitations=['Only one new source parent, not 10 independent objects','Complete narrow slats retain original small angles',
        'Curved inputs are real selected surfaces; the full unknown object is not fed into inference',
        'More input coverage is not yet evidence of model improvement or broad generalization'])
    write(OUT/'DATASET_READY.json',summary)
    rr=[]
    for r in m['conditions']:rr.append('| '+r['parent_id'][:6]+' | '+r['region_name']+' | '+str(r['K'])+' | '+str(r['boundary_edges'])+' | '+str(r['source_owner_count'])+'/'+str(r['source_alias_count'])+'/'+str(r['source_null_count'])+' | '+r['review_status']+' |')
    report='''# Bench 完整部件与曲面条件重构

已实际构造11条候选：10条通过几何资格、逐图审阅和原生产CPU打包，1条隔离。合格输入来自4个来源，其中只有9e429b是旧25父池之外的新父。独立的合并提案为 **26父、91条条件**；原25父/81条件和39条DEV原件保持不变，未切换活动池。

本轮是数据准备，零训练、零模型前向、零GPU计算。不能据此宣布生成质量提升或数据覆盖已经充分。

## 实际构造

从已核验完整目标M的现有面选择C；Y为精确补集。没有移动顶点、裁断三角形、修补M、重新三角化或按面数比例强凑。目标仍为原N189/256、原规范化和角序版本。下面的面ID指冻结target_case.source_faces行号；历史QEM之前的来源关系沿原lineage.json查询，不冒称原始高面数网格的一一对应面ID。

| 来源 | 条件 | K | 开放边 | owner/alias/null | 录用 |
|---|---|---:|---:|---:|---|
'''+ '\n'.join(rr)+'''

a8的39面完整木条组虽然视觉完整，却有非流形顶点，保留为HELD_GEOMETRY，不进入新训练清单。没有通过删掉局部面伪装成合格完整输入。7b已有28面木条组与旧条件完全重复，未计新增。两条K=2输入是整根木条的真实上表面，长宽比分别约23和84，不是随机细条；它们仍含自然小角，不能宣称所有新输入都低小角。端部和曲面条目按表面边界明确命名，开放外曲面不声称是闭合实体端块。

943新增真正的上座面、内曲背及连续组合；9e新增三湾座面、端部外曲面及组合。它们补充了以前只含平座底/外背、截断下端块等输入时缺少的结构关系。四父每父2或3条，并不等于新增10种独立Bench款式。

## 核验与数据合同

所有新C保持原顶点、面身份及可逆来源角序。C/Y源面ID互斥且并集等于M，数组逐位还原；10条合格C均为一个边连通分量，原有限三角面检查通过。完整目标复用已通过的原几何与视觉审阅，不重复跑全目标审计。

实际查看了11张最终打包数组的M/C/Y/C+Y双视图，并为窄木条补顶视图。记录为AI_PROXY，不冒称人工评审，也没有把几何PASS直接当作语义标签。

对最终10条各执行一次原生产SEAM数据打包和CPU受限OT：10/10 PASS。含C/N推理文件检查、来源owner/alias引用、目标面各出现一次、固定角点与FM掩码、OT后HYBRID0.25、联合尺度增强检查。仅生成20个CPU数据验证噪声数组，不是训练/评测噪声，不运行生成器、去噪器或新评价面板。正式推理输入只含C/N所需字段；部位名称、Y、完整M和owner/alias来源标签仅用于整理与训练监督。

39条原DEV记录（8父）逐条不变，TRAIN/DEV来源隔离检查通过。旧文件与正式生产代码hash复核未变。评测来源是一种参考完成，不作为开放生成唯一GT。

## 交付与使用

- `manifest.json`：本轮10条录用输入及1条隔离证据，配`DATASET_READY.json`和`validation/latest.json`使用。manifest封存于CPU验收前，最终PASS由带hash的外部回执确认。
- `proposed_merged_manifest.json`：旧81＋新10＝91条、26父的独立合并提案；未覆盖旧活动池。
- `conditions.csv`：全11条的K/N、面ID、拓扑、来源接口、描述与录用状态。
- `conditions/`：每条实际C.obj、Y.obj、input.npz、paired.npz、case.npz、supervision_only.npz、quality.json与selection.json；隔离条目也保留。
- `review.html`：最终图册；`visual_review.json`记录实际图审和图片hash。
- `protected_source_hashes.json`、`source_hashes.json`、`delivery_hashes.json`：输入/代码/交付完整性；`ledger.jsonl`记录阶段。
- `r2/data_tools/bench_user_relation_rebuild_v1/resume.ps1`：默认只验证封存结果，不重算OT、不训练。

资源记录在resources文件中。两条独立区域设计并行，几何打包和正式CPU OT各单worker、每进程一个数值线程；此规模CPU工作不需要GPU。没有用额外GPU探针，不干预用户的GPU进程。构图/图审耗时不能当作模型训练吞吐。

## 这一步能支持的下一项决定

现在已有一批可直接进入现有数据入口的完整部件和非平面输入，值得作为后续配对数据覆盖实验的增量。应继续用原R2/SEAM模型与训练预算检验其作用；不能把一父三个条件算作三倍独立数据，也不宜只用这10条替换旧81条。训练时继续按父组织曝光，并单独统计新区域出现次数。本轮没有注册或启动下一次训练。
'''
    bytes_new(OUT/'REPORT.md',report.encode('utf-8'))
    cards=[]
    for r in m['conditions']:
        name=r['condition_id'];cards.append('<article><h2>'+html.escape(name)+' — '+r['review_status']+'</h2><p>'+html.escape(r['review']['notes'])+'</p><a href="gallery/'+name+'.png"><img loading="lazy" src="gallery/'+name+'.png"></a></article>')
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Bench 完整输入重构</title><style>body{font:16px sans-serif;max-width:1500px;margin:auto;padding:24px;background:#f4f6f8}article{background:white;padding:16px;margin:20px 0}h2{font-size:18px;overflow-wrap:anywhere}img{width:100%}</style><h1>4父 · 11候选 · 10通过 · 1隔离</h1><p>蓝色C，灰色Y；双视图使用同一尺度。点击看原图。没有模型生成结果。</p><img src="gallery/slat_top_details.png">'+''.join(cards)+'</html>'
    bytes_new(OUT/'review.html',page.encode('utf-8'))
    write(OUT/'resources_after.json',resources())
    sources=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.ps1'))+[ROOT/'r2/data_tools/bench_user_v2/validate.py',ROOT/'r2/data_tools/bench_user_v2/render.py',ROOT/'r2/data_tools/bench_parts_v1/quality.py',ROOT/'r2/data_tools/bench_parts_v1/construct.py']
    write(OUT/'source_hashes.json',{str(p):sha(p) for p in sources})
    event('COMPLETE',accepted=10,held=1,merged_parents=26,merged_conditions=91,old_assets_unchanged=True)
    artifacts={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_hashes.json'}
    write(OUT/'delivery_hashes.json',artifacts)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
def verify_delivery():
    verify()
    for p,h in read(OUT/'source_hashes.json').items():assert sha(p)==h,p
    for p,h in read(OUT/'delivery_hashes.json').items():assert sha(OUT/p)==h,p
    print('DELIVERY_VERIFIED_NO_RECOMPUTE')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['details','admit','finalize','verify_delivery']);args=ap.parse_args();globals()[args.stage]()
if __name__=='__main__':main()
