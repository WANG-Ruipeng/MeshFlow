"""Build a local-only review index after the registered figures/tables exist.

Default execution fails with an explicit missing-file inventory. It never labels
rendered-but-unreviewed figures as visually reviewed or invents absent records.
"""
from __future__ import annotations
import argparse,html,json,os
from pathlib import Path
from .evaluation import C,ROOT,OUT,ARMS,requests
from .case_sections import CASES

VERSION='seam_hinge_quality_local_review_index_v1'


def inventory():
    groups=[]
    def add(group,title,path,required=True):
        p=Path(path);groups.append(dict(group=group,title=title,path=p,required=required,exists=p.is_file()))
    add('report','完整实验报告',OUT/'REPORT.md')
    add('report','冻结实验合同',OUT/'protocol.json')
    add('report','完整数值与身份复算',OUT/'evaluation_audit/completion.json')
    for name,title in [('per_output','完整逐输出指标'),('key_results','TRAIN / DEV / ALL关键汇总'),
                       ('summary','全部指标汇总'),('paired_per_noise','逐噪声配对差值'),
                       ('paired_per_parent','逐父配对差值'),('illegal_pairs_by_role','非法面对角色分解'),
                       ('deleted_faces','删面明细')]:
        add('tables',title,OUT/'results/final'/(name+'.csv'))
    for name,title in [('per_output','完整索引拓扑逐输出'),('per_face_connectivity','逐面完整连通账本')]:
        add('tables',title,OUT/'topology/final_v2'/(name+'.csv'))
    for arm in ARMS:
        add('four_stage_index',arm+'：RAW / D0 / RESTORED / POST',OUT/'sections'/arm/'index.html')
        for q in requests():add('four_stage_page',arm+' '+q['request_id'],OUT/'sections'/arm/(q['request_id']+'.png'))
    for kind in ('RAW','POST','BOUND015',*['HINGE_'+a for a in ('PLAIN_S2000',*ARMS)]):
        add('gallery_index',kind+' 外观与同位置截面',OUT/'gallery'/kind/'index.html')
        for q in requests():add('gallery_page',kind+' '+q['request_id'],OUT/'gallery'/kind/(q['request_id']+'.png'))
    for rid,view,xlim,ylim in CASES:
        for suffix,title in [('_stages','三模型四阶段'),('_hinge','三模型HINGE前后')]:
            add('fixed_cases',rid+' '+title,OUT/'sections/fixed_cases'/(rid+suffix+'.png'))
        add('fixed_cases',rid+' 原基线HINGE局部放大',OUT/'topology/baseline_hinge_followup_v2'/(rid+'_fixed_hinge.png'))
    # Only existing records are advertised. The index does not manufacture a
    # CLEAN/SMOOTH review status from the presence of a PNG.
    for p in sorted((OUT/'visual').glob('*')):
        if p.is_file() and p.suffix in ('.csv','.json','.md'):add('review_records',p.name,p,False)
    p=OUT/'topology/baseline_hinge_followup_v2/actual_image_review.json'
    add('review_records','原基线指定局部窗口的实际图审',p,False)
    add('review_records','三模型指定局部窗口的实际四阶段图审',OUT/'sections/fixed_cases/actual_four_stage_review.json',False)
    add('review_records','三模型指定局部窗口的实际HINGE图审',OUT/'sections/fixed_cases/actual_hinge_review.json',False)
    add('review_records','HINGE面积尾部两例的局部实际图审',OUT/'visual/hinge_shape_tails/actual_review.json',False)
    for name in ('73b369b9_n3_slot192','608b46b0_n0_slot54'):
        add('optional','局部形变放大：'+name,OUT/'visual/hinge_shape_tails'/(name+'.png'),False)
    add('optional','HINGE独立审计',OUT/'hinge/baseline_audit_findings.txt',False)
    return groups


def run(check=False):
    entries=inventory();missing=[r for r in entries if r['required'] and not r['exists']]
    status=dict(status='BLOCKED_MISSING_ARTIFACTS' if missing else 'READY_TO_BUILD',
        expected_required=sum(r['required'] for r in entries),existing_required=sum(r['required'] and r['exists'] for r in entries),
        missing=[dict(title=r['title'],path=str(r['path'])) for r in missing])
    if check:return status
    if missing:raise FileNotFoundError(json.dumps(status,ensure_ascii=False,indent=2))
    dest=OUT/'review';dest.mkdir(parents=True,exist_ok=True)
    def href(path):return html.escape(os.path.relpath(Path(path),dest).replace(os.sep,'/'),quote=True)
    def link(title,path):return '<a href="'+href(path)+'">'+html.escape(title)+'</a>'
    def group_links(group):
        rr=[r for r in entries if r['group']==group and r['exists']]
        return '<ul>'+''.join('<li>'+link(r['title'],r['path'])+'</li>' for r in rr)+'</ul>' if rr else '<p class="pending">暂无真实记录。</p>'
    content=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>SEAM Hinge Quality 本地审阅</title>',
        '<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1450px;margin:2rem auto;padding:0 1rem;color:#222;background:#fff}a{color:#165d9c}h1{font-size:1.8rem}h2{margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.35rem}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ddd;padding:.5rem;text-align:left}th{background:#f4f6f8}img{width:100%;height:auto}p.note{background:#eef5fa;padding:1rem}.pending{color:#9c4f08}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}figure{margin:0}figcaption{font-size:14px}</style>',
        '<h1>SEAM Hinge Quality 本地审阅</h1>',
        '<p class="note">28个固定请求（16 TRAIN / 12 DEV）。新臂各新增1,000更新，与 PLAIN_S2000 作同预算比较。页面不调用网络或模型。所有链接在构建时核验存在；“已渲染”不等于“已图审”，实际观察以独立 AI_PROXY 记录为准。</p>',
        '<p>蓝色为 C，灰色为自由面；D0 提议中的蓝色 C 尚未恢复。截面与相机沿用旧面板的同位置、同尺度设置。截面短线不能单独证明悬片，合法共边图与无过滤实际索引边图分别报告。同一槽位编号仅用于单份输出的阶段追踪，不代表跨模型同语义面。</p>',
        '<p class="note">噪声身份说明：旧复用绘图器标题中的 “fixed original Gaussian” 指共享的底层固定高斯数组。SMOOTH_PRIOR 在 bulk 槽上实际使用 T₀.₂₅(Z)=mean_corner(Z)+0.25·(Z−mean_corner(Z))；其实际初始坐标与其余两模型不同，这是本臂的明确干预。C和seam角色未作该变换；PNG保留原渲染记录，不将其标题解释为三模型起点逐位相同。</p>',
        '<h2>报告与冻结合同</h2>',group_links('report'),
        '<h2>图集入口</h2>',group_links('four_stage_index'),group_links('gallery_index'),
        '<h2>56份新模型四阶段图与逐请求对比</h2>',
        '<p>四阶段图包含 front / side 有限截线及真实半空间剖切。RAW、POST、BOUND015 对比列为原基线和两新臂；HINGE 每页比较同模型 POST、1.5%L、2.5%L。</p>',
        '<table><thead><tr><th>请求</th><th>分组</th><th>CLEAN 四阶段</th><th>SMOOTH 四阶段</th><th>RAW / POST / BOUND</th><th>原基线 HINGE</th><th>CLEAN HINGE</th><th>SMOOTH HINGE</th></tr></thead><tbody>']
    for q in requests():
        rid=q['request_id'];r=[html.escape(rid),q['split']]
        r.extend(link(arm,OUT/'sections'/arm/(rid+'.png')) for arm in ARMS)
        r.append(' · '.join(link(s,OUT/'gallery'/s/(rid+'.png')) for s in ('RAW','POST','BOUND015')))
        r.extend(link('POST / 015 / 025',OUT/'gallery'/('HINGE_'+arm)/(rid+'.png')) for arm in ('PLAIN_S2000',*ARMS))
        content.append('<tr>'+''.join('<td>'+x+'</td>' for x in r)+'</tr>')
    content.extend(['</tbody></table>','<h2>指定位置放大截面</h2>',
        '<p>608/n3 使用原 front 位置；3f/n2 使用原 side 位置。下列为预先指定的局部窗口，不代替完整图集。空窗口也可能表示表面移动到窗口外，不能单独当作碎片消失。</p>','<div class="grid">'])
    for r in entries:
        if r['group']=='fixed_cases':
            content.append('<figure><a href="'+href(r['path'])+'"><img loading="lazy" alt="'+html.escape(r['title'],quote=True)+'" src="'+href(r['path'])+'"></a><figcaption>'+link(r['title'],r['path'])+'</figcaption></figure>')
    content.extend(['</div>','<h2>完整数值、分母和配对记录</h2>',group_links('tables'),
        '<h2>实际图审记录</h2>','<p>以下只列已存在的真实记录；不会由指标自动生成视觉判断。AI_PROXY 为非盲评，不能冒称人类独立评审。</p>',group_links('review_records'),
        '<h2>补充审计</h2>',group_links('optional'),'</html>'])
    page=dest/'index.html';page.write_text('\n'.join(content),encoding='utf-8')
    files={str(r['path']):C.sha(r['path']) for r in entries if r['exists']}
    result=dict(status='COMPLETE_LINKS_VERIFIED',version=VERSION,code_sha256=C.sha(__file__),index_path=str(page),index_sha256=C.sha(page),
        four_stage_new_pages=56,gallery_pages=168,required_links=len([r for r in entries if r['required']]),
        files=files,model_calls=0,network_calls=0,review_status='Consult linked actual observation records; rendered images alone do not certify review')
    C.write(dest/'completion.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();r=run(a.check)
    print(json.dumps(r if a.check else {k:v for k,v in r.items() if k!='files'},ensure_ascii=False,indent=2),flush=True)
