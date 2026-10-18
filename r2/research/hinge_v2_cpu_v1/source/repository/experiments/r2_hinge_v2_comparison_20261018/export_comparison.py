"""Read frozen metrics only; no mesh/model execution or historical mutation."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/e/MeshFlow')
SRC=ROOT/'experiments/r2_hinge_v2_cpu_v1'
OUT=Path(__file__).parent

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def num(x):return None if x=='' else float(x)
def show(x):return 'NA' if x is None else f'{x:.10g}'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    sources=[SRC/'metric_summary.csv',SRC/'paired_summary.csv',SRC/'paired_per_parent.csv',SRC/'scores/per_output.csv']
    before={str(p):sha(p) for p in sources}
    data=read(sources[0]);paired=read(sources[1]);parents=read(sources[2]);raw=read(sources[3])
    lookup={(r['stage'],r['split'],r['metric']):r for r in data}
    pairs={(r['split'],r['metric']):r for r in paired if r['comparison']=='ALPHA_HOLE_vs_BASELINE'}
    metrics=[r['metric'] for r in data if r['stage']=='BASELINE' and r['split']=='ALL']
    assert len(metrics)==179 and len(set(metrics))==179
    rows=[];lines=['# 原HINGE与CPU组合方案：完整179字段对照','',
      '原HINGE = PLAIN_S2000 + 单次D0/GEO_C + HINGE_OPP015。CPU组合 = 同一保存结果 + ALPHA + HOLE（ALPHA_HOLE）。同28请求、同C、同噪声，生成器与D0未重新运行。',
      '', '本表由既有完整数值表读取，不新增评分。原实验报告及结果保持只读。用户接受本轮I的小幅变化，此处仍完整展示数值。',
      '', '以下所有字段按原始尺度列出逐输出宏均值：fraction、A-VMR、PMR、I的0.01表示1%；计数字段也是每输出均值。CSV另含全组总数。变化=CPU组合−原HINGE；胜/平/负采用原字典方向，context_only没有优劣方向。NA保留原原因，新增桥形状仅12份有桥输出适用。',
      '', '碎片口径特别区分：任意共边图单面分量121→102；合法共边图单面分量221→198。两项不是同一指标。']
    for split in ('ALL','TRAIN','DEV'):
        lines+=['',f'## {split}：完整179字段','', '| 字段 | 原HINGE | CPU组合 | 变化 | 胜/平/负 | 有效样本 原/新 |','|---|---:|---:|---:|---:|---:|']
        for m in metrics:
            b=lookup['BASELINE',split,m];a=lookup['ALPHA_HOLE',split,m];p=pairs[split,m]
            x,y=num(b['mean']),num(a['mean']);d=None if x is None or y is None else y-x
            rows.append(dict(split=split,metric=m,direction=b['direction'],inherited_135=b['inherited_135'],
                baseline_mean=x,cpu_mean=y,delta_mean=d,baseline_total=num(b['total']),cpu_total=num(a['total']),
                win=p['win'],tie=p['tie'],loss=p['loss'],baseline_valid=b['n_valid'],cpu_valid=a['n_valid']))
            lines.append(f"| `{m}` | {show(x)} | {show(y)} | {show(d)} | {p['win']}/{p['tie']}/{p['loss']} | {b['n_valid']}/{a['n_valid']} |")
    lines+=['','## 逐父与逐请求','',
        '详见同目录 `per_parent_comparison.csv`（全部7父×179字段）和 `per_output_two_stages.csv`（28请求×2阶段）。相同父四噪声配对，没有换请求或删除坏例。','',
        '本轮新增补面不计入原bulk面积；固定原父FF/CF分母不含新桥。actual FF/CF改善中由新桥扩大分母造成的部分不能算旧面质量改善。','']
    def write_csv(name,rr):
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    write_csv('all_179_metrics.csv',rows)
    write_csv('per_parent_comparison.csv',[r for r in parents if r['comparison']=='ALPHA_HOLE_vs_BASELINE'])
    write_csv('per_output_two_stages.csv',[r for r in raw if r['stage'] in ('BASELINE','ALPHA_HOLE')])
    (OUT/'FULL_COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')
    assert before=={str(p):sha(p) for p in sources}
    files={str(p):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='receipt.json'}
    (OUT/'receipt.json').write_text(json.dumps(dict(status='COMPLETE_READ_ONLY_DERIVATION',metrics=179,group_rows=len(rows),
        full_scores=56,input_hashes=before,outputs=files,model_calls=0,new_geometry_or_scoring=0),indent=2),encoding='utf-8')
    print('COMPLETE_READ_ONLY_COMPARISON: 179 metrics x ALL/TRAIN/DEV')

if __name__=='__main__':main()
