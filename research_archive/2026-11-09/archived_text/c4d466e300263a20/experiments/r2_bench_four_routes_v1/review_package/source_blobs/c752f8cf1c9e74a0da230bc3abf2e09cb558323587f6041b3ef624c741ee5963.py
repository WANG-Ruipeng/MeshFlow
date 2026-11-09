"""Prepare an isolated copy of the qualified same-panel evaluator."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'r2/pilots/bench_group_weight_match_v1'
DST=Path(__file__).parent/'exposure'


def fn_replace(text,name,new):
    tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+new.strip()+'\n'+''.join(lines[node.end_lineno:])


def save(name,text):
    p=DST/name
    if p.exists() and p.read_text(encoding='utf-8')!=text:raise ValueError('Existing prepared file differs: '+name)
    if not p.exists():p.write_text(text,encoding='utf-8',newline='\n')


def main():
    text=(SRC/'evaluation.py').read_text(encoding='utf-8')
    text=text.replace('parents[3]','parents[4]').replace("OUT = ROOT/'experiments/r2_bench_group_weight_match_v1'","OUT = ROOT/'experiments/r2_bench_four_routes_v1/exposure'\nOLD_M = ROOT/'experiments/r2_bench_group_weight_match_v1'")
    text=text.replace("LABELS = ('B', 'W', 'M')","LABELS = ('M', 'EXPOSURE')").replace("NEW_LABELS = ('M',)","NEW_LABELS = ('EXPOSURE',)")
    text=text.replace("ARMS = {'M':'W_MATCHED'}","ARMS = {'EXPOSURE':'M_PLUS3000'}")
    text=text.replace("label=='M'","label=='EXPOSURE'").replace("('B','W')","('M',)")
    text=text.replace("from .runtime import ","from r2.pilots.bench_natural_n_recipe_v1.runtime import ")
    text=text.replace("from . import runtime as n_runtime","from r2.pilots.bench_natural_n_recipe_v1 import runtime as n_runtime")
    text=text.replace('bench_group_weight_match_evaluation_v1','bench_four_routes_exposure_evaluation_v1')
    text=text.replace('BW_cache_audit.json','reference_cache_audit.json')
    text=text.replace('reused_score_rows=6*n,total_score_rows=9*n','reused_score_rows=3*n,total_score_rows=6*n')
    text=text.replace('reused_score_rows=276','reused_score_rows=138').replace('score_rows_reused=276','score_rows_reused=138')
    text=text.replace('old_reused_requests=92','old_reused_requests=46')
    text=text.replace("oldrows=read(WIDE/'results/metric_rows.json')","oldrows=[r for r in read(OLD_M/'results/metric_rows.json') if r['label']=='M']")
    text=text.replace("WIDE/'results/metric_rows.json'","OLD_M/'results/metric_rows.json'")
    text=text.replace("WIDE/'results/completion.json'","OLD_M/'results/completion.json'")
    text=text.replace('len(oldrows)!=276','len(oldrows)!=138').replace('len(cells)!=276','len(cells)!=138')
    text=text.replace("visual_labels=['W','M']","visual_labels=['M','EXPOSURE']")
    text=text.replace("primary_contrast='M-W',secondary_quantitative_contrast='M-B'","primary_contrast='EXPOSURE-M',secondary_quantitative_contrast=None")
    text=text.replace("for ref,cand in (('W','M'),('B','M')):","for ref,cand in (('M','EXPOSURE'),):")
    text=fn_replace(text,'storage_label',"def storage_label(label,q=None):return label")
    text=fn_replace(text,'root_for',"""
def root_for(label,q=None):
    if label not in LABELS:raise ValueError('Unknown label')
    return OLD_M if label=='M' else OUT
""")
    text=fn_replace(text,'raw_completion',"def raw_completion(label):return root_for(label)/'evaluation'/label/'raw_complete_full.json'")
    text=fn_replace(text,'label_spec',"""
def label_spec(label):
    if label=='M':
        from r2.pilots.bench_group_weight_match_v1.evaluation import label_spec as prior
        return prior('M')
    if label!='EXPOSURE':raise ValueError(label)
    return dict(mode='PLAIN',arm='M_PLUS3000',local_step=3000,global_step=25500,subset='full',
        schema='bench_four_routes_exposure_generator_v1',generator_path=str(OUT/'training/M_PLUS3000/generator_step3000.pt'))
""")
    save('evaluation.py',text)
    report=(SRC/'report_results.py').read_text(encoding='utf-8')
    report=report.replace("CONTRASTS=(('W','M'),('B','M'))","CONTRASTS=(('M','EXPOSURE'),)")
    report=report.replace("VISUAL_LABELS=('W','M')","VISUAL_LABELS=('M','EXPOSURE')")
    report=report.replace("VISUAL_CONTRASTS=(('W','M'),)","VISUAL_CONTRASTS=(('M','EXPOSURE'),)")
    report=report.replace('bench_group_weight_match_report_v2','bench_four_routes_exposure_report_v2')
    save('report_results.py',report)


if __name__=='__main__':main()
