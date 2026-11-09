"""Small deterministic report tables from completed full metrics, no rescoring."""
import argparse,csv,json
from pathlib import Path
from .protocol import OUT,WAVE1,WAVE2

ORDER=('LONG','B_LONG_LAYOUT','PLAIN_S1000','PLAIN_S2000','PLAIN3000','WISE_B_PLAIN_050',*WAVE2)
SHORT={'B_LONG_LAYOUT':'B(+0)','PLAIN_S1000':'PLAIN+1000','PLAIN_S2000':'PLAIN+2000','PLAIN3000':'PLAIN+3000',
       'WISE_B_PLAIN_050':'WISE0.5','CLEAN_BULK_FM_S1000':'CLEAN+1000','SURGICAL_LATE4_S1000':'SURGICAL+1000','LONG':'LONG'}
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def num(row,key,percent=False):
    x=row.get(key,'')
    if x in ('','None','NA'):return 'NA'
    return f'{float(x)*(100 if percent else 1):.3f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,r))+' |' for r in rows]])+'\n'
def run(scope):
    folder=OUT/'results'/scope;records=read(folder/'key_results.csv')
    lookup={(r['label'],r['stage'],r['split']):r for r in records}; labels=[a for a in ORDER if (a,'POST','DEV') in lookup]
    pieces=['# 可复算关键表','全部比率显示为百分数。逐输出原始比率与全部124字段保留在同目录CSV。I为ConflictCov_C，非整个网格冲突面积。\n']
    for split in ('TRAIN','DEV'):
        for stage in ('RAW','POST','BOUND015'):
            pieces+=['## '+split+' / '+stage]
            rows=[]
            for a in labels:
                r=lookup[a,stage,split]
                rows.append([SHORT[a],num(r,'FF_illegal_affected_free_fraction',True),num(r,'CF_illegal_affected_free_fraction',True),
                    num(r,'ConflictCov_C',True),num(r,'free_only_reached_bulk_area_fraction',True),
                    num(r,'free_only_reached_bulk_area_fraction_median',True),f"{r['bulk_reach30_count']}/{r['n']}",
                    num(r,'legal_seam_bulk_edge_count'),num(r,'C_F_valid_edge_count')])
            pieces.append(table(['模型','FF%','CF%','I%','bulk可达均值%','中位数%','≥30%请求','合法seam-bulk边','合法C/F边'],rows))
            rows=[]
            for a in labels:
                r=lookup[a,stage,split]
                rows.append([SHORT[a],num(r,'EXACT_COORD_face_components'),num(r,'EXACT_COORD_boundary_edges'),
                    num(r,'EXACT_COORD_nonmanifold_edges'),num(r,'free_minimum_angle_lt5_fraction',True),num(r,'free_q_gt10_fraction',True),
                    num(r,'actual_free_faces'),num(r,'deleted_free_faces')])
            pieces.append(table(['模型','连通分量','开放边','非流形边','小角%','q>10%','保留free','删除free'],rows))
            if stage=='BOUND015':
                rows=[]
                for a in labels:
                    r=lookup[a,stage,split]
                    rows.append([SHORT[a],r['sum_accepted_tip'],r['sum_accepted_edge'],r['sum_FF_new_illegal_pairs'],
                         r['sum_FF_illegal_pairs_removed'],r['sum_CF_new_illegal_pairs'],r['sum_CF_lost_legal_edges'],
                         r['sum_FF_lost_legal_edges'],r['sum_deleted_source_faces']])
                pieces.append(table(['模型','TIP接受','整侧边接受','新增FF对','消除FF对','新增CF对','丢合法CF边','丢合法FF边','删源面'],rows))
    if scope=='final':
        parent=read(folder/'paired_per_parent.csv')
        wanted=[r for r in parent if r['purpose']=='PRIMARY_EQUAL_1000_UPDATES' and r['stage']=='POST']
        key={(r['candidate'],r['parent_id'],r['metric']):r for r in wanted}
        rows=[]
        for arm in WAVE2:
            for par in dict.fromkeys(r['parent_id'] for r in wanted if r['candidate']==arm):
                def delta(metric,p=False):return num(key[arm,par,metric],'delta',p)
                rows.append([SHORT[arm],par[:8],key[arm,par,'ConflictCov_C']['split'],delta('FF_illegal_affected_free_fraction',True),
                    delta('illegal_pairs_bulk-bulk'),delta('CF_illegal_affected_free_fraction',True),delta('ConflictCov_C',True),
                    delta('EXACT_COORD_face_components'),delta('actual_free_faces'),delta('free_only_reached_bulk_area_fraction',True)])
        pieces+=['## Wave2逐父POST差值：候选 − PLAIN+1000',
                 table(['候选','父','split','ΔFF pp','Δbulk-bulk对','ΔCF pp','ΔI pp','Δ分量','Δ保留free','Δbulk可达 pp'],rows)]
    path=folder/'KEY_TABLES.md';path.write_text('\n\n'.join(pieces),encoding='utf-8')
    print(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope',choices=['wave1','final']);run(p.parse_args().scope)
