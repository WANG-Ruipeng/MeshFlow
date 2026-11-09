"""CPU-only, same-time-bin window summaries from saved training diagnostics."""
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[4]
EXP=ROOT/'experiments/r2_seam_quality_four_v1'


def main():
    definitions=[('SEAM200_SHARED_HISTORY',ROOT/'experiments/r2_seam_slot_c200_v1/training/SEAM_SLOT_C200/geometry_diagnostics.jsonl',200,0)]
    definitions += [(a,EXP/'training'/a/'geometry_diagnostics.jsonl',800,200) for a in ('LONG','ANTI_COVER','LOCAL_NOISE')]
    rows=[];sources={}
    for arm,path,count,offset in definitions:
        payload=path.read_bytes();sources[str(path.relative_to(ROOT))]=hashlib.sha256(payload).hexdigest()
        records=[json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
        assert len(records)==count*8
        for window,first,last in [('first50',1,50),('last50',count-49,count)]:
            for time_bin in range(5):
                selected=[r for r in records if first<=r['local_step']<=last and r['time_bin']==time_bin]
                assert selected
                row=dict(arm=arm,window=window,local_first=first,local_last=last,
                    seam_total_first=offset+first,seam_total_last=offset+last,time_bin=time_bin,
                    t_lower=selected[0]['time_bin_lower'],t_upper=selected[0]['time_bin_upper'],microbatches=len(selected))
                for group in ('source_interface_third','bulk'):
                    denominator=sum(r[group+'_scalar_count'] for r in selected)
                    row[group+'_scalar_count']=denominator
                    for space in ('velocity','endpoint'):
                        row[group+'_'+space+'_mse']=sum(r[group+'_'+space+'_sse'] for r in selected)/denominator
                rows.append(row)
    out=EXP/'training_analysis/time_bin_window_summary.csv'
    with out.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    receipt=dict(status='PASS',rows=len(rows),definition='Pooled coordinate SSE / scalar count within fixed update window and fixed t bin; not a microbatch macro mean',
        old_time_bin_comparison='Old report .2528606193 -> .0930237050 is SEAM200 t in [0.4,0.6), first/last50. All-t .326099528 -> .112029071 is a different marginal.',
        source_hashes=sources,output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),model_calls=0,updates=0)
    (EXP/'training_analysis/time_bin_window_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps([r for r in rows if r['time_bin']==2],indent=2))


if __name__=='__main__':main()
