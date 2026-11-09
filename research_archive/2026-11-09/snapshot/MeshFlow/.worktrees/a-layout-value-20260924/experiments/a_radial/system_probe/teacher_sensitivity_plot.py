from pathlib import Path
import json,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..io_utils import file_hash,write_json

def main(run):
    run=Path(run);s=json.loads((run/'SUMMARY.json').read_text());fig,ax=plt.subplots(figsize=(9,4.8))
    for uid in s['moving_parents']:
        rows=[r for r in s['per_parent'] if r['uid']==uid and r['fraction']>0]
        ax.plot([r['fraction'] for r in rows],[r['angular_pass']/16 for r in rows],marker='.',alpha=.55,label=uid[:8])
    rows=[r for r in s['moving_aggregate'] if r['fraction']>0]
    ax.plot([r['fraction'] for r in rows],[r['angular_pass_fraction'] for r in rows],color='black',linewidth=2.5,label='equal-parent average')
    ax.set_xscale('log');ax.set(xlabel='Tangent perturbation RMS / teacher displacement RMS',ylabel='Fraction passing both angular conditions',ylim=(-.03,1.03))
    fig.suptitle('Teacher angular sensitivity\n11 moving training parents, 16 fixed perturbations each',fontsize=12)
    ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2,bbox_to_anchor=(1,1),loc='upper left');fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(run/('sensitivity_v2.'+suffix),dpi=160,bbox_inches='tight')
    plt.close(fig)
    write_json(run/'FIGURE_V2.json',{'reason':'Fix clipped title; data unchanged','summary_sha256':file_hash(run/'SUMMARY.json'),
        'source_sha256':file_hash(__file__),'files':{f'sensitivity_v2.{ext}':file_hash(run/f'sensitivity_v2.{ext}') for ext in ['png','pdf']}})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();main(a.run)