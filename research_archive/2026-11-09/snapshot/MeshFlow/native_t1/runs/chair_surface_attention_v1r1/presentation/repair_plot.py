"""Read stored scientific results and move legends clear of data; no model calls."""
from pathlib import Path
import hashlib
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

r=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=r/'precheck/result.json';q=r/'reference/reference_result.json';before={str(p):sha(p),str(q):sha(q)}
pre,ref=read(p),read(q);target=r/'figures/repair_evidence.png';original=r/'presentation/repair_evidence.initial.png'
if not original.exists():shutil.copyfile(target,original)
tasks=pre['cost']['per_fixture'];fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
x=np.arange(len(tasks))
axes[0].bar(x-.18,[t['surface_to_H_CONT_forward_median_ratio'] for t in tasks],.36,label='Full forward')
axes[0].bar(x+.18,[t['surface_to_H_CONT_forward_backward_median_ratio'] for t in tasks],.36,label='Full F+B')
axes[0].axhline(3,color='black',linestyle='--');axes[0].set_xticks(x,[t['fixture_id'] for t in tasks],rotation=20)
axes[0].set_ylabel('SURFACE / H_CONT (same task)');axes[0].legend(loc='upper left');axes[0].set_title('Measured whole-model cost')
for n in (8,64,128):
    vals=[next(c['mu_L2'] for c in ref['comparisons'] if c['pair_id']==v['pair']['pair_id'] and c['rule_n']==n) for v in ref['pairs'][:4]]
    axes[1].plot(range(4),vals,marker='o',label=str(3*n*n)+' nodes')
axes[1].axhline(.01,color='black',linestyle='--');axes[1].set_yscale('log')
axes[1].set_xticks(range(4),[v['pair']['pair_id'] for v in ref['pairs'][:4]],rotation=20)
axes[1].set_ylabel('Mean-vector L2 difference');axes[1].legend(loc='lower right');axes[1].set_title('Against independent adaptive reference')
fig.savefig(target,dpi=160);plt.close(fig)
assert before=={str(p):sha(p),str(q):sha(q)}
(r/'presentation/plot_review.json').write_text(json.dumps(dict(status='PASS_PRESENTATION_ONLY',
    changes='Move both legends clear of observed data; numerical arrays unchanged',original_figure_sha256=sha(original),
    corrected_figure_sha256=sha(target),input_hashes=before,script_sha256=sha(Path(__file__)),model_calls=0,
    relation_calls=0),indent=2)+'\n',encoding='utf8')
print('PLOT_LABELS_REPOSITIONED_NO_SCIENTIFIC_CHANGE')
