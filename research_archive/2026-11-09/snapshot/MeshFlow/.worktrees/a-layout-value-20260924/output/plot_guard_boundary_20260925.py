from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.a_radial.io_utils import file_hash,write_json,new_dir
source=Path('output/a_balanced_guard_replay_20260925_01/REPLAY.json')
r=json.loads(source.read_text());rows=r['records'];out=new_dir('output/a_balanced_guard_figure_20260925_01')
assert r['guard_calls']==29 and r['expected_failure_reproduced'] and len(rows)==29
x=[v['step_one_based'] for v in rows]
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
axes[0].semilogy(x,[v['input']['minimum'] for v in rows],'o-',label='Float64 input minimum det',ms=3)
axes[0].scatter([29],[rows[-1]['failed_candidate_decimal']['minimum_float']],marker='*',s=100,c='#aa3377',label='Failed candidate: 100-digit readback')
axes[0].annotate('Float64 candidate = 0; assertion fails',xy=(29,rows[-1]['failed_candidate_decimal']['minimum_float']),xytext=(6,1e-17),arrowprops={'arrowstyle':'->'},fontsize=9)
axes[0].set_ylabel('Minimum oriented face determinant');axes[0].set_xlabel('Original rollout step (1-based)');axes[0].legend(fontsize=8,loc='upper right')
axes[1].semilogy(x,[v.get('alpha',v.get('alpha_after_original_loop')) for v in rows],'o-',ms=3,c='#228833')
axes[1].set_ylabel('Original guard accepted / attempted alpha');axes[1].set_xlabel('Original rollout step (1-based)')
axes[1].annotate('Step29: alpha = 7.1e-15',xy=(29,7.105427357601002e-15),xytext=(4,1e-11),arrowprops={'arrowstyle':'->'},fontsize=9)
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Balanced seed11 / internal object 394302e2: one unchanged CPU4 failure replay',fontsize=11)
fig.text(.5,.02,'Near-collapse reaches floating-point cancellation; positive high-precision sign does not certify final mesh quality. No guard change, no retry.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(out/'guard_boundary.png',dpi=170);fig.savefig(out/'guard_boundary.pdf');plt.close(fig)
write_json(out/'FIGURE_METADATA.json',{'source_sha256':file_hash(source),'source_code_sha256':file_hash(__file__),'no_model_inference_or_geometry_recalculation':True,'only_saved29records_used':True})
print(out)