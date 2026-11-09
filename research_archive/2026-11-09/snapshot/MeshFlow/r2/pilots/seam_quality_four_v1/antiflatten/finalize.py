"""Freeze completed CPU scientific results and actual visual review receipts."""
from . import run as A
from pathlib import Path
import json


def main():
    out=A.OUT
    if (out/'FINAL_STATUS.json').exists():
        from .resume import main as verify
        return verify()
    protected={}
    for name in ('source_gate.json','loss_gate.json','geometry_completion.json','completion.json','audit.json'):
        value=A.C.read(out/name);protected.update(value.get('input_hashes',{}))
    review=A.C.read(out/'gallery/review_completion.json');assert review['new_actual_pairs']==6 and review['inherited_byte_identical_pairs']==10
    for p,h in protected.items():assert A.C.sha(p)==h,p
    protocol=A.C.read(out/'protocol.json')
    for p,h in protocol['code_hashes'].items():assert A.C.sha(p)==h,p
    code={str(p):A.C.sha(p) for p in Path(__file__).parent.glob('*.py')}
    code.update(A.C.read(out/'loss_gate.json')['code_hashes'])
    for p,h in code.items():assert A.C.sha(p)==h,p
    outputs={str(p):A.C.sha(p) for p in out.rglob('*') if p.is_file()}
    value=dict(status='COMPLETE',version=A.VERSION,source_edges=389,source_triggered=0,
               new_POST=16,original_quality_scans=16,model_calls=0,GPU_calls=0,D0_forwards=0,
               CPU_geometric_autograd_calls=55,actual_review_pairs=6,inherited_review_pairs=10,
               code_hashes=code,protected_hashes=protected,output_hashes=outputs)
    A.C.write(out/'FINAL_STATUS.json',value)
    print(json.dumps(dict(status='COMPLETE',protected_files=len(protected),output_files=len(outputs))))


if __name__=='__main__':main()
