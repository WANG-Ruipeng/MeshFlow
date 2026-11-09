"""Complete unexecuted final delivery audit coverage without model work."""
from pathlib import Path
p=Path(__file__).with_name('delivery_audit.py');s=p.read_text()
s=s.replace("'protocol/protected_sources_before.json'):","'protocol/protected_sources_before.json','protocol/evaluation_sources_before.json'):")
s=s.replace("out/'raw_index.json'","out/'raw_integrity_index.json'")
s=s.replace("out/'raw_index.csv'","out/'raw_integrity_index.csv'")
s=s.replace("'experiments/mf_h1_local0/guided_native_geometry_geometry.py']", "'experiments/mf_h1_local0/guided_native_geometry_geometry.py',\n        'experiments/mf_h1_local0/control_mvp48/eval_scoring.py',\n        'experiments/mf_h1_local0/gpu_guidance_evaluation.py',\n        'native_t1/runs/edge_factorial_20261001_123000/source_snapshot/experiments/mf_h1_local0/edge_factorial_pilot/edge_metric.py']")
p.write_text(s);compile(s,str(p),'exec')
print('Final delivery audit source coverage ready')
