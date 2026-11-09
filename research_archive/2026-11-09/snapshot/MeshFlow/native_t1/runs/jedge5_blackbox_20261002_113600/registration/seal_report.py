from pathlib import Path
p=Path('experiments/mf_h1_local0/jedge5_blackbox/report.py');s=p.read_text();s=s.replace("status='COMPLETED_PENDING_FINAL_AUDIT'", "status='PASS_EXECUTION',final_audit='final_audit.json'");p.write_text(s,encoding='utf-8')
