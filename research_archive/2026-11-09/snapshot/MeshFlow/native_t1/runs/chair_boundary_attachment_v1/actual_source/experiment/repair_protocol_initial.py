"""Pre-dispatch source/metric receipt paths; zero model calls."""
from pathlib import Path
p=Path(__file__).with_name('protocol.py');s=p.read_text()
s=s.replace("'worker.py','resources.py')","'worker.py','resources.py','protocol.py','control_metrics.py','metric_tests.py')")
s=s.replace("tests/metrics/results.json","metric_tests_final/metric_test_results.json")
p.write_text(s)
print('Protocol receipt paths ready')
