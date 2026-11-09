from pathlib import Path
p=Path('experiments/mf_h1_local0/jedge5_blackbox/report.py');s=p.read_text(encoding='utf-8-sig');s=s.replace('未计算这些真实例的连续积分，不能把份额叫37.5倍。','这些是固定teacher状态的离线复算，不是历史逐次optimizer训练的实时日志；t<.5的原始项也可作观测，但不表示训练时门控激活。未计算这些真实例的连续积分，不能把份额叫37.5倍。');p.write_text(s,encoding='utf-8')
