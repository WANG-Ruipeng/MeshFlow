
from pathlib import Path
p=Path("native_t1/train_cli.py")
s=p.read_text()
s=s.replace("\n\n\ndef encoder_update_metrics", "\n\ndef encoder_update_metrics")
s=s.replace('''    if args.init_geo is not None and args.context_encoder != "geo":''','''    if args.seed is not None and not 0 <= args.seed <= 2**32-1-70000:
        raise ValueError("--seed must be between 0 and 2**32-1-70000 for the five input RNGs")
    if args.encoder_seed is not None and not 0 <= args.encoder_seed < 2**63:
        raise ValueError("--encoder-seed must be between 0 and 2**63-1")
    if args.init_geo is not None and args.context_encoder != "geo":''')
p.write_text(s)
p=Path("native_t1/tests/test_geo_cli.py")
s=p.read_text().replace('''                ["--init-geo", "geo.pt"],''','''                ["--init-geo", "geo.pt"],
                ["--init-t1", "pure.pt", "--seed", "-1"],
                ["--init-t1", "pure.pt", "--seed", str(2**32-70000)],
                ["--init-t1", "pure.pt", "--context-encoder", "geo", "--encoder-seed", "-1"],
                ["--init-t1", "pure.pt", "--context-encoder", "geo", "--encoder-seed", str(2**63)],''')
p.write_text(s)
