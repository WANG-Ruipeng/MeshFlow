
from pathlib import Path
p=Path("native_t1/train_cli.py")
s=p.read_text()
s=s.replace('"""Explicit portable pure-FM training; no work is performed at import time."""','"""Explicit portable pure-FM training for Native T1 and its optional T_geo branch."""')
s=s.replace('from .model import native_execution','from .model import native_execution\nfrom .accounting import count_ot_calls')
s=s.replace('description="Pure Native T1 training on the registered N112 two-parent task"','description="Native T1 / T_geo pure-FM training on the registered N112 two-parent task"')
s=s.replace('    parser.add_argument("--config",', '''    init.add_argument("--init-geo", type=Path, help="A T_geo checkpoint; fresh AdamW")
    parser.add_argument("--context-encoder", choices=("none", "geo"), default="none",
                        help="Default: unchanged pure T1; geo adds the known-face geometry encoder")
    parser.add_argument("--encoder-seed", type=int,
                        help="Fresh geo encoder initialization seed (default 1010); not used with --init-geo")
    parser.add_argument("--config",''')
s=s.replace('help="Restore input RNG from --init-t1; optimizer is still reset"','help="Restore input RNG from --init-t1 or --init-geo; optimizer is still reset"')
s=s.replace('''        if args.init_t1 is None or args.seed is not None:
            raise ValueError("--continue-stream requires --init-t1 and cannot be combined with --seed")
        payload = torch.load(args.init_t1, map_location="cpu", weights_only=True, mmap=True)
        if payload.get("schema") != SCHEMA or payload.get("input_sha256") != archive.sha256:''','''        checkpoint = args.init_t1 or args.init_geo
        if checkpoint is None or args.seed is not None:
            raise ValueError("--continue-stream requires --init-t1 or --init-geo and no --seed")
        from .geometry_checkpoint import SCHEMA as GEO_SCHEMA, LEGACY_SCHEMA
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        schemas = (SCHEMA,) if args.init_t1 else (GEO_SCHEMA, LEGACY_SCHEMA)
        if payload.get("schema") not in schemas or payload.get("input_sha256") != archive.sha256:''')
s=s.replace('raise ValueError("Only correct-context pure T1 training is supported")','raise ValueError("Only correct-context Native training is supported")')
s=s.replace('''    if args.continue_stream and (args.init_t1 is None or args.seed is not None):
        raise ValueError("--continue-stream requires --init-t1 and no --seed")''','''    if args.init_geo is not None and args.context_encoder != "geo":
        raise ValueError("--init-geo requires --context-encoder geo")
    if args.encoder_seed is not None and (args.context_encoder != "geo" or args.init_geo is not None):
        raise ValueError("--encoder-seed is only for a newly attached geo encoder")
    if args.continue_stream and ((args.init_t1 is None and args.init_geo is None) or args.seed is not None):
        raise ValueError("--continue-stream requires --init-t1 or --init-geo and no --seed")''')
s=s.replace('''                  input_path=str(archive.path),''','''                  context_encoder=args.context_encoder, input_path=str(archive.path),''')
s=s.replace('''        else:
            model, report["initialization"] = load_trained_checkpoint(args.init_t1, args.config)
            base_updates = int(report["initialization"]["cumulative_updates"])
        report["base_cumulative_updates"] = base_updates''','''        elif args.init_geo is not None:
            from .geometry_checkpoint import load_geometry_checkpoint
            model, report["initialization"] = load_geometry_checkpoint(args.init_geo, args.config)
            base_updates = int(report["initialization"]["cumulative_updates"])
        else:
            model, report["initialization"] = load_trained_checkpoint(args.init_t1, args.config)
            base_updates = int(report["initialization"]["cumulative_updates"])
        if args.context_encoder == "geo" and args.init_geo is None:
            from .context_geometry import attach_context_encoder
            report["encoder_initialization"] = attach_context_encoder(
                model, "geo", seed=1010 if args.encoder_seed is None else args.encoder_seed)
        report["base_cumulative_updates"] = base_updates''')
s=s.replace('''            samples = stream.next_effective_batch()
            report["counts"]["OT_calls"] += 8''','''            with count_ot_calls(report["counts"]):
                samples = stream.next_effective_batch()''')
s=s.replace('''                    samples = stream.next_effective_batch()
                    report["counts"]["OT_calls"] += 8''','''                    with count_ot_calls(report["counts"]):
                        samples = stream.next_effective_batch()''')
s=s.replace('''                    metrics = train_step(model, optimizer, samples, counts=report["counts"])
                    report["completed_updates"] = step''','''                    log_step = step in (1,100,250,args.updates)
                    encoder = getattr(model, "context_encoder", None)
                    before_encoder = ({name: value.detach().clone() for name, value in encoder.named_parameters()}
                                      if log_step and encoder is not None else None)
                    metrics = train_step(model, optimizer, samples, counts=report["counts"])
                    if before_encoder is not None:
                        metrics["encoder"] = encoder_update_metrics(encoder, before_encoder)
                    report["completed_updates"] = step''')
s=s.replace('''                    if step in (1,100,250,args.updates):''','''                    if log_step:''')
s=s.replace('''                        report["checkpoints"].append(save_checkpoint(path, model, optimizer, stream, archive,
                                                                    args.config, base_updates, step))''','''                        if args.context_encoder == "geo":
                            from .geometry_checkpoint import save_geometry_checkpoint
                            saved = save_geometry_checkpoint(path, model, optimizer, stream, archive,
                                config_path=args.config, base_identity=report["initialization"],
                                base_updates=base_updates, completed=step)
                        else:
                            saved = save_checkpoint(path, model, optimizer, stream, archive,
                                                    args.config, base_updates, step)
                        report["checkpoints"].append(saved)''')
anchor='\ndef run(args):'
helper='''
def encoder_update_metrics(encoder, before):
    """Selected-step, post-clip gradients and actual branch parameter deltas."""
    groups = {}
    for name, parameter in encoder.named_parameters():
        group = groups.setdefault(name.split(".")[0], dict(parameter_squared=0., gradient_squared=0., update_squared=0.))
        group["parameter_squared"] += float(parameter.detach().square().sum())
        group["gradient_squared"] += 0. if parameter.grad is None else float(parameter.grad.detach().square().sum())
        group["update_squared"] += float((parameter.detach()-before[name]).square().sum())
    return {name: dict(parameter_l2=values["parameter_squared"]**.5,
                       gradient_l2_after_clip=values["gradient_squared"]**.5,
                       actual_update_l2=values["update_squared"]**.5)
            for name, values in groups.items()}

'''
assert anchor in s
s=s.replace(anchor,'\n'+helper+anchor)
p.write_text(s)
p=Path("native_t1/__main__.py")
s=p.read_text().replace('description="Pure Native T1, fixed N112 and 50 Euler steps"','description="Native T1 / T_geo, fixed N112 and 50 Euler steps"')
s=s.replace('choices=("a-continue", "original-t1", "trained")','choices=("a-continue", "original-t1", "trained", "trained-geo")')
s=s.replace('help="Required for trained; otherwise a relocated copy of the pinned profile checkpoint"','help="Required for trained/trained-geo; otherwise a relocated copy of the pinned profile checkpoint"')
s=s.replace('''    raise ValueError("Unknown checkpoint profile")''','''    if args.profile == "trained-geo":
        if args.checkpoint is None:
            raise ValueError("--profile trained-geo requires --checkpoint")
        from .geometry_checkpoint import load_geometry_checkpoint
        return load_geometry_checkpoint, args.checkpoint
    raise ValueError("Unknown checkpoint profile")''')
p.write_text(s)
p=Path("native_t1/accounting.py")
s=p.read_text().replace("the pilot's original","the original").replace("The pilot consumes","The portable trainer consumes")
p.write_text(s)
p=Path("native_t1/tests/test_accounting.py")
s=p.read_text().replace("native_t1.geometry_accounting","native_t1.accounting")
p.write_text(s)
p=Path(".gitignore")
s=p.read_text()
if "\n/experiments/\n" not in s:
    s+='\n# Local research history: not part of the portable Native T1 package.\n/experiments/\n'
p.write_text(s)
