from pathlib import Path
import hashlib
root=Path("/mnt/e/MeshFlow")
p=root/"native_t1/__main__.py"
t=p.read_text(encoding="utf-8-sig")
t=t.replace('import argparse\n','import argparse\nimport sys\n')
t=t.replace('choices=("a-continue", "original-t1")','choices=("a-continue", "original-t1", "trained")')
t=t.replace('help="Optional relocated copy of the selected profile\'s pinned checkpoint"',
            'help="Required for trained; otherwise a relocated copy of the pinned profile checkpoint"')
t=t.replace('    return parser\n','    commands.add_parser("prepare", add_help=False, help="Prepare the registered task from external official data")\n'
            '    commands.add_parser("train", add_help=False, help="Explicit pure-FM training or a zero-update preflight")\n    return parser\n',1)
t=t.replace('    raise ValueError("Unknown checkpoint profile")',
            '    if args.profile == "trained":\n'
            '        if args.checkpoint is None:\n'
            '            raise ValueError("--profile trained requires --checkpoint")\n'
            '        from .portable_checkpoint import load_trained_checkpoint\n'
            '        return load_trained_checkpoint, args.checkpoint\n'
            '    raise ValueError("Unknown checkpoint profile")')
t=t.replace('def main(argv=None):\n    args = build_parser().parse_args(argv)',
            'def main(argv=None):\n'
            '    argv = list(sys.argv[1:] if argv is None else argv)\n'
            '    if argv and argv[0] == "prepare":\n'
            '        from .prepare import main as prepare_main\n'
            '        return prepare_main(argv[1:])\n'
            '    if argv and argv[0] == "train":\n'
            '        from .train_cli import main as train_main\n'
            '        return train_main(argv[1:])\n'
            '    args = build_parser().parse_args(argv)')
p.write_text(t,encoding="utf-8")
p=root/"native_t1/train_cli.py"
t=p.read_text(encoding="utf-8")
t=t.replace('    payload = dict(schema=SCHEMA, model=model.state_dict(), optimizer=optimizer.state_dict(),\n'
            '                   config_sha256=file_sha256(config_path), completed_updates=completed,',
            '    from .portable_checkpoint import NATIVE_META, config_sha256\n'
            '    payload = dict(schema=SCHEMA, native_meta=dict(NATIVE_META), model=model.state_dict(), optimizer=optimizer.state_dict(),\n'
            '                   config_sha256=config_sha256(config_path), completed_updates=completed,')
p.write_text(t,encoding="utf-8")
p=root/"native_t1/training.py"
t=p.read_text(encoding="utf-8-sig").replace("never called by sampler/CLI.","called only by an explicit trainer.")
p.write_text(t,encoding="utf-8")
p=root/"native_t1/.gitignore"
p.write_text("__pycache__/\n*.py[cod]\nruns/\nvalidation/\ndata/\ncheckpoints/\n",encoding="utf-8")
p=root/".gitignore"
t=p.read_text(encoding="utf-8")
t += ("\n# External ML assets and local audit products never belong in code commits.\n"
      "*.pt\n*.pt.partial\n*.pth\n*.ckpt\n*.safetensors\n*.npz\n*.npy\n*.py[cod]\n*.tar.gz\n"
      "/WORKSPACE.md\n")
p.write_text(t,encoding="utf-8")
p=root/"README.md"
t=p.read_text(encoding="utf-8")
t = ('> **Native T1 extension:** code-only conditional completion and training from external official assets: '
     '[native_t1/README.md](native_t1/README.md). No weights or datasets are bundled.\n\n')+t
p.write_text(t,encoding="utf-8")
config=root/"configs/snet/base-120m-ot-v-chair.yaml"
print("canonical config SHA256:",hashlib.sha256(config.read_text(encoding="utf-8").encode()).hexdigest())
