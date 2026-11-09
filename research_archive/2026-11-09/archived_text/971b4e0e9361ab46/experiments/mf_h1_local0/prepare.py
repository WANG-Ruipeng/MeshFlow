"""Freeze real Phase 2 parents and diagnostic prompts before any generation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .prompt_dataset import (RealChairStream, select_real_parents, make_prompt,
                             sample_surface_primitive, file_sha256, PROMPT_TYPES, numpy_stream)
from .evaluate import surface_distance_squared


def prepare(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = select_real_parents(data, output/"split_manifest.json")
    stream = RealChairStream(manifest["diagnostic"], seed=20260928)
    # Canonical official target coordinates, without augmentation for the fixed
    # diagnostics. Official normalization/OT are still called without alteration.
    stream.dataset.use_permut_aug = False
    records = []
    fixed_dir = output/"fixed_inputs"
    fixed_dir.mkdir()
    for i, parent in enumerate(manifest["diagnostic"]):
        with numpy_stream(stream.data_rng):
            item = stream.dataset[i]
        tokens = item["coords"].astype(np.float32)
        vertices = stream.dataset.data[i]["vertices"]
        bounds = np.array([vertices.min(0), vertices.max(0)])
        reversible = dict(source_bbox_center=((bounds[0]+bounds[1])/2).tolist(),
                          source_bbox_max_extent=float((bounds[1]-bounds[0]).max()),
                          final_target_multiplier=2/.3762,
                          transform="(source-center)/max_extent * (2/0.3762)")
        diagonal = float(np.linalg.norm(np.ptp(tokens.reshape(-1,3),axis=0)))
        for kind, kind_name in enumerate(PROMPT_TYPES):
            rng = np.random.RandomState(7100+i*10+kind)
            prompt = make_prompt(tokens,kind,rng)
            check_rng = np.random.RandomState(8100+i*10+kind)
            checks = np.stack([sample_surface_primitive(v,int(a),128,check_rng,False)
                               for v,a in zip(prompt["vertices"],prompt["arity"])])
            distance = surface_distance_squared(checks.reshape(-1,3),tokens.reshape(-1,3,3))
            max_distance = float(np.sqrt(distance.max()))
            if max_distance > 1e-5:
                raise ValueError("Prompt coordinate-frame/surface check failed")
            path = fixed_dir/f"parent{i}_{kind_name}.npz"
            np.savez_compressed(path,target=tokens,check_points=checks,**prompt)
            records.append(dict(parent_index=i,parent_uid=parent["uid"],kind=kind_name,
                path=str(path.resolve()),sha256=file_sha256(path),bbox_diagonal=diagonal,
                transform=reversible,prompt_surface_max_distance=max_distance,
                prompt_surface_threshold=1e-5, check_seed=8100+i*10+kind,context_seed=7100+i*10+kind))
    prompt_manifest = dict(status="PASS",records=records,diagnostic_noise_seeds=[4101,4102],
        source="four fixed adapter-unseen author-train parents",pretraining_unseen_verified=False,
        correct_and_mismatch="cyclic next parent for the same prompt type",phase3="NOT_AUTHORIZED")
    (output/"prompt_manifest.json").write_text(json.dumps(prompt_manifest,indent=2),encoding="utf-8")
    print(json.dumps(dict(status="PASS",train_parents=len(manifest["train"]),
                         diagnostic_parents=len(manifest["diagnostic"]),prompts=len(records),output=str(output))),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    prepare(args.data,args.output)
