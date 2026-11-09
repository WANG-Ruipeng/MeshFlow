import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d


MESH_EXTENSIONS = {".obj", ".off", ".ply", ".stl"}


def find_mesh_files(mesh_dir: Path, max_meshes: Optional[int] = None) -> List[Path]:
    """Find supported mesh files recursively in a deterministic order."""
    mesh_dir = Path(mesh_dir).expanduser().resolve()
    if not mesh_dir.is_dir():
        raise NotADirectoryError(f"Mesh directory does not exist: {mesh_dir}")

    mesh_files = sorted(
        path for path in mesh_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MESH_EXTENSIONS
    )
    if max_meshes is not None:
        if max_meshes <= 0:
            raise ValueError("max_meshes must be greater than zero")
        mesh_files = mesh_files[:max_meshes]

    if not mesh_files:
        extensions = ", ".join(sorted(MESH_EXTENSIONS))
        raise FileNotFoundError(
            f"No supported mesh files found in {mesh_dir} (expected: {extensions})"
        )
    return mesh_files


def evaluate_mesh(mesh_path: Path) -> Dict[str, float]:
    """Compute topology and self-intersection metrics for one triangle mesh."""
    mesh_path = Path(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    num_faces = len(mesh.triangles)
    num_vertices = len(mesh.vertices)
    if num_faces == 0 or num_vertices == 0:
        raise ValueError("mesh has no vertices or triangle faces")

    intersecting_pairs = np.asarray(mesh.get_self_intersecting_triangles())
    num_intersecting_pairs = len(intersecting_pairs)
    if num_intersecting_pairs:
        num_intersecting_faces = len(np.unique(intersecting_pairs))
    else:
        num_intersecting_faces = 0

    possible_pairs = num_faces * (num_faces - 1) / 2
    return {
        "num_vertices": float(num_vertices),
        "num_faces": float(num_faces),
        "intersecting_face_ratio": num_intersecting_faces / num_faces,
        "intersecting_pair_ratio": (
            num_intersecting_pairs / possible_pairs if possible_pairs else 0.0
        ),
        "vertex_face_corner_ratio": num_vertices / (num_faces * 3),
    }


def evaluate_mesh_dir(
    mesh_dir: Path,
    max_meshes: Optional[int] = None,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Evaluate every supported mesh under mesh_dir and return means plus records."""
    mesh_dir = Path(mesh_dir).expanduser().resolve()
    mesh_files = find_mesh_files(mesh_dir, max_meshes=max_meshes)
    records: List[Dict[str, object]] = []
    failures: List[str] = []

    for mesh_path in mesh_files:
        try:
            metrics = evaluate_mesh(mesh_path)
        except (RuntimeError, ValueError) as error:
            failures.append(f"{mesh_path}: {error}")
            continue
        records.append({"file": str(mesh_path.relative_to(mesh_dir)), **metrics})

    if not records:
        details = "\n".join(failures[:10])
        raise RuntimeError(f"Failed to evaluate every mesh in {mesh_dir}:\n{details}")

    metric_names = tuple(key for key in records[0] if key != "file")
    summary = {
        "num_meshes": float(len(records)),
        "num_failed": float(len(failures)),
        **{
            f"mean_{name}": float(np.mean([record[name] for record in records]))
            for name in metric_names
        },
    }
    for failure in failures:
        print(f"Warning: {failure}")
    return summary, records


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate topology and self-intersection metrics for a mesh directory."
    )
    parser.add_argument("mesh_dir", type=Path, help="Directory searched recursively for meshes")
    parser.add_argument("--max-meshes", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for summary and per-mesh metrics",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary, records = evaluate_mesh_dir(args.mesh_dir, max_meshes=args.max_meshes)
    print(json.dumps(summary, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "meshes": records}, indent=2),
            encoding="utf-8",
        )
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()