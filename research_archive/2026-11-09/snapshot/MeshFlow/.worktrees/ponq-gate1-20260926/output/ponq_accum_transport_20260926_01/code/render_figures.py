from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trimesh
root = Path(__file__).resolve().parents[1]
figdir = root / "figures"
figdir.mkdir(exist_ok=True)
ledger = pd.read_csv(root / "LEDGER.csv")
measured = ledger[ledger.status == "EXTRACTED"].copy()
names = {"sphere":"Sphere", "box":"Box", "thin_u":"Closed thin-wall U", "torus":"Torus"}
states = ["clean128","clean256","position_light","position_strong","normal_light",
          "normal_strong","qem_light","qem_strong","joint_light"]
# Retry only the visualization step after fixing the output-directory permission.
plt.close("all")

def render_object(obj: str) -> None:
    for state in ["reference"] + states:
        path = (
            root / "inputs" / obj / "reference.obj"
            if state == "reference"
            else root / "runs" / f"{obj}__{state}" / "mesh.obj"
        )
        mesh = trimesh.load(path, process=False, force="mesh")
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        fig = plt.figure(figsize=(6.4, 5.4))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f,
                        linewidth=0.15, antialiased=True, shade=True)
        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-0.85, 0.85)
        ax.set_zlim(-0.85, 0.85)
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
        ax.view_init(elev=58 if obj == "thin_u" else 35, azim=-65)
        ax.set_axis_off()
        if state == "reference":
            subtitle = f"Reference triangle mesh | {len(f)} faces"
        else:
            r = measured[(measured.object == obj) & (measured.state == state)].iloc[0]
            subtitle = (
                f"{state} | {len(f)} faces | RMS {r.rms_pct_bboxdiag:.4f}%\n"
                f"Betti ({int(r.beta0)}, {int(r.beta1)}, {int(r.beta2)})"
                f" | nonmanifold vertices {int(r.nonmanifold_vertices)}"
            )
        ax.set_title(f"{names[obj]}\n{subtitle}", fontsize=11)
        fig.tight_layout()
        fig.savefig(figdir / f"{obj}__{state}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

for obj in names:
    render_object(obj)

print(f"已保存 {len(list(figdir.glob('*__*.png')))} 张实际网格视图。")

curves = [
    ("position", [0, 2, 10], ["clean256", "position_light", "position_strong"],
     "Position displacement (% of local spacing)", "Position sensitivity"),
    ("normal", [0, 2, 5], ["clean256", "normal_light", "normal_strong"],
     "Normal rotation (degrees)", "Normal sensitivity"),
    ("qem", [0, 2, 10], ["clean256", "qem_light", "qem_strong"],
     "Full homogeneous QEM factor perturbation (%)", "QEM factor sensitivity"),
]
for tag, levels, conditions, xlabel, title in curves:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for obj, name in names.items():
        values = [
            float(measured[(measured.object == obj) & (measured.state == state)]
                  .rms_pct_bboxdiag.iloc[0])
            for state in conditions
        ]
        ax.plot(levels, values, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Surface RMS (% of reference bbox diagonal)")
    ax.set_title(f"{title}\n256 elements; one fixed perturbation seed per object")
    ax.set_xticks(levels)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / f"curve_{tag}.png", dpi=170)
    if tag == "qem":
        plt.show()
    plt.close(fig)
