from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

root = Path('/mnt/e/MeshFlow/output/wsl5080-overfit1000')
items = [
    ('Ground truth', root/'current_weights_preview/ground_truth.obj'),
    ('Current weights: 1000 steps', root/'current_weights_preview/sample.obj'),
    ('EMA: 1000 steps', root/'infer_00001000/000000.obj'),
]
meshes = [trimesh.load(path, force='mesh', process=False) for _, path in items]
limit = max(float(np.abs(mesh.vertices).max()) for mesh in meshes) * 1.05
fig = plt.figure(figsize=(13, 4.6), facecolor='white')
for i, ((title, _), mesh) in enumerate(zip(items, meshes), 1):
    ax = fig.add_subplot(1, 3, i, projection='3d')
    triangles = np.asarray(mesh.triangles)
    # Plot x,z,y to keep the dataset's y axis vertical.
    triangles = triangles[:, :, [0, 2, 1]]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.3, -0.5, 0.8]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65*np.abs(normals@light)
    colors = np.c_[np.array([0.20, 0.55, 0.80])[None, :]*shade[:, None], np.ones(len(shade))]
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors=(0.05, 0.1, 0.15, 0.25), linewidths=0.15))
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-55)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
fig.suptitle('MeshFlow / RTX 5080 / single-shape overfit check', fontsize=15)
fig.tight_layout()
path = root/'mesh_comparison.png'
fig.savefig(path, dpi=140)
fig.savefig(path.with_suffix('.jpg'), dpi=100, pil_kwargs={'quality': 85})
print(path)