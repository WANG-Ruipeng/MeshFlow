# Unmodified dependency definitions extracted from official PoNQ mesh_tools.py.
# Unused igl/sklearn/skimage dependencies intentionally not imported.
import numpy as np
SIGNS = np.array(
    [
        [(-1) ** i, (-1) ** j, (-1) ** k]
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
)  # +1 or -1 for all coordinates


def tet_circumcenter(verts):
    # ba = b - a
    ba_x = verts[:, 1, 0] - verts[:, 0, 0]
    ba_y = verts[:, 1, 1] - verts[:, 0, 1]
    ba_z = verts[:, 1, 2] - verts[:, 0, 2]
    # ca = c - a
    ca_x = verts[:, 2, 0] - verts[:, 0, 0]
    ca_y = verts[:, 2, 1] - verts[:, 0, 1]
    ca_z = verts[:, 2, 2] - verts[:, 0, 2]
    # da = d - a
    da_x = verts[:, 3, 0] - verts[:, 0, 0]
    da_y = verts[:, 3, 1] - verts[:, 0, 1]
    da_z = verts[:, 3, 2] - verts[:, 0, 2]
    # Squares of lengths of the edges incident to `a'.
    len_ba = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z
    len_ca = ca_x * ca_x + ca_y * ca_y + ca_z * ca_z
    len_da = da_x * da_x + da_y * da_y + da_z * da_z
    # Cross products of these edges.
    # c cross d
    cross_cd_x = ca_y * da_z - da_y * ca_z
    cross_cd_y = ca_z * da_x - da_z * ca_x
    cross_cd_z = ca_x * da_y - da_x * ca_y
    # d cross b
    cross_db_x = da_y * ba_z - ba_y * da_z
    cross_db_y = da_z * ba_x - ba_z * da_x
    cross_db_z = da_x * ba_y - ba_x * da_y
    # b cross c
    cross_bc_x = ba_y * ca_z - ca_y * ba_z
    cross_bc_y = ba_z * ca_x - ca_z * ba_x
    cross_bc_z = ba_x * ca_y - ca_x * ba_y
    # Calculate the denominator of the formula.
    div_den = (ba_x * cross_cd_x + ba_y *
               cross_cd_y + ba_z * cross_cd_z)
    # coplanar vertices
    mask_div_den = np.abs(div_den) == 0
    div_den[mask_div_den] = 1
    denominator = 0.5 / div_den
    # Calculate offset (from `a') of circumcenter.
    circ_x = (len_ba * cross_cd_x + len_ca * cross_db_x +
              len_da * cross_bc_x) * denominator
    circ_y = (len_ba * cross_cd_y + len_ca * cross_db_y +
              len_da * cross_bc_y) * denominator
    circ_z = (len_ba * cross_cd_z + len_ca * cross_db_z +
              len_da * cross_bc_z) * denominator

    out = np.column_stack((circ_x, circ_y, circ_z))+verts[:, 0]
    out[mask_div_den] = verts[mask_div_den].mean(1)
    return out
