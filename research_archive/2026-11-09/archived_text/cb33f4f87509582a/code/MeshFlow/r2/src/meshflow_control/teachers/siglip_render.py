"""Deterministic CPU orthographic training-mesh rendering, without geometry edits."""
from __future__ import annotations
import hashlib,json
import numpy as np
from PIL import Image

RENDER_CONFIG=dict(version="coal_training_software_raster_v1",size=256,
    azimuth_degrees=[45.,135.,225.,315.],elevation_degrees=20.,
    coordinate_up="Y",projection="orthographic",margin_factor=1.10,
    material_rgb=[175,175,175],background_rgb=[245,245,245],
    light_world=[1.,2.,3.],ambient=.35,diffuse=.65,two_sided=True,
    shading="flat ambient plus absolute-normal Lambert",supersampling=1,
    pixel_sample="center",backface_culling=False,
    normals="per original triangle; no smoothing",geometry_edits=False)
def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def array_hash(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(json.dumps(dict(dtype=value.dtype.str,shape=list(value.shape)),sort_keys=True).encode())
    h.update(value.tobytes());return h.hexdigest()
def basis(azimuth,elevation=20.):
    az,el=np.deg2rad([azimuth,elevation])
    view=np.array([np.cos(el)*np.sin(az),np.sin(el),np.cos(el)*np.cos(az)])
    right=np.cross([0.,1.,0.],view);right/=np.linalg.norm(right)
    up=np.cross(view,right)
    return np.stack((right,up,view))
def camera_from_training(meshes):
    if not meshes:raise ValueError("No actual training targets")
    low=np.full(3,np.inf);high=-low
    for mesh in meshes:
        a=np.asarray(mesh)
        if a.dtype!=np.float32 or a.ndim!=3 or a.shape[1:]!=(3,3) or not np.isfinite(a).all():raise ValueError("Invalid clean training mesh")
        low=np.minimum(low,a.reshape(-1,3).min(0));high=np.maximum(high,a.reshape(-1,3).max(0))
    center=(low+high)/2
    half=0.
    for az in RENDER_CONFIG["azimuth_degrees"]:
        matrix=basis(az)
        for mesh in meshes:
            projected=(mesh.reshape(-1,3).astype(np.float64)-center)@matrix.T
            half=max(half,float(np.abs(projected[:,:2]).max()))
    if not half>0:raise ValueError("Degenerate total training frame")
    return dict(version="train_global_frame_v1",target=center.tolist(),half_extent=half*1.10,
        unpadded_projected_half_extent=half,training_xyz_min=low.tolist(),training_xyz_max=high.tolist(),
        margin_factor=1.10,fit_source="Only complete alpha1 TRAINING target geometry, all four fixed views",
        basis_by_view=[basis(az).tolist() for az in RENDER_CONFIG["azimuth_degrees"]])
def render(mesh,camera,view_index):
    a=np.asarray(mesh)
    if a.dtype!=np.float32 or a.ndim!=3 or a.shape[1:]!=(3,3) or not np.isfinite(a).all():raise ValueError("Expected finite FP32[N,3,3]")
    n=RENDER_CONFIG["size"];half=float(camera["half_extent"])
    matrix=np.asarray(camera["basis_by_view"][view_index]);center=np.asarray(camera["target"])
    projected=(a.astype(np.float64)-center)@matrix.T
    if np.abs(projected[...,:2]).max()>half:raise ValueError("Training frame clips source geometry")
    screen=projected.copy();screen[...,0]=(projected[...,0]/(2*half)+.5)*n
    screen[...,1]=(.5-projected[...,1]/(2*half))*n
    image=np.broadcast_to(np.array(RENDER_CONFIG["background_rgb"],np.uint8),(n,n,3)).copy()
    depth=np.full((n,n),-np.inf);light=np.asarray(RENDER_CONFIG["light_world"]);light/=np.linalg.norm(light)
    drawn=0;projected_degenerate=0
    for original,p in zip(a.astype(np.float64),screen):
        x0,y0=p[0,:2];x1,y1=p[1,:2];x2,y2=p[2,:2]
        den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if den==0:projected_degenerate+=1;continue
        xmin=max(0,int(np.ceil(p[:,0].min()-.5)));xmax=min(n-1,int(np.floor(p[:,0].max()-.5)))
        ymin=max(0,int(np.ceil(p[:,1].min()-.5)));ymax=min(n-1,int(np.floor(p[:,1].max()-.5)))
        if xmin>xmax or ymin>ymax:continue
        yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1];xx=xx+.5;yy=yy+.5
        w0=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den
        w1=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;w2=1-w0-w1
        inside=(w0>=-1e-12)&(w1>=-1e-12)&(w2>=-1e-12)
        zz=w0*p[0,2]+w1*p[1,2]+w2*p[2,2];zview=depth[ymin:ymax+1,xmin:xmax+1]
        take=inside&(zz>zview)
        if not take.any():continue
        normal=np.cross(original[1]-original[0],original[2]-original[0]);norm=np.linalg.norm(normal)
        value=RENDER_CONFIG["ambient"]+RENDER_CONFIG["diffuse"]*(abs(float(normal@light/norm)) if norm>0 else 0.)
        color=np.rint(np.array(RENDER_CONFIG["material_rgb"])*value).clip(0,255).astype(np.uint8)
        zview[take]=zz[take];image[ymin:ymax+1,xmin:xmax+1][take]=color;drawn+=1
    occupied=np.isfinite(depth)
    if not occupied.any():raise ValueError("Blank rendered training image")
    if occupied[[0,-1],:].any() or occupied[:,[0,-1]].any():raise ValueError("Foreground reaches frame edge")
    yy,xx=np.where(occupied)
    audit=dict(input_triangles=len(a),rendered_triangles_with_visible_pixels=drawn,
        projected_zero_area_triangles=projected_degenerate,foreground_pixels=int(occupied.sum()),
        foreground_bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
        finite_frame=True,geometry_sha256=array_hash(a),rgb_sha256=array_hash(image))
    return Image.fromarray(image,"RGB"),audit
