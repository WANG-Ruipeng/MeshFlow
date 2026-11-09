"""Immutable triangle-soup VTK renderer for paper panels (one GL worker).

No mesh filters, welding, normal smoothing, model execution or mesh repair.
Camera fitting operates in a rotated projection basis without moving geometry.
"""
from __future__ import annotations

import atexit
from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import threading
import time

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

VERSION = 'chair_paper_vtk_triangle_soup_v1'
PALETTE = {'context': '#287BB5', 'free': '#E79535', 'gt': '#A9AFB7', 'background': '#FFFFFF'}
STYLE = dict(projection='parallel', shading='flat_per_triangle', double_sided=True,
             edges=False, ambient=.36, diffuse=.64, specular=0., multisamples=4,
             camera_margin=1.10, opacity=1., background='white', up_axis='Y')
_SESSION = None


def _triangles(value):
    if value is None:
        return np.empty((0, 3, 3), dtype=np.float32)
    a = np.asarray(value)
    if a.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise TypeError('Render coordinates must be the original FP32 or FP64 values')
    if a.ndim == 2 and a.shape[1:] == (9,):
        a = a.reshape(-1, 3, 3)
    if a.ndim != 3 or a.shape[1:] != (3, 3):
        raise ValueError('Expected triangles [N,3,3] or [N,9]')
    if not np.isfinite(a).all():
        raise ValueError('Rendering does not discard nonfinite geometry')
    return a


def _hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((str(a.dtype), a.shape)).encode()+a.tobytes()).hexdigest()


def _rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16)/255. for i in (1, 3, 5))


def frame_for(arrays, azimuth=45, elevation=22, aspect=1):
    """Fit one orthographic camera to a whole row's C/F/GT union.

    Azimuth zero looks from +Z; positive azimuth moves toward +X. Elevation
    moves toward +Y. No coordinate transform is applied to any mesh.
    Pass every row method plus context and GT to this function exactly once.
    """
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    elif isinstance(arrays, Mapping):
        arrays = list(arrays.values())
    else:
        arrays = list(arrays)
    aspect = float(aspect)
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError('Positive finite camera aspect required')
    azimuth, elevation = float(azimuth), float(elevation)
    if not math.isfinite(azimuth) or not math.isfinite(elevation) or abs(elevation) >= 89.9:
        raise ValueError('Finite view angles, elevation strictly between -89.9 and 89.9')
    tri = [_triangles(x) for x in arrays]
    nonempty = [x.reshape(-1, 3) for x in tri if len(x)]
    if not nonempty:
        raise ValueError('At least one nonempty geometry is needed to fit a camera')
    points = np.concatenate(nonempty).astype(np.float64, copy=False)
    az, el = np.radians([azimuth, elevation])
    eye_direction = np.array([np.cos(el)*np.sin(az), np.sin(el), np.cos(el)*np.cos(az)], np.float64)
    eye_direction /= np.linalg.norm(eye_direction)
    right = np.cross(np.array([0., 1., 0.]), eye_direction)
    right /= np.linalg.norm(right)
    up = np.cross(eye_direction, right)
    basis = np.stack([right, up, eye_direction])
    projected = points @ basis.T
    low, high = projected.min(0), projected.max(0)
    span = high-low
    bbox_low, bbox_high = points.min(0), points.max(0)
    diagonal = float(np.linalg.norm(bbox_high-bbox_low))
    if diagonal <= 0:
        raise ValueError('Zero-extent geometry cannot define a publication frame')
    center = ((low+high)*.5) @ basis
    half_height = 1.10*max(float(span[1])*.5, float(span[0])/(2*aspect), diagonal*1e-8)
    distance = max(3*diagonal, 3*half_height)
    depth_pad = max(.10*diagonal, 1e-9)
    clipping = [max(distance-float(span[2])*.5-depth_pad, diagonal*1e-8), distance+float(span[2])*.5+depth_pad]
    uv = np.column_stack([.5+(projected[:, 0]-(low[0]+high[0])*.5)/(2*half_height*aspect),
                          .5+(projected[:, 1]-(low[1]+high[1])*.5)/(2*half_height)])
    bounds = [uv.min(0).tolist(), uv.max(0).tolist()]
    if not (np.all(uv >= 0) and np.all(uv <= 1)):
        raise AssertionError('Frame fitting clipped geometry')
    return dict(version=VERSION, projection='parallel', aspect=aspect,
        azimuth_degrees=azimuth, elevation_degrees=elevation, margin=1.10,
        position=(center+eye_direction*distance).tolist(), focal_point=center.tolist(),
        view_up=up.tolist(), reference_up=[0., 1., 0.],
        right=right.tolist(), eye_direction=eye_direction.tolist(),
        parallel_scale=half_height, clipping_range=clipping,
        union_bbox=[bbox_low.tolist(), bbox_high.tolist()],
        projection_bounds=[low.tolist(), high.tolist()],
        screen_bounds_normalized=bounds, all_union_vertices_inside=True,
        union_face_count=sum(len(t) for t in tri), union_vertex_count=len(points),
        fitted_input_geometry_sha256=[_hash(t) for t in tri],
        geometry_coordinates_transformed=False)


class _RenderSession:
    def __init__(self):
        self.thread_id = threading.get_ident()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*_rgb(PALETTE['background']))
        self.renderer.SetBackgroundAlpha(1.)
        self.renderer.TwoSidedLightingOn()
        self.renderer.AutomaticLightCreationOff()
        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetWindowName('Immutable chair publication panel')
        self.window.SetMultiSamples(STYLE['multisamples'])
        self.window.SetAlphaBitPlanes(0)
        self.window.AddRenderer(self.renderer)
        # Camera-space lights track the common camera; no shadows/floor geometry.
        for position, intensity in [((-0.35, 0.65, 1.), .85), ((.8, .2, .6), .35)]:
            light = vtk.vtkLight()
            light.SetLightTypeToCameraLight()
            light.SetPosition(*position)
            light.SetFocalPoint(0., 0., 0.)
            light.SetIntensity(intensity)
            self.renderer.AddLight(light)
        self.capabilities = None

    def check_thread(self):
        if threading.get_ident() != self.thread_id:
            raise RuntimeError('VTK context must remain on its single owning GL worker thread')

    def close(self):
        if self.window is not None:
            self.window.Finalize()
            self.window = None


def close_renderer():
    """Explicitly release the sole offscreen GL context after the batch."""
    global _SESSION
    if _SESSION is not None:
        _SESSION.check_thread()
        _SESSION.close()
        _SESSION = None


atexit.register(close_renderer)


def _actor(triangles, hex_color):
    vertices = np.ascontiguousarray(triangles.reshape(-1, 3))
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    offsets = np.arange(0, len(vertices)+1, 3, dtype=np.int64)
    indices = np.arange(len(vertices), dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(indices, deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*_rgb(hex_color))
    prop.SetOpacity(1.)
    prop.SetInterpolationToFlat()
    prop.SetAmbient(STYLE['ambient'])
    prop.SetDiffuse(STYLE['diffuse'])
    prop.SetSpecular(0.)
    prop.BackfaceCullingOff()
    prop.FrontfaceCullingOff()
    prop.EdgeVisibilityOff()
    returned = vtk_to_numpy(poly.GetPoints().GetData()).reshape(triangles.shape)
    returned_indices = vtk_to_numpy(poly.GetPolys().GetConnectivityArray())
    if returned.dtype != triangles.dtype or returned.tobytes() != np.ascontiguousarray(triangles).tobytes():
        raise AssertionError('VTK point storage changed geometry')
    if not np.array_equal(returned_indices, indices):
        raise AssertionError('VTK triangle-soup indices changed')
    return actor, dict(face_count=len(triangles), vertex_count=len(vertices),
        input_geometry_sha256=_hash(triangles), vtk_geometry_sha256=_hash(returned),
        indices_sha256=_hash(returned_indices), triangles_unchanged=True,
        triangle_soup_3N_vertices=True, color=hex_color,
        point_normals_present=poly.GetPointData().GetNormals() is not None,
        cell_normals_present=poly.GetCellData().GetNormals() is not None,
        mesh_filters_used=[], actor_transform='identity')


def render_panel(C, F, path, camera, size=(900, 900), mode='output', GT=None):
    """Render one panel with the caller's shared row camera and immutable inputs.

    output: blue C and orange F; context: blue C; gt: neutral gray complete GT.
    Returns a JSON-safe receipt. Does not write labels, alter meshes or overwrite
    an existing image. Call close_renderer() at the end of the single-worker batch.
    """
    global _SESSION
    started = time.perf_counter()
    inputs = dict(C=_triangles(C), F=_triangles(F), GT=_triangles(GT))
    input_hashes = {k:_hash(v) for k, v in inputs.items()}
    if mode not in ('output', 'context', 'gt'):
        raise ValueError('mode must be output, context or gt')
    if mode == 'gt' and GT is None:
        raise ValueError('The complete GT array must be explicit for a gt panel')
    selected = [('context', inputs['C']), ('free', inputs['F'])] if mode=='output' else [
        ('context', inputs['C'])] if mode=='context' else [('gt', inputs['GT'])]
    selected = [(label, t) for label, t in selected if len(t)]
    if not selected:
        raise ValueError('Selected panel has no geometry')
    width, height = map(int, size)
    if min(width, height) <= 0 or (width, height) != tuple(size):
        raise ValueError('Positive integer raster dimensions required')
    if not math.isclose(width/height, float(camera['aspect']), rel_tol=0., abs_tol=1e-12):
        raise ValueError('Panel aspect must equal the shared camera aspect')
    if camera.get('projection') != 'parallel':
        raise ValueError('Only the registered parallel camera is supported')
    path = Path(path)
    if path.exists():
        raise FileExistsError('Render output already exists; do not silently replace a panel')
    path.parent.mkdir(parents=True, exist_ok=True)
    if _SESSION is None:
        _SESSION = _RenderSession()
    session = _SESSION
    session.check_thread()
    session.renderer.RemoveAllViewProps()
    session.window.SetSize(width, height)
    cam = session.renderer.GetActiveCamera()
    cam.ParallelProjectionOn()
    cam.SetPosition(*camera['position'])
    cam.SetFocalPoint(*camera['focal_point'])
    cam.SetViewUp(*camera['view_up'])
    cam.SetParallelScale(float(camera['parallel_scale']))
    cam.SetClippingRange(*camera['clipping_range'])
    actors = []
    geometry = {}
    for label, triangles in selected:
        actor, receipt = _actor(triangles, PALETTE[label])
        actors.append(actor)
        session.renderer.AddActor(actor)
        geometry[label] = receipt
    prepared = time.perf_counter()
    session.window.Render()
    rendered = time.perf_counter()
    if session.capabilities is None:
        text = session.window.ReportCapabilities()
        session.capabilities = [line.strip() for line in text.splitlines() if any(
            token in line.lower() for token in ('vendor string', 'renderer string', 'version string'))]
    # Verify the actual VTK camera, not just our analytic fit, clips no vertex.
    screen = []
    for _, triangles in selected:
        for point in triangles.reshape(-1, 3):
            session.renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.)
            session.renderer.WorldToDisplay()
            screen.append(session.renderer.GetDisplayPoint())
    screen = np.asarray(screen, np.float64)
    low, high = screen.min(0), screen.max(0)
    inside = bool(np.all(low >= [-1e-6, -1e-6, -1e-9]) and np.all(high <= [width+1e-6, height+1e-6, 1.+1e-9]))
    if not inside:
        session.renderer.RemoveAllViewProps()
        raise AssertionError('At least one actual geometry vertex lies outside the shared VTK frame/depth range')
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(session.window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    capture.ShouldRerenderOff()
    capture.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.SetCompressionLevel(6)
    writer.Write()
    if not path.exists() or path.stat().st_size == 0:
        raise IOError('VTK did not produce the requested PNG')
    pixels = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
    if pixels.shape != (width*height, 3):
        raise AssertionError('Unexpected RGB capture shape')
    if not np.any(pixels != 255):
        raise AssertionError('Blank white render is not an acceptable panel')
    after_hashes = {k:_hash(v) for k, v in inputs.items()}
    if input_hashes != after_hashes:
        raise AssertionError('Renderer mutated an input array')
    finished = time.perf_counter()
    receipt = dict(version=VERSION, mode=mode, path=str(path), size=[width, height],
        png_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), png_bytes=path.stat().st_size,
        pixel_sha256=_hash(pixels), camera=dict(camera), palette=dict(PALETTE), style=dict(STYLE),
        input_geometry_sha256=input_hashes, input_geometry_sha256_after=after_hashes,
        input_arrays_unchanged=True, geometry=geometry,
        rendered_face_count=sum(len(t) for _, t in selected), rendered_vertex_count=sum(3*len(t) for _, t in selected),
        screen_bounds_pixels=[low[:2].tolist(), high[:2].tolist()],
        screen_bounds_normalized=[(low[:2]/[width, height]).tolist(), (high[:2]/[width, height]).tolist()],
        depth_bounds=[float(low[2]), float(high[2])], all_vertices_inside=inside,
        depth_buffer_occlusion=True, vtk_version=vtk.vtkVersion.GetVTKVersion(),
        render_window_class=session.window.GetClassName(), gl_capabilities=session.capabilities,
        worker_threads=1, context_reused=True,
        seconds=dict(prepare=prepared-started, render=rendered-prepared,
                     capture_validation_write=finished-rendered, total=finished-started),
        model_forwards=0, model_backwards=0, mesh_processing_calls=0,
        mesh_repairs=0, welding_calls=0, normal_smoothing_calls=0)
    session.renderer.RemoveAllViewProps()
    return receipt
