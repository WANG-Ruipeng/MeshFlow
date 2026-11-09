"""CPU-only six-stage gallery for twelve frozen student-denoising requests.
No model, score, GT, smoothing, geometry normalization or hidden-face deletion.
The geometric reader/shader/draw functions preserve the previous renderer.
"""
from __future__ import annotations
import argparse,csv,hashlib,html,io,json,os,re,subprocess,time
from pathlib import Path
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_student_multistep_denoise_v1'
GALLERY=OUT/'gallery'
OLD=ROOT/'experiments/r2_multistep_denoise_v1'
ORIGIN=ROOT/'r2/tools/multistep_denoise_v1/gallery.py'
SCHEMA='r2_student_multistep_denoise_gallery_v1'
STAGES=('RAW','IDENTITY_GEO','D1','S1','S3','S5')
TITLES={'RAW':'RAW: no postprocessing','IDENTITY_GEO':'IDENTITY: GEO only',
 'D1':'Official: 1 pass + GEO','S1':'Student: 1 pass + GEO',
 'S3':'Student: 3 passes + GEO','S5':'Student: 5 passes + GEO'}
CAMERAS=((22.0,-60.0),(22.0,120.0))
COLORS={'C':'#2387c9','F':'#b8bec4','edges':'#38424b'}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def path_value(value):
    value=str(value)
    if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',value):
        value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    return Path(value)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def checked(path):
    lexical=Path(path).absolute()
    for part in (lexical,*lexical.parents):
        if part.is_symlink():raise ValueError('SYMLINK_OUTPUT: '+str(part))
    result=lexical.resolve();target=GALLERY.resolve()
    if result==target or not result.is_relative_to(target):raise ValueError('WRITE_OUTSIDE_GALLERY: '+str(result))
    return result

def atomic(path,writer,binary=False):
    p=checked(path);p.parent.mkdir(parents=True,exist_ok=True);temporary=checked(p.with_name(p.name+'.tmp'))
    with temporary.open('wb' if binary else 'w',**({} if binary else {'encoding':'utf-8','newline':''})) as f:
        writer(f);f.flush();os.fsync(f.fileno())
    temporary.replace(p)

def json_write(path,obj):atomic(path,lambda f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False))
def text_write(path,text):atomic(path,lambda f:f.write(text))
def save_npz(path,arrays):atomic(path,lambda f:np.savez_compressed(f,**arrays),True)
def csv_write(path,rows):
    def writer(f):
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic(path,writer)

def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def triangles(array, name):
    a = np.asarray(array)
    if a.ndim == 2 and a.shape[1:] == (9,):
        a = a.reshape(-1, 3, 3)
    if a.ndim != 3 or a.shape[1:] != (3, 3) or a.dtype.kind != "f" or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite float triangle array")
    return a.copy()


def load_mesh(path, k, stage):
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if stage == "RAW":
            if "raw" not in keys:
                raise ValueError(f"{path}: missing RAW array")
            mesh = triangles(data["raw"], "raw")
        elif {"vertices", "faces"} <= keys:
            vertices = np.asarray(data["vertices"])
            faces = np.asarray(data["faces"])
            if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
                raise ValueError(f"{path}: invalid indexed mesh shape")
            if not np.issubdtype(faces.dtype, np.integer) or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices))):
                raise ValueError(f"{path}: invalid vertex indices")
            mesh = triangles(vertices[faces], "indexed POST")
            if "output" in keys and not same_bits(mesh, triangles(data["output"], "output")):
                raise ValueError(f"{path}: indexed POST differs from output triangles")
        elif "output" in keys:
            mesh = triangles(data["output"], "output")
        elif {"C", "free"} <= keys:
            mesh = np.concatenate([triangles(data["C"], "C"), triangles(data["free"], "free")])
        else:
            raise ValueError(f"{path}: unsupported explicit POST schema: {sorted(keys)}")
        if k <= 0 or k > len(mesh):
            raise ValueError(f"{path}: invalid K")
        if "C" in keys and not same_bits(mesh[:k], triangles(data["C"], "C")):
            raise ValueError(f"{path}: C identity mismatch")
        if "known_faces" in keys and int(data["known_faces"]) != k:
            raise ValueError(f"{path}: K differs from stored known_faces")
        if "known_face_indices" in keys and not np.array_equal(np.asarray(data["known_face_indices"]), np.arange(k)):
            raise ValueError(f"{path}: C not in declared prefix order")
    return mesh


def shaded(tri, color):
    tri = np.asarray(tri, dtype=float)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normal, axis=1)
    normal = np.divide(normal, norms[:, None], out=np.zeros_like(normal), where=norms[:, None] > 0)
    light = np.asarray([.3, .7, .6]); light /= np.linalg.norm(light)
    factor = .70 + .30 * np.abs(normal @ light)
    return np.clip(np.asarray(to_rgb(color))[None, :] * factor[:, None], 0, 1)


def draw(ax, mesh, k, bounds, camera):
    color = shaded(mesh, COLORS["F"])
    color[:k] = shaded(mesh[:k], COLORS["C"])
    ax.add_collection3d(Poly3DCollection(mesh[:, :, [0, 2, 1]].astype(float), facecolors=color,
                                        edgecolors=COLORS["edges"], linewidths=.35, alpha=1., zsort="average"))
    center = np.asarray(bounds["center_model_xyz"])[[0, 2, 1]]
    half = bounds["span"] / 2
    ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half), zlim=(center[2]-half, center[2]+half))
    ax.set_proj_type("ortho"); ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=camera[0], azim=camera[1]); ax.set_axis_off()




def frozen_inputs():
    registration=OUT/'registration.json';index=OUT/'post_index.csv'
    if not registration.is_file() or not index.is_file():raise RuntimeError('BLOCKED_MISSING_REGISTERED_POST_INDEX')
    reg=read(registration);original=read(OLD/'registration.json')
    records=sorted(reg['records'],key=lambda r:r['request_id'])
    if len(records)!=12 or any(r['label']!='R2_USER_MIX' for r in records):raise ValueError('Expected12USERMIX')
    if {r['request_id'] for r in records}!={r['request_id'] for r in original['records']}:raise ValueError('Request panel differs from frozen original12')
    with index.open(encoding='utf-8-sig',newline='') as f:post=list(csv.DictReader(f))
    if len(post)!=72:raise ValueError('Expected72explicitRAWandPOSTpaths')
    groups={}
    for record in records:
        rid=record['request_id']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',rid):raise ValueError('Unsafe request ID')
        group=[r for r in post if r['request_id']==rid]
        if len(group)!=6 or {r['stage'] for r in group}!=set(STAGES):raise ValueError('Missing/duplicate stage '+rid)
        paths={};hashes={}
        for row in group:
            stage=row['stage']
            if row.get('label','R2_USER_MIX')!='R2_USER_MIX':raise ValueError('Wrong generator')
            if row.get('parent_id',record['parent_id'])!=record['parent_id']:raise ValueError('Wrong parent')
            value=row.get('mesh_path') or row.get('post_path') or (row.get('raw_path') if stage=='RAW' else None)
            expected=row.get('mesh_sha256') or row.get('post_sha256') or (row.get('raw_sha256') if stage=='RAW' else None)
            if not value or not expected or not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('Missing path/hash '+rid+'/'+stage)
            p=path_value(value)
            if sha256(p)!=expected:raise ValueError('Input geometry hash changed '+str(p))
            paths[stage]=p;hashes[str(p)]=expected
        if paths['RAW'].resolve()!=path_value(record['raw_path']).resolve():raise ValueError('RAW identity changed')
        if 'D1_post_path' in record and paths['D1'].resolve()!=path_value(record['D1_post_path']).resolve():raise ValueError('D1 identity changed')
        if 'S1_post_path' in record and paths['S1'].resolve()!=path_value(record['S1_post_path']).resolve():raise ValueError('S1 identity changed')
        groups[rid]=dict(record=record,paths=paths,source_hashes=hashes)
    return records,groups,{str(registration):sha256(registration),str(index):sha256(index)}


def render_request(record,group,definition_hashes):
    rid=record['request_id'];k=int(record['K']);paths=group['paths']
    loaded={s:load_mesh(p,k,'RAW' if s=='RAW' else 'POST') for s,p in paths.items()}
    if not all(same_bits(m[:k],loaded['RAW'][:k]) for m in loaded.values()):raise ValueError('C changed '+rid)
    coordinates=np.concatenate([m.reshape(-1,3) for m in loaded.values()]).astype(float)
    lo,hi=coordinates.min(0),coordinates.max(0);span=float(np.max(hi-lo))*1.12
    if span<=0:raise ValueError('Zero extent')
    bounds=dict(bounds_model_xyz=[lo.tolist(),hi.tolist()],center_model_xyz=((lo+hi)/2).tolist(),span=span,
        coordinate_rescaling=False,same_bounds_all_six_panels=True,projection='orthographic',axis_transform='model[x,y,z] -> plot[x,z,y]; modelYvertical')
    arrays_path=GALLERY/'view_arrays'/f'{rid}.npz'
    arrays={**loaded,'C':loaded['RAW'][:k], 'K':np.asarray(k,dtype=np.int64)}
    if arrays_path.exists():
        with np.load(arrays_path,allow_pickle=False) as existing:
            if set(existing.files)!=set(arrays) or any(not same_bits(existing[x],arrays[x]) for x in arrays):raise ValueError('Saved rendering arrays differ')
    else:save_npz(arrays_path,arrays)
    arrays_hash=sha256(arrays_path);receipts=[]
    for view,camera in enumerate(CAMERAS):
        png=GALLERY/f'{rid}__v{view}.png';jpg=GALLERY/'review_jpg'/f'{rid}__v{view}.jpg';sidecar=png.with_suffix('.png.json')
        identity=dict(schema=SCHEMA,request_id=rid,parent_id=record['parent_id'],condition_id=record.get('condition_id'),region_type=record['region_type'],
            label='R2_USER_MIX',stages=list(STAGES),K=k,N_by_stage={s:len(loaded[s]) for s in STAGES},camera_elev_azim=list(camera),view_id=view,
            bounds=bounds,source_hashes=group['source_hashes'],definition_hashes=definition_hashes,source_code_sha256=sha256(__file__),
            renderer_origin=str(ORIGIN),renderer_origin_sha256=sha256(ORIGIN),render_arrays_path=str(arrays_path),render_arrays_sha256=arrays_hash,
            matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,colors=COLORS,pixels=[2400,1600],
            hidden_triangles_removed=False,smoothing=False,GT_read=False,model_calls=0,GPU=0,C_bitwise_unchanged_all_stages=True,topology_claim_from_picture=False)
        if any(p.exists() for p in (png,jpg,sidecar)):
            if not all(p.is_file() for p in (png,jpg,sidecar)):raise RuntimeError('Incomplete render requires explicit recovery: '+str(png))
            prior=read(sidecar)
            if prior['identity']!=identity or sha256(png)!=prior['sha256'] or sha256(jpg)!=prior['jpg_sha256']:raise RuntimeError('Cached render identity differs')
            receipts.append(prior);continue
        fig=plt.figure(figsize=(24,16),dpi=100,facecolor='white')
        for i,s in enumerate(STAGES):
            ax=fig.add_subplot(2,3,i+1,projection='3d');draw(ax,loaded[s],k,bounds,camera)
            ax.set_title(f'{s}: {TITLES[s]}\nN={len(loaded[s])} | C={k}, F={len(loaded[s])-k}',fontsize=14,pad=0)
        fig.suptitle(f"R2_USER_MIX | {record['parent_id']} | {record['region_type']} {record.get('tier','')} | {rid.rsplit('__',1)[-1]}\nFixed six-path comparison | view {view+1}: elev {camera[0]:g}, azim {camera[1]:g}",fontsize=16,y=.99)
        fig.text(.5,.025,'BLUE = unchanged C | GRAY = generated free | Same camera, bounds and material across all six panels',ha='center',fontsize=12)
        fig.text(.5,.009,'C restored after every learned pass; one GEO cleanup at the selected endpoint. No GT, smoothing or hidden-face deletion. Pictures do not certify attachment.',ha='center',fontsize=10)
        fig.subplots_adjust(left=.003,right=.997,top=.93,bottom=.045,hspace=.015,wspace=0)
        atomic(png,lambda f:fig.savefig(f,format='png',dpi=100,facecolor='white'),True);plt.close(fig)
        with Image.open(png) as im:
            compact=im.convert('RGB');compact.thumbnail((1800,1200),Image.Resampling.LANCZOS)
            atomic(jpg,lambda f:compact.save(f,format='JPEG',quality=84,optimize=True,subsampling=0),True)
        if any(sha256(p)!=h for p,h in group['source_hashes'].items()):raise RuntimeError('Source changed during render')
        receipt=dict(status='COMPLETE',identity=identity,image=str(png),png=str(png),sha256=sha256(png),jpg=str(jpg),jpg_sha256=sha256(jpg),visual_review='PENDING_AI_PROXY')
        json_write(sidecar,receipt);receipts.append(receipt)
    return receipts


def make_html(records,groups,portable=False):
    intro=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Student multistep denoising</title>',
      '<style>body{font:15px Arial;margin:24px;max-width:2450px}img{width:100%;height:auto}section{margin:36px 0}a{color:#176299}</style>',
      '<h1>R2_USER_MIX: RAW / IDENTITY / official D1 / student S1, S3, S5</h1>',
      '<p>All twelve fixed requests, two opposite cameras each. Six panels share camera and union bounds; blue unchanged C, gray free. Geometry is rendered directly without smoothing or hidden-face deletion. No source-parent GT. Pictures do not certify connectivity.</p>']
    for r in records:
        rid=r['request_id'];intro.append(f'<section><h2>{html.escape(rid)}</h2>')
        if not portable:
            links=[]
            for s,p in groups[rid]['paths'].items():
                relative=Path(os.path.relpath(p,GALLERY)).as_posix();links.append(f'<a href="{html.escape(relative,quote=True)}">{s} exact NPZ</a>')
            intro.append('<p>'+' | '.join(links)+'</p>')
        for view in range(2):
            name=('review_jpg/' if portable else '')+f'{rid}__v{view}'+('.jpg' if portable else '.png')
            intro.append(f'<a href="{name}"><img src="{name}" alt="{html.escape(rid)} opposite view {view+1}"></a>')
        intro.append('</section>')
    intro.append('</html>');return '\n'.join(intro)


def cpu_resources():
    def cmd(args):
        p=subprocess.run(args,text=True,capture_output=True,timeout=20);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    result=dict(logical_CPU=os.cpu_count(),CPU_workers=1,CPU_threads=1,pid=os.getpid(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if hasattr(os,'getloadavg'):result['load_1_5_15']=list(os.getloadavg())
    if Path('/proc/meminfo').exists():
        result['memory']={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    result['GPU_observation_only']=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
    result['GPU_processes']=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'])
    return result


def verify_complete(definitions,groups):
    manifest_path=GALLERY/'gallery_manifest.json'
    if not manifest_path.exists():return False
    m=read(manifest_path)
    if m['status']!='COMPLETE' or m['schema']!=SCHEMA or m['source_code_sha256']!=sha256(__file__):raise ValueError('Completed gallery definition differs')
    if m['definition_hashes']!=definitions or (m['requests'],m['PNG_count'],m['JPEG_count'])!=(12,24,24):raise ValueError('Completed gallery identity differs')
    expected={(rid,v) for rid in groups for v in range(2)};seen=set()
    for r in m['receipts']:
        ident=r['identity'];key=(ident['request_id'],ident['view_id'])
        if key not in expected or key in seen or ident['source_hashes']!=groups[key[0]]['source_hashes']:raise ValueError('Completed source identity differs')
        if sha256(r['image'])!=r['sha256'] or sha256(r['jpg'])!=r['jpg_sha256']:raise ValueError('Completed image changed')
        if sha256(ident['render_arrays_path'])!=ident['render_arrays_sha256']:raise ValueError('Completed render arrays changed')
        sidecar=path_value(r['image']).with_suffix('.png.json')
        if read(sidecar)!=r:raise ValueError('Completed sidecar changed')
        seen.add(key)
    if seen!=expected:raise ValueError('Missing completed views')
    for p,h in m['support_hashes'].items():
        if sha256(p)!=h:raise ValueError('Completed HTML/template changed')
    print(json.dumps(dict(status='SKIP_VERIFIED_COMPLETE',PNG_count=24,JPEG_count=24,model_calls=0,renders=0)));return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    for forbidden in (OLD/'gallery/test.png',OUT/'registration.json',ROOT/'WORKSPACE.md'):
        try:checked(forbidden)
        except ValueError:pass
        else:raise AssertionError('Write guard failed')
    if args.check_only:
        print(json.dumps(dict(status='PASS_CPU_IMPORT_AND_WRITE_GUARD',schema=SCHEMA,CPU_workers=1,CPU_threads=1,model_calls=0,renders=0)));return
    records,groups,definitions=frozen_inputs()
    if verify_complete(definitions,groups):return
    start=time.monotonic();resources=cpu_resources();json_write(GALLERY/'resources_before.json',resources)
    receipts=[]
    for r in records:
        receipts.extend(render_request(r,groups[r['request_id']],definitions));print(json.dumps(dict(request_id=r['request_id'],completed_images=len(receipts))),flush=True)
    if len(receipts)!=24 or any(sha256(p)!=h for p,h in definitions.items()):raise ValueError('Completion/source mismatch')
    text_write(GALLERY/'index.html',make_html(records,groups));text_write(GALLERY/'index_portable.html',make_html(records,groups,True))
    pending=[]
    for r in records:
        for stage in STAGES[1:]:
            pending.append(dict(request_id=r['request_id'],parent_id=r['parent_id'],region_type=r['region_type'],stage=stage,
              review_kind='AI_PROXY',review_status='PENDING',blind=False,overall_shape='NOT_REVIEWED',C_use='NOT_REVIEWED',
              boundary_continuation='NOT_REVIEWED',thin_structure='NOT_REVIEWED',change_vs_S1='NOT_REVIEWED',change_vs_D1='NOT_REVIEWED',
              uncertainty='NOT_REVIEWED',observations='NOT_REVIEWED',figure_v0=r['request_id']+'__v0.png',figure_v1=r['request_id']+'__v1.png'))
    csv_write(GALLERY/'review_pending.csv',pending)
    supports={str(p):sha256(p) for p in (GALLERY/'index.html',GALLERY/'index_portable.html',GALLERY/'review_pending.csv')}
    json_write(GALLERY/'resources_after.json',cpu_resources())
    manifest=dict(schema=SCHEMA,status='COMPLETE',requests=12,stages=list(STAGES),input_meshes=72,PNG_count=24,JPEG_count=24,
      definition_hashes=definitions,source_code_sha256=sha256(__file__),wall_seconds=time.monotonic()-start,CPU_workers=1,CPU_threads=1,
      resources_before=resources,resources_after=read(GALLERY/'resources_after.json'),GPU=0,model_calls=0,GT_read=False,source_geometry_modified=False,
      visual_review_status='PENDING_AI_PROXY',receipts=receipts,support_hashes=supports,JPEG_total_bytes=sum(path_value(r['jpg']).stat().st_size for r in receipts))
    json_write(GALLERY/'gallery_manifest.json',manifest)
    print(json.dumps(dict(status='COMPLETE',PNG_count=24,JPEG_count=24,wall_seconds=manifest['wall_seconds'],JPEG_total_bytes=manifest['JPEG_total_bytes'])),flush=True)

if __name__=='__main__':main()
