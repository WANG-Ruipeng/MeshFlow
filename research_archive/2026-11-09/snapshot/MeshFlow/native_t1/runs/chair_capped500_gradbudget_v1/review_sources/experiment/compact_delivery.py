"""Lossless container conversion and compact gallery receipts for the small review ZIP."""
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'native_t1/runs/chair_capped500_gradbudget_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')

def run():
    folder=OUT/'delivery_compact';folder.mkdir(exist_ok=True)
    records=[]
    for panel in ('OLD','FRESH'):
        pointer=read(OUT/'visuals'/f'latest_{panel}.json')
        original=Path(pointer['contact_sheet']);source=Image.open(original).convert('RGBA')
        target=folder/f'contact_{panel}.webp'
        source.save(target,format='WEBP',lossless=True,quality=100,method=6,exact=True)
        decoded=Image.open(target).convert('RGBA')
        assert source.size==decoded.size and source.tobytes()==decoded.tobytes()
        manifest_path=Path(pointer['manifest']);manifest=read(manifest_path)
        compact={k:v for k,v in manifest.items() if k not in ('render_receipts','overlay_receipts')}
        compact.update(original_full_manifest=str(manifest_path),original_full_manifest_sha256=sha(manifest_path),
            compact_scope='Full local per-mesh render receipts omitted only from ZIP; page hashes, camera identity and all review observations retained.',
            full_manifest_in_zip=False)
        save(folder/f'gallery_{panel}.json',compact)
        records.append(dict(panel=panel,source=str(original),source_sha256=sha(original),source_bytes=original.stat().st_size,
            target=str(target),target_sha256=sha(target),target_bytes=target.stat().st_size,dimensions=list(source.size),
            decoded_RGBA_pixels_identical=True,full_gallery_manifest_sha256=sha(manifest_path)))
    for name,rel in [('training_overview','training_diagnostics/final/training_overview.png'),
                     ('primary500_control_and_shape','report_figures/primary500_control_and_shape.png'),
                     ('stability500_to1000','report_figures/stability500_to1000.png')]:
        original=OUT/rel;source=Image.open(original).convert('RGBA');target=folder/(name+'.webp')
        source.save(target,format='WEBP',lossless=True,quality=100,method=6,exact=True)
        decoded=Image.open(target).convert('RGBA')
        assert source.size==decoded.size and source.tobytes()==decoded.tobytes()
        records.append(dict(figure=name,source=str(original),source_sha256=sha(original),source_bytes=original.stat().st_size,
            target=str(target),target_sha256=sha(target),target_bytes=target.stat().st_size,dimensions=list(source.size),
            decoded_RGBA_pixels_identical=True))
    receipt=dict(status='PASS',conversion='PNG to lossless WebP; identical decoded RGBA pixels, no resize/crop/render',
        records=records,model_calls=0,scoring_calls=0,geometry_changes=0,full_originals_retained=True)
    save(folder/'receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':run()
