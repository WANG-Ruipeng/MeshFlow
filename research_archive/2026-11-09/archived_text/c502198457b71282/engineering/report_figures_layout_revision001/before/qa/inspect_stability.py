"""Pixel-exact vertical splits for viewing the tall final PNG; no rendering."""
from pathlib import Path
import hashlib
import json
from PIL import Image
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'stability500_to1000.png'
im=Image.open(SOURCE)
rows=[]
for i,(y0,y1) in enumerate([(0,1200),(1200,2400),(2400,im.height)]):
    path=HERE/f'stability_part{i+1}.png'
    im.crop((0,y0,im.width,y1)).save(path)
    rows.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     original_pixel_bounds=[0,y0,im.width,y1],resized=False))
(HERE/'split_receipt.json').write_text(json.dumps(dict(source=str(SOURCE),
    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),original_size=list(im.size),
    all_source_pixels_included_once=True,splits=rows,render_calls=0,model_calls=0),indent=2)+'\n')
print(json.dumps(rows))
