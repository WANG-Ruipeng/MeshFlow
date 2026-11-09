"""Lossless image-only crops for the assigned AI_PROXY visual review."""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'visual_review'/'crops_part01'
OUT.mkdir(exist_ok=True)
for p in sorted((ROOT/'gallery').glob('*/comparison.png')):
    if not any(s in p.parent.name for s in ['a8e987','27eb79']):
        continue
    im=Image.open(p).convert('RGB')
    canvas=Image.new('RGB',(1040,920),'white')
    draw=ImageDraw.Draw(canvas)
    for i,method in enumerate(['EXPOSURE','PLAIN1000','LAYOUT1000','LAYOUT_NULL']):
        draw.text((8,i*230+5),method+' | '+p.parent.name.split('__')[0],fill='black')
        crop=im.crop((100,250+400*i,1140,450+400*i))
        canvas.paste(crop,(0,i*230+25))
    canvas.save(OUT/(p.parent.name+'.png'))
print('created 12 unscaled crop pages')
