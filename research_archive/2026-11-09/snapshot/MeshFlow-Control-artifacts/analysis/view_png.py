import sys,io,base64
from PIL import Image
im=Image.open(sys.argv[1]); b=io.BytesIO(); im.save(b,format='WEBP',lossless=True); print(base64.b64encode(b.getvalue()).decode())