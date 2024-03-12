import os
from pathlib import Path
from PIL import Image
import numpy as np

img_dir = Path(r"/home/luca/Desktop/IMC24/GlassCup/original")
size = (1300, 3288)
out_dir = Path(r"/home/luca/Desktop/IMC24/GlassCup/outs")

images = os.listdir(img_dir)

for img_name in images:
    img = Image.open(img_dir / img_name)
    mask = Image.new("RGB", (size[0], size[1]), color="black")
    img.paste(mask, (0, 0))
    img.save(out_dir / img_name)
