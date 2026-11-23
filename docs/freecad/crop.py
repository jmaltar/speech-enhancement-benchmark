"""Trim the white border off each render in docs/freecad/render/, keeping a small margin.
   usage: .venv/bin/python docs/freecad/crop.py"""
import os
from PIL import Image, ImageChops

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render")
for f in sorted(os.listdir(D)):
    if not f.endswith(".png") or f.endswith("_crop.png"):
        continue
    im = Image.open(os.path.join(D, f)).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    m = 40
    box = (max(box[0] - m, 0), max(box[1] - m, 0), min(box[2] + m, im.width), min(box[3] + m, im.height))
    out = im.crop(box)
    out.save(os.path.join(D, f.replace(".png", "_crop.png")))
    print(f, im.size, "->", out.size)
