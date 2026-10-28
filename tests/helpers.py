"""Shared test fixtures: temporary services and synthetic book images (no external files needed)."""
import os
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from services.catalogue_service import CatalogueService
from services.db import Database
from services.image_scanner import ImageScanner
from services.metadata_service import MetadataService
from services.sample_data import load_sample
from services.user_service import UserService


def make_services(sample=False, online=False):
    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "test.db"))
    cat, users = CatalogueService(db), UserService(db)
    meta = MetadataService(catalogue_provider=cat.list_books, online=online)
    if sample:
        load_sample(cat, users)
    return SimpleNamespace(tmp=tmp, db=db, cat=cat, users=users, meta=meta, scanner=ImageScanner(meta),
                           cleanup=lambda: shutil.rmtree(tmp, ignore_errors=True))


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def make_cover(title_lines, author, size=(700, 1000), bg=(30, 60, 120)):
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    y = 150
    for t in title_lines:
        d.text((50, y), t, font=font(90), fill=(255, 255, 255))
        y += 110
    d.text((50, y + 120), author, font=font(48), fill=(230, 230, 100))
    return im


def to_bgr(im):
    return np.array(im)[:, :, ::-1].copy()


def table_image(covers, size=(520, 740)):
    canvas = Image.new("RGB", (60 + len(covers) * 600, 900), (190, 190, 190))
    for i, c in enumerate(covers):
        canvas.paste(c.resize(size), (60 + i * 600, 80))
    return canvas
