"""Procedurally generated sample images, so demos and tests need no downloaded photos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 480, 320


def _gradient(top: tuple[int, int, int], bottom: tuple[int, int, int], w: int = W, h: int = H) -> Image.Image:
    t = np.linspace(0, 1, h)[:, None, None]
    arr = (np.array(top) * (1 - t) + np.array(bottom) * t).repeat(w, axis=1)
    return Image.fromarray(arr.astype(np.uint8))


def _noise(img: Image.Image, amount: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = np.asarray(img, dtype=np.float64)
    a = a + rng.normal(0, amount, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def sunset() -> Image.Image:
    img = _gradient((70, 30, 110), (250, 140, 40))
    d = ImageDraw.Draw(img)
    d.ellipse((190, 150, 290, 250), fill=(255, 210, 90))
    d.polygon([(0, 250), (90, 215), (200, 245), (330, 205), (480, 240), (480, 320), (0, 320)], fill=(45, 20, 40))
    return _noise(img.filter(ImageFilter.GaussianBlur(1.2)), 3, 1)


def forest() -> Image.Image:
    img = _gradient((90, 160, 235), (190, 225, 250))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 170, W, H), fill=(60, 120, 50))
    rng = np.random.default_rng(2)
    for _ in range(26):
        x, base, hgt = int(rng.integers(-20, W)), int(rng.integers(190, 320)), int(rng.integers(70, 150))
        g = int(rng.integers(70, 130))
        d.polygon([(x, base), (x + 24, base - hgt), (x + 48, base)], fill=(20, g, 40))
    return _noise(img, 6, 2)


def beach() -> Image.Image:
    img = _gradient((70, 150, 230), (170, 215, 245))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 140, W, 220), fill=(20, 110, 170))
    d.rectangle((0, 215, W, H), fill=(230, 205, 150))
    for y in range(150, 215, 12):
        d.line((0, y, W, y + 2), fill=(90, 170, 210), width=1)
    return _noise(img, 4, 3)


def night_city() -> Image.Image:
    img = _gradient((5, 8, 25), (20, 25, 55))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(4)
    x = 0
    while x < W:
        bw, bh = int(rng.integers(35, 70)), int(rng.integers(90, 230))
        d.rectangle((x, H - bh, x + bw, H), fill=(12, 14, 22))
        for wy in range(H - bh + 8, H - 6, 14):
            for wx in range(x + 5, x + bw - 6, 10):
                if rng.random() < 0.35:
                    d.rectangle((wx, wy, wx + 4, wy + 6), fill=(255, 200, 90))
        x += bw + 4
    return img


def document() -> Image.Image:
    img = Image.new("RGB", (300, 420), (250, 250, 248))
    d = ImageDraw.Draw(img)
    d.rectangle((28, 30, 200, 44), fill=(40, 40, 45))
    rng = np.random.default_rng(5)
    for y in range(70, 390, 16):
        d.rectangle((28, y, int(rng.integers(180, 272)), y + 5), fill=(120, 120, 125))
    return img


def snow_mountain() -> Image.Image:
    img = _gradient((110, 165, 225), (200, 225, 245))
    d = ImageDraw.Draw(img)
    d.polygon([(0, 260), (140, 90), (230, 180), (330, 70), (480, 230), (480, 320), (0, 320)], fill=(120, 125, 140))
    d.polygon([(105, 125), (140, 90), (175, 125), (150, 118)], fill=(250, 250, 252))
    d.polygon([(290, 110), (330, 70), (370, 115), (330, 105)], fill=(250, 250, 252))
    d.rectangle((0, 215, W, H), fill=(245, 247, 250))
    return _noise(img, 3, 6)


def abstract() -> Image.Image:
    img = Image.new("RGB", (400, 400), (250, 240, 230))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(7)
    palette = [(230, 40, 90), (30, 140, 230), (250, 200, 30), (40, 190, 120), (150, 60, 200), (250, 110, 30)]
    for _ in range(16):
        x, y = (int(v) for v in rng.integers(0, 400, 2))
        r = int(rng.integers(30, 90))
        d.ellipse((x - r, y - r, x + r, y + r), fill=palette[int(rng.integers(len(palette)))])
    return img


def monochrome_blocks() -> Image.Image:
    rng = np.random.default_rng(8)
    img = Image.new("L", (360, 480), 140)
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = int(rng.integers(0, 360)), int(rng.integers(0, 480))
        d.rectangle((x, y, x + int(rng.integers(10, 80)), y + int(rng.integers(10, 120))),
                    fill=int(rng.integers(20, 235)))
    return img.convert("RGB")


SAMPLES = {
    "sunset.jpg": sunset, "forest.jpg": forest, "beach.jpg": beach, "night-city.png": night_city,
    "document.png": document, "snow-mountain.jpg": snow_mountain, "abstract.png": abstract,
    "monochrome-blocks.jpg": monochrome_blocks,
}


def write_samples(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in SAMPLES.items():
        path = out / name
        img = fn()
        if path.suffix == ".jpg":
            img.save(path, quality=88)
        else:
            img.save(path, optimize=True)
        paths.append(path)
    return paths
