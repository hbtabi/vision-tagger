"""Hand-crafted image features computed with Pillow + NumPy (no deep learning)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .colours import kmeans_colours, name_colour, rgb_to_hsv, to_hex

ANALYSIS_SIZE = 256


@dataclass
class DominantColour:
    name: str
    hex: str
    share: float


@dataclass
class ImageFeatures:
    width: int
    height: int
    orientation: str
    brightness: float  # mean luma 0-1
    contrast: float  # luma std 0-1
    saturation: float  # mean HSV saturation 0-1
    colourfulness: float  # Hasler & Suesstrunk metric (0 ~ grey, >60 very colourful)
    edge_density: float  # share of pixels on a strong Sobel edge
    entropy: float  # luma histogram entropy in bits (0-8)
    warmth: float  # warm minus cool share of saturated pixels, -1..1
    regions: dict[str, float] = field(default_factory=dict)
    colours: list[DominantColour] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        d["regions"] = {k: round(v, 4) for k, v in self.regions.items()}
        return d


def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # respect camera orientation
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def _sobel_edges(luma: np.ndarray) -> np.ndarray:
    p = np.pad(luma, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.hypot(gx, gy)


def _colourfulness(rgb: np.ndarray) -> float:
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def _region_masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = (s > 0.2) & (v > 0.25)
    return {
        "vivid": vivid,
        "dusk": vivid & ((h < 50) | (h >= 260)),
        "blue": vivid & (h >= 185) & (h < 255),
        "green": vivid & (h >= 70) & (h < 170),
        "warm": (s > 0.35) & (v > 0.35) & ((h < 50) | (h >= 330)),
        "white": (s < 0.12) & (v > 0.85),
        "dark": v < 0.18,
        "sand": (h >= 25) & (h < 55) & (s > 0.15) & (s < 0.55) & (v > 0.6),
    }


def extract_features(image: Image.Image | str | Path, n_colours: int = 5) -> ImageFeatures:
    img = load_image(image) if not isinstance(image, Image.Image) else image.convert("RGB")
    width, height = img.size
    small = img.copy()
    small.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE), Image.Resampling.BILINEAR)
    rgb = np.asarray(small, dtype=np.uint8)
    luma = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) / 255.0
    hsv = rgb_to_hsv(rgb)

    edges = _sobel_edges(luma)
    hist, _ = np.histogram(luma, bins=256, range=(0, 1))
    prob = hist[hist > 0] / hist.sum()
    masks = _region_masks(hsv)
    third = max(1, rgb.shape[0] // 3)
    regions: dict[str, float] = {}
    for name, mask in masks.items():
        regions[name] = float(mask.mean())
        regions[f"{name}_top"] = float(mask[:third].mean())
        regions[f"{name}_bottom"] = float(mask[-third:].mean())
    cool = masks["blue"] | masks["green"]
    sat_px = max(1, int((hsv[..., 1] > 0.2).sum()))
    warmth = float((masks["warm"].sum() - cool.sum()) / sat_px)

    ratio = width / height if height else 1.0
    orientation = "square" if 0.9 <= ratio <= 1.1 else ("landscape" if ratio > 1.1 else "portrait")
    colours: list[DominantColour] = []
    for c, share in kmeans_colours(rgb.reshape(-1, 3), k=n_colours):
        name = name_colour(c)
        twin = next((x for x in colours if x.name == name), None)
        if twin:  # two clusters with the same name: merge their shares
            twin.share = round(twin.share + share, 4)
        else:
            colours.append(DominantColour(name, to_hex(c), round(share, 4)))
    colours.sort(key=lambda x: -x.share)
    return ImageFeatures(
        width=width, height=height, orientation=orientation,
        brightness=float(luma.mean()), contrast=float(luma.std()), saturation=float(hsv[..., 1].mean()),
        colourfulness=_colourfulness(rgb), edge_density=float((edges > 0.5).mean()),
        entropy=float(-(prob * np.log2(prob)).sum()), warmth=max(-1.0, min(1.0, warmth)),
        regions=regions, colours=colours,
    )
