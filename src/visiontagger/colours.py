"""Colour-space helpers and a small named palette (matched in CIE Lab, not RGB)."""

from __future__ import annotations

import numpy as np

PALETTE: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 22), "charcoal": (60, 62, 66), "grey": (128, 128, 128), "silver": (192, 192, 196),
    "white": (245, 245, 245), "cream": (240, 228, 196), "beige": (215, 192, 150), "brown": (120, 72, 40),
    "tan": (196, 150, 100), "red": (200, 30, 35), "maroon": (110, 20, 30), "pink": (240, 140, 170),
    "magenta": (200, 40, 160), "purple": (110, 50, 150), "lavender": (180, 160, 220), "navy": (20, 30, 90),
    "blue": (40, 90, 200), "sky blue": (120, 180, 235), "teal": (20, 130, 130), "turquoise": (60, 200, 200),
    "green": (40, 150, 60), "dark green": (20, 70, 35), "olive": (120, 120, 40), "lime": (160, 210, 60),
    "yellow": (245, 215, 50), "gold": (215, 165, 40), "orange": (240, 130, 30), "coral": (240, 110, 90),
}


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an (..., 3) array of 0-255 sRGB values to CIE Lab (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


_NAMES = list(PALETTE)
_LAB = srgb_to_lab(np.array([PALETTE[n] for n in _NAMES]))


def name_colour(rgb: tuple[int, int, int] | np.ndarray) -> str:
    lab = srgb_to_lab(np.asarray(rgb, dtype=np.float64).reshape(1, 3))[0]
    return _NAMES[int(np.argmin(np.linalg.norm(_LAB - lab, axis=1)))]


def to_hex(rgb) -> str:
    r, g, b = (int(round(v)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    """Vectorised RGB (0-255) -> HSV with H in degrees [0, 360), S and V in [0, 1]."""
    a = arr.astype(np.float64) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    delta = mx - mn
    h = np.zeros_like(mx)
    nz = delta > 1e-9
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / delta[rm]) % 6
    h[gm] = (b - r)[gm] / delta[gm] + 2
    h[bm] = (r - g)[bm] / delta[bm] + 4
    s = np.where(mx > 1e-9, delta / np.maximum(mx, 1e-9), 0.0)
    return np.stack([h * 60.0, s, mx], axis=-1)


def kmeans_colours(pixels: np.ndarray, k: int = 5, iters: int = 12, seed: int = 7) -> list[tuple[np.ndarray, float]]:
    """Dominant colours by k-means (k-means++ init) in Lab space. Returns ``[(rgb, share), ...]``."""
    rng = np.random.default_rng(seed)
    if len(pixels) > 6000:
        pixels = pixels[rng.choice(len(pixels), 6000, replace=False)]
    lab = srgb_to_lab(pixels)
    k = max(1, min(k, len(np.unique(pixels, axis=0))))
    centres = [lab[rng.integers(len(lab))]]
    for _ in range(1, k):
        d = np.min(np.linalg.norm(lab[:, None] - np.array(centres)[None], axis=2), axis=1) ** 2
        centres.append(lab[rng.choice(len(lab), p=d / d.sum())] if d.sum() > 0 else lab[rng.integers(len(lab))])
    centres_arr = np.array(centres)
    labels = np.zeros(len(lab), dtype=int)
    for _ in range(iters):
        labels = np.argmin(np.linalg.norm(lab[:, None] - centres_arr[None], axis=2), axis=1)
        new = np.array([lab[labels == i].mean(0) if np.any(labels == i) else centres_arr[i] for i in range(k)])
        if np.allclose(new, centres_arr, atol=0.5):
            break
        centres_arr = new
    out = []
    for i in range(k):
        mask = labels == i
        if mask.any():
            out.append((pixels[mask].mean(0), float(mask.mean())))
    return sorted(out, key=lambda x: -x[1])
