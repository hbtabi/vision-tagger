"""Turn features into scored tags and a natural-language caption."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .features import ImageFeatures, extract_features


@dataclass
class Tag:
    name: str
    score: float
    kind: str = "scene"  # scene | colour | style | layout


@dataclass
class TagResult:
    file: str
    caption: str
    tags: list[Tag]
    features: ImageFeatures | None = None
    source: str = "heuristic"
    error: str | None = None

    @property
    def tag_names(self) -> list[str]:
        return [t.name for t in self.tags]

    def to_dict(self, include_features: bool = False) -> dict:
        d = {"file": self.file, "caption": self.caption, "source": self.source,
             "tags": [{"name": t.name, "score": round(t.score, 3), "kind": t.kind} for t in self.tags]}
        if self.features is not None:
            d["colours"] = [c.__dict__ for c in self.features.colours]
            d["size"] = [self.features.width, self.features.height]
            if include_features:
                d["features"] = self.features.to_dict()
        if self.error:
            d["error"] = self.error
        return d


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0..1 ramp; a soft threshold that doubles as a confidence score."""
    if hi == lo:
        return float(x >= hi)
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def score_tags(f: ImageFeatures) -> list[Tag]:
    r = f.regions
    s: dict[str, tuple[float, str]] = {}

    def add(name: str, score: float, kind: str = "scene") -> None:
        if score >= 0.35:
            s[name] = (round(min(1.0, score), 3), kind)

    sky = _ramp(r["blue_top"], 0.15, 0.5) * (1 - _ramp(f.edge_density, 0.25, 0.45))
    add("sky", sky)
    add("vegetation", _ramp(r["green"], 0.12, 0.4))
    add("water", _ramp(r["blue_bottom"], 0.2, 0.5) * _ramp(r["blue_top"] + r["white_top"], 0.05, 0.3))
    add("beach", _ramp(r["sand_bottom"], 0.15, 0.45) * _ramp(r["blue"], 0.1, 0.3))
    cool_ratio = (r["blue"] + r["green"]) / max(r["vivid"], 1e-6)
    graphic = _ramp(f.colourfulness, 80, 120) * _ramp(5.0 - f.entropy, 0, 1.5)
    add("sunset", _ramp(r["dusk_top"], 0.3, 0.7) * _ramp(r["warm"], 0.15, 0.35) * (1 - _ramp(cool_ratio, 0.1, 0.25)))
    add("night", _ramp(0.3 - f.brightness, 0.0, 0.12) * _ramp(r["dark"], 0.4, 0.7))
    doc = _ramp(r["white"], 0.45, 0.75) * _ramp(f.edge_density, 0.01, 0.05) * (1 - _ramp(f.saturation, 0.08, 0.2))
    add("document", doc)
    add("snow", _ramp(r["white_bottom"], 0.4, 0.75) * (1 - doc) * _ramp(r["blue_top"] + r["white_top"], 0.2, 0.6))
    add("graphic", graphic, "style")
    outdoor = max(s.get(k, (0.0, ""))[0] for k in ("sky", "vegetation", "water", "beach", "sunset", "snow"))
    if outdoor:
        add("outdoor", outdoor)
    if "sky" in s and ("vegetation" in s or "snow" in s or "water" in s or "sunset" in s):
        add("landscape", min(s["sky"][0], max(s.get(k, (0.0, ""))[0] for k in ("vegetation", "snow", "water"))))
    if "night" in s and r["warm"] + r["white"] > 0.01:
        add("city lights", _ramp(r["warm"] + r["white"], 0.01, 0.06) * s["night"][0])

    # style
    add("monochrome", _ramp(12 - f.colourfulness, 0, 8), "style")
    add("vibrant", _ramp(f.colourfulness, 55, 90), "style")
    add("high-key", _ramp(f.brightness, 0.72, 0.88), "style")
    add("low-key", _ramp(0.3 - f.brightness, 0.02, 0.15), "style")
    add("high contrast", _ramp(f.contrast, 0.27, 0.36), "style")
    add("minimal", _ramp(0.04 - f.edge_density, 0, 0.03) * _ramp(6.5 - f.entropy, 0, 2), "style")
    if "document" not in s:
        add("detailed", _ramp(f.edge_density, 0.15, 0.3), "style")
    if f.colourfulness > 15:
        add("warm tones", _ramp(f.warmth, 0.15, 0.5), "style")
        add("cool tones", _ramp(-f.warmth, 0.15, 0.5), "style")
    # layout + colours
    s[f"{f.orientation} format"] = (1.0, "layout")
    for c in f.colours[:3]:
        if c.share >= 0.12 and c.name not in s:
            s[c.name] = (round(min(1.0, c.share * 2.5), 3), "colour")
    order = {"scene": 0, "style": 1, "colour": 2, "layout": 3}
    return sorted((Tag(n, sc, k) for n, (sc, k) in s.items()), key=lambda t: (order[t.kind], -t.score, t.name))


SCENE_PHRASES = [
    ("graphic", "flat graphic shapes or an illustration"),
    ("sunset", "a sunset sky"),
    ("city lights", "a night-time scene with city lights"),
    ("beach", "a beach by the water"),
    ("document", "a document or page of text"),
    ("snow", "a snowy landscape"),
    ("landscape", "an outdoor landscape"),
    ("water", "open water"),
    ("vegetation", "greenery or foliage"),
    ("night", "a dark, low-light scene"),
    ("sky", "open sky"),
]


def caption(f: ImageFeatures, tags: list[Tag]) -> str:
    names = {t.name for t in tags}
    if "graphic" in names:
        names -= {k for k, _ in SCENE_PHRASES}
        names.add("graphic")
    light = "dark" if f.brightness < 0.3 else ("bright" if f.brightness > 0.65 else "softly lit")
    tone = ("black-and-white" if "monochrome" in names else "vivid" if "vibrant" in names
            else "warm-toned" if "warm tones" in names else "cool-toned" if "cool tones" in names else "muted")
    colours = [c.name for c in f.colours if c.share >= 0.1][:2] or [f.colours[0].name] if f.colours else []
    scene = next((p for key, p in SCENE_PHRASES if key in names), None)
    text = f"A {light}, {tone} {f.orientation} image"
    if colours and "monochrome" not in names:
        text += " dominated by " + " and ".join(colours)
    if scene:
        text += f", likely showing {scene}"
    return text + "."


def tag_image(path: str | Path, n_colours: int = 5) -> TagResult:
    try:
        f = extract_features(path, n_colours=n_colours)
    except Exception as exc:  # noqa: BLE001 - unreadable/corrupt files become error rows in batch output
        return TagResult(str(path), "", [], None, "heuristic", error=f"{type(exc).__name__}: {exc}")
    tags = score_tags(f)
    return TagResult(str(path), caption(f, tags), tags, f)
