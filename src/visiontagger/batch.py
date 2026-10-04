"""Batch processing and exporters (JSON, JSONL, CSV and an HTML contact sheet)."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .features import load_image
from .tagger import TagResult, tag_image
from .vision_llm import VisionLLM, enrich

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def find_images(paths: Iterable[str | Path], recursive: bool = False) -> list[Path]:
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            found.extend(sorted(x for x in it if x.is_file() and x.suffix.lower() in IMAGE_EXTS))
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(p)
    return list(dict.fromkeys(found))


def run_batch(paths: list[Path], workers: int = 4, llm: VisionLLM | None = None, n_colours: int = 5,
              min_score: float = 0.0) -> list[TagResult]:
    def work(path: Path) -> TagResult:
        res = tag_image(path, n_colours=n_colours)
        if llm is not None:
            res = enrich(res, llm)
        res.tags = [t for t in res.tags if t.score >= min_score]
        return res

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(work, paths))


def to_json(results: list[TagResult], include_features: bool = False) -> str:
    return json.dumps([r.to_dict(include_features) for r in results], indent=2)


def to_jsonl(results: list[TagResult], include_features: bool = False) -> str:
    return "".join(json.dumps(r.to_dict(include_features)) + "\n" for r in results)


def to_csv(results: list[TagResult]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["file", "width", "height", "caption", "tags", "colours", "source", "error"])
    for r in results:
        f = r.features
        w.writerow([r.file, f.width if f else "", f.height if f else "", r.caption, "; ".join(r.tag_names),
                    "; ".join(f"{c.name} {c.hex} {c.share:.0%}" for c in f.colours) if f else "", r.source,
                    r.error or ""])
    return buf.getvalue()


def summary(results: list[TagResult]) -> Counter[str]:
    return Counter(t.name for r in results for t in r.tags if t.kind != "layout")


def _thumb(path: str, side: int = 260) -> str:
    img = load_image(path)
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


CSS = """
body{font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:#0f1218;color:#e6e8ee;margin:0;padding:28px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#8c93a3;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:16px}
.card{background:#171b24;border:1px solid #262c3a;border-radius:12px;padding:12px}
.img{height:170px;display:flex;align-items:center;justify-content:center;background:#0b0d12;border-radius:8px;overflow:hidden}
.img img{max-width:100%;max-height:170px}
.name{font-weight:600;margin:10px 0 6px}
.name small{color:#8c93a3;font-weight:400}
.bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin-bottom:8px}
.sw{display:block}
p{margin:6px 0 8px;color:#c5cad6}
.tag{display:inline-block;padding:2px 8px;margin:0 4px 4px 0;border-radius:99px;font-size:12px}
.tag i{font-style:normal;opacity:.55;margin-left:5px}
.s{background:#1f3b2d;color:#8ee6b0}
.y{background:#3a2f4d;color:#d3b8ff}
.c{background:#2a3446;color:#9fc3ff}
.l{background:#2b2b2b;color:#aaa}
.err{border-color:#6b2a2a}
"""


def to_html(results: list[TagResult], title: str = "vision-tagger report") -> str:
    """A self-contained contact sheet with thumbnails, captions, colour swatches and tags."""
    cards = []
    kind_cls = {"scene": "s", "style": "y", "colour": "c", "layout": "l", "llm": "s"}
    for r in results:
        if r.error:
            cards.append(f'<div class="card err"><div class="name">{html.escape(Path(r.file).name)}</div>'
                         f'<p>{html.escape(r.error)}</p></div>')
            continue
        sw = "".join(f'<span class="sw" title="{c.name} {c.share:.0%}" style="background:{c.hex};flex:{c.share}">'
                     "</span>" for c in r.features.colours)
        tags = "".join(f'<span class="tag {kind_cls.get(t.kind, "s")}">{html.escape(t.name)}'
                       f'<i>{t.score:.2f}</i></span>' for t in r.tags)
        cards.append(
            f'<div class="card"><div class="img"><img src="data:image/jpeg;base64,{_thumb(r.file)}"></div>'
            f'<div class="name">{html.escape(Path(r.file).name)} <small>{r.features.width}×{r.features.height}'
            f'</small></div><div class="bar">{sw}</div><p>{html.escape(r.caption)}</p><div>{tags}</div></div>')
    top = ", ".join(f"{k} ({v})" for k, v in summary(results).most_common(8))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>
{CSS}</style></head><body><h1>{html.escape(title)}</h1>
<div class="sub">{len(results)} images · top tags: {html.escape(top)}</div>
<div class="grid">{"".join(cards)}</div></body></html>"""
