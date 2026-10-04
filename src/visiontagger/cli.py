"""``vision-tagger`` command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
import time
from pathlib import Path

from . import __version__
from .batch import find_images, run_batch, summary, to_csv, to_html, to_json, to_jsonl
from .samples import write_samples
from .tagger import tag_image
from .vision_llm import VisionLLM

KIND_COLOUR = {"scene": "32", "llm": "32", "style": "35", "colour": "34", "layout": "2"}


def _colour_on() -> bool:
    return os.getenv("FORCE_COLOR", "") not in ("", "0") or (sys.stdout.isatty() and not os.getenv("NO_COLOR"))


def _paint(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _colour_on() else text


def cmd_tag(args: argparse.Namespace) -> int:
    try:
        paths = find_images(args.paths, recursive=args.recursive)
    except FileNotFoundError as exc:
        print(f"error: no such file or directory: {exc}", file=sys.stderr)
        return 2
    if not paths:
        print("error: no images found", file=sys.stderr)
        return 2
    llm = None
    if args.vision_llm:
        llm = VisionLLM()
        if not llm.available:
            print("note: --vision-llm needs OPENAI_API_KEY; using offline heuristics", file=sys.stderr)
            llm = None
    t0 = time.perf_counter()
    results = run_batch(paths, workers=args.workers, llm=llm, n_colours=args.colours, min_score=args.min_score)
    elapsed = time.perf_counter() - t0

    fmt = args.format or (Path(args.out).suffix.lstrip(".") if args.out else "table")
    if fmt == "table":
        width = max(len(Path(r.file).name) for r in results)
        for r in results:
            name = Path(r.file).name.ljust(width)
            if r.error:
                print(f"{name}  ERROR {r.error}")
                continue
            tags = ", ".join(_paint(KIND_COLOUR.get(t.kind, "0"), t.name) for t in r.tags[:8])
            print(f"{_paint('1', name)}  {tags}")
            cols = max(60, shutil.get_terminal_size((110, 24)).columns) - width - 2
            for line in textwrap.wrap(r.caption, cols):
                print(f"{' ' * width}  {_paint('2', line)}")
        sys.stdout.flush()
        out_text = None
    else:
        render = {"json": lambda: to_json(results, args.features), "jsonl": lambda: to_jsonl(results, args.features),
                  "csv": lambda: to_csv(results), "html": lambda: to_html(results)}
        if fmt not in render:
            print(f"error: unknown format {fmt!r}", file=sys.stderr)
            return 2
        out_text = render[fmt]()
    if out_text is not None:
        if args.out:
            Path(args.out).write_text(out_text, encoding="utf-8")
        else:
            sys.stdout.write(out_text)
    if args.html:
        Path(args.html).write_text(to_html(results), encoding="utf-8")
    errors = sum(1 for r in results if r.error)
    top = ", ".join(f"{k}×{v}" for k, v in summary(results).most_common(5))
    print(f"tagged {len(results) - errors}/{len(results)} images in {elapsed:.2f}s"
          + (f" -> {args.out}" if args.out else "") + (f" (+ {args.html})" if args.html else "")
          + (f" · top: {top}" if top else ""), file=sys.stderr)
    return 1 if errors == len(results) else 0


def cmd_inspect(args: argparse.Namespace) -> int:
    r = tag_image(args.image)
    if r.error:
        print(f"error: {r.error}", file=sys.stderr)
        return 2
    print(json.dumps(r.to_dict(include_features=True), indent=2))
    return 0


def cmd_samples(args: argparse.Namespace) -> int:
    for p in write_samples(args.out_dir):
        print(p)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vision-tagger", description="Batch image tagging and captioning.")
    p.add_argument("--version", action="version", version=f"vision-tagger {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("tag", help="tag images or folders")
    sp.add_argument("paths", nargs="+")
    sp.add_argument("-r", "--recursive", action="store_true")
    sp.add_argument("-f", "--format", choices=["table", "json", "jsonl", "csv", "html"],
                    help="output format (default: from --out extension, else table)")
    sp.add_argument("-o", "--out", help="output file")
    sp.add_argument("--html", help="also write an HTML contact sheet here")
    sp.add_argument("-w", "--workers", type=int, default=4)
    sp.add_argument("--colours", type=int, default=5, help="dominant colours to extract")
    sp.add_argument("--min-score", type=float, default=0.0, help="drop tags below this confidence")
    sp.add_argument("--features", action="store_true", help="include raw features in JSON output")
    sp.add_argument("--vision-llm", action="store_true", help="caption with an OpenAI-compatible vision model")
    sp.set_defaults(func=cmd_tag)

    sp = sub.add_parser("inspect", help="print every feature, tag and score for one image")
    sp.add_argument("image")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("samples", help="generate the procedural sample images")
    sp.add_argument("out_dir", nargs="?", default="examples/images")
    sp.set_defaults(func=cmd_samples)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
