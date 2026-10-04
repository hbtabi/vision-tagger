import csv
import io
import json

from visiontagger.batch import find_images, run_batch, to_csv, to_html, to_jsonl
from visiontagger.cli import main
from visiontagger.tagger import Tag
from visiontagger.vision_llm import VisionLLM, VisionLLMError, enrich, parse_reply


def test_find_images_recursive(sample_dir, tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "x.png").write_bytes((sample_dir / "abstract.png").read_bytes())
    (tmp_path / "notes.txt").write_text("hi")
    assert find_images([tmp_path]) == []
    assert [p.name for p in find_images([tmp_path], recursive=True)] == ["x.png"]
    assert len(find_images([sample_dir])) == 8


def test_exports(sample_dir):
    results = run_batch(find_images([sample_dir]), workers=2, min_score=0.5)
    assert all(t.score >= 0.5 for r in results for t in r.tags)
    rows = list(csv.DictReader(io.StringIO(to_csv(results))))
    assert len(rows) == 8 and rows[0]["caption"] and "#" in rows[0]["colours"]
    lines = to_jsonl(results).splitlines()
    assert len(lines) == 8 and json.loads(lines[0])["tags"]
    page = to_html(results)
    assert page.count('class="card"') == 8 and "data:image/jpeg;base64," in page


def test_cli_json_csv_html(sample_dir, tmp_path, capsys):
    out = tmp_path / "tags.json"
    html = tmp_path / "r.html"
    assert main(["tag", str(sample_dir), "-o", str(out), "--html", str(html), "--features"]) == 0
    data = json.loads(out.read_text())
    assert len(data) == 8 and "features" in data[0] and html.exists()
    assert main(["tag", str(sample_dir / "beach.jpg"), "-f", "csv"]) == 0
    assert "beach.jpg" in capsys.readouterr().out
    assert main(["tag", str(tmp_path / "missing")]) == 2


def test_cli_inspect_and_samples(tmp_path, capsys):
    assert main(["samples", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["inspect", str(tmp_path / "forest.jpg")]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["features"]["regions"]["green"] > 0.2


def test_parse_reply():
    cap, tags = parse_reply('```json\n{"caption": "A cat.", "tags": ["Cat", "cat", "sofa"]}\n```')
    assert cap == "A cat." and tags == ["cat", "sofa"]
    for bad in ("nothing", '{"tags": []}', "{bad json}"):
        try:
            parse_reply(bad)
            raise AssertionError("expected failure")
        except VisionLLMError:
            pass


class FakeLLM(VisionLLM):
    def __init__(self, reply):
        super().__init__(api_key="k")
        self.reply = reply

    def describe(self, path):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_enrich_merges_llm_and_colour_tags(sample_dir):
    res = run_batch([sample_dir / "beach.jpg"])[0]
    out = enrich(res, FakeLLM(("Waves on a sandy beach.", ["beach", "sea", "summer"])))
    assert out.caption == "Waves on a sandy beach." and out.source == "llm:gpt-4o-mini"
    assert out.tag_names[:3] == ["beach", "sea", "summer"]
    assert any(t.kind == "colour" for t in out.tags) and all(isinstance(t, Tag) for t in out.tags)


def test_enrich_falls_back_on_error(sample_dir):
    res = run_batch([sample_dir / "forest.jpg"])[0]
    out = enrich(res, FakeLLM(VisionLLMError("timeout")))
    assert out.source.startswith("heuristic") and "vegetation" in out.tag_names


def test_llm_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not VisionLLM().available
