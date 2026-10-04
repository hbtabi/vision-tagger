import pytest

from visiontagger.tagger import tag_image

EXPECT = {
    "sunset.jpg": {"sunset", "warm tones"},
    "forest.jpg": {"vegetation", "sky", "outdoor"},
    "beach.jpg": {"beach", "sky"},
    "night-city.png": {"night", "city lights", "low-key"},
    "document.png": {"document", "monochrome"},
    "snow-mountain.jpg": {"snow", "sky"},
    "abstract.png": {"graphic", "vibrant"},
    "monochrome-blocks.jpg": {"monochrome"},
}
FORBID = {
    "abstract.png": {"sunset", "outdoor"},
    "document.png": {"snow", "detailed"},
    "night-city.png": {"sky", "sunset"},
    "monochrome-blocks.jpg": {"vibrant", "outdoor"},
}


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_sample_tags(sample_dir, name):
    res = tag_image(sample_dir / name)
    assert res.error is None
    tags = set(res.tag_names)
    assert EXPECT[name] <= tags, f"{name}: {tags}"
    assert not (FORBID.get(name, set()) & tags), f"{name}: {tags}"
    assert all(0 < t.score <= 1 for t in res.tags)
    assert res.caption.startswith("A ") and res.caption.endswith(".")


def test_caption_mentions_scene_and_layout(sample_dir):
    res = tag_image(sample_dir / "beach.jpg")
    assert "beach" in res.caption and "landscape" in res.caption
    assert "landscape format" in res.tag_names


def test_corrupt_file_becomes_error(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    res = tag_image(bad)
    assert res.error and res.tags == [] and res.to_dict()["error"]
