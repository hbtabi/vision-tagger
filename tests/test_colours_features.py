import numpy as np
import pytest
from PIL import Image

from visiontagger.colours import kmeans_colours, name_colour, rgb_to_hsv, srgb_to_lab, to_hex
from visiontagger.features import extract_features


def test_lab_reference_values():
    assert np.allclose(srgb_to_lab([255, 255, 255]), [100, 0, 0], atol=0.1)
    assert np.allclose(srgb_to_lab([0, 0, 0]), [0, 0, 0], atol=0.1)


@pytest.mark.parametrize("rgb,name", [((250, 250, 250), "white"), ((10, 10, 12), "black"), ((205, 25, 30), "red"),
                                      ((35, 150, 60), "green"), ((245, 130, 25), "orange"), ((25, 30, 95), "navy")])
def test_name_colour(rgb, name):
    assert name_colour(rgb) == name


def test_hsv_and_hex():
    hsv = rgb_to_hsv(np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]], dtype=np.uint8))
    assert np.allclose(hsv[:, 0][:3], [0, 120, 240]) and hsv[3, 1] == 0
    assert to_hex((255, 128, 0)) == "#ff8000"


def test_kmeans_finds_two_halves():
    px = np.array([[255, 0, 0]] * 300 + [[0, 0, 255]] * 100, dtype=np.uint8)
    cols = kmeans_colours(px, k=3)
    assert name_colour(cols[0][0]) == "red" and cols[0][1] == pytest.approx(0.75)
    assert sum(s for _, s in cols) == pytest.approx(1.0)


def test_features_on_flat_and_split_images():
    flat = extract_features(Image.new("RGB", (100, 60), (128, 128, 128)))
    assert flat.orientation == "landscape" and flat.edge_density == 0 and flat.colourfulness < 1
    assert flat.entropy == 0 and len(flat.colours) == 1
    split = Image.new("RGB", (60, 100), (0, 0, 0))
    split.paste((255, 255, 255), (30, 0, 60, 100))
    f = extract_features(split)
    assert f.orientation == "portrait" and f.contrast > 0.45 and f.edge_density > 0


def test_rgba_is_composited_on_white(tmp_path):
    p = tmp_path / "t.png"
    Image.new("RGBA", (40, 40), (0, 0, 0, 0)).save(p)
    assert extract_features(p).brightness > 0.99
