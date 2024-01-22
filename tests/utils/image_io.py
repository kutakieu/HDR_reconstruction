import tempfile

import numpy as np

from reconsthdr.utils import load_bgr_hdr, load_rgb_hdr, save_hdr


def test_hdr_is_rgb_format():
    hdr = load_rgb_hdr("tests/data/lonely_road_afternoon_2k.hdr")
    assert hdr[0,0,0] < hdr[0,0,2]
    assert hdr[0,0,1] < hdr[0,0,2]

def test_load_bgr_hdr():
    hdr = load_bgr_hdr("tests/data/lonely_road_afternoon_2k.hdr")
    assert hdr[0,0,1] < hdr[0,0,0]
    assert hdr[0,0,2] < hdr[0,0,0]

def test_save_hdr_in_rgb_format():
    hdr = load_rgb_hdr("tests/data/lonely_road_afternoon_2k.hdr")
    with tempfile.TemporaryDirectory() as dname:
        save_hdr(f"{dname}/saved.hdr", hdr)
        hdr_saved = load_rgb_hdr(f"{dname}/saved.hdr")
    assert np.allclose(hdr, hdr_saved)
