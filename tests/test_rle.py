
import numpy as np
from cosmofire.rle import decode_rle, encode_rle
def test_rle_roundtrip():
    mask = np.zeros((7, 9), dtype=np.uint8)
    mask[0, :3] = 1
    mask[2:5, 4] = 1
    mask[-1, -2:] = 1
    encoded = encode_rle(mask)
    np.testing.assert_array_equal(decode_rle(encoded, mask.shape), mask)
def test_empty_rle():
    mask = np.zeros((3, 4), dtype=np.uint8)
    assert encode_rle(mask) == ""
    np.testing.assert_array_equal(decode_rle("", mask.shape), mask)
