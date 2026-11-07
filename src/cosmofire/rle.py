from __future__ import annotations
import numpy as np
def encode_rle(mask: np.ndarray) -> str:
    """Encode a binary mask row-major, with 1-based starts."""
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1)
    padded = np.pad(pixels, (1, 1), constant_values=0)
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[::2] + 1
    lengths = changes[1::2] - changes[::2]
    return " ".join(f"{s} {length}" for s, length in zip(starts, lengths))
def decode_rle(rle: str | float | None, shape: tuple[int, int]) -> np.ndarray:
    """Decode competition RLE to a uint8 binary mask."""
    result = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        return result.reshape(shape)
    text = str(rle).strip()
    if not text:
        return result.reshape(shape)
    numbers = np.fromstring(text, sep=" ", dtype=np.int64)
    if numbers.size % 2:
        raise ValueError("RLE must contain start-length pairs")
    starts = numbers[0::2] - 1
    lengths = numbers[1::2]
    if np.any(starts < 0) or np.any(lengths <= 0):
        raise ValueError("Invalid RLE values")
    for start, length in zip(starts, lengths):
        end = start + length
        if end > result.size:
            raise ValueError("RLE exceeds mask bounds")
        result[start:end] = 1
    return result.reshape(shape)
