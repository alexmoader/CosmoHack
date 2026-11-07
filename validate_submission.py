from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cosmofire.rle import decode_rle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict submission validator")
    parser.add_argument("--submission", type=Path, default=ROOT / "submissions" / "submission.csv")
    parser.add_argument("--sample", type=Path, default=ROOT / "data" / "test" / "sample_submission.csv")
    parser.add_argument("--meta", type=Path, default=ROOT / "data" / "test" / "meta.csv")
    args = parser.parse_args()

    if not args.submission.exists():
        raise FileNotFoundError(f"Submission file does not exist: {args.submission}")
    if not args.sample.exists():
        raise FileNotFoundError(f"Sample file does not exist: {args.sample}")
    if not args.meta.exists():
        raise FileNotFoundError(f"Meta file does not exist: {args.meta}")

    submission = pd.read_csv(args.submission, keep_default_na=False)
    sample = pd.read_csv(args.sample, keep_default_na=False)
    meta = pd.read_csv(args.meta).set_index("chip_id")

    required = ["chip_id", "class_id", "rle"]
    if list(submission.columns) != required:
        raise ValueError(f"Columns must be exactly {required}, got {list(submission.columns)}")

    if len(submission) != len(sample):
        raise ValueError(f"Expected {len(sample)} rows, got {len(submission)}")

    expected = list(zip(sample.chip_id, sample.class_id.astype(int)))
    actual = list(zip(submission.chip_id, submission.class_id.astype(int)))
    if actual != expected:
        raise ValueError("(chip_id, class_id) rows/order differ from sample_submission.csv")

    if submission.duplicated(["chip_id", "class_id"]).any():
        raise ValueError("Duplicate chip/class pairs")

    for chip_id, group in submission.groupby("chip_id", sort=False):
        row = meta.loc[chip_id]
        shape = (int(row.height), int(row.width))
        occupied = np.zeros(shape, dtype=bool)
        for record in group.itertuples(index=False):
            mask = decode_rle(record.rle, shape).astype(bool)
            if np.any(occupied & mask):
                raise ValueError(f"Overlapping classes in {chip_id}")
            occupied |= mask

    print(f"VALID: {args.submission} ({len(submission)} rows, {submission.chip_id.nunique()} unique chips)")


if __name__ == "__main__":
    main()
