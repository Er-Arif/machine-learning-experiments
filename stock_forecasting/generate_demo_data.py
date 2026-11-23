"""Create explicitly synthetic CSV for testing the pipeline, not for investment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo_synthetic.csv"))
    parser.add_argument("--days", type=int, default=400)
    args = parser.parse_args()
    if args.days < 50:
        parser.error("--days must be at least 50")
    rng = np.random.default_rng(42)
    synthetic_close = 100 + np.cumsum(rng.normal(0.05, 1.1, args.days))
    dates = pd.bdate_range("2020-01-01", periods=args.days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Date": dates, "Close": synthetic_close}).to_csv(
        args.output, index=False
    )
    print(f"Saved SYNTHETIC demonstration data: {args.output}")


if __name__ == "__main__":
    main()
