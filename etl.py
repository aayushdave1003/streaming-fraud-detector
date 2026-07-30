"""Build the curated, columnar dataset from the raw Spotify Charts CSV.

The raw ``charts_combined.csv`` is multi-GB; this reads it in bounded-memory
chunks, keeps only the columns and regions the pipeline needs, drops rows
without a stream count, and writes a compact Parquet file that loads an order
of magnitude faster than re-parsing the CSV every run.

    python etl.py                                   # uses config defaults
    python etl.py --raw charts_combined.csv --out charts_us_global.parquet
    python etl.py --regions "United States" Global "United Kingdom"
"""
from __future__ import annotations

import argparse

import pandas as pd

import config

USECOLS = ["title", "date", "artist", "region", "streams"]


def build(raw: str = config.RAW_CHARTS,
          out: str = config.CURATED_PARQUET,
          regions: list[str] | None = None,
          chunksize: int = 500_000,
          all_regions: bool = False) -> pd.DataFrame:
    """Filter ``raw`` to ``regions`` in chunks and write ``out`` (parquet or csv).

    ``all_regions=True`` keeps every country chart and drops only ``Global`` —
    that row is itself an aggregate of the countries, so keeping it would
    double-count worldwide streams.
    """
    regions = regions or config.REGIONS
    kept: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(
        raw,
        usecols=lambda c: c in USECOLS,
        chunksize=chunksize,
        parse_dates=["date"],
    ):
        total += len(chunk)
        chunk = chunk[chunk["region"] != "Global"] if all_regions else chunk[chunk["region"].isin(regions)]
        chunk = chunk.dropna(subset=["streams"])
        if len(chunk):
            kept.append(chunk)
        print(f"  ...scanned {total:,} rows, kept {sum(len(k) for k in kept):,}", end="\r")

    df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=USECOLS)
    print()
    if out.endswith(".parquet"):
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"✅ Read {total:,} raw rows → {len(df):,} rows for {regions}. Wrote {out}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default=config.RAW_CHARTS)
    p.add_argument("--out", default=config.CURATED_PARQUET)
    p.add_argument("--regions", nargs="+", default=None)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--all-regions", action="store_true",
                   help="Keep every country chart (drop only the double-counting Global aggregate).")
    args = p.parse_args()
    build(args.raw, args.out, args.regions, args.chunksize, args.all_regions)


if __name__ == "__main__":
    main()
