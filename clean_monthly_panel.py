"""Clean data/monthly_panel.parquet: fill tp_target_price_stdev nulls (single-analyst rows,
where a stdev is mathematically undefined) with 0. No rows are dropped -- internal coverage
gaps (e.g. the Oct 2000 vendor-data hole affecting ~34 securities, or genuine multi-year
lapses like FUNC/GRA/WPC/NAVI) are real absences in the source data, not bad rows to excise,
so every row from monthly_panel.parquet is kept as-is.

Usage: .venv/bin/python3 src/clean_monthly_panel.py
"""

from pathlib import Path

import pandas as pd

SOURCE = Path(__file__).resolve().parent.parent / "data" / "monthly_panel.parquet"
DEST = Path(__file__).resolve().parent.parent / "data" / "monthly_panel_clean.parquet"


def main() -> None:
    df = pd.read_parquet(SOURCE)

    n_null_stdev = df["tp_target_price_stdev"].isna().sum()
    cleaned = df.copy()
    cleaned["tp_target_price_stdev"] = cleaned["tp_target_price_stdev"].fillna(0)

    cleaned.to_parquet(DEST, compression="snappy", index=False)

    print(f"Rows: {len(cleaned)} (unchanged from source, none dropped)")
    print(f"Securities: {cleaned['security_id'].nunique()}")
    print(f"tp_target_price_stdev nulls filled with 0: {n_null_stdev}")
    print(f"Wrote {DEST} -- {len(cleaned)} rows, {cleaned.shape[1]} columns")


if __name__ == "__main__":
    main()
