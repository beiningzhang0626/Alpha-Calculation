"""Clean data/monthly_panel.parquet: for each security, keep only its longest run of
consecutive months (dropping shorter, discontinuous fringe segments caused by an internal
coverage gap), and fill tp_target_price_stdev nulls (single-analyst rows, where a stdev is
mathematically undefined) with 0.

See the coverage-gap and null-cell analysis in this session for background: 104/1094 securities
have at least one internal gap in their monthly history (e.g. the Oct 2000 panel-wide data
artifact, or genuine multi-year coverage lapses like FUNC/GRA/WPC/NAVI). Rather than dropping
those securities' rows entirely, each security's history is split at its gap(s) into contiguous
segments and only the longest segment is kept -- so every ticker stays in the panel, just possibly
with a shorter (but internally continuous) history. tp_target_price_stdev is null exactly when
tp_single_analyst_flag is True (5,094 rows) -- filled with 0 here rather than left null.

Usage: .venv/bin/python3 src/clean_monthly_panel.py
"""

from pathlib import Path

import pandas as pd

SOURCE = Path(__file__).resolve().parent.parent / "data" / "monthly_panel.parquet"
DEST = Path(__file__).resolve().parent.parent / "data" / "monthly_panel_clean.parquet"


def longest_contiguous_segment(months: list) -> tuple:
    """Given sorted unique pd.Period months, return the (start, end) of the longest run of
    consecutive months. Ties broken in favor of the most recent segment."""
    segments = []
    seg_start = months[0]
    prev = months[0]
    for m in months[1:]:
        if m == prev + 1:
            prev = m
        else:
            segments.append((seg_start, prev))
            seg_start = m
            prev = m
    segments.append((seg_start, prev))
    return max(segments, key=lambda se: ((se[1] - se[0]).n + 1, se[1]))


def main() -> None:
    df = pd.read_parquet(SOURCE)
    month = pd.to_datetime(df["date"]).dt.to_period("M")

    n_before = len(df)
    n_securities = df["security_id"].nunique()
    n_gapped = 0
    keep_mask = pd.Series(False, index=df.index)

    for sid, idx in df.groupby("security_id").groups.items():
        sec_months = month.loc[idx]
        months = sorted(sec_months.unique())
        start, end = longest_contiguous_segment(months)
        if len(months) < (end - start).n + 1 or len(pd.period_range(months[0], months[-1], freq="M")) > len(months):
            n_gapped += 1
        keep_mask.loc[idx] = (sec_months >= start) & (sec_months <= end)

    cleaned = df[keep_mask].copy()
    cleaned["tp_target_price_stdev"] = cleaned["tp_target_price_stdev"].fillna(0)

    cleaned.to_parquet(DEST, compression="snappy", index=False)

    print(f"Securities with an internal coverage gap: {n_gapped} of {n_securities}")
    print(f"Rows dropped (discontinuous fringe segments): {n_before - len(cleaned)} of {n_before} "
          f"({100*(n_before-len(cleaned))/n_before:.2f}%)")
    print(f"Securities retained in output: {cleaned['security_id'].nunique()} of {n_securities}")
    print(f"tp_target_price_stdev nulls filled with 0: {df['tp_target_price_stdev'].isna().sum()}")
    print(f"Wrote {DEST} -- {len(cleaned)} rows, {cleaned.shape[1]} columns")


if __name__ == "__main__":
    main()
