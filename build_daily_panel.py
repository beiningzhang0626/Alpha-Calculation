"""Extract the agreed daily columns from a merged Parquet dataset and clean them.

This tool merges two previously separate steps into one run:

1. Extraction (no data modification)
   Select the agreed daily columns from a merged Parquet dataset. This stage
   performs no cleaning, conversion, sorting, validation, or filtering. It
   preserves the original row order and values, streaming one row group at a
   time so the full (wide) source file never has to be materialized in memory.
   Its only QC output is a CSV listing requested columns absent from the source.

2. Cleaning (rules R1, R2, R3)
   R1. Raw OHLC inconsistency:
       Delete the entire row when high < low, or when open/close is outside
       [low, high].
   R2. Ex-dividend return identity:
       Calculate the return from consecutive adjusted closes within each
       security. If equity_price_return_ex_div differs from the calculated
       return by more than five basis points, set only that column to missing.
       equity_total_return is not tested against adjusted close because it may
       include dividends.
   R3. Price-adjustment identity:
       Delete the entire row when
           adjusted_close * price_adjustment_factor != raw_close
       beyond the configured relative and absolute tolerances.
   The quote-midpoint flag and every other surviving column are preserved.
   The original row order is preserved, except that R1/R3 rows are removed.

Outputs written to --outdir:
  - daily_raw.parquet            (selected columns, unmodified)
  - missing_daily_columns.csv    (requested columns absent from the source)
  - daily_clean.parquet          (R1/R2/R3-cleaned panel)

Example:
  python build_daily_panel.py \
      --input D:\\icarus_alpha_calculation\\all_merged_repaired.parquet \
      --outdir D:\\icarus_alpha_calculation\\daily_data
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Extraction: requested daily columns
# ---------------------------------------------------------------------------

# Keys, metadata, and classification columns.
DAILY_KEYS_META = [
    "date",
    "security_id",
    "ticker",
    "company_id",
    "gics_sector",
    "gics_industry",
    "gics_subindustry",
]

# Security metadata columns.
DAILY_SECURITY_META = [
    "equity_ticker",
    "equity_company_name_pit",
    "equity_share_class_code",
    "equity_exchange_code",
    "equity_sic_code_pit",
    "equity_naics_code_pit",
    "equity_cusip_pit",
]

# Raw daily price and volume columns.
DAILY_PRICE = [
    "equity_open_raw",
    "equity_high_raw",
    "equity_low_raw",
    "equity_close_raw",
    "equity_volume_raw",
    "equity_dollar_volume",
    "equity_vwap_proxy",
    "equity_close_is_quote_midpoint",
]

# Adjusted daily price and volume columns.
DAILY_ADJUSTED_PRICE = [
    "equity_adj_open",
    "equity_adj_high",
    "equity_adj_low",
    "equity_adj_close",
    "equity_adj_volume",
]

# Daily return columns.
DAILY_RETURN = [
    "equity_total_return",
    "equity_price_return_ex_div",
]

# Shares-outstanding and adjustment-factor columns.
DAILY_SHARES_ADJ = [
    "equity_shares_outstanding_k",
    "equity_price_adj_factor",
    "equity_share_adj_factor",
]

# All explicitly requested daily columns.
REQUESTED_DAILY_COLUMNS = (
    DAILY_KEYS_META
    + DAILY_SECURITY_META
    + DAILY_PRICE
    + DAILY_ADJUSTED_PRICE
    + DAILY_RETURN
    + DAILY_SHARES_ADJ
)


# ---------------------------------------------------------------------------
# Cleaning: required columns and tolerances
# ---------------------------------------------------------------------------

# Required columns for the cleaning stage.
REQUIRED_COLUMNS = [
    "date",
    "security_id",
    "equity_open_raw",
    "equity_high_raw",
    "equity_low_raw",
    "equity_close_raw",
    "equity_adj_close",
    "equity_price_adj_factor",
    "equity_price_return_ex_div",
]

# R2: five basis points of absolute return difference.
RETURN_IDENTITY_ATOL = 0.0005

# R3: 0.5% relative tolerance plus a one-cent absolute tolerance.
PRICE_IDENTITY_RTOL = 0.005
PRICE_IDENTITY_ATOL = 0.01


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def find_source_equity_flags(source_columns: list[str]) -> list[str]:
    """Return existing source columns that are equity QC flags."""
    return [
        column
        for column in source_columns
        if column.startswith("equity_") and column.endswith("_flag")
    ]


def keep_existing_columns(
    requested_columns: list[str], source_columns: list[str]
) -> list[str]:
    """Return requested columns that exist, without duplicates."""
    source_set = set(source_columns)
    selected_columns = []

    for column in requested_columns:
        if column in source_set and column not in selected_columns:
            selected_columns.append(column)

    return selected_columns


def write_missing_columns_csv(
    output_path: Path, requested_columns: list[str], source_columns: list[str]
) -> None:
    """Write requested columns absent from the source to a CSV file."""
    source_set = set(source_columns)
    missing_columns = [
        column for column in requested_columns if column not in source_set
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["missing_column"])
        writer.writerows([[column] for column in missing_columns])


def extract_parquet_columns(
    parquet_file: pq.ParquetFile,
    selected_columns: list[str],
    output_path: Path,
) -> None:
    """Copy selected columns to a new Parquet file one row group at a time."""
    source_schema = parquet_file.schema_arrow
    output_schema = pa.schema(
        [source_schema.field(column) for column in selected_columns]
    )

    with pq.ParquetWriter(output_path, output_schema, compression="snappy") as writer:
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(
                row_group_index,
                columns=selected_columns,
            )
            writer.write_table(table)


def extract_daily_columns(input_path: Path, raw_path: Path, missing_path: Path) -> None:
    """Stage 1: select the requested daily columns and write raw + missing outputs."""
    parquet_file = pq.ParquetFile(input_path)
    source_columns = list(parquet_file.schema_arrow.names)

    # Include all existing equity flag columns after the explicitly requested columns.
    source_equity_flags = find_source_equity_flags(source_columns)
    requested_columns = REQUESTED_DAILY_COLUMNS + source_equity_flags
    selected_columns = keep_existing_columns(requested_columns, source_columns)

    # The missing-column CSV is the extraction stage's only QC output.
    write_missing_columns_csv(missing_path, REQUESTED_DAILY_COLUMNS, source_columns)

    # Extract columns without modifying, sorting, or filtering any data rows.
    extract_parquet_columns(parquet_file, selected_columns, raw_path)

    print("=== DAILY EXTRACTION REPORT ===")
    print(f"Source: {input_path}")
    print(f"Source columns: {len(source_columns):,}")
    print(f"Requested daily columns: {len(REQUESTED_DAILY_COLUMNS):,}")
    print(f"Equity flag columns carried over: {len(source_equity_flags):,}")
    print(f"Selected columns written: {len(selected_columns):,}")
    print(f"Raw output: {raw_path}")
    print(f"Missing-column CSV: {missing_path}")


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def finite(series: pd.Series) -> pd.Series:
    """Return True for finite numeric observations."""
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    return pd.Series(np.isfinite(values), index=series.index)


def require_columns(df: pd.DataFrame) -> None:
    """Stop with a clear error if a required column is absent."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_r1_mask(df: pd.DataFrame) -> pd.Series:
    """Return rows with internally inconsistent raw OHLC values."""
    columns = [
        "equity_open_raw",
        "equity_high_raw",
        "equity_low_raw",
        "equity_close_raw",
    ]

    eligible = df[columns].notna().all(axis=1)
    for column in columns:
        eligible &= finite(df[column])

    open_price = df["equity_open_raw"]
    high_price = df["equity_high_raw"]
    low_price = df["equity_low_raw"]
    close_price = df["equity_close_raw"]

    return eligible & (
        (high_price < low_price)
        | (open_price > high_price)
        | (open_price < low_price)
        | (close_price > high_price)
        | (close_price < low_price)
    )


def build_r2_mask(df: pd.DataFrame) -> pd.Series:
    """Return rows whose ex-dividend return violates the adjusted-close identity."""
    ordered = pd.DataFrame(
        {
            "__row_id": np.arange(len(df), dtype=np.int64),
            "security_id": df["security_id"].to_numpy(),
            "__date": pd.to_datetime(df["date"], errors="coerce").to_numpy(),
            "equity_adj_close": df["equity_adj_close"].to_numpy(),
            "equity_price_return_ex_div": df[
                "equity_price_return_ex_div"
            ].to_numpy(),
        }
    )

    # A global trading-date rank prevents returns from being compared across a
    # missing market date for an individual security.
    valid_dates = np.sort(ordered["__date"].dropna().unique())
    date_rank = pd.Series(np.arange(len(valid_dates)), index=valid_dates)
    ordered["__date_rank"] = ordered["__date"].map(date_rank)

    ordered = ordered.sort_values(
        ["security_id", "__date", "__row_id"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    grouped = ordered.groupby("security_id", sort=False)
    previous_adj_close = grouped["equity_adj_close"].shift(1)
    previous_date_rank = grouped["__date_rank"].shift(1)

    consecutive = ordered["__date_rank"] - previous_date_rank == 1
    calculated_return = ordered["equity_adj_close"] / previous_adj_close - 1.0
    reported_return = ordered["equity_price_return_ex_div"]

    eligible = (
        consecutive
        & ordered["equity_adj_close"].notna()
        & previous_adj_close.notna()
        & reported_return.notna()
        & finite(ordered["equity_adj_close"])
        & finite(previous_adj_close)
        & finite(reported_return)
        & (ordered["equity_adj_close"] > 0)
        & (previous_adj_close > 0)
    )

    mismatch_ordered = eligible & (
        (reported_return - calculated_return).abs() > RETURN_IDENTITY_ATOL
    )

    # Map the mask back to the source row order without modifying the source.
    mismatch_source_order = np.zeros(len(df), dtype=bool)
    mismatch_source_order[
        ordered.loc[mismatch_ordered, "__row_id"].to_numpy(dtype=np.int64)
    ] = True
    return pd.Series(mismatch_source_order, index=df.index)


def build_r3_mask(df: pd.DataFrame) -> pd.Series:
    """Return rows that violate the adjusted-price reconstruction identity."""
    required = [
        "equity_adj_close",
        "equity_price_adj_factor",
        "equity_close_raw",
    ]

    eligible = pd.Series(True, index=df.index)
    for column in required:
        eligible &= df[column].notna() & finite(df[column])
    eligible &= df["equity_close_raw"] > 0

    reconstructed_raw = (
        df["equity_adj_close"] * df["equity_price_adj_factor"]
    )
    identity_holds = np.isclose(
        reconstructed_raw.to_numpy(dtype="float64", na_value=np.nan),
        df["equity_close_raw"].to_numpy(dtype="float64", na_value=np.nan),
        rtol=PRICE_IDENTITY_RTOL,
        atol=PRICE_IDENTITY_ATOL,
        equal_nan=False,
    )

    return eligible & ~pd.Series(identity_holds, index=df.index)


def verify_cleaned_data(df: pd.DataFrame) -> None:
    """Verify that R1/R3 violations are gone from the cleaned panel."""
    remaining_r1 = int(build_r1_mask(df).sum())
    remaining_r3 = int(build_r3_mask(df).sum())

    print("\n=== VERIFICATION ===")
    print(f"Remaining R1 rows: {remaining_r1:,}")
    print(f"Remaining R3 rows: {remaining_r3:,}")


def clean_daily_panel(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Stage 2: apply R1/R2/R3 to a daily panel and return the cleaned copy."""
    require_columns(df)

    # Calculate every QC mask before changing the data so rule order cannot
    # change the reported counts.
    mask_r1 = build_r1_mask(df)
    mask_r2 = build_r2_mask(df)
    mask_r3 = build_r3_mask(df)

    delete_mask = mask_r1 | mask_r3
    r1_r3_overlap = mask_r1 & mask_r3
    r2_deleted_by_r1_or_r3 = mask_r2 & delete_mask
    r2_surviving = mask_r2 & ~delete_mask

    print("\n=== DAILY R1/R2/R3 CLEANING REPORT ===")
    print(f"Source: {source_label}")
    print(f"Input rows: {len(df):,}")
    print(f"Input columns: {df.shape[1]:,}")
    print(f"R1 raw-OHLC inconsistent rows: {int(mask_r1.sum()):,}")
    print(f"R2 ex-dividend return-identity violations: {int(mask_r2.sum()):,}")
    print(f"R3 price-adjustment identity violations: {int(mask_r3.sum()):,}")
    print(f"R1/R3 overlapping rows: {int(r1_r3_overlap.sum()):,}")
    print(f"Rows deleted by R1 or R3: {int(delete_mask.sum()):,}")
    print(
        "R2 rows also deleted by R1/R3: "
        f"{int(r2_deleted_by_r1_or_r3.sum()):,}"
    )
    print(
        "Surviving R2 values set to missing: "
        f"{int(r2_surviving.sum()):,}"
    )

    # R1 and R3 remove entire rows. R2 changes only the ex-dividend return.
    cleaned = df.loc[~delete_mask].copy()
    cleaned.loc[r2_surviving, "equity_price_return_ex_div"] = np.nan

    # Preserve the original relative row order while assigning a fresh index.
    cleaned = cleaned.reset_index(drop=True)

    verify_cleaned_data(cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    """Extract the daily columns, then check, clean, verify, and write the panel."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract the agreed daily columns from a merged Parquet file and "
            "clean them with rules R1, R2, and R3."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    raw_path = args.outdir / "daily_raw.parquet"
    missing_path = args.outdir / "missing_daily_columns.csv"
    clean_path = args.outdir / "daily_clean.parquet"

    # Stage 1: extract selected columns (streamed, no modification).
    extract_daily_columns(args.input, raw_path, missing_path)

    # Stage 2: load only the selected columns, then clean.
    df = pd.read_parquet(raw_path)
    cleaned = clean_daily_panel(df, source_label=str(raw_path))
    cleaned.to_parquet(clean_path, index=False)

    print("\n=== OUTPUT ===")
    print(f"Raw output: {raw_path}")
    print(f"Missing-column CSV: {missing_path}")
    print(f"Clean output: {clean_path}")
    print(f"Clean rows: {len(cleaned):,}")
    print(f"Clean columns: {cleaned.shape[1]:,}")
    print(f"Rows removed: {len(df) - len(cleaned):,}")
    print("equity_total_return was preserved.")
    print("The quote-midpoint flag and all other surviving columns were preserved.")


if __name__ == "__main__":
    main()