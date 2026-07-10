
"""Build the monthly-grained panel from data/all_merged_repaired.parquet.

Column selection is the verified list from ts_alpha_features.MONTHLY_COLUMNS (see that module and
.claude/skills/ts-feature-reference/SKILL.md for how it was derived). Row grain: one row per
security per calendar month, taken from that month's last trading day in the daily PIT panel, per
spec section 3 ("Take the last trading day of each month from the PIT daily panel for
recommendation and target-price variables").

Usage: .venv/bin/python3 src/build_monthly_panel.py
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


SOURCE = Path(r"D:\icarus_alpha_calculation\all_merged_repaired.parquet")
DEST = Path(r"D:\icarus_alpha_calculation\data_separation\monthly_panel.parquet")

KEY_COLUMNS = ["security_id", "ticker", "date"]
DATE_COLUMNS = ["date", "tp_statistic_date"]  # cast timestamp -> date32 for viewer compatibility

MONTHLY_RECOMMENDATION_COLUMNS = [
    "rec_mean",
    "rec_median",
    "rec_stdev",
    "rec_count",
    "rec_buypct",
    "rec_holdpct",
    "rec_sellpct",
    "rec_unanimous_coverage_flag",
    "tp_statistic_date",
    "tp_analyst_count",
    "tp_target_price_mean",
    "tp_target_price_median",
    "tp_target_price_high",
    "tp_target_price_low",
    "tp_target_price_stdev",
    "tp_single_analyst_flag",
    "tp_zero_price_flag",
]

MONTHLY_MACRO_COLUMNS = [
    "macro_cpi_index",
    "macro_cpi_yoy",
    "macro_unemployment_rate",
    "macro_industrial_production",
    "macro_manuf_employment_k",
]

MONTHLY_COLUMNS = MONTHLY_RECOMMENDATION_COLUMNS + MONTHLY_MACRO_COLUMNS


def main() -> None:
    dataset = ds.dataset(SOURCE)
    columns = KEY_COLUMNS + MONTHLY_COLUMNS
    df = dataset.to_table(columns=columns).to_pandas()

    df = df.sort_values(["security_id", "date"])
    month_key = df["date"].dt.to_period("M")
    monthly = df.groupby(
        ["security_id", month_key],
        sort=False,
        as_index=False,
    ).tail(1)
    monthly = monthly.sort_values(
        ["security_id", "date"]
    ).reset_index(drop=True)

    table = pa.Table.from_pandas(monthly, preserve_index=False)
    for col in DATE_COLUMNS:
        idx = table.column_names.index(col)
        table = table.set_column(
            idx,
            col,
            table.column(col).cast(pa.date32()),
        )

    pq.write_table(table, DEST, compression="snappy")
    print(
        f"Wrote {DEST} — "
        f"{table.num_rows} rows, {table.num_columns} columns"
    )


if __name__ == "__main__":
    main()

