from pathlib import Path

import pandas as pd


DAILY_PATH = Path(
    r"D:\icarus_alpha_calculation\data_separation\daily_clean.parquet"
)

MONTHLY_PATH = Path(
    r"D:\icarus_alpha_calculation\data_separation\monthly_panel.parquet"
)

QUARTERLY_PATH = Path(
    r"D:\icarus_alpha_calculation\data_separation\fmt_quarterly_panel.parquet"
)


# =========================
# 1. Daily panel
# =========================
daily = pd.read_parquet(DAILY_PATH)

daily["date"] = (
    pd.to_datetime(
        daily["date"],
        errors="raise",
        utc=True,
    )
    .dt.tz_convert(None)
    .dt.normalize()
)

daily.to_parquet(DAILY_PATH, index=False)


# =========================
# 2. Monthly panel
# =========================
monthly = pd.read_parquet(MONTHLY_PATH)

monthly["date"] = (
    pd.to_datetime(
        monthly["date"],
        errors="raise",
        utc=True,
    )
    .dt.tz_convert(None)
    .dt.normalize()
)

monthly.to_parquet(MONTHLY_PATH, index=False)


# =========================
# 3. Quarterly panel
# =========================
quarterly = pd.read_parquet(QUARTERLY_PATH)

# Explicitly rename rebalance_date to date.
quarterly = quarterly.rename(
    columns={"rebalance_date": "date"}
)

# rebalance_date was originally stored as a Unix timestamp in milliseconds.
quarterly["date"] = (
    pd.to_datetime(
        quarterly["date"],
        unit="ms",
        errors="raise",
        utc=True,
    )
    .dt.tz_convert(None)
    .dt.normalize()
)

quarterly.to_parquet(QUARTERLY_PATH, index=False)


# =========================
# Verification
# =========================
for name, path in {
    "daily": DAILY_PATH,
    "monthly": MONTHLY_PATH,
    "quarterly": QUARTERLY_PATH,
}.items():
    df = pd.read_parquet(path)

    print(f"\n{name}")
    print("date column exists:", "date" in df.columns)
    print(
        "rebalance_date column exists:",
        "rebalance_date" in df.columns,
    )
    print("date dtype:", df["date"].dtype)
    print("date examples:")
    print(df["date"].head())


print("\nAll three files have been standardized successfully.")

