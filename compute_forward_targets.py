"""Create separate monthly and quarterly forward-return target files.

Outputs
-------
monthly_targets.parquet
    One row per security at each market month-end. target_1m is the cumulative
    total return from that month-end close to the next month-end close.

quarterly_targets.parquet
    One row per security at each March, June, September, and December market
    month-end. target_3m is the cumulative total return from that quarter-end
    close to the next quarter-end close.

Important behavior
------------------
- Targets are not forward-filled.
- Missing or invalid daily returns are not treated as zero.
- A missing security observation inside a holding interval invalidates the
  corresponding target.
- The target files remain separate from all factor files.
- Security identifiers and available ticker/company metadata are preserved.
- Output dates are stored as readable YYYY-MM-DD strings.
- equity_total_return is expected to be in fraction form by default, such as
  0.01 for 1%. Use --return-unit percent only if 1% is stored as 1.0.

Run
---
python compute_monthly_quarterly_targets.py \
    --daily daily_clean.parquet \
    --outdir targets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REQUIRED_COLUMNS = ["security_id", "ticker", "date", "equity_total_return"]

# These metadata columns are retained when they exist in the daily source.
OPTIONAL_IDENTIFIER_COLUMNS = [
    "company_id",
    "equity_ticker",
    "equity_company_name_pit",
]


def finite(series: pd.Series) -> pd.Series:
    """Return True for finite numeric observations."""
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    return pd.Series(np.isfinite(values), index=series.index)


def validate_input(daily: pd.DataFrame) -> None:
    """Validate the minimum schema and the security-date primary key."""
    missing = [column for column in REQUIRED_COLUMNS if column not in daily.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    parsed_date = pd.to_datetime(daily["date"], errors="coerce")
    invalid_dates = int(parsed_date.isna().sum())
    if invalid_dates:
        raise ValueError(f"Found {invalid_dates:,} invalid or missing dates")

    duplicate_rows = int(
        pd.DataFrame(
            {
                "security_id": daily["security_id"],
                "date": parsed_date,
            }
        ).duplicated(["security_id", "date"]).sum()
    )
    if duplicate_rows:
        raise ValueError(
            f"Found {duplicate_rows:,} duplicate (security_id, date) rows"
        )


def build_return_state(
    daily: pd.DataFrame,
    return_unit: str,
    identifier_columns: list[str],
) -> pd.DataFrame:
    """Build cumulative log-return and invalid-increment counters."""
    selected_columns = list(
        dict.fromkeys(identifier_columns + ["date", "equity_total_return"])
    )
    d = daily.loc[:, selected_columns].copy()
    d["date"] = pd.to_datetime(d["date"], errors="raise")
    d = d.sort_values(["security_id", "date"], kind="stable").reset_index(drop=True)

    returns = d["equity_total_return"].astype("float64")
    if return_unit == "percent":
        returns = returns / 100.0

    # Rank the market's observed trading dates. A rank gap within one security
    # indicates a missing observation in that security's daily path.
    market_dates = np.sort(d["date"].unique())
    market_date_rank = pd.Series(np.arange(len(market_dates)), index=market_dates)
    d["__date_rank"] = d["date"].map(market_date_rank)

    previous_rank = d.groupby("security_id", sort=False)["__date_rank"].shift(1)
    consecutive_observation = d["__date_rank"] - previous_rank == 1

    gross_return = 1.0 + returns
    valid_return = returns.notna() & finite(returns) & (gross_return > 0)
    valid_increment = consecutive_observation & valid_return

    # Zero is used only as a neutral internal accumulator. The cumulative
    # invalid counter ensures that any target crossing such an increment is NaN.
    log_increment = pd.Series(0.0, index=d.index)
    log_increment.loc[valid_increment] = np.log1p(returns.loc[valid_increment])
    invalid_increment = (~valid_increment).astype("int64")

    d["__cum_log_return"] = log_increment.groupby(d["security_id"]).cumsum()
    d["__cum_invalid"] = invalid_increment.groupby(d["security_id"]).cumsum()

    print(f"[STATE] market trading dates: {len(market_dates):,}")
    print(f"[STATE] missing daily returns: {int(returns.isna().sum()):,}")
    print(f"[STATE] non-finite daily returns: {int((returns.notna() & ~finite(returns)).sum()):,}")
    print(f"[STATE] daily returns with 1 + r <= 0: {int((gross_return <= 0).sum()):,}")
    print(
        "[STATE] nonconsecutive security observations: "
        f"{int((previous_rank.notna() & ~consecutive_observation).sum()):,}"
    )

    return d


def build_market_month_end_state(
    state: pd.DataFrame,
    identifier_columns: list[str],
) -> pd.DataFrame:
    """Keep security observations that occur on the market's month-end date."""
    d = state.copy()
    d["__month"] = d["date"].dt.to_period("M")

    market_month_end = d.groupby("__month", sort=True)["date"].max()
    d["__market_month_end"] = d["__month"].map(market_month_end)

    month_end_columns = list(
        dict.fromkeys(
            identifier_columns
            + [
                "date",
                "__month",
                "__cum_log_return",
                "__cum_invalid",
            ]
        )
    )
    month_end = d.loc[
        d["date"] == d["__market_month_end"],
        month_end_columns,
    ].copy()
    month_end = month_end.rename(columns={"date": "sample_date"})
    return month_end.reset_index(drop=True)


def build_horizon_targets(
    month_end_state: pd.DataFrame,
    horizon_months: int,
    quarter_end_only: bool,
    identifier_columns: list[str],
) -> pd.DataFrame:
    """Create one horizon's target table from market month-end states."""
    samples = month_end_state.copy()

    if quarter_end_only:
        samples = samples.loc[samples["sample_date"].dt.month.isin([3, 6, 9, 12])]

    target_name = f"target_{horizon_months}m"
    target_end_name = f"target_{horizon_months}m_end_date"

    samples["__target_month"] = samples["__month"] + horizon_months
    samples = samples.rename(
        columns={
            "__cum_log_return": "__start_cum_log",
            "__cum_invalid": "__start_cum_invalid",
        }
    )

    endpoints = month_end_state.loc[
        :,
        [
            "security_id",
            "sample_date",
            "__month",
            "__cum_log_return",
            "__cum_invalid",
        ],
    ].rename(
        columns={
            "sample_date": target_end_name,
            "__month": "__target_month",
            "__cum_log_return": "__end_cum_log",
            "__cum_invalid": "__end_cum_invalid",
        }
    )

    out = samples.merge(
        endpoints,
        on=["security_id", "__target_month"],
        how="left",
        validate="one_to_one",
    )

    invalid_increment_count = (
        out["__end_cum_invalid"] - out["__start_cum_invalid"]
    )
    endpoint_exists = out[target_end_name].notna()
    interval_complete = endpoint_exists & (invalid_increment_count == 0)

    out[target_name] = np.nan
    out.loc[interval_complete, target_name] = np.expm1(
        out.loc[interval_complete, "__end_cum_log"]
        - out.loc[interval_complete, "__start_cum_log"]
    )

    output_identifier_columns = list(dict.fromkeys(identifier_columns))
    result = out.loc[
        :,
        output_identifier_columns
        + ["sample_date", target_end_name, target_name],
    ].sort_values(["sample_date", "security_id"], kind="stable")

    # Use an intuitive date column and readable ISO date strings in the target
    # files instead of viewer-dependent integer timestamp representations.
    result = result.rename(columns={"sample_date": "date"})
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result[target_end_name] = result[target_end_name].dt.strftime("%Y-%m-%d")

    missing_endpoint = int((~endpoint_exists).sum())
    invalid_interval = int((endpoint_exists & ~interval_complete).sum())
    valid_targets = int(result[target_name].notna().sum())

    print(f"\n[{target_name}] rows: {len(result):,}")
    print(f"[{target_name}] valid targets: {valid_targets:,}")
    print(f"[{target_name}] missing target endpoints: {missing_endpoint:,}")
    print(f"[{target_name}] intervals containing invalid/missing increments: {invalid_interval:,}")

    target_values = result[target_name].dropna()
    if not target_values.empty:
        print(
            f"[{target_name}] mean={target_values.mean():.6f}, "
            f"median={target_values.median():.6f}, "
            f"p01={target_values.quantile(0.01):.6f}, "
            f"p99={target_values.quantile(0.99):.6f}, "
            f"min={target_values.min():.6f}, "
            f"max={target_values.max():.6f}"
        )

    return result.reset_index(drop=True)


def main() -> None:
    """Load daily returns and write independent monthly and quarterly targets."""
    parser = argparse.ArgumentParser(
        description="Create separate month-end and quarter-end return targets."
    )
    parser.add_argument("--daily", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--return-unit",
        choices=["fraction", "percent"],
        default="fraction",
        help="Use fraction when 1%% is stored as 0.01; use percent when stored as 1.0.",
    )
    args = parser.parse_args()

    print("=== FORWARD TARGET GENERATION ===")
    print(f"Daily source: {args.daily}")
    print(f"Return unit: {args.return_unit}")

    source_columns = list(pq.read_schema(args.daily).names)
    read_columns = REQUIRED_COLUMNS + [
        column
        for column in OPTIONAL_IDENTIFIER_COLUMNS
        if column in source_columns
    ]
    daily = pd.read_parquet(args.daily, columns=read_columns)
    print(f"Daily rows: {len(daily):,}")

    validate_input(daily)
    identifier_columns = ["security_id", "ticker"] + [
        column
        for column in OPTIONAL_IDENTIFIER_COLUMNS
        if column in daily.columns
    ]

    state = build_return_state(daily, args.return_unit, identifier_columns)
    month_end_state = build_market_month_end_state(state, identifier_columns)

    monthly_targets = build_horizon_targets(
        month_end_state,
        horizon_months=1,
        quarter_end_only=False,
        identifier_columns=identifier_columns,
    )
    quarterly_targets = build_horizon_targets(
        month_end_state,
        horizon_months=3,
        quarter_end_only=True,
        identifier_columns=identifier_columns,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    monthly_path = args.outdir / "monthly_targets.parquet"
    quarterly_path = args.outdir / "quarterly_targets.parquet"

    monthly_targets.to_parquet(monthly_path, index=False)
    quarterly_targets.to_parquet(quarterly_path, index=False)

    print("\n=== OUTPUT ===")
    print(f"[WRITE] {monthly_path}")
    print(f"[WRITE] {quarterly_path}")
    print("Targets were not merged into any factor file and were not forward-filled.")


if __name__ == "__main__":
    main()