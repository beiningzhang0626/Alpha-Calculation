"""
factor_utils.py

Shared factor utility functions for daily, monthly, and quarterly panels.

Conventions:
- by: entity key, default 'security_id'
- order: time-order column
- All functions return Series aligned with the original DataFrame index
"""

import numpy as np
import pandas as pd


# ======================================================================
# 1. Generic grouped shift
# ======================================================================

def group_shift(df, col, periods, by='security_id', order='date'):
    """Shift a column within each entity after sorting by time."""
    d = df.sort_values([by, order])
    return (
        d.groupby(by, sort=False)[col]
        .shift(periods)
        .reindex(df.index)
    )


# ======================================================================
# 2. Quarterly lags and TTM
# ======================================================================

def lag_nq(df, col, n, by='security_id', order='rebalance_date'):
    """Value from n quarterly rows ago."""
    return group_shift(df, col, n, by, order)


def lag_1q(df, col, by='security_id', order='rebalance_date'):
    """Previous-quarter value."""
    return group_shift(df, col, 1, by, order)


def lag_4q(df, col, by='security_id', order='rebalance_date'):
    """Value from four quarters ago."""
    return group_shift(df, col, 4, by, order)


def rolling_sum_4q(
    df,
    col,
    by='security_id',
    order='rebalance_date',
    min_valid=3
):
    """Trailing four-quarter sum, allowing one missing quarter."""
    d = df.sort_values([by, order])
    g = d.groupby(by, sort=False)

    result = (
        g[col]
        .rolling(4, min_periods=min_valid)
        .mean()
        .reset_index(level=0, drop=True)
        * 4
    )

    result = result.where(g.cumcount() >= 3)
    return result.reindex(df.index)


ttm = rolling_sum_4q


# ======================================================================
# 3. Monthly lags
# ======================================================================

def lag_nm(df, col, n, by='security_id', order='date'):
    """Value from n monthly rows ago."""
    return group_shift(df, col, n, by, order)


def lag_1m(df, col, by='security_id', order='date'):
    """Previous-month value."""
    return group_shift(df, col, 1, by, order)


def lag_3m(df, col, by='security_id', order='date'):
    """Value from three months ago."""
    return group_shift(df, col, 3, by, order)


# ======================================================================
# 4. Daily lags
# ======================================================================

def lag_21d(df, col, by='security_id', order='date'):
    """Value from 21 trading rows ago."""
    return group_shift(df, col, 21, by, order)


def lag_63d(df, col, by='security_id', order='date'):
    """Value from 63 trading rows ago."""
    return group_shift(df, col, 63, by, order)


# ======================================================================
# 5. Derived variables
# ======================================================================

def market_cap_mn(
    df,
    close='equity_close_raw',
    shares_k='equity_shares_outstanding_k'
):
    """Market capitalization in millions."""
    return df[close] * df[shares_k] / 1000.0


def avg_total_assets(df, by='security_id', order='rebalance_date'):
    return 0.5 * (
        df['total_assets']
        + lag_4q(df, 'total_assets', by, order)
    )


def avg_book_equity(df, by='security_id', order='rebalance_date'):
    return 0.5 * (
        df['book_equity']
        + lag_4q(df, 'book_equity', by, order)
    )


def avg_inventory(df, by='security_id', order='rebalance_date'):
    return 0.5 * (
        df['inventory']
        + lag_4q(df, 'inventory', by, order)
    )


# ======================================================================
# 6. Safe division
# ======================================================================

def safe_divide(num, den):
    """Return num / den, with invalid or zero denominators set to NaN."""
    num = pd.Series(
        np.asarray(num, dtype='float64'),
        index=num.index if isinstance(num, pd.Series) else None
    )

    den = pd.Series(
        np.asarray(den, dtype='float64'),
        index=den.index if isinstance(den, pd.Series) else None
    )

    den = den.where(den != 0, np.nan)
    return num / den


# ======================================================================
# 7. Cross-sectional scoring
# ======================================================================

def winsorize(s, lower=0.01, upper=0.99):
    """Clip a Series at the specified quantiles."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def _xs_score(s):
    return winsorize(s).rank(pct=True) - 0.5


def score(df, col, by='rebalance_date'):
    """Cross-sectional percentile score; higher values receive higher scores."""
    return df.groupby(by, sort=False)[col].transform(_xs_score)


def score_low(df, col, by='rebalance_date'):
    """Cross-sectional percentile score where lower values are better."""
    neg = -df[col]
    return neg.groupby(df[by], sort=False).transform(_xs_score)


# ======================================================================
# 8. Stock-level time-series normalization
# ======================================================================

def _rolling_history_rank(s, window, min_valid):
    """Rank each value against the previous window observations."""
    values = pd.to_numeric(s, errors='coerce').to_numpy(dtype='float64')
    result = np.full(len(values), np.nan, dtype='float64')

    for i, current in enumerate(values):
        if not np.isfinite(current):
            continue

        start = max(0, i - window)
        history = values[start:i]
        history = history[np.isfinite(history)]

        if len(history) < min_valid:
            continue

        if np.unique(history).size < 2:
            continue

        below = np.sum(history < current)
        equal = np.sum(history == current)

        percentile = (below + 0.5 * equal) / len(history)
        result[i] = percentile - 0.5

    return pd.Series(result, index=s.index)


def rolling_ts_rank(
    df,
    col,
    window,
    min_valid,
    by='security_id',
    order='date'
):
    """
    Centered historical percentile rank in [-0.5, 0.5].

    The current value is compared with previous observations only.
    """
    d = df.sort_values([by, order])

    result = (
        d.groupby(by, sort=False, group_keys=False)[col]
        .apply(
            lambda s: _rolling_history_rank(
                s,
                window=window,
                min_valid=min_valid
            )
        )
    )

    return result.reindex(df.index)


def ts_rank_quarterly(
    df,
    col,
    window=20,
    min_valid=12,
    by='security_id',
    order='rebalance_date'
):
    """Five-year rolling rank for quarterly factors."""
    return rolling_ts_rank(
        df=df,
        col=col,
        window=window,
        min_valid=min_valid,
        by=by,
        order=order
    )


def ts_rank_monthly(
    df,
    col,
    window=60,
    min_valid=36,
    by='security_id',
    order='date'
):
    """Five-year rolling rank for monthly or month-end sampled factors."""
    return rolling_ts_rank(
        df=df,
        col=col,
        window=window,
        min_valid=min_valid,
        by=by,
        order=order
    )
