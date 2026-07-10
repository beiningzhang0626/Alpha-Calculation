"""Time-series ML alpha model formulas.

Implements every formula from `lit/Normalization Instruction.pdf` ("Time-Series ML Alpha Model -
Intern Formula Checklist"). See `.claude/skills/ts-feature-reference/SKILL.md` for a formula ->
function lookup table.

Convention: all functions operate on a single firm's time series — a `pd.Series` indexed by date
and sorted ascending (or by fiscal quarter, ascending, for the section-6 fundamental helpers).
Apply per-firm, e.g. `df.groupby("firm_id")["price"].apply(lambda s: momentum(s, 20))`.

Raw rolling features (sections 1-3) use windows that include the current observation `t`, since
those quantities are contemporaneously known. Normalization statistics (section 4) and target
lookback statistics (section 7) instead use the prior window `[t-L, t-1]`, excluding `t` — this
distinction is deliberate and preserved throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Core setup constants
# ---------------------------------------------------------------------------

FEATURE_WINDOWS = (5, 10, 20, 30, 63, 126, 252)
NORM_WINDOWS = (63, 126, 252)
DEFAULT_HORIZON = 5  # weekly horizon, in trading days
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Monthly-cadence columns in data/all_merged_repaired.parquet
# ---------------------------------------------------------------------------
# Verified empirically: for the securities with the longest history, found each column's
# change-points (dates where its value actually differs from the prior row) and measured the
# trading-day gap between consecutive changes. Genuine monthly series cluster tightly around a
# ~20-44 row-gap; quarterly accounting fields cluster at ~63; daily fields at ~1.
#
# MONTHLY_RECOMMENDATION_COLUMNS = the analyst recommendation / target-price snapshot panel from
# spec section 2.4 (lit/fundamental_factor_calculation_plan_simplified_workflow_fixed.pdf). All
# 14 of section 2.4's columns confirmed at ~20-44 row-gap. Also includes 3 companion QC flags
# that section 2.4's table omits but that section 2.7 assigns to the same monthly rec/tp pattern
# (`rec_unanimous_coverage_flag`, `tp_single_analyst_flag`, `tp_zero_price_flag`) — these rarely
# flip so they look "rare" rather than "monthly" under a naive gap measurement, but they are
# recomputed every month alongside their parent variables and belong on the same monthly panel.
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

# MONTHLY_MACRO_COLUMNS = macro series that empirically update ~monthly (~21 row-gap), matching
# their real-world release cadence (CPI, unemployment, industrial production, manufacturing
# employment are all monthly government releases). Not listed anywhere in the fundamental-factor
# spec's raw-data tables (section 2) — that's an omission in that PDF, confirmed here from data.
# The other 3 macro_* columns (macro_vix, macro_ig_credit_spread_longhist,
# macro_hy_credit_spread_proxy_longhist) update ~daily and are NOT included here.
MONTHLY_MACRO_COLUMNS = [
    "macro_cpi_index",
    "macro_cpi_yoy",
    "macro_unemployment_rate",
    "macro_industrial_production",
    "macro_manuf_employment_k",
]

# Combined monthly column list for convenience. Deliberately excludes: consensus_* fields (several
# empirically show ~20-30 day gaps, but the spec explicitly treats them as quarterly-native for
# base-panel construction — that design choice is preserved here rather than overridden);
# equity_shares_outstanding_k (gap percentiles [19, 22, 43, 64, 80] — an irregular cadence
# uncorrelated with quarterly filings, not a clean monthly series); and
# sector_returns_constituent_count / industry_returns_industry_constituent_count (event-driven
# universe-membership counts with extreme long-tailed gaps and no day-of-month clustering — not
# periodic at all, let alone monthly).
MONTHLY_COLUMNS = MONTHLY_RECOMMENDATION_COLUMNS + MONTHLY_MACRO_COLUMNS


# ---------------------------------------------------------------------------
# 1. Price and return features
# ---------------------------------------------------------------------------

def log_price(price: pd.Series) -> pd.Series:
    """p_t = ln(P_t)"""
    return np.log(price)


def log_return(price: pd.Series) -> pd.Series:
    """r_t = ln(P_t / P_{t-1})"""
    return np.log(price / price.shift(1))


def momentum(price: pd.Series, n: int) -> pd.Series:
    """mom_N,t = ln(P_t / P_{t-N})"""
    return np.log(price / price.shift(n))


def simple_return(price: pd.Series, n: int) -> pd.Series:
    """R_N,t = P_t / P_{t-N} - 1"""
    return price / price.shift(n) - 1


def moving_average(price: pd.Series, n: int) -> pd.Series:
    """MA_N,t = mean(P_{t-N+1:t})"""
    return price.rolling(n).mean()


def moving_average_log_price(price: pd.Series, n: int) -> pd.Series:
    """MAlog_N,t = mean(p_{t-N+1:t})"""
    return log_price(price).rolling(n).mean()


def price_distance_from_ma(price: pd.Series, n: int) -> pd.Series:
    """dist_N,t = ln(P_t / MA_N,t)"""
    return np.log(price / moving_average(price, n))


def log_price_zscore(price: pd.Series, n: int) -> pd.Series:
    """z_price_N,t = (p_t - mean(p_{t-N+1:t})) / std(p_{t-N+1:t})"""
    p = log_price(price)
    return (p - p.rolling(n).mean()) / p.rolling(n).std()


def return_volatility(price: pd.Series, n: int) -> pd.Series:
    """vol_N,t = std(r_{t-N+1:t})"""
    return log_return(price).rolling(n).std()


def vol_normalized_ma_distance(price: pd.Series, n: int) -> pd.Series:
    """dist_vol_N,t = ln(P_t / MA_N,t) / vol_N,t"""
    return price_distance_from_ma(price, n) / return_volatility(price, n)


def high_proximity(price: pd.Series, n: int) -> pd.Series:
    """highprox_N,t = ln(P_t / max(P_{t-N+1:t}))"""
    return np.log(price / price.rolling(n).max())


def low_proximity(price: pd.Series, n: int) -> pd.Series:
    """lowprox_N,t = ln(P_t / min(P_{t-N+1:t}))"""
    return np.log(price / price.rolling(n).min())


def ma_crossover(price: pd.Series, n_short: int, n_long: int) -> pd.Series:
    """ma_cross_short_long,t = ln(MA_short,t / MA_long,t)"""
    return np.log(moving_average(price, n_short) / moving_average(price, n_long))


# ---------------------------------------------------------------------------
# 2. Volume and liquidity features
# ---------------------------------------------------------------------------

def log_volume(volume: pd.Series) -> pd.Series:
    """lv_t = ln(1 + V_t)"""
    return np.log1p(volume)


def average_volume(volume: pd.Series, n: int) -> pd.Series:
    """AvgVol_N,t = mean(V_{t-N+1:t})"""
    return volume.rolling(n).mean()


def relative_volume(volume: pd.Series, n: int) -> pd.Series:
    """relvol_N,t = ln(V_t / AvgVol_N,t)"""
    return np.log(volume / average_volume(volume, n))


def dollar_volume(price: pd.Series, volume: pd.Series) -> pd.Series:
    """DV_t = P_t * V_t"""
    return price * volume


def log_dollar_volume(price: pd.Series, volume: pd.Series) -> pd.Series:
    """ldv_t = ln(1 + DV_t)"""
    return np.log1p(dollar_volume(price, volume))


def average_dollar_volume(price: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    """AvgDV_N,t = mean(DV_{t-N+1:t})"""
    return dollar_volume(price, volume).rolling(n).mean()


def relative_dollar_volume(price: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    """relDV_N,t = ln(DV_t / AvgDV_N,t)"""
    return np.log(dollar_volume(price, volume) / average_dollar_volume(price, volume, n))


def turnover(volume: pd.Series, shares_outstanding: pd.Series) -> pd.Series:
    """TO_t = V_t / shares_outstanding_t"""
    return volume / shares_outstanding


def log_turnover(volume: pd.Series, shares_outstanding: pd.Series) -> pd.Series:
    """lto_t = ln(1 + TO_t)"""
    return np.log1p(turnover(volume, shares_outstanding))


def turnover_change(volume: pd.Series, shares_outstanding: pd.Series, n: int) -> pd.Series:
    """dTO_N,t = TO_t / mean(TO_{t-N+1:t}) - 1"""
    to = turnover(volume, shares_outstanding)
    return to / to.rolling(n).mean() - 1


def volume_price_confirmation(price: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    """vp_mom_N,t = mom_N,t * relvol_N,t"""
    return momentum(price, n) * relative_volume(volume, n)


# ---------------------------------------------------------------------------
# 3. Volatility, range, and risk features
# ---------------------------------------------------------------------------

def annualized_volatility(price: pd.Series, n: int, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """ann_vol_N,t = vol_N,t * sqrt(252)"""
    return return_volatility(price, n) * np.sqrt(periods_per_year)


def downside_volatility(price: pd.Series, n: int) -> pd.Series:
    """downvol_N,t = std(min(r_tau, 0)) for tau in [t-N+1, t]"""
    downside = log_return(price).clip(upper=0)
    return downside.rolling(n).std()


def high_low_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """range_t = ln(H_t / L_t)"""
    return np.log(high / low)


def average_range(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    """avg_range_N,t = mean(range_{t-N+1:t})"""
    return high_low_range(high, low).rolling(n).mean()


def close_open_return(close: pd.Series, open_: pd.Series) -> pd.Series:
    """co_t = ln(C_t / O_t)"""
    return np.log(close / open_)


def open_close_gap(open_: pd.Series, close: pd.Series) -> pd.Series:
    """gap_t = ln(O_t / C_{t-1})"""
    return np.log(open_ / close.shift(1))


def rolling_skew(price: pd.Series, n: int) -> pd.Series:
    """skew_N,t = skew(r_{t-N+1:t})"""
    return log_return(price).rolling(n).skew()


def rolling_kurtosis(price: pd.Series, n: int) -> pd.Series:
    """kurt_N,t = kurtosis(r_{t-N+1:t})"""
    return log_return(price).rolling(n).kurt()


# ---------------------------------------------------------------------------
# 4. General feature normalization
# ---------------------------------------------------------------------------
# Statistics here use the prior window [t-L, t-1], excluding t: hist = x.shift(1).

def rolling_z(x: pd.Series, l: int) -> pd.Series:
    """x_z,t = (x_raw,t - mu_x,t) / sigma_x,t, with mu/sigma over {t-L:t-1}"""
    hist = x.shift(1)
    mu = hist.rolling(l).mean()
    sigma = hist.rolling(l).std()
    return (x - mu) / sigma


def clip_series(x: pd.Series, lower: float = -5, upper: float = 5) -> pd.Series:
    """x_norm,t = clip(x_z,t, -5, 5)"""
    return x.clip(lower=lower, upper=upper)


def robust_center(x: pd.Series, l: int) -> pd.Series:
    """med_x,t = median(x_raw,{t-L:t-1})"""
    return x.shift(1).rolling(l).median()


def robust_scale_mad(x: pd.Series, l: int) -> pd.Series:
    """MAD_x,t = median(|x_raw,tau - med_x,t|) for tau in [t-L, t-1]"""
    hist = x.shift(1)
    return hist.rolling(l).apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)


def robust_zscore(x: pd.Series, l: int) -> pd.Series:
    """x_robust_z,t = (x_raw,t - med_x,t) / (1.4826 * MAD_x,t)"""
    return (x - robust_center(x, l)) / (1.4826 * robust_scale_mad(x, l))


def rolling_minmax(x: pd.Series, l: int) -> pd.Series:
    """x_minmax,t = (x_raw,t - min(x_raw,{t-L:t-1})) / (max(x_raw,{t-L:t-1}) - min(x_raw,{t-L:t-1}))"""
    hist = x.shift(1)
    lo = hist.rolling(l).min()
    hi = hist.rolling(l).max()
    return (x - lo) / (hi - lo)


def ewma_mean_var(x: pd.Series, alpha: float) -> tuple[pd.Series, pd.Series]:
    """ew_mu_t = alpha*x_raw,t-1 + (1-alpha)*ew_mu_t-1
    ew_var_t = alpha*(x_raw,t-1 - ew_mu_t)^2 + (1-alpha)*ew_var_t-1

    Recursive, so computed iteratively rather than vectorized. Seeds the recursion at the first
    non-null lagged observation (ew_mu = that value, ew_var = 0).
    """
    x_lag = x.shift(1).to_numpy()
    n = len(x)
    ew_mu = np.full(n, np.nan)
    ew_var = np.full(n, np.nan)

    valid = np.flatnonzero(~np.isnan(x_lag))
    if valid.size == 0:
        return pd.Series(ew_mu, index=x.index), pd.Series(ew_var, index=x.index)

    start = valid[0]
    ew_mu[start] = x_lag[start]
    ew_var[start] = 0.0
    for i in range(start + 1, n):
        if np.isnan(x_lag[i]):
            ew_mu[i] = ew_mu[i - 1]
            ew_var[i] = ew_var[i - 1]
            continue
        ew_mu[i] = alpha * x_lag[i] + (1 - alpha) * ew_mu[i - 1]
        ew_var[i] = alpha * (x_lag[i] - ew_mu[i]) ** 2 + (1 - alpha) * ew_var[i - 1]

    return pd.Series(ew_mu, index=x.index), pd.Series(ew_var, index=x.index)


def ewma_zscore(x: pd.Series, alpha: float) -> pd.Series:
    """x_ewz,t = (x_raw,t - ew_mu_t) / sqrt(ew_var_t)"""
    ew_mu, ew_var = ewma_mean_var(x, alpha)
    return (x - ew_mu) / np.sqrt(ew_var)


# ---------------------------------------------------------------------------
# 5. Suggested normalization by feature type
# ---------------------------------------------------------------------------
# Each wraps a raw formula from sections 1-3 with rolling_z(raw, L).

def feature_price_trend(price: pd.Series, n: int, l: int) -> pd.Series:
    return rolling_z(price_distance_from_ma(price, n), l)


def feature_momentum(price: pd.Series, n: int, l: int) -> pd.Series:
    return rolling_z(momentum(price, n), l)


def feature_price_overextension(price: pd.Series, n: int, l: int) -> pd.Series:
    return rolling_z(vol_normalized_ma_distance(price, n), l)


def feature_volume_shock(volume: pd.Series, n: int, l: int) -> pd.Series:
    return rolling_z(relative_volume(volume, n), l)


def feature_dollar_volume_shock(price: pd.Series, volume: pd.Series, n: int, l: int) -> pd.Series:
    return rolling_z(relative_dollar_volume(price, volume, n), l)


def feature_turnover_shock(volume: pd.Series, shares_outstanding: pd.Series, n: int, l: int) -> pd.Series:
    return rolling_z(turnover_change(volume, shares_outstanding, n), l)


def feature_volatility_regime(price: pd.Series, n: int, l: int) -> pd.Series:
    return rolling_z(np.log(return_volatility(price, n)), l)


def feature_range_regime(high: pd.Series, low: pd.Series, n: int, l: int) -> pd.Series:
    return rolling_z(np.log(average_range(high, low, n)), l)


def feature_gap(open_: pd.Series, close: pd.Series, l: int) -> pd.Series:
    return rolling_z(open_close_gap(open_, close), l)


def feature_fundamental(raw_quarterly: pd.Series, l_quarters: int) -> pd.Series:
    return rolling_z(raw_quarterly, l_quarters)


def feature_macro(raw_macro: pd.Series, l_macro: int) -> pd.Series:
    return rolling_z(raw_macro, l_macro)


# ---------------------------------------------------------------------------
# 6. Fundamental and macro feature formulas
# ---------------------------------------------------------------------------
# Operate on a Series indexed by fiscal quarter, ascending (the unique quarterly base panel).

def ttm_flow(x: pd.Series) -> pd.Series:
    """x_TTM,q = x_q + x_{q-1} + x_{q-2} + x_{q-3}"""
    return x.rolling(4).sum()


def qoq_change(x: pd.Series) -> pd.Series:
    """delta_qoq_q = x_q - x_{q-1}"""
    return x - x.shift(1)


def yoy_change(x: pd.Series) -> pd.Series:
    """delta_yoy_q = x_q - x_{q-4}"""
    return x - x.shift(4)


def qoq_growth(x: pd.Series) -> pd.Series:
    """growth_qoq_q = (x_q - x_{q-1}) / |x_{q-1}|"""
    return (x - x.shift(1)) / x.shift(1).abs()


def yoy_growth(x: pd.Series) -> pd.Series:
    """growth_yoy_q = (x_q - x_{q-4}) / |x_{q-4}|"""
    return (x - x.shift(4)) / x.shift(4).abs()


def roa_ttm(net_income_ttm: pd.Series, total_assets: pd.Series) -> pd.Series:
    """ROA_TTM = net_income_TTM / total_assets"""
    return net_income_ttm / total_assets


def roe_ttm(net_income_ttm: pd.Series, book_equity: pd.Series) -> pd.Series:
    """ROE_TTM = net_income_TTM / book_equity"""
    return net_income_ttm / book_equity


def cfoa(operating_cash_flow_ttm: pd.Series, total_assets: pd.Series) -> pd.Series:
    """CFOA = operating_cash_flow_TTM / total_assets"""
    return operating_cash_flow_ttm / total_assets


def debt_to_assets(total_liabilities: pd.Series, total_assets: pd.Series) -> pd.Series:
    """Debt-to-assets = total_liabilities / total_assets"""
    return total_liabilities / total_assets


def macro_change(macro: pd.Series) -> pd.Series:
    """dMacro_t = Macro_t - Macro_{t-1}"""
    return macro - macro.shift(1)


def macro_pct_change(macro: pd.Series) -> pd.Series:
    """pctMacro_t = Macro_t / Macro_{t-1} - 1"""
    return macro / macro.shift(1) - 1


def macro_surprise(actual: pd.Series, consensus: pd.Series) -> pd.Series:
    """surprise_t = actual_t - consensus_t"""
    return actual - consensus


def macro_normalized_surprise(surprise: pd.Series, l: int) -> pd.Series:
    """surprise_z_t = surprise_t / rolling_std(surprise_{t-L:t-1})"""
    return surprise / surprise.shift(1).rolling(l).std()


# ---------------------------------------------------------------------------
# 7. Target y construction and normalization
# ---------------------------------------------------------------------------

def raw_weekly_target(price: pd.Series, h: int = DEFAULT_HORIZON) -> pd.Series:
    """y_raw,t = ln(P_{t+h} / P_t)"""
    return np.log(price.shift(-h) / price)


def simple_weekly_target(price: pd.Series, h: int = DEFAULT_HORIZON) -> pd.Series:
    """y_simple,t = P_{t+h} / P_t - 1"""
    return price.shift(-h) / price - 1


def weekly_volatility_estimate(price: pd.Series, l: int, h: int = DEFAULT_HORIZON) -> pd.Series:
    """week_vol_L,t = vol_L,t * sqrt(h)"""
    return return_volatility(price, l) * np.sqrt(h)


def vol_normalized_target(price: pd.Series, l: int, h: int = DEFAULT_HORIZON) -> pd.Series:
    """y_norm,t = y_raw,t / (vol_L,t * sqrt(h))"""
    return raw_weekly_target(price, h) / weekly_volatility_estimate(price, l, h)


def market_excess_target(y_raw: pd.Series, y_market: pd.Series) -> pd.Series:
    """y_excess,t = y_raw,t - y_market,t"""
    return y_raw - y_market


def beta_adjusted_target(y_raw: pd.Series, y_market: pd.Series, beta_mkt: pd.Series) -> pd.Series:
    """y_beta_adj,t = y_raw,t - beta_mkt,t * y_market,t"""
    return y_raw - beta_mkt * y_market


def past_label_mean(y_raw: pd.Series, l: int) -> pd.Series:
    """mu_y,t = mean(y_raw,{t-L:t-1})"""
    return y_raw.shift(1).rolling(l).mean()


def past_label_std(y_raw: pd.Series, l: int) -> pd.Series:
    """sigma_y,t = std(y_raw,{t-L:t-1})"""
    return y_raw.shift(1).rolling(l).std()


def rolling_label_zscore(y_raw: pd.Series, l: int) -> pd.Series:
    """y_z,t = (y_raw,t - mu_y,t) / sigma_y,t"""
    return (y_raw - past_label_mean(y_raw, l)) / past_label_std(y_raw, l)


# clip_series (section 4) also implements "Clipped target: y_final,t = clip(y_norm,t, -5, 5)"


# ---------------------------------------------------------------------------
# 8. Time-series IC, ICIR, and alpha screening
# ---------------------------------------------------------------------------

def time_series_ic(feature: pd.Series, target: pd.Series) -> float:
    """IC_i,j = corr_t(x_i,j,norm,t, y_i,norm,t)"""
    return feature.corr(target)


def rolling_ic(feature: pd.Series, target: pd.Series, block_size: int) -> pd.Series:
    """IC_i,j,s = corr_t in block s of (x_i,j,norm,t, y_i,norm,t)

    Blocks are consecutive, non-overlapping chunks of `block_size` jointly-valid observations.
    Returned series is indexed by each block's last date.
    """
    df = pd.concat([feature, target], axis=1).dropna()
    df.columns = ["x", "y"]
    n_blocks = len(df) // block_size
    values, idx = [], []
    for b in range(n_blocks):
        block = df.iloc[b * block_size:(b + 1) * block_size]
        values.append(block["x"].corr(block["y"]))
        idx.append(block.index[-1])
    return pd.Series(values, index=idx)


def icir(rolling_ic_series: pd.Series) -> float:
    """ICIR_i,j = mean_s(IC_i,j,s) / std_s(IC_i,j,s)"""
    return rolling_ic_series.mean() / rolling_ic_series.std()


def ic_hit_rate(rolling_ic_series: pd.Series) -> float:
    """hit_i,j = count(IC_i,j,s > 0) / number_of_blocks"""
    return (rolling_ic_series > 0).sum() / len(rolling_ic_series)


def univariate_regression(feature: pd.Series, target: pd.Series) -> tuple[float, float, float]:
    """y_norm,t = a + b_j*x_j,norm,t + error_t; t_b = b_j / SE(b_j)

    Returns (intercept a, slope b_j, t-stat of b_j).
    """
    df = pd.concat([feature, target], axis=1).dropna()
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()
    n = len(x)
    x_mean, y_mean = x.mean(), y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    b = np.sum((x - x_mean) * (y - y_mean)) / sxx
    a = y_mean - b * x_mean
    resid = y - (a + b * x)
    dof = n - 2
    sse = np.sum(resid ** 2)
    se_b = np.sqrt(sse / dof / sxx)
    t_b = b / se_b
    return a, b, t_b


def forward_horizon_decay(feature: pd.Series, price: pd.Series, horizons=(1, 5, 10, 20)) -> dict[int, float]:
    """IC_h = corr_t(x_norm,t, ln(P_{t+h}/P_t)) for h in {1, 5, 10, 20}"""
    return {h: feature.corr(np.log(price.shift(-h) / price)) for h in horizons}


# ---------------------------------------------------------------------------
# 9. Feature correlation and redundancy
# ---------------------------------------------------------------------------

def feature_correlation_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """rho_j,k = corr_t(x_j,norm,t, x_k,norm,t)"""
    return features.corr()


def high_correlation_flags(corr: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """|rho_j,k| > 0.75 or 0.80"""
    return corr.abs() > threshold


def regression_r_squared(y: pd.Series, x: pd.DataFrame) -> float:
    """R_j^2 from regression: x_j = a + sum_{k != j} b_k*x_k + error"""
    df = pd.concat([y, x], axis=1).dropna()
    yv = df.iloc[:, 0].to_numpy()
    xv = df.iloc[:, 1:].to_numpy()
    xv = np.column_stack([np.ones(len(xv)), xv])
    beta, *_ = np.linalg.lstsq(xv, yv, rcond=None)
    y_hat = xv @ beta
    ss_res = np.sum((yv - y_hat) ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    return 1 - ss_res / ss_tot


def variance_inflation_factor(features: pd.DataFrame, column: str) -> float:
    """VIF_j = 1 / (1 - R_j^2)"""
    y = features[column]
    x = features.drop(columns=[column])
    return 1 / (1 - regression_r_squared(y, x))


def cluster_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """distance_j,k = 1 - |rho_j,k|"""
    return 1 - corr.abs()


# ---------------------------------------------------------------------------
# 10. Linear regression baselines
# ---------------------------------------------------------------------------
# scikit-learn is imported lazily inside each fit_* function so this module still imports
# without scikit-learn installed.

def fit_ols(x: pd.DataFrame, y: pd.Series):
    """OLS: minimize sum_t (y_norm,t - beta_0 - sum_j beta_j*x_j,norm,t)^2"""
    from sklearn.linear_model import LinearRegression

    return LinearRegression().fit(x, y)


def fit_ridge(x: pd.DataFrame, y: pd.Series, alpha: float = 1.0):
    """Ridge: minimize SSE + lambda * sum_j beta_j^2"""
    from sklearn.linear_model import Ridge

    return Ridge(alpha=alpha).fit(x, y)


def fit_lasso(x: pd.DataFrame, y: pd.Series, alpha: float = 1.0):
    """Lasso: minimize SSE + lambda * sum_j |beta_j|"""
    from sklearn.linear_model import Lasso

    return Lasso(alpha=alpha).fit(x, y)


def fit_elastic_net(x: pd.DataFrame, y: pd.Series, alpha: float = 1.0, l1_ratio: float = 0.5):
    """Elastic Net: minimize SSE + lambda_1*sum_j|beta_j| + lambda_2*sum_j beta_j^2"""
    from sklearn.linear_model import ElasticNet

    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(x, y)


def predicted_score(model, x: pd.DataFrame) -> pd.Series:
    """y_hat_t = beta_0 + sum_j beta_j*x_j,norm,t"""
    return pd.Series(model.predict(x), index=x.index)


def trading_signal(y_hat: pd.Series) -> pd.Series:
    """signal_t = sign(y_hat_t)"""
    return np.sign(y_hat)


def position_size(y_hat: pd.Series, target_vol: pd.Series, w_max: float) -> pd.Series:
    """w_t = clip(y_hat_t / target_vol_t, -w_max, w_max)"""
    return (y_hat / target_vol).clip(lower=-w_max, upper=w_max)


# ---------------------------------------------------------------------------
# 11. Walk-forward validation and backtest evaluation
# ---------------------------------------------------------------------------

def walk_forward_windows(n_obs: int, train_size: int, test_size: int, step: int):
    """Train window: [t - T_train, t - 1]; test window: [t, t + T_test]; rolling step S.

    Yields (train_idx, test_idx) integer-position arrays into a length-`n_obs` series.
    """
    start = train_size
    while start + test_size <= n_obs:
        train_idx = np.arange(start - train_size, start)
        test_idx = np.arange(start, start + test_size)
        yield train_idx, test_idx
        start += step


def prediction_correlation(y_hat: pd.Series, y_norm: pd.Series) -> float:
    """pred_IC = corr_t(y_hat_t, y_norm,t)"""
    return y_hat.corr(y_norm)


def directional_accuracy(y_hat: pd.Series, y_norm: pd.Series) -> float:
    """accuracy = count(sign(y_hat_t) = sign(y_norm,t)) / n"""
    df = pd.concat([y_hat, y_norm], axis=1).dropna()
    return (np.sign(df.iloc[:, 0]) == np.sign(df.iloc[:, 1])).mean()


def strategy_return(position: pd.Series, y_raw: pd.Series) -> pd.Series:
    """strat_ret_t = position_t * y_raw,t"""
    return position * y_raw


def sharpe_ratio(strat_ret: pd.Series, h: int = DEFAULT_HORIZON, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Sharpe = mean(strat_ret_t) / std(strat_ret_t) * sqrt(252 / h)"""
    return strat_ret.mean() / strat_ret.std() * np.sqrt(periods_per_year / h)


def turnover_series(position: pd.Series) -> pd.Series:
    """turnover_t = |position_t - position_{t-1}|"""
    return position.diff().abs()


def cost_adjusted_return(strat_ret: pd.Series, turnover: pd.Series, cost_per_turnover: float) -> pd.Series:
    """net_ret_t = strat_ret_t - cost_per_turnover * turnover_t"""
    return strat_ret - cost_per_turnover * turnover
