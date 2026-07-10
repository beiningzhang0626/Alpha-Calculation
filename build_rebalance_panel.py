"""
按调仓日(每年 3/31, 6/30, 9/30, 12/31)重建 per-stock 面板。
每个 (security_id, rebalance_date) 一行,三块分别对齐:

  FMT(fundamentals) : 按 announce_date          取 <= 调仓日 的最近一条
  actuals           : 按 actuals_announce_date   取 <= 调仓日 的最近一条 -> 得到 (期P, 公布日A)
  consensus         : 取 forecast_period_end == P 且 statistic_date < A 的最后一条
                      (= 该期财报公布前最后一次一致预期更新)
"""
import numpy as np
import pandas as pd

REBAL_MD = [(3, 31), (6, 30), (9, 30), (12, 31)]


def build_rebalance_panel(df):
    df = df.sort_values(['security_id', 'date']).reset_index(drop=True)

    # 统一所有 datetime 列精度到 [ns],避免 merge_asof 的 us/ns dtype 冲突
    # (用 dtype.kind=='M' 判断,能覆盖 [us]/[ms]/[ns];select_dtypes('datetime') 会漏掉非 ns)
    for c in df.columns:
        if df[c].dtype.kind == 'M':
            df[c] = df[c].astype('datetime64[ns]')

    consensus_cols = [c for c in df.columns if c.startswith('consensus_')]
    actuals_cols   = [c for c in df.columns if c.startswith('actuals_')]
    fmt_cols       = [c for c in df.columns
                      if c not in consensus_cols + actuals_cols + ['security_id', 'date']]

    # ---------- 调仓日网格(限制在每只票的数据区间内) ----------
    yrs = range(df['date'].dt.year.min(), df['date'].dt.year.max() + 1)
    all_rebal = pd.to_datetime([f'{y}-{m:02d}-{d:02d}' for y in yrs
                                for m, d in REBAL_MD]).astype('datetime64[ns]')
    rng = df.groupby('security_id')['date'].agg(['min', 'max'])
    grid = (rng.assign(key=1).reset_index()
              .merge(pd.DataFrame({'rebalance_date': all_rebal, 'key': 1}), on='key')
              .query('min <= rebalance_date <= max')
              [['security_id', 'rebalance_date']]
              .sort_values(['security_id', 'rebalance_date']).reset_index(drop=True))

    # ---------- FMT: announce_date <= 调仓日 的最近一条 ----------
    fmt = (df[['security_id'] + fmt_cols]
           .dropna(subset=['announce_date'])
           .sort_values('announce_date')
           .drop_duplicates(['security_id', 'announce_date'], keep='last'))
    res = pd.merge_asof(grid.sort_values('rebalance_date'), fmt.sort_values('announce_date'),
                        by='security_id', left_on='rebalance_date', right_on='announce_date',
                        direction='backward')

    # ---------- actuals: actuals_announce_date <= 调仓日 的最近一条 ----------
    act = (df[['security_id'] + actuals_cols]
           .dropna(subset=['actuals_announce_date'])
           .sort_values('actuals_announce_date')
           .drop_duplicates(['security_id', 'actuals_announce_date'], keep='last'))
    res = pd.merge_asof(res.sort_values('rebalance_date'), act.sort_values('actuals_announce_date'),
                        by='security_id', left_on='rebalance_date', right_on='actuals_announce_date',
                        direction='backward')

    # ---------- consensus: 期P 相同 且 statistic_date < A 的最后一条 ----------
    cons = (df[['security_id'] + consensus_cols]
            .dropna(subset=['consensus_forecast_period_end', 'consensus_statistic_date'])
            .drop_duplicates(['security_id', 'consensus_forecast_period_end',
                              'consensus_statistic_date'], keep='last')
            .copy())
    cons['_pe'] = cons['consensus_forecast_period_end'].astype('datetime64[ns]')

    left = res[['security_id', 'rebalance_date', 'actuals_period_end', 'actuals_announce_date']].copy()
    left['_pe'] = left['actuals_period_end'].astype('datetime64[ns]')
    left_ok = left.dropna(subset=['actuals_period_end', 'actuals_announce_date'])

    cm = pd.merge_asof(
        left_ok.sort_values('actuals_announce_date'),
        cons.sort_values('consensus_statistic_date'),
        by=['security_id', '_pe'],
        left_on='actuals_announce_date', right_on='consensus_statistic_date',
        direction='backward', allow_exact_matches=True)   # <= A(公布当天或之前)

    res = res.merge(cm[['security_id', 'rebalance_date'] + consensus_cols],
                    on=['security_id', 'rebalance_date'], how='left')
    return res.sort_values(['security_id', 'rebalance_date']).reset_index(drop=True)
