"""
factor_utils.py —— 全项目共用的因子中间算法(Section 1: Shared Raw Variables)
三种频率(daily / monthly / quarterly)的所有 alpha 都可调用。
约定:
- by    : 实体键,默认 'security_id'
- order : 时间排序键。季度面板用 'rebalance_date',日/月频面板用 'date'
所有返回值都是与输入 df 行对齐的 pd.Series。
★重要前提(相对旧版的改动):
  build_rebalance_panel 的过期守卫已保证季度面板每个季度槽位只有一条
  (缺报的是 NaN 行,不是重复行),因此季度函数不再按 report_date 去重,
  统一按行 shift。固定每年 4 行的结构让 lag_4q(shift 4 行)天然对齐去年同期,
  缺季的 NaN 行作为占位保留,不会使 YoY 错位。
  -> 季度 / 月频 / 日频全部用同一个 group_shift,只是 periods 和 order 不同。
"""
import numpy as np
import pandas as pd
# ======================================================================
# 各频率 alpha 用到的 func 速查(func 可重复;★=特殊用法)
# ----------------------------------------------------------------------
# 【季度 Quarterly · target_63d · 季末采样】 df_q (order='rebalance_date')
#   rolling_sum_4q(TTM) · lag_1q · lag_4q · safe_divide · score/score_low
#
# 【日频价格挂钩 · target_21d · 月末筛选】 df_d (order='date')
#   market_cap_mn · rolling_sum_4q(季度面板算好,再对齐日频) · safe_divide · score/score_low
#
# 【月频 · target_21d · 月末采样】 df_m (order='date')
#   group_shift(periods=1或3) · safe_divide · score/score_low
#   ★ RatingChange_1M/3M 必须 group_shift(df_m,'rec_mean',1或3);✗ 别用 lag_21d/63d
#
# 【复合 Composites】 score/score_low(横截面 by=采样日) + 各成分最终值
# ======================================================================
# ======================================================================
# 一、通用分组滞后 group_shift(季度/月/日频都用它)
# ======================================================================
def group_shift(df, col, periods, by='security_id', order='date'):
    """按 by 分组、order 排序后 shift periods 行,结果对齐回原 index。"""
    d = df.sort_values([by, order])
    return d.groupby(by, sort=False)[col].shift(periods).reindex(df.index)
# ======================================================================
# 二、季度类:lag_1q / lag_4q / rolling_sum_4q(TTM)
#    直接按行 shift(order='rebalance_date');前提见文件头 ★。
# ======================================================================
def lag_nq(df, col, n, by='security_id', order='rebalance_date'):
    """x 的 n 期前季度值(按行)。"""
    return group_shift(df, col, n, by, order)
def lag_1q(df, col, by='security_id', order='rebalance_date'):
    """上一季度值(QoQ 用)。"""
    return group_shift(df, col, 1, by, order)
def lag_4q(df, col, by='security_id', order='rebalance_date'):
    """四个季度前的值(YoY 用,同一财季去年同期)。"""
    return group_shift(df, col, 4, by, order)
def rolling_sum_4q(df, col, by='security_id', order='rebalance_date', min_valid=3):
    """TTM:最近四个季度之和。
    - 公司历史不足 4 个季度(组内前 3 行)-> NaN(窗口还没满 4 季);
    - 已满 4 季、缺 1 个洞 -> 用可得季度均值补(等价 mean(有效)*4);
    - 缺 2 个及以上(有效 < min_valid=3)-> NaN。
    """
    d = df.sort_values([by, order])
    g = d.groupby(by, sort=False)
    m = (g[col].rolling(4, min_periods=min_valid).mean()
              .reset_index(level=0, drop=True)) * 4
    m = m.where(g.cumcount() >= 3)          # 组内不足 4 季(前3行)-> NaN
    return m.reindex(df.index)
# TTM 别名
ttm = rolling_sum_4q
# ======================================================================
# 三、日/月频滞后:lag_21d / lag_63d(日频面板;月频用 group_shift(periods=1/3))
# ======================================================================
def lag_21d(df, col, by='security_id', order='date'):
    """21 个交易日前(约 1 个月,日频面板用)。"""
    return group_shift(df, col, 21, by, order)
def lag_63d(df, col, by='security_id', order='date'):
    """63 个交易日前(约 3 个月,日频面板用)。"""
    return group_shift(df, col, 63, by, order)
# ======================================================================
# 四、派生量:market_cap_mn / avg_*(流量对存量的期初期末平均)
# ======================================================================
def market_cap_mn(df, close='equity_close_raw', shares_k='equity_shares_outstanding_k'):
    """市值(百万):close * shares(千股) / 1000。"""
    return df[close] * df[shares_k] / 1000.0
def avg_total_assets(df, by='security_id', order='rebalance_date'):
    return 0.5 * (df['total_assets'] + lag_4q(df, 'total_assets', by, order))
def avg_book_equity(df, by='security_id', order='rebalance_date'):
    return 0.5 * (df['book_equity'] + lag_4q(df, 'book_equity', by, order))
def avg_inventory(df, by='security_id', order='rebalance_date'):
    return 0.5 * (df['inventory'] + lag_4q(df, 'inventory', by, order))
# ======================================================================
# 五、除法安全:Zero numerator rule
#    分母为 0 / 无效 -> NaN;零分子(debt/PPE/capex=0)有经济意义,保留为 0。
# ======================================================================
def safe_divide(num, den):
    """num/den;den==0 或无效 -> NaN。零分子自然得 0(保留)。"""
    num = pd.Series(np.asarray(num, dtype='float64'),
                    index=num.index if isinstance(num, pd.Series) else None)
    den = pd.Series(np.asarray(den, dtype='float64'),
                    index=den.index if isinstance(den, pd.Series) else None)
    den = den.where(den != 0, np.nan)              # 分母 0 -> 无效
    return num / den
# ======================================================================
# 六、打分:score / score_low(横截面 winsorize + 百分位排名,复合因子内部用)
#    score(x) = percentile_rank(winsorize(x,1%,99%)) - 0.5,越高越好。
#    score_low(x) = score(-x),用于杠杆/离散度等"越低越好"的因子。
# ======================================================================
def winsorize(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)
def _xs_score(s):
    return winsorize(s).rank(pct=True) - 0.5
def score(df, col, by='rebalance_date'):
    """横截面(按 by,默认采样日)百分位打分。"""
    return df.groupby(by, sort=False)[col].transform(_xs_score)
def score_low(df, col, by='rebalance_date'):
    """score(-x):越低越好。"""
    neg = (-df[col])
    return neg.groupby(df[by], sort=False).transform(_xs_score)
