"""真实 A 股数据上的机器学习信号研究（截面打分 + 严格样本外评估）。

运行::

    python examples/real_ml_signal.py --data-dir /path/to/kairos-data/data/ashare

数据为**只读**的真实 A 股日线（后复权 hfq，公开行情），本脚本不修改任何行情文件；
全流程离线、确定性、仅 numpy/pandas（模型为本包 numpy 自研，不依赖 sklearn）。

流程
----
1. :func:`kairos_ml.realdata.load_close_panel` 读 CSV 目录 → 收盘价面板
   （index=交易日, columns=标的，按上市日对齐）。
2. 用本包 :mod:`kairos_ml.features` 构造 16 个无量纲特征：滚动动量 (1/5/10/20/60)、
   已实现波动率 (10/20)、RSI(14)、均线乖离 (5/20)、EMA 快慢差、归一化 MACD 柱，
   外加 4 个「当日截面 z-score」版本（动量/波动/RSI/乖离）。全部只用截至当日的数据。
3. 用本包 :mod:`kairos_ml.labels` 构造标签：``forward_returns(H)`` 的 H 日远期收益
   （``--label barrier`` 可切换为 ``triple_barrier`` 的障碍出场收益）。
4. 用本包 :mod:`kairos_ml.cv` 做**严格样本外**切分：
   - ``walk_forward_splits``：滚动窗口（训练长度恒定）；
   - ``purged_kfold``（purge + embargo）：扩展窗口（训练起点恒为 0）。
   两套切分都再施加「前向约束」：训练样本位置 p 必须满足 ``p + H < 测试期首日``，
   即训练只用**截至测试期开始前已完全实现**的样本（不训练在未来数据上）。
5. 训练 ``RidgeRegression`` / ``LogisticRegression`` / ``GradientBoostingRegressor``。
6. 用本包 :mod:`kairos_ml.evaluate` 汇总样本外指标：hit_rate、precision/recall/F1、
   pooled IC、日频截面 IC（均值/ICIR/t 值）、top-K 多头与多空的信号 PnL 及 t 值。
7. 结果写入 ``research/real_ml/``：``REPORT.md``（中文报告）、``metrics.json``、
   ``feature_importance.csv``（置换重要性，按 IC 下降幅度度量）。

性能说明：38 标的 × ~1900 交易日 ≈ 7 万条池化样本；线性模型直接向量化求解，
梯度提升树（Python 逐点分裂，较慢）按固定步长子采样训练行至 ``--gbm-max-rows``。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import kairos_ml as kml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(_REPO_ROOT, "..", "kairos-data", "data", "ashare"))
if not os.path.isdir(_DEFAULT_DATA_DIR):  # 兜底：绝对路径
    _DEFAULT_DATA_DIR = "/home/zhuoming.wang/quant-hub/kairos/kairos-data/data/ashare"
_DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "research", "real_ml")


# ---------------------------------------------------------------------------
# 特征工程（全部复用 kairos_ml.features，无未来函数）
# ---------------------------------------------------------------------------
def build_feature_panels(close: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """由收盘价面板构造特征面板字典 ``{特征名: DataFrame(index=日期, columns=标的)}``。

    所有特征均为**无量纲**（收益率/比值/z-score），可跨标的池化训练；
    仅使用「截至当日 t」及以前的价格，天然无未来函数。
    """
    sma5 = kml.sma(close, 5)
    sma20 = kml.sma(close, 20)
    ema12 = kml.ema(close, 12)
    ema26 = kml.ema(close, 26)
    mom20 = kml.rolling_momentum(close, 20)
    rv10 = kml.realized_volatility(close, 10)
    rv20 = kml.realized_volatility(close, 20)
    rsi14 = kml.rsi(close, 14)
    bias5 = close / sma5 - 1.0
    bias20 = close / sma20 - 1.0
    # kml.macd 仅接受单资产 Series，逐列计算后按价格归一化 -> 无量纲
    macd_hist = pd.DataFrame({c: kml.macd(close[c])["hist"] for c in close.columns},
                             index=close.index)

    panels: Dict[str, pd.DataFrame] = {
        "mom1": kml.rolling_momentum(close, 1),
        "mom5": kml.rolling_momentum(close, 5),
        "mom10": kml.rolling_momentum(close, 10),
        "mom20": mom20,
        "mom60": kml.rolling_momentum(close, 60),
        "rv10": rv10,
        "rv20": rv20,
        "rsi14": rsi14,
        "bias5": bias5,
        "bias20": bias20,
        "ema_gap": ema12 / ema26 - 1.0,
        "macd_hist": macd_hist / close,
        # 当日截面 z-score：剥离个股长期水平差异，突出「今天谁相对更强/更贵」
        "cs_mom20": kml.cross_sectional_zscore(mom20),
        "cs_rv20": kml.cross_sectional_zscore(rv20),
        "cs_rsi14": kml.cross_sectional_zscore(rsi14 - 50.0),
        "cs_bias20": kml.cross_sectional_zscore(bias20),
    }
    return panels


FEATURE_DESC = {
    "mom1": "1 日动量 rolling_momentum(close,1)",
    "mom5": "5 日动量 rolling_momentum(close,5)",
    "mom10": "10 日动量 rolling_momentum(close,10)",
    "mom20": "20 日动量 rolling_momentum(close,20)",
    "mom60": "60 日动量 rolling_momentum(close,60)",
    "rv10": "10 日已实现波动率 realized_volatility(close,10)",
    "rv20": "20 日已实现波动率 realized_volatility(close,20)",
    "rsi14": "14 日 RSI（Wilder 平滑）rsi(close,14)",
    "bias5": "5 日均线乖离 close/sma(close,5)-1",
    "bias20": "20 日均线乖离 close/sma(close,20)-1",
    "ema_gap": "EMA 快慢差 ema(close,12)/ema(close,26)-1",
    "macd_hist": "MACD 柱 / 收盘价（macd(close)['hist']/close）",
    "cs_mom20": "mom20 的当日截面 z-score cross_sectional_zscore",
    "cs_rv20": "rv20 的当日截面 z-score cross_sectional_zscore",
    "cs_rsi14": "rsi14-50 的当日截面 z-score cross_sectional_zscore",
    "cs_bias20": "bias20 的当日截面 z-score cross_sectional_zscore",
}


def panel_to_long(panels: Dict[str, pd.DataFrame], dates: pd.Index,
                  symbols: Sequence[str]) -> pd.DataFrame:
    """把特征面板堆叠成池化长表 ``DataFrame(index=MultiIndex(date, symbol))``。

    按「日期主序」展平，与 ``MultiIndex.from_product([dates, symbols])`` 一致，
    因此长表的日期位置天然单调不减，便于按时间位置切分训练/测试。
    """
    idx = pd.MultiIndex.from_product([dates, list(symbols)], names=["date", "symbol"])
    data = {}
    for name, df in panels.items():
        arr = df.reindex(index=dates, columns=list(symbols)).to_numpy(dtype="float64")
        data[name] = arr.ravel()
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# 标签（复用 kairos_ml.labels）
# ---------------------------------------------------------------------------
def build_label_panels(close: pd.DataFrame, horizon: int, kind: str,
                       pt: float = 1.5, sl: float = 1.5, vol_window: int = 20
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """构造标签面板 ``(回归目标, 分类目标)``。

    - ``kind='forward'``：``forward_returns(close, horizon)`` 的 H 日远期收益；
      分类目标 = 远期收益是否为正。
    - ``kind='barrier'``：逐标的 ``triple_barrier`` 的障碍出场收益 ``ret``；
      分类目标 = 是否先触达止盈上轨（``label == 1``）。

    无未来数据的位置（末尾 H 期等）标签为 NaN，后续统一丢弃。
    """
    if kind == "forward":
        ret = kml.forward_returns(close, periods=horizon)
        cls_vals = np.where(ret.notna(), (ret > 0).astype("float64"), np.nan)
        cls = pd.DataFrame(cls_vals, index=ret.index, columns=ret.columns)
        return ret, cls

    if kind != "barrier":
        raise ValueError(f"未知标签类型: {kind}（可选 forward / barrier）")

    rets: Dict[str, pd.Series] = {}
    clss: Dict[str, pd.Series] = {}
    for c in close.columns:
        tb = kml.triple_barrier(close[c], pt=pt, sl=sl, vertical=horizon,
                                vol_window=vol_window)
        lab = tb["label"].reindex(close.index) if not tb.empty else \
            pd.Series(np.nan, index=close.index)
        rets[c] = tb["ret"].reindex(close.index) if not tb.empty else \
            pd.Series(np.nan, index=close.index)
        clss[c] = pd.Series(np.where(lab.notna(), (lab == 1).astype("float64"), np.nan),
                            index=close.index)
    ret = pd.DataFrame(rets, index=close.index)[list(close.columns)]
    cls = pd.DataFrame(clss, index=close.index)[list(close.columns)]
    return ret, cls


# ---------------------------------------------------------------------------
# 严格样本外切分
# ---------------------------------------------------------------------------
def make_splits(n_dates: int, scheme: str, horizon: int, train_size: int,
                test_size: int, n_splits: int, min_train_dates: int
                ) -> Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]]:
    """生成两套「严格前向」切分，返回 ``{方案名: [(折名, 训练位置, 测试位置)]}``。

    对 ``walk_forward_splits`` / ``purged_kfold`` 给出的训练位置再施加 purge 约束
    ``p + horizon < 测试期首日``：训练样本的标签期必须在测试期开始之前**完全实现**，
    同时保证训练集永远在测试集之前（不使用未来数据训练）。
    """
    out: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {}

    def _purge(tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        return tr[tr + horizon < te[0]]

    if scheme in ("walk_forward", "both"):
        folds = []
        for i, (tr, te) in enumerate(
                kml.walk_forward_splits(n_dates, train_size=train_size, test_size=test_size)):
            tr = _purge(tr, te)
            if tr.size >= min_train_dates:
                folds.append((f"wf{i + 1}", tr, te))
        out["walk_forward"] = folds

    if scheme in ("purged", "both"):
        folds = []
        for i, (tr, te) in enumerate(
                kml.purged_kfold(n_dates, n_splits=n_splits, embargo=horizon,
                                 label_horizon=horizon)):
            tr = _purge(tr, te)
            if tr.size >= min_train_dates:
                folds.append((f"pk{i + 1}", tr, te))
        out["purged"] = folds
    return out


def _pos_mask(tpos: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """按「日期位置集合」生成长表行掩码（长表按日期主序，故掩码按位置查表即可）。"""
    size = int(max(int(tpos.max()), int(positions.max()))) + 1
    ok = np.zeros(size, dtype=bool)
    ok[positions] = True
    return ok[tpos]


# ---------------------------------------------------------------------------
# 训练与样本外打分
# ---------------------------------------------------------------------------
ModelSpec = Tuple[str, object, str, Optional[int]]  # (名称, 工厂, clf/reg, 训练行上限)


def model_specs(with_gbm: bool, gbm_max_rows: int, gbm_estimators: int) -> List[ModelSpec]:
    """待评估模型清单（工厂函数，保证每折都是全新未拟合的模型）。"""
    specs: List[ModelSpec] = [
        ("Ridge", lambda: kml.RidgeRegression(alpha=100.0, standardize=True), "reg", None),
        ("Logistic", lambda: kml.LogisticRegression(C=1.0, max_iter=50), "clf", None),
    ]
    if with_gbm:
        specs.append(("GBM",
                      lambda: kml.GradientBoostingRegressor(
                          n_estimators=gbm_estimators, learning_rate=0.05, max_depth=2,
                          random_state=0),
                      "reg", gbm_max_rows))
    return specs


def fit_predict_folds(folds: List[Tuple[str, np.ndarray, np.ndarray]], X: pd.DataFrame,
                      tpos: np.ndarray, y_ret: np.ndarray, y_cls: np.ndarray,
                      specs: List[ModelSpec]) -> Dict[str, np.ndarray]:
    """在给定切分上逐折训练并收集**样本外**打分，返回 ``{模型名: 打分数组}``。

    打分口径统一为「越大越看多」的连续分值：回归模型直接输出预测收益；
    逻辑回归输出 ``P(上涨) - 0.5``（中心化后符号可判方向）。未落在任何测试折的行
    为 NaN。
    """
    n_rows = X.shape[0]
    Xv = X.to_numpy(dtype="float64")
    out: Dict[str, np.ndarray] = {}
    for name, factory, kind, max_rows in specs:
        score = np.full(n_rows, np.nan)
        for _fname, tr, te in folds:
            m_tr = _pos_mask(tpos, tr)
            m_te = _pos_mask(tpos, te)
            if max_rows is not None:      # 慢模型：确定性步长子采样训练行
                rows = np.nonzero(m_tr)[0]
                step = max(1, int(np.ceil(rows.size / float(max_rows))))
                m_tr = np.zeros_like(m_tr)
                m_tr[rows[::step]] = True
            target = y_cls if kind == "clf" else y_ret
            model = factory().fit(Xv[m_tr], target[m_tr])
            if kind == "clf":
                score[m_te] = model.predict_proba(Xv[m_te])[:, 1] - 0.5
            else:
                score[m_te] = model.predict(Xv[m_te])
        out[name] = score
    return out


# ---------------------------------------------------------------------------
# 样本外评估（复用 kairos_ml.evaluate）
# ---------------------------------------------------------------------------
def daily_ic(score: pd.Series, actual: pd.Series, min_assets: int = 10) -> pd.Series:
    """逐日截面 IC（spearman）：每个交易日在标的截面上算预测与实际收益的秩相关。"""
    df = pd.concat([score.rename("p"), actual.rename("a")], axis=1).dropna()
    vals = {}
    for dt, g in df.groupby(level=0, sort=True):
        if g.shape[0] < min_assets:
            continue
        vals[dt] = kml.ic(g["p"], g["a"], method="spearman")
    return pd.Series(vals, name="daily_ic").sort_index()


def portfolio_pnl(score: pd.Series, actual: pd.Series, horizon: int, top_k: int,
                  bottom_k: int) -> pd.DataFrame:
    """top-K 多头 / top-K−bottom-K 多空的**不重叠**持有期 PnL。

    每 ``horizon`` 个交易日调仓一次（与标签持有期一致 → 各期不重叠），
    多头等权 1/top_k，空头等权 −1/bottom_k；单期 PnL 用
    :func:`kairos_ml.evaluate.signal_pnl` 计算（信号 × 远期收益）。
    同时给出同期「全体标的等权」基准，用于剥离市场 beta。
    """
    df = pd.concat([score.rename("p"), actual.rename("a")], axis=1).dropna()
    dates = df.index.get_level_values(0).unique().sort_values()
    rebalance = list(dates)[::max(1, int(horizon))]
    rows = []
    for dt in rebalance:
        cs = df.xs(dt, level=0)
        if cs.shape[0] < max(top_k, bottom_k) + 1:
            continue
        order = cs["p"].sort_values(ascending=False, kind="mergesort")
        w = pd.Series(0.0, index=cs.index)
        w[order.index[:top_k]] = 1.0 / top_k
        if bottom_k > 0:
            w[order.index[-bottom_k:]] = -1.0 / bottom_k
        long_only = pd.Series(0.0, index=cs.index)
        long_only[order.index[:top_k]] = 1.0 / top_k
        rows.append((dt,
                     float(kml.signal_pnl(w, cs["a"]).sum()),
                     float(kml.signal_pnl(long_only, cs["a"]).sum()),
                     float(cs["a"].mean())))
    return pd.DataFrame(rows, columns=["date", "long_short", "long_only", "benchmark"])


def _t(x: pd.Series) -> Tuple[float, float]:
    """盈亏/IC 序列的 t 统计量与 p 值（样本不足时返回 (0, 1)）。"""
    if x is None or x.dropna().shape[0] < 2:
        return 0.0, 1.0
    return kml.t_statistic(x)


def evaluate_score(score: pd.Series, actual: pd.Series, horizon: int, top_k: int,
                   bottom_k: int, min_assets: int, fold_names: Sequence[str],
                   dates_index: pd.DatetimeIndex) -> Dict[str, object]:
    """把一个模型的样本外打分汇总成指标字典（全部用 kairos_ml.evaluate 计算）。"""
    mask = score.notna() & actual.notna()
    s, a = score[mask], actual[mask]
    n = int(mask.sum())
    if n < 2:
        return {"n_oos": n, "error": "样本外样本不足"}

    oos_dates = s.index.get_level_values(0)
    cls_pred = (s.to_numpy() > 0).astype(int)      # 分值符号 -> 预测方向
    cls_true = (a.to_numpy() > 0).astype(int)      # 实际方向
    pnl = portfolio_pnl(s, a, horizon, top_k, bottom_k)

    dic = daily_ic(s, a, min_assets)
    t_ic, p_ic = _t(dic)
    t_lo, p_lo = _t(pnl["long_only"])
    t_ls, p_ls = _t(pnl["long_short"])
    excess = pnl["long_only"] - pnl["benchmark"]
    t_ex, p_ex = _t(excess)

    res: Dict[str, object] = {
        "n_oos": n,
        "n_oos_dates": int(oos_dates.nunique()),
        "oos_start": str(oos_dates.min().date()),
        "oos_end": str(oos_dates.max().date()),
        "n_folds": len(fold_names),
        "folds": list(fold_names),
        "hit_rate": kml.hit_rate(a, s),
        "precision": kml.precision(cls_true, cls_pred, pos_label=1),
        "recall": kml.recall(cls_true, cls_pred, pos_label=1),
        "f1": kml.f1(cls_true, cls_pred, pos_label=1),
        "ic_pooled_spearman": kml.ic(s, a, method="spearman"),
        "ic_pooled_pearson": kml.ic(s, a, method="pearson"),
        "ic_daily_mean": float(dic.mean()) if dic.size else 0.0,
        "ic_daily_std": float(dic.std(ddof=1)) if dic.size > 1 else 0.0,
        "ic_daily_n": int(dic.size),
        "icir": float(dic.mean() / dic.std(ddof=1)) if dic.size > 1 and dic.std(ddof=1) > 0 else 0.0,
        "ic_daily_t": t_ic,
        "ic_daily_p": p_ic,
        "ic_daily_pos_ratio": float((dic > 0).mean()) if dic.size else 0.0,
        "actual_mean": float(a.mean()),
        "signal_pnl_mean": float(kml.signal_pnl(np.sign(s.to_numpy()), a.to_numpy()).mean()),
        "top_k": top_k,
        "bottom_k": bottom_k,
        "n_rebalance": int(pnl.shape[0]),
        "long_only_period_mean": float(pnl["long_only"].mean()) if pnl.shape[0] else 0.0,
        "long_only_cum": float(pnl["long_only"].sum()) if pnl.shape[0] else 0.0,
        "long_only_t": t_lo,
        "long_only_p": p_lo,
        "benchmark_period_mean": float(pnl["benchmark"].mean()) if pnl.shape[0] else 0.0,
        "benchmark_cum": float(pnl["benchmark"].sum()) if pnl.shape[0] else 0.0,
        "excess_period_mean": float(excess.mean()) if pnl.shape[0] else 0.0,
        "excess_cum": float(excess.sum()) if pnl.shape[0] else 0.0,
        "excess_t": t_ex,
        "excess_p": p_ex,
        "excess_win_rate": float((excess > 0).mean()) if pnl.shape[0] else 0.0,
        "long_short_period_mean": float(pnl["long_short"].mean()) if pnl.shape[0] else 0.0,
        "long_short_cum": float(pnl["long_short"].sum()) if pnl.shape[0] else 0.0,
        "long_short_t": t_ls,
        "long_short_p": p_ls,
        "long_only_var95": kml.value_at_risk(pnl["long_only"], 0.95) if pnl.shape[0] else 0.0,
        "benchmark_var95": kml.value_at_risk(pnl["benchmark"], 0.95) if pnl.shape[0] else 0.0,
        "long_short_var95": kml.value_at_risk(pnl["long_short"], 0.95) if pnl.shape[0] else 0.0,
    }
    return res


# ---------------------------------------------------------------------------
# 特征重要性（置换重要性，按「平均日频截面 IC 下降幅度」度量）
# ---------------------------------------------------------------------------
def make_daily_ic_scorer(dates: np.ndarray, min_assets: int = 10):
    """构造「平均日频截面 IC」scorer（供 :func:`kairos_ml.permutation_importance` 使用）。

    置换只打乱某一列的取值，因此 scorer 必须在**每个日期截面内部**重算 IC 再取平均，
    才能反映该特征对截面排序的真实贡献；直接用池化 IC 会把跨日期的水平差异也算进来，
    从而系统性夸大重要性。返回 ``(scorer, 有效截面数)``。
    """
    groups = [np.nonzero(dates == d)[0] for d in pd.unique(dates)]
    groups = [g for g in groups if g.size >= min_assets]

    def scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if not groups:
            return 0.0
        vals = [kml.ic(y_pred[g], y_true[g], method="spearman") for g in groups]
        return float(np.mean(vals))

    return scorer, len(groups)


def feature_importance_last_folds(scheme: str,
                                  folds: List[Tuple[str, np.ndarray, np.ndarray]],
                                  X: pd.DataFrame, tpos: np.ndarray, y_ret: np.ndarray,
                                  min_assets: int = 10, max_rows: int = 6000,
                                  n_repeats: int = 6, n_folds: int = 3
                                  ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """在**最后若干折**的样本外数据上做置换重要性并取平均（严格样本外，不含训练期）。

    每折各训练一个岭回归（只用该折训练位置的数据），在该折**测试位置**的样本上度量：
    重要性 = 打乱该列前的平均日频截面 IC − 打乱后的 IC。跨折取平均可避免结论被
    单一窗口主导（单折 IC 可能为负，此时重要性应理解为「驱动了该窗口的 IC」）。
    同时给出各折岭回归标准化系数的均值，便于交叉印证方向。
    """
    Xv = X.to_numpy(dtype="float64")
    frames: List[pd.DataFrame] = []
    coefs: List[pd.Series] = []
    baselines: Dict[str, float] = {}
    n_rows_total = 0
    n_dates_total = 0
    eval_start = eval_end = ""
    for fname, tr, te in folds[-max(1, int(n_folds)):]:
        m_tr, m_te = _pos_mask(tpos, tr), _pos_mask(tpos, te)
        rows = np.nonzero(m_te)[0]
        step = max(1, int(np.ceil(rows.size / float(max_rows))))
        rows = rows[::step]
        if rows.size < min_assets * 2:
            continue

        model = kml.RidgeRegression(alpha=100.0, standardize=True).fit(Xv[m_tr], y_ret[m_tr])
        Xte = X.iloc[rows]
        scorer, n_groups = make_daily_ic_scorer(
            Xte.index.get_level_values(0).to_numpy(), min_assets)
        if n_groups == 0:
            continue
        imp_i = kml.permutation_importance(model, Xte,
                                           pd.Series(y_ret[rows], index=Xte.index),
                                           scorer=scorer, n_repeats=n_repeats,
                                           random_state=0)
        frames.append(imp_i)
        coefs.append(pd.Series(model.coef_, index=list(X.columns)).reindex(imp_i.index))
        baselines[fname] = scorer(y_ret[rows], model.predict(Xte.to_numpy(dtype="float64")))
        n_rows_total += int(rows.size)
        n_dates_total += int(n_groups)
        d0 = str(Xte.index.get_level_values(0).min().date())
        d1 = str(Xte.index.get_level_values(0).max().date())
        eval_start = d0 if not eval_start else min(eval_start, d0)
        eval_end = d1 if not eval_end else max(eval_end, d1)

    if not frames:
        raise RuntimeError("没有任何一折能计算置换重要性（样本或截面数不足）")

    imp = pd.concat(frames).groupby(level=0).mean()
    imp = imp.sort_values("importance_mean", ascending=False)
    imp["ridge_coef_std"] = pd.concat(coefs, axis=1).mean(axis=1).reindex(imp.index)
    meta = {
        "scheme": scheme,
        "folds": list(baselines),
        "fold": "/".join(baselines),
        "n_folds": len(frames),
        "model": "RidgeRegression(alpha=100, standardize=True)",
        "scorer": "平均日频截面 spearman IC（重要性 = 打乱前 IC − 打乱后 IC，跨折平均）",
        "n_eval_rows": n_rows_total,
        "n_eval_dates": n_dates_total,
        "n_repeats": int(n_repeats),
        "eval_start": eval_start,
        "eval_end": eval_end,
        "baseline_ic_per_fold": baselines,
        "baseline_ic": float(np.mean(list(baselines.values()))),
        "ridge_coef_unit": "每 1 个特征标准差对应的 H 日收益（各折岭回归标准化系数的均值）",
    }
    return imp, meta


# ---------------------------------------------------------------------------
# 诊断：单因子裸 IC、逐折稳定性、稳健性设定
# ---------------------------------------------------------------------------
def xs_excess_panel(ret_panel: pd.DataFrame) -> pd.DataFrame:
    """把远期收益按「当日截面均值」去均值 → 截面超额收益（剥离市场分量）。

    多头 top-K 相对等权基准的超额收益才是模型的贡献，因此这是一个与评估口径
    一致的、**先验**（不看结果）的替代训练目标。
    """
    return ret_panel.sub(ret_panel.mean(axis=1), axis=0)


def daily_ic_multi(scores: pd.DataFrame, actual: pd.Series,
                   min_assets: int = 10) -> pd.DataFrame:
    """逐日截面 IC 面板：``index=日期, columns=各打分列``（一次 groupby 算完多列）。"""
    df = scores.copy()
    df["__y__"] = actual
    df = df.dropna()
    cols = list(scores.columns)
    out = {}
    for dt, g in df.groupby(level=0, sort=True):
        if g.shape[0] < min_assets:
            continue
        y = g["__y__"].to_numpy(dtype="float64")
        out[dt] = {c: kml.ic(g[c].to_numpy(dtype="float64"), y, method="spearman") for c in cols}
    return pd.DataFrame(out).T.sort_index()


def ic_summary(ic_panel: pd.DataFrame) -> pd.DataFrame:
    """把逐日 IC 面板汇总成「均值 / 标准差 / ICIR / t 值 / IC>0 占比」表。"""
    rows = []
    for c in ic_panel.columns:
        s = ic_panel[c].dropna()
        t, p = _t(s)
        sd = float(s.std(ddof=1)) if s.size > 1 else 0.0
        rows.append((str(c), float(s.mean()) if s.size else 0.0, sd,
                     float(s.mean() / sd) if sd > 1e-12 else 0.0,
                     t, p, float((s > 0).mean()) if s.size else 0.0, int(s.size)))
    df = pd.DataFrame(rows, columns=["score", "ic_daily_mean", "ic_daily_std", "icir",
                                     "ic_daily_t", "ic_daily_p", "ic_pos_ratio", "n_dates"])
    return df.set_index("score")


def fold_ic_stability(folds: List[Tuple[str, np.ndarray, np.ndarray]], X: pd.DataFrame,
                      tpos: np.ndarray, y_ret: np.ndarray, watch: Sequence[str],
                      min_assets: int = 10) -> pd.DataFrame:
    """逐折诊断：每折岭回归的样本外日频 IC、关注特征的系数与该窗口内特征自身的 IC。

    用于检验「特征—收益关系是否随时间翻转」：若某折系数符号与该折窗口内该特征的
    实现 IC 相反，说明模型在训练窗口学到、到测试窗口已经反转的关系（A 股典型的
    风格轮动 / 因子拥挤），这是样本外失效的主要机制之一。
    """
    Xv = X.to_numpy(dtype="float64")
    date_idx = X.index.get_level_values(0)
    date_arr = date_idx.to_numpy()
    rows = []
    for fname, tr, te in folds:
        m_tr, m_te = _pos_mask(tpos, tr), _pos_mask(tpos, te)
        scorer, n_groups = make_daily_ic_scorer(date_arr[m_te], min_assets)
        if n_groups == 0:
            continue
        model = kml.RidgeRegression(alpha=100.0, standardize=True).fit(Xv[m_tr], y_ret[m_tr])
        yte = y_ret[m_te]
        coef = pd.Series(model.coef_, index=list(X.columns))
        te_dates = date_idx[m_te]
        row = {
            "fold": fname,
            "start": str(te_dates.min().date()),
            "end": str(te_dates.max().date()),
            "n_te": int(m_te.sum()),
            "model_ic": scorer(yte, model.predict(Xv[m_te])),
        }
        for f in watch:
            if f in coef.index:
                row[f"{f}_coef"] = float(coef[f])
                row[f"{f}_ic"] = scorer(yte, X[f].to_numpy(dtype="float64")[m_te])
        rows.append(row)
    return pd.DataFrame(rows)


def run_walk_forward_variant(X_all: pd.DataFrame, tpos_all: np.ndarray, dates: pd.DatetimeIndex,
                             y_tgt: pd.Series, y_act: pd.Series, horizon: int, args,
                             factory, kind: str = "reg") -> Dict[str, object]:
    """跑一遍「替换训练目标或持有期」的 walk-forward 岭回归样本外评估（稳健性检验）。

    评估真值始终用**原始 H 日远期收益**（``y_act``），以便与主表口径一致可比。
    """
    valid = X_all.notna().all(axis=1) & y_tgt.notna() & y_act.notna()
    X = X_all[valid]
    tpos = tpos_all[valid.to_numpy()]
    yt = y_tgt[valid].to_numpy(dtype="float64")
    ya = y_act[valid].to_numpy(dtype="float64")
    yc = (yt > 0).astype("float64")     # 分类目标（回归变体不用，保持接口一致）
    folds = make_splits(len(dates), "walk_forward", horizon, args.train_size, args.test_size,
                        args.n_splits, args.min_train_dates)["walk_forward"]
    if not folds:
        return {"error": "无可用折"}
    raw = fit_predict_folds(folds, X, tpos, yt, yc, [("M", factory, kind, None)])["M"]
    m = evaluate_score(pd.Series(raw, index=X.index), pd.Series(ya, index=X.index), horizon,
                       args.top_k, args.bottom_k, args.min_assets, [f[0] for f in folds], dates)
    m["n_folds"] = len(folds)
    m["oos_start"] = str(pd.Series(raw, index=X.index).dropna().index.get_level_values(0).min().date())
    m["oos_end"] = str(pd.Series(raw, index=X.index).dropna().index.get_level_values(0).max().date())
    return m


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def _fmt(x: object, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _model_table(models: Dict[str, Dict[str, object]]) -> str:
    """渲染单个 CV 方案下各模型的样本外指标 Markdown 表。"""
    head = ("| 模型 | 样本外样本 | hit_rate | precision | F1 | pooled IC | 日频 IC 均值 | ICIR | IC t 值 | "
            "多头单期均值 | 多头累计 | 多头 t | 多头 VaR95 | 超额累计 | 超额 t | 多空累计 | 多空 t |\n")
    head += "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    lines = [head]
    for name, m in models.items():
        if "error" in m:
            lines.append(f"| {name} | {m.get('n_oos', 0)} |" + " — |" * 15 + "\n")
            continue
        lines.append(
            f"| {name} | {m['n_oos']} | {_fmt(m['hit_rate'])} | {_fmt(m['precision'])} | "
            f"{_fmt(m['f1'])} | {_fmt(m['ic_pooled_spearman'], 5)} | {_fmt(m['ic_daily_mean'], 5)} | "
            f"{_fmt(m['icir'], 3)} | {_fmt(m['ic_daily_t'], 2)} | "
            f"{_fmt(m['long_only_period_mean'], 5)} | {_fmt(m['long_only_cum'], 3)} | "
            f"{_fmt(m['long_only_t'], 2)} | {_fmt(m.get('long_only_var95', 0.0), 4)} | "
            f"{_fmt(m['excess_cum'], 3)} | {_fmt(m['excess_t'], 2)} | "
            f"{_fmt(m['long_short_cum'], 3)} | {_fmt(m['long_short_t'], 2)} |\n")
    return "".join(lines)


def _verdict(m: Dict[str, object]) -> str:
    """按样本外证据的**方向与强度**给出单模型结论（数据驱动，避免夸大）。"""
    if "error" in m:
        return "样本外样本不足，无法评估。"
    ic = float(m["ic_daily_mean"])
    t_ic = float(m["ic_daily_t"])
    t_ex = float(m["excess_t"])
    hr = float(m["hit_rate"])
    base = (f"日频截面 IC={ic:+.5f}（t={t_ic:+.2f}），hit_rate={hr:.4f}，"
            f"超额收益 t={t_ex:+.2f}")
    if t_ic >= 3.0 and ic >= 0.02 and t_ex >= 2.0:
        return f"证据较强（正向）：{base}；样本外存在可用但幅度有限的预测力。"
    if t_ic >= 3.0 and ic >= 0.02:
        return (f"**截面 IC 显著、但无法变现**：{base}；排序能力微弱到扣掉等权基准后"
                f"就不显著了（超额 t={t_ex:+.2f} < 2），且尚未计入任何交易成本。")
    if t_ic >= 2.0 and ic >= 0.01:
        return f"证据偏弱（正向）：{base}；勉强显著，扣成本后很可能不具实操价值。"
    if t_ic <= -2.0 and ic <= -0.01:
        return (f"**显著为负（反向）**：{base}；即模型在样本外系统性做错方向——"
                f"训练窗口学到的关系在测试窗口反转，这是过拟合/风格轮动的典型表现，"
                f"**不是**可以反着用的 alpha（反向使用等价于又一次样本内选择）。")
    return f"无稳定预测力：{base}；|t|<2，与随机猜测无实质差异。"


def render_report(ctx: Dict[str, object]) -> str:
    """把研究上下文渲染成中文 Markdown 报告。"""
    cfg = ctx["config"]
    data = ctx["data"]
    imp: pd.DataFrame = ctx["importance"]
    imp_meta = ctx["importance_meta"]
    horizon = int(cfg["horizon"])
    label_zh = "H 日远期收益 forward_returns" if cfg["label"] == "forward" else \
        "三重障碍出场收益 triple_barrier"

    L: List[str] = []
    A = L.append
    A("# 真实 A 股数据上的机器学习信号研究（严格样本外）\n\n")
    A("> 本报告由 `examples/real_ml_signal.py` 自动生成，全流程离线、确定性、可复现。\n")
    A("> 模型均为 `kairos_ml` 的 **numpy 自研实现**，不依赖 sklearn。\n")
    A("> **数据为公开行情、仅用于研究，不构成任何投资建议。**\n\n")

    A("## 1. 数据与口径\n\n")
    A(f"- 数据目录（只读）：`{data['dir']}`\n")
    A(f"- 标的数：**{data['n_assets']}** 只 A 股（沪深主板/创业板，列按文件名排序）\n")
    A(f"- 交易日数：**{data['n_dates']}**（{data['start']} ~ {data['end']}，已按全体上市日对齐）\n")
    A("- 复权口径：**后复权 hfq**——价格水平被放大，但**收益率正确**，适合特征/标签研究；\n")
    A("  非正价与缺失值置 NaN 后按列 `ffill`（停牌沿用最后有效价）。\n")
    A(f"- 池化样本量（日期 × 标的，去除特征预热期与标签末尾期后）：**{data['n_samples']}**\n\n")

    A("## 2. 特征（全部只用「截至当日」的数据，无未来函数）\n\n")
    A("| 特征 | 定义（复用 `kairos_ml.features`） |\n|---|---|\n")
    for f in ctx["features"]:
        A(f"| `{f}` | {FEATURE_DESC.get(f, '')} |\n")
    A("\n所有特征均为无量纲量（收益率/比值/z-score），可跨标的池化训练；"
      "其中 `cs_*` 为**当日截面** z-score，剥离个股长期水平差异。\n\n")

    A("## 3. 标签与严格样本外设计\n\n")
    A(f"- 标签：{label_zh}，持有期 H = **{horizon}** 个交易日"
      f"（`--label` 可切换 forward / barrier）。\n")
    A(f"- 分类目标：{'远期收益 > 0' if cfg['label'] == 'forward' else '先触达止盈上轨'}。\n")
    A("- 切分（`kairos_ml.cv`）：\n")
    A(f"  - **walk_forward**：滚动窗口，`train_size={cfg['train_size']}`、`test_size={cfg['test_size']}`，"
      f"共 {ctx['n_folds'].get('walk_forward', 0)} 折；\n")
    A(f"  - **purged**：`purged_kfold(n_splits={cfg['n_splits']}, embargo={horizon}, label_horizon={horizon})`，"
      f"共 {ctx['n_folds'].get('purged', 0)} 折（扩展窗口）。\n")
    A(f"- **防未来（purge）**：两套切分的训练位置 p 均额外要求 `p + H < 测试期首日`，"
      f"即训练样本的标签在测试期开始前已**完全实现**；训练集永远在测试集之前，"
      f"不使用任何未来数据。\n")
    A("- **无任何超参搜索**：模型超参数在看过样本外结果之前固定（见下），"
      "避免选择性偏差；因此结果不含调参带来的虚高。\n")
    A(f"- 模型：`RidgeRegression(alpha=100, standardize=True)`、`LogisticRegression(C=1.0)`"
      f"{('、`GradientBoostingRegressor(n_estimators=%d, lr=0.05, max_depth=2)`' % cfg['gbm_estimators']) if cfg['with_gbm'] else ''}"
      f"；标准化统计量在**每折训练集内部**计算，不泄漏测试信息。\n\n")

    A("## 4. 样本外绩效\n\n")
    A("指标口径：`hit_rate` = 预测分值符号与实际 H 日收益符号一致的比例；"
      "`pooled IC` = 全部样本外样本的 spearman 秩相关；`日频 IC` = 每个交易日在标的截面上的 "
      "spearman IC（其均值/ICIR/t 值为标准因子评价口径）；"
      f"多头 = 按打分选 **top {ctx['top_k']}** 等权做多、每 {horizon} 日调仓（各期不重叠），"
      f"多空 = 同时做空 bottom {ctx['bottom_k']}；"
      "超额 = 多头 − 同期全体标的等权基准（剥离市场 beta）；t 值由 `kml.t_statistic` 给出；"
      "`VaR95` = 单期收益的 95% 参数法风险价值（`kml.value_at_risk`，正数表示该分位下的损失）。\n\n")
    scheme_titles = {"walk_forward": "walk-forward（滚动窗口，主结果）",
                     "purged": "purged K-fold（purge + embargo，扩展窗口）"}
    for i, scheme in enumerate(["walk_forward", "purged"], start=1):
        if scheme not in ctx["models"]:
            continue
        A(f"### 4.{i} {scheme_titles[scheme]}\n\n")
        A(f"折数：{ctx['n_folds'].get(scheme, 0)}；样本外区间：{ctx['oos_range'][scheme]}；"
          f"调仓期数：{ctx['n_rebalance'][scheme]}。\n\n")
        A(_model_table(ctx["models"][scheme]))
        A("\n")

    A("## 5. 诊断与对照（单因子裸 IC / 逐折稳定性 / 稳健性设定）\n\n")

    fac: pd.DataFrame = ctx["factor_ic"]
    A("### 5.1 单因子「裸 IC」对照（完全不建模，把特征值直接当打分）\n\n")
    A(f"同一段样本外区间（{ctx['factor_ic_range']}），与模型打分**完全同口径**计算日频截面 IC；"
      f"`MODEL:*` 行即第 4.1 节各模型的打分，便于直接对比。\n\n")
    A("| 打分 | 日频 IC 均值 | IC 标准差 | ICIR | IC t 值 | IC>0 占比 | 截面数 |\n")
    A("|---|---|---|---|---|---|---|\n")
    for name, row in fac.iterrows():
        A(f"| `{name}` | {row['ic_daily_mean']:+.5f} | {row['ic_daily_std']:.4f} | "
          f"{row['icir']:+.3f} | {row['ic_daily_t']:+.2f} | {row['ic_pos_ratio']:.3f} | "
          f"{int(row['n_dates'])} |\n")
    n_sig = int((fac["ic_daily_t"].abs() >= 2.0).sum())
    A(f"\n解读：IC 绝对值最大的打分是 `{fac.index[0]}`（IC={fac.iloc[0]['ic_daily_mean']:+.5f}，"
      f"t={fac.iloc[0]['ic_daily_t']:+.2f}）；全表 |t|≥2 的仅 {n_sig} 个。"
      f"在 {len(ctx['features'])} 个特征 + {ctx['n_models']} 个模型上同时做 t 检验属于**多重检验**，"
      f"未做校正时 |t|≈2~3 的个别显著极可能是偶然。\n")
    if ctx.get("factor_ic_note"):
        A(f"\n注：{ctx['factor_ic_note']}\n")
    A("\n")

    st: pd.DataFrame = ctx["fold_stability"]
    watch = ctx["watch_features"]
    A("### 5.2 逐折稳定性（岭回归，walk-forward）\n\n")
    head = "| 折 | 样本外区间 | 测试样本 | 模型日频 IC |"
    for f in watch:
        head += f" {f} 系数 | {f} 同期 IC |"
    A(head + "\n")
    A("|---|---|---|---|" + "---|" * (2 * len(watch)) + "\n")
    for _i, row in st.iterrows():
        line = (f"| {row['fold']} | {row['start']} ~ {row['end']} | {int(row['n_te'])} | "
                f"{row['model_ic']:+.4f} |")
        for f in watch:
            line += f" {row[f + '_coef']:+.5f} | {row[f + '_ic']:+.4f} |"
        A(line + "\n")
    ics = st["model_ic"]
    t_fold, _p_fold = _t(ics)
    mism = {f: int(((st[f + "_coef"] * st[f + "_ic"]) < 0).sum()) for f in watch}
    n_pos_fold, n_neg_fold = int((ics > 0).sum()), int((ics < 0).sum())
    ratio = abs(float(ics.mean())) / float(ics.std(ddof=1)) if ics.size > 1 else 0.0
    A(f"\n解读：{len(st)} 折中模型 IC 为正 {n_pos_fold} 折、为负 {n_neg_fold} 折，"
      f"极差 {ics.min():+.4f} ~ {ics.max():+.4f}，跨折 IC 均值 {ics.mean():+.5f}、"
      f"标准差 {ics.std(ddof=1):.4f}（|均值|/标准差 = {ratio:.2f}）、t={t_fold:+.2f}。\n")
    if n_pos_fold > 0 and n_neg_fold > 0:
        A(f"- 逐折 IC **符号在窗口之间反复翻转**（{n_pos_fold} 正 / {n_neg_fold} 负），"
          f"跨折波动幅度远大于其均值，长期平均后信号基本被抵消。\n")
    else:
        A(f"- 逐折 IC 符号一致（{n_pos_fold} 正 / {n_neg_fold} 负），"
          f"但幅度很小且跨折 t={t_fold:+.2f}，仍不足以支撑稳定 alpha 的结论。\n")
    A("- 关注特征（IC 绝对值最大的 3 个）在各折的岭回归系数 vs 该折窗口内的实现 IC"
      "（注意特征间存在共线，如 rv10/rv20/cs_rv20 高度相关，单个系数符号不等于该特征的"
      "边际效应，本项只作定性诊断）：\n")
    for f in watch:
        A(f"  - `{f}`：{mism[f]}/{len(st)} 折出现「系数符号与同期实现 IC 符号相反」，"
          f"即模型带着上一个窗口学到的关系进入已经反转的新窗口。\n")
    worst = max(mism, key=lambda k: mism[k]) if mism else None
    if worst is not None and mism[worst] * 2 >= len(st):
        A(f"\n`{worst}` 有超过半数折出现「系数符号 vs 同期实现 IC 符号」相反，"
          f"这是 A 股短周期技术类因子的典型现象：因子方向随风格轮动翻转，"
          f"滚动重训只能追认**已经发生**的关系，无法预判下一次翻转。\n\n")
    else:
        A("\n系数符号与同期实现 IC 的方向大体一致，说明失效并非来自「关系翻转」，"
          "更可能是信号强度本身太弱、被交易噪声淹没。\n\n")

    A("### 5.3 稳健性设定（同一 walk-forward 切分，只替换训练目标 / 持有期）\n\n")
    A("| 设定 | 日频 IC 均值 | IC t 值 | hit_rate | 超额累计 | 超额 t |\n|---|---|---|---|---|---|\n")
    for r in ctx["robustness"]:
        if "error" in r:
            A(f"| {r['name']} | — | — | — | — | — |\n")
            continue
        A(f"| {r['name']} | {r['ic_daily_mean']:+.5f} | {r['ic_daily_t']:+.2f} | "
          f"{r['hit_rate']:.4f} | {r['excess_cum']:+.3f} | {r['excess_t']:+.2f} |\n")
    rb_ok = [r for r in ctx["robustness"] if "error" not in r]
    rb_sig = [r for r in rb_ok if abs(float(r["ic_daily_t"])) >= 2.0]
    rb_ex_sig = [r for r in rb_sig if abs(float(r["excess_t"])) >= 2.0]
    if rb_ok:
        ts = [float(r["ic_daily_t"]) for r in rb_ok]
        A(f"\n各设定的日频 IC t 值在 {min(ts):+.2f} ~ {max(ts):+.2f} 之间摆动，"
          f"其中 {len(rb_sig)}/{len(rb_ok)} 个达到 |t|≥2。\n")
        if not rb_sig:
            A("没有任何设定显著：结论**不是**某一种目标函数或持有期的偶然产物，"
              "换目标、换持有期都救不回来。\n\n")
        elif not rb_ex_sig:
            A("需要警惕：「换个设定就显著」正是**设定搜索 (specification searching)** 的典型陷阱——"
              "在没有事先唯一指定设定的情况下，这类显著性不能作为 alpha 的证据；"
              "而且这些设定的**超额收益 t 值全部 < 2**（见上表），"
              "即一旦扣掉等权基准、落到组合层面，显著性就消失了。\n\n")
        else:
            names = "、".join(r["name"] for r in rb_ex_sig)
            A(f"其中 {names} 的超额收益也达到 |t|≥2，值得进一步用更长样本、更多标的与"
              f"含成本的回测复核；但仍需先排除设定搜索与多重检验的影响。\n\n")

    A("## 6. 特征重要性（置换重要性，样本外日频截面 IC 下降幅度）\n\n")
    per_fold = "，".join(f"{k}={v:+.4f}" for k, v in imp_meta["baseline_ic_per_fold"].items())
    A(f"在 `{imp_meta['scheme']}` 的**最后 {imp_meta['n_folds']} 折**（{imp_meta['fold']}，评估区间 "
      f"{imp_meta['eval_start']} ~ {imp_meta['eval_end']}，合计 {imp_meta['n_eval_rows']} 条样本外样本 / "
      f"{imp_meta['n_eval_dates']} 个截面日，每列 {imp_meta['n_repeats']} 次置换）上度量并跨折平均；"
      f"scorer = 平均日频截面 spearman IC，各折基线值为 {per_fold}"
      f"（均值 {_fmt(imp_meta['baseline_ic'], 5)}）。重要性 > 0 表示打乱该列后截面 IC 下降；"
      f"若某折基线为负，则该折的重要性应理解为「驱动了这个负 IC」。\n\n")
    A("| 排名 | 特征 | 重要性均值（IC 下降） | 标准差 | 岭回归标准化系数 |\n|---|---|---|---|---|\n")
    for i, (fname, row) in enumerate(imp.head(10).iterrows(), start=1):
        A(f"| {i} | `{fname}` | {row['importance_mean']:+.5f} | {row['importance_std']:.5f} | "
          f"{row['ridge_coef_std']:+.5f} |\n")
    A(f"\n注：`ridge_coef_std` 为各折岭回归的标准化系数均值，含义是「该特征变动 1 个标准差，"
      f"预测的 {horizon} 日收益变动多少（绝对收益率）」。\n")
    if float(imp_meta["baseline_ic"]) < 0:
        A(f"\n**读数警告**：这几折模型的基线 IC 为**负**（{float(imp_meta['baseline_ic']):+.5f}），"
          f"因此本表的「重要性」应读作「该特征对这个负 IC 的贡献」：数值为负（如置换后 IC 反而变好）"
          f"意味着模型在该特征上的载荷方向与当期实现关系相反。"
          f"要判断特征本身是否有效，请以第 5.1 节的**裸因子 IC** 为准。\n")
    A("\n")

    A("## 7. 诚实结论\n\n")
    for scheme in ("walk_forward", "purged"):
        if scheme not in ctx["models"]:
            continue
        zh = "walk-forward（滚动）" if scheme == "walk_forward" else "purged K-fold（扩展）"
        A(f"**{zh}**\n\n")
        for name, m in ctx["models"][scheme].items():
            A(f"- `{name}`：{_verdict(m)}\n")
        A("\n")
    A("**总体**\n\n")
    for line in ctx["conclusions"]:
        A(f"- {line}\n")
    A("\n")

    A("## 8. 复现\n\n")
    A("```bash\n")
    A("cd kairos-ml\n")
    A("python3 -m pytest -q                      # 离线单元测试（含 tests/test_realdata.py）\n")
    A(f"python3 examples/real_ml_signal.py --data-dir {data['dir']}\n")
    A("```\n\n")
    A(f"- 运行耗时：{ctx['runtime_seconds']} 秒；环境：Python {ctx['env']['python']}、"
      f"numpy {ctx['env']['numpy']}、pandas {ctx['env']['pandas']}、"
      f"scipy {'可用' if ctx['env']['scipy'] else '不可用（自动回退 numpy 近似）'}。\n")
    A(f"- 关键参数：`--horizon {horizon} --train-size {cfg['train_size']} "
      f"--test-size {cfg['test_size']} --n-splits {cfg['n_splits']} --top-k {ctx['top_k']} "
      f"--bottom-k {ctx['bottom_k']} --label {cfg['label']}`。\n")
    A("- 产出：`REPORT.md`（本文）、`metrics.json`（全部指标）、`feature_importance.csv`。\n\n")

    A("## 9. 局限与免责声明\n\n")
    A(f"- **成本未计入**：PnL 为毛收益，未扣佣金、印花税、冲击成本与融券成本；"
      f"H={horizon} 日调仓下的换手不低，实盘净收益会显著低于本文数字。\n")
    A(f"- **样本与生存偏差**：{data['n_assets']} 只标的均为当前仍活跃的大市值个股，未含退市/长期停牌标的，"
      f"存在生存偏差；样本区间 {data['start']} ~ {data['end']} 只覆盖一段特定行情"
      f"（含 2019-2021 的结构性行情与其后的多轮回撤），"
      "结论不具备跨市场/跨周期的普适性。\n")
    A("- **后复权口径**：hfq 价格水平被放大，本文只使用收益率/比值类特征，不受水平影响；"
      "但数据源本身可能存在少量口径误差。\n")
    A("- **统计显著 ≠ 可盈利**：即使 t 值显著，也可能来自微小且难以覆盖成本的 alpha，"
      f"或来自多重检验（本文评估了 {ctx['n_models']} 个模型 × {ctx['n_schemes']} 套 CV × "
      f"{len(ctx['features'])} 个特征，未做多重检验校正）。\n")
    A("- 本文所有结果仅为**方法论演示与研究记录**，不构成投资建议。\n")
    return "".join(L)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="真实 A 股数据上的 ML 信号研究（严格样本外，离线可复现）")
    p.add_argument("--data-dir", default=_DEFAULT_DATA_DIR,
                   help="行情 CSV 目录（只读，默认 kairos-data/data/ashare）")
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR,
                   help="产出目录（REPORT.md / metrics.json / feature_importance.csv）")
    p.add_argument("--horizon", type=int, default=5, help="标签持有期 H（交易日），默认 5")
    p.add_argument("--train-size", type=int, default=500, help="walk-forward 训练窗口（交易日）")
    p.add_argument("--test-size", type=int, default=120, help="walk-forward 测试窗口（交易日）")
    p.add_argument("--n-splits", type=int, default=5, help="purged K-fold 折数")
    p.add_argument("--cv", default="both", choices=["walk_forward", "purged", "both"],
                   help="使用哪套样本外切分")
    p.add_argument("--label", default="forward", choices=["forward", "barrier"],
                   help="标签类型：forward=H 日远期收益；barrier=三重障碍出场收益")
    p.add_argument("--top-k", type=int, default=5, help="多头持仓标的数（截面 top-K）")
    p.add_argument("--bottom-k", type=int, default=5, help="空头持仓标的数（0=只做多）")
    p.add_argument("--min-assets", type=int, default=10, help="计算日频截面 IC 所需的最少标的数")
    p.add_argument("--min-train-dates", type=int, default=250,
                   help="一折可用的最少训练交易日数（不足则丢弃该折）")
    p.add_argument("--max-assets", type=int, default=0,
                   help=">0 时按步长子采样标的数（提速用；默认 0=全部）")
    p.add_argument("--no-gbm", action="store_true", help="跳过较慢的梯度提升模型")
    p.add_argument("--gbm-max-rows", type=int, default=2500, help="GBM 每折训练行上限")
    p.add_argument("--gbm-estimators", type=int, default=50, help="GBM 提升轮数")
    p.add_argument("--imp-repeats", type=int, default=6, help="置换重要性的重复次数")
    p.add_argument("--imp-folds", type=int, default=3,
                   help="置换重要性使用的末尾折数（跨折平均，默认 3）")
    p.add_argument("--quiet", action="store_true", help="只打印关键结果")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """研究主流程：加载真实数据 → 特征/标签 → 严格样本外训练 → 评估 → 写报告。"""
    t_start = time.time()
    args = parse_args(argv)

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    print("=" * 78)
    print("Kairos ML · 真实 A 股数据 ML 信号研究（严格样本外）")
    print("=" * 78)

    # ① 加载真实收盘价面板
    t0 = time.time()
    close_full = kml.load_close_panel(args.data_dir, drop_incomplete=True)
    symbols = list(close_full.columns)
    if args.max_assets and args.max_assets < len(symbols):
        step = max(1, int(np.ceil(len(symbols) / float(args.max_assets))))
        symbols = symbols[::step][:args.max_assets]
        close_full = close_full[symbols]
    close = close_full
    print(f"[数据] {close.shape[1]} 只标的 × {close.shape[0]} 个交易日 "
          f"({close.index[0].date()} ~ {close.index[-1].date()}), "
          f"NaN={int(close.isna().sum().sum())}, 耗时 {time.time() - t0:.1f}s")

    # ② 特征
    t0 = time.time()
    panels = build_feature_panels(close)
    feature_names = list(panels)
    log(f"[特征] {len(feature_names)} 个: {feature_names} ({time.time() - t0:.1f}s)")

    # ③ 标签
    t0 = time.time()
    ret_panel, cls_panel = build_label_panels(close, args.horizon, args.label)
    log(f"[标签] kind={args.label}, H={args.horizon} ({time.time() - t0:.1f}s)")

    # ④ 堆叠成池化长表（保留未过滤版本 X_all，供稳健性变体复用）
    dates = close.index
    X_all = panel_to_long(panels, dates, symbols)
    y_ret_all = panel_to_long({"ret": ret_panel}, dates, symbols)["ret"]
    y_cls_all = panel_to_long({"cls": cls_panel}, dates, symbols)["cls"]
    tpos_map = pd.Series(np.arange(len(dates)), index=dates)
    tpos_all = tpos_map.reindex(X_all.index.get_level_values(0)).to_numpy(dtype="int64")

    valid = X_all.notna().all(axis=1) & y_ret_all.notna()
    if args.label == "barrier":
        valid &= y_cls_all.notna()
    X = X_all[valid]
    y_ret = y_ret_all[valid].to_numpy(dtype="float64")
    y_cls = y_cls_all[valid].to_numpy(dtype="float64")
    tpos = tpos_all[valid.to_numpy()]
    actual = pd.Series(y_ret, index=X.index)
    print(f"[样本] 池化样本 {X.shape[0]} 条 = {X.index.get_level_values(0).nunique()} 个交易日 × "
          f"{X.index.get_level_values(1).nunique()} 只标的；特征 {X.shape[1]} 个")
    if X.shape[0] < 500:
        print("样本过少，退出。")
        return 2

    # ⑤ 严格样本外切分
    n_dates = len(dates)
    schemes = make_splits(n_dates, args.cv, args.horizon, args.train_size, args.test_size,
                          args.n_splits, args.min_train_dates)
    for name, folds in schemes.items():
        if not folds:
            print(f"[CV] {name}: 无可用折（请减小 --min-train-dates 或 --train-size）")
            continue
        te_lo = dates[min(f[2][0] for f in folds)]
        te_hi = dates[max(f[2][-1] for f in folds)]
        log(f"[CV] {name}: {len(folds)} 折，训练位置 "
            f"{min(f[1][0] for f in folds)}~{max(f[1][-1] for f in folds)}，"
            f"样本外区间 {te_lo.date()} ~ {te_hi.date()}")

    # ⑥ 训练 + 样本外打分 + 评估
    specs = model_specs(not args.no_gbm, args.gbm_max_rows, args.gbm_estimators)
    models_out: Dict[str, Dict[str, Dict[str, object]]] = {}
    scores_out: Dict[str, Dict[str, pd.Series]] = {}
    oos_range: Dict[str, str] = {}
    n_rebalance: Dict[str, int] = {}
    for scheme, folds in schemes.items():
        if not folds:
            continue
        t0 = time.time()
        raw = fit_predict_folds(folds, X, tpos, y_ret, y_cls, specs)
        log(f"[训练] {scheme}: {len(specs)} 个模型 × {len(folds)} 折，耗时 {time.time() - t0:.1f}s")
        per_model: Dict[str, Dict[str, object]] = {}
        scores: Dict[str, pd.Series] = {}
        for name, arr in raw.items():
            sc = pd.Series(arr, index=X.index)
            scores[name] = sc
            m = evaluate_score(sc, actual, args.horizon, args.top_k, args.bottom_k,
                               args.min_assets, [f[0] for f in folds], dates)
            if args.label == "forward":
                m["label_cls_mean"] = float(np.nanmean(y_cls))
            per_model[name] = m
        models_out[scheme] = per_model
        scores_out[scheme] = scores
        first = min(m.get("oos_start", "") for m in per_model.values() if "error" not in m)
        last = max(m.get("oos_end", "") for m in per_model.values() if "error" not in m)
        oos_range[scheme] = f"{first} ~ {last}"
        n_rebalance[scheme] = max(int(m.get("n_rebalance", 0)) for m in per_model.values())

        print(f"\n--- {scheme} 样本外结果（{oos_range[scheme]}，H={args.horizon}，"
              f"多头 top{args.top_k}）---")
        for name, m in per_model.items():
            if "error" in m:
                print(f"  {name:<9} 样本不足")
                continue
            print(f"  {name:<9} hit_rate={m['hit_rate']:.4f}  precision={m['precision']:.4f}  "
                  f"F1={m['f1']:.4f}  pooledIC={m['ic_pooled_spearman']:+.5f}  "
                  f"日频IC={m['ic_daily_mean']:+.5f}(t={m['ic_daily_t']:+.2f})  "
                  f"多头累计={m['long_only_cum']:+.3f}(t={m['long_only_t']:+.2f})  "
                  f"超额累计={m['excess_cum']:+.3f}(t={m['excess_t']:+.2f})  "
                  f"多空累计={m['long_short_cum']:+.3f}(t={m['long_short_t']:+.2f})")

    if not models_out:
        print("没有任何可用的样本外切分，退出。")
        return 3

    # ⑦ 诊断：单因子裸 IC / 逐折稳定性 / 稳健性设定
    t0 = time.time()
    diag_scheme = "walk_forward" if schemes.get("walk_forward") else \
        next(s for s, f in schemes.items() if f)
    diag_scores = scores_out[diag_scheme]
    oos_mask = diag_scores[next(iter(diag_scores))].notna() & actual.notna()
    score_cols = pd.DataFrame({f"MODEL:{n}": s for n, s in diag_scores.items()})[oos_mask]
    # cs_* 特征是同日截面单调变换，逐日截面 spearman IC 与原特征**恒等**，故不重复列出
    raw_cols = [c for c in X.columns if not str(c).startswith("cs_")]
    n_cs_dropped = X.shape[1] - len(raw_cols)
    ic_panel = daily_ic_multi(pd.concat([X[oos_mask][raw_cols], score_cols], axis=1),
                              actual[oos_mask], args.min_assets)
    ic_tab = ic_summary(ic_panel)
    feat_rows = ic_tab.loc[[c for c in ic_tab.index if not str(c).startswith("MODEL:")]]
    feat_rows = feat_rows.reindex(feat_rows["ic_daily_mean"].abs()
                                  .sort_values(ascending=False).index)
    model_rows = ic_tab.loc[[c for c in ic_tab.index if str(c).startswith("MODEL:")]]
    factor_ic = pd.concat([feat_rows, model_rows])
    factor_ic_range = (f"{ic_panel.index.min().date()} ~ {ic_panel.index.max().date()}，"
                       f"{ic_panel.shape[0]} 个截面日，{diag_scheme}")
    factor_ic_note = (f"表中未列出 {n_cs_dropped} 个 `cs_*` 特征：它们是同日截面的单调变换"
                      f"（z-score），逐日截面 spearman IC 与对应原特征**完全相同**。"
                      if n_cs_dropped > 0 else "")
    watch = [str(c) for c in feat_rows.index[:3]]
    stability = fold_ic_stability(schemes[diag_scheme], X, tpos, y_ret, watch, args.min_assets)

    ridge_factory = lambda: kml.RidgeRegression(alpha=100.0, standardize=True)  # noqa: E731
    robustness: List[Dict[str, object]] = []
    ref = models_out.get(diag_scheme, {}).get("Ridge")
    if ref is not None and "error" not in ref:
        robustness.append(dict(ref, name=f"主设定：H={args.horizon}，目标=原始收益，"
                                       f"Ridge，{diag_scheme}"))
    xs_long = panel_to_long({"y": xs_excess_panel(ret_panel)}, dates, symbols)["y"]
    v1 = run_walk_forward_variant(X_all, tpos_all, dates, xs_long, y_ret_all, args.horizon,
                                  args, ridge_factory)
    v1["name"] = f"目标=截面超额收益（当日去均值），H={args.horizon}，Ridge，walk_forward"
    robustness.append(v1)
    h2 = args.horizon * 2
    ret2, _cls2 = build_label_panels(close, h2, args.label)
    y2 = panel_to_long({"y": ret2}, dates, symbols)["y"]
    v2 = run_walk_forward_variant(X_all, tpos_all, dates, y2, y2, h2, args, ridge_factory)
    v2["name"] = f"持有期翻倍 H={h2}，目标=原始收益，Ridge，walk_forward"
    robustness.append(v2)
    log(f"[诊断] 单因子 IC {factor_ic.shape[0]} 行 / 逐折 {stability.shape[0]} 折 / "
        f"稳健性 {len(robustness)} 个设定，耗时 {time.time() - t0:.1f}s")

    print(f"\n--- 单因子裸 IC（{diag_scheme} 样本外日频截面 IC，按 |IC| 降序前 6）---")
    print(factor_ic.head(6).round(5).to_string())
    print("\n--- 逐折模型 IC（岭回归）与关注特征的系数 vs 同期实现 IC ---")
    print(stability.round(5).to_string(index=False))
    print("\n--- 稳健性设定 ---")
    for r in robustness:
        if "error" in r:
            print(f"  {r['name']}: {r['error']}")
            continue
        print(f"  {r['name']}: 日频IC={r['ic_daily_mean']:+.5f}(t={r['ic_daily_t']:+.2f})  "
              f"hit_rate={r['hit_rate']:.4f}  超额累计={r['excess_cum']:+.3f}"
              f"(t={r['excess_t']:+.2f})")

    # ⑧ 特征重要性（末尾若干折的样本外数据，跨折平均）
    imp_scheme = diag_scheme
    t0 = time.time()
    imp, imp_meta = feature_importance_last_folds(imp_scheme, schemes[imp_scheme], X, tpos, y_ret,
                                                  min_assets=args.min_assets, max_rows=6000,
                                                  n_repeats=args.imp_repeats,
                                                  n_folds=args.imp_folds)
    log(f"[重要性] {imp_scheme} 最后 {imp_meta['n_folds']} 折，平均基线日频截面 IC="
        f"{imp_meta['baseline_ic']:+.5f}，耗时 {time.time() - t0:.1f}s")
    print("\n--- 置换重要性 top 8（样本外日频截面 IC 下降幅度）---")
    print(imp.head(8).round(5).to_string())

    # ⑨ 结论（全部由样本外数字驱动，不做主观拔高）
    flat = [(sc, nm, m) for sc, pm in models_out.items() for nm, m in pm.items()
            if "error" not in m]
    conclusions: List[str] = []
    if flat:
        best = max(flat, key=lambda x: float(x[2]["ic_daily_t"]))
        worst = min(flat, key=lambda x: float(x[2]["ic_daily_t"]))
        bm, wm = best[2], worst[2]
        n_pos = sum(1 for x in flat if float(x[2]["ic_daily_t"]) >= 2.0)
        n_neg = sum(1 for x in flat if float(x[2]["ic_daily_t"]) <= -2.0)
        n_flat = len(flat) - n_pos - n_neg
        max_ex_t0 = max(abs(float(x[2]["excess_t"])) for x in flat)
        if n_pos == 0 and n_neg == 0:
            verdict = "**总体判断：样本外没有可用的预测力**（所有组合的日频 IC 均 |t|<2）。"
        elif n_pos == 0:
            verdict = ("**总体判断：模型在样本外没有正向预测力**——达到 |t|≥2 的组合全部为**负**"
                       "（即系统性做错方向），其余落在噪声区间，不构成可实盘的 alpha。")
        elif n_neg > 0:
            verdict = ("**总体判断：没有稳定的样本外信号**——不同模型/切分下 IC 的显著方向相反，"
                       "正负相互抵消，属于典型的噪声形态。")
        elif max_ex_t0 < 2.0:
            verdict = ("**总体判断：样本外只有微弱且不稳定的排序能力**——个别组合的截面 IC 达到 "
                       "|t|≥2，但落到组合层面（扣除同期等权基准）无一显著，"
                       "不构成可实盘的 alpha。")
        else:
            verdict = ("**总体判断：存在值得进一步复核的微弱样本外信号**，"
                       "但在计入交易成本、容量约束与多重检验校正之前，不应视为可实盘的 alpha。")
        conclusions.append(verdict)
        conclusions.append(
            f"最正向的组合是 **{best[0]} / {best[1]}**：日频截面 IC="
            f"{float(bm['ic_daily_mean']):+.5f}（t={float(bm['ic_daily_t']):+.2f}），"
            f"hit_rate={float(bm['hit_rate']):.4f}，多头累计={float(bm['long_only_cum']):+.3f}"
            f"（同期等权基准累计={float(bm['benchmark_cum']):+.3f}），"
            f"超额累计={float(bm['excess_cum']):+.3f}（t={float(bm['excess_t']):+.2f}）。")
        if float(wm["ic_daily_t"]) <= -2.0:
            conclusions.append(
                f"最差的组合 **{worst[0]} / {worst[1]}** 的日频 IC 显著为**负**"
                f"（IC={float(wm['ic_daily_mean']):+.5f}，t={float(wm['ic_daily_t']):+.2f}）："
                f"模型在样本外系统性做错方向，这是训练窗口关系在新窗口反转的结果，"
                f"不能当成「反着用的 alpha」。")
        t_lo = min(float(x[2]["ic_daily_t"]) for x in flat)
        t_hi = max(float(x[2]["ic_daily_t"]) for x in flat)
        n_ic_pos = sum(1 for x in flat if float(x[2]["ic_daily_mean"]) > 0)
        n_ic_neg = len(flat) - n_ic_pos
        if n_pos > 0 and n_neg > 0:
            shape_zh = ("**显著方向在不同模型/切分之间翻转**（既有显著为正、也有显著为负），"
                        "这本身就是「没有稳定信号」的直接证据")
        elif n_pos == 0 and n_neg == 0:
            shape_zh = "**没有任何一个组合达到 |t|≥2**，全部落在噪声区间"
        elif n_neg > 0 and n_pos == 0:
            shape_zh = (f"达到显著的组合全部为负（模型系统性做错方向），其余 {n_flat} 个不显著；"
                        f"IC 均值符号为 {n_ic_pos} 正 / {n_ic_neg} 负，跨模型与切分并不一致")
        else:
            mix = (f"，但 IC 均值符号为 {n_ic_pos} 正 / {n_ic_neg} 负"
                   if (n_ic_pos > 0 and n_ic_neg > 0) else "")
            shape_zh = (f"显著项方向一致为正{mix}；幅度随切分剧烈摆动（见第 5.2 节）、"
                        f"且超额收益无一显著，稳健性存疑")
        conclusions.append(
            f"{len(flat)} 个（模型 × CV 方案）组合中：日频 IC t≥+2 的 {n_pos} 个、"
            f"t≤−2 的 {n_neg} 个、|t|<2 的 {n_flat} 个；IC t 值区间 {t_lo:+.2f} ~ {t_hi:+.2f}。"
            f"{shape_zh}。")
        pos_ratio = float(np.mean(actual[oos_mask].to_numpy() > 0)) if oos_mask.any() else 0.5
        naive = max(pos_ratio, 1.0 - pos_ratio)
        max_hr = max(float(x[2]["hit_rate"]) for x in flat)
        if max_hr <= naive + 0.005:
            cmp_hr = (f"**不高于**无脑单边基准（同期「H 日收益为正」的自然占比 "
                      f"{pos_ratio:.4f}，即全押涨/全押跌可拿到 {naive:.4f} 的命中率）")
        else:
            cmp_hr = (f"仅比无脑单边基准（{naive:.4f}）高 {max_hr - naive:.4f}，"
                      f"这点优势远小于其跨窗口波动")
        conclusions.append(
            f"方向命中率最高仅 {max_hr:.4f}，{cmp_hr}——"
            f"「预测未来 {args.horizon} 日涨跌」在这批标的上基本不可行。")
        if factor_ic.shape[0]:
            frows = factor_ic.loc[[c for c in factor_ic.index if not str(c).startswith("MODEL:")]]
            mrows = factor_ic.loc[[c for c in factor_ic.index if str(c).startswith("MODEL:")]]
            fbest = str(frows["ic_daily_mean"].abs().idxmax())
            fb = frows.loc[fbest]
            best_factor_ic = float(fb["ic_daily_mean"])
            best_factor_t = float(fb["ic_daily_t"])
            best_model_ic = float(mrows["ic_daily_mean"].abs().max()) if mrows.shape[0] else 0.0
            best_model_t = float(mrows["ic_daily_t"].abs().max()) if mrows.shape[0] else 0.0
            if abs(best_factor_ic) >= best_model_ic:
                cmp_zh = (f"裸因子就已经**不弱于**任何模型打分（模型最大 |IC|={best_model_ic:.5f}）："
                          f"把 {len(feature_names)} 个特征交给模型组合，并没有比直接用单个技术特征更稳")
            else:
                cmp_zh = (f"模型打分（最大 |IC|={best_model_ic:.5f}）确实**优于**最强裸因子，"
                          f"但优势仅 {best_model_ic - abs(best_factor_ic):.5f}，"
                          f"远小于其跨窗口波动幅度，不足以支撑实盘")
            conclusions.append(
                f"不建模的裸因子对照：IC 绝对值最大的是 `{fbest}`"
                f"（{FEATURE_DESC.get(fbest, '')}；IC={best_factor_ic:+.5f}，t={best_factor_t:+.2f}）。"
                f"{cmp_zh}。")
            if abs(best_factor_t) >= 3.0 and abs(best_factor_t) > best_model_t:
                sign_zh = ("（IC 为**负**，即应反着用该特征打分：做多低值、做空高值——"
                           "对波动率类特征这就是经典的**低波动异象**）"
                           if best_factor_ic < 0 else
                           "（IC 为正，即直接按该特征从大到小排序做多）")
                conclusions.append(
                    f"**最值得注意的对照**：完全不用模型、直接把 `{fbest}` 当打分（按其符号定向），"
                    f"在同一段样本外的日频 IC 就有 {best_factor_ic:+.5f}（t={best_factor_t:+.2f}）"
                    f"{sign_zh}，其绝对值强于任何模型打分（模型最大 |t|={best_model_t:.2f}）。"
                    f"也就是说，在这批标的上**单个技术特征的样本外有效性高于 "
                    f"{len(feature_names)} 特征的 ML 组合**：模型未能稳定提取该信号，"
                    f"反而被其余特征的时变噪声抵消。这提示「加更多特征 + 更复杂模型」"
                    f"并不是无条件的改进。")
        top_lo = max(flat, key=lambda x: float(x[2]["long_only_cum"]))
        tm = top_lo[2]
        max_ex_t = max(abs(float(x[2]["excess_t"])) for x in flat)
        ex_word = ("仅" if abs(float(tm["excess_cum"])) < abs(float(tm["long_only_cum"]))
                   else "为")
        beta_zh = (f"同期等权基准累计 {float(tm['benchmark_cum']):+.3f}，"
                   f"超额（多头 − 基准）{ex_word} {float(tm['excess_cum']):+.3f}"
                   f"（t={float(tm['excess_t']):+.2f}）")
        if abs(float(tm["benchmark_cum"])) > abs(float(tm["excess_cum"])):
            head = f"多头累计收益最高的是 **{top_lo[0]} / {top_lo[1]}**（{float(tm['long_only_cum']):+.3f}），但其中大部分来自**市场 beta**：{beta_zh}"
        else:
            head = f"多头累计收益最高的是 **{top_lo[0]} / {top_lo[1]}**（{float(tm['long_only_cum']):+.3f}），{beta_zh}"
        tail = ("；所有组合的超额 |t| 均 < 2，无一显著。" if max_ex_t < 2.0
                else f"；全部组合中最大的超额 |t| 也仅 {max_ex_t:.2f}。")
        conclusions.append(head + tail)
        st_ics = stability["model_ic"] if stability.shape[0] else pd.Series(dtype="float64")
        if st_ics.size > 2:
            t_fold, _p = _t(st_ics)
            conclusions.append(
                f"逐折诊断：岭回归在 {st_ics.size} 个滚动窗口上的日频 IC 为 "
                f"{st_ics.min():+.4f} ~ {st_ics.max():+.4f}（正 {int((st_ics > 0).sum())} 折 / "
                f"负 {int((st_ics < 0).sum())} 折，跨折 t={t_fold:+.2f}），"
                f"波动幅度远大于均值——**信号被窗口间的不稳定完全抵消**。")
        conclusions.append(
            "所有超参数在看到任何样本外结果之前就已固定（未做调参、未做模型选择），"
            "因此这是一次**无偏的一次性检验**。"
            + ("这个「没有可用 alpha」的结论，比反复调参后得到的漂亮数字更可信。"
               if (n_pos == 0 and n_neg == 0) else
               "即便如此，上文任何达到 |t|≥2 的项都还必须先通过交易成本、容量与"
               "多重检验的考验；由于它们的超额收益 t 值均未达标，本文不把它们当作可实盘的 alpha。"))

    # ⑩ 写产出
    os.makedirs(args.out_dir, exist_ok=True)
    runtime = time.time() - t_start
    env = {"python": platform.python_version(), "numpy": np.__version__,
           "pandas": pd.__version__, "scipy": kml.evaluate.has_scipy(),
           "platform": platform.platform()}
    config = {"horizon": args.horizon, "train_size": args.train_size,
              "test_size": args.test_size, "n_splits": args.n_splits, "cv": args.cv,
              "label": args.label, "top_k": args.top_k, "bottom_k": args.bottom_k,
              "min_assets": args.min_assets, "min_train_dates": args.min_train_dates,
              "max_assets": args.max_assets, "with_gbm": not args.no_gbm,
              "gbm_max_rows": args.gbm_max_rows, "gbm_estimators": args.gbm_estimators,
              "imp_repeats": args.imp_repeats,
              "models": [s[0] for s in specs]}
    data_info = {"dir": os.path.abspath(args.data_dir), "n_assets": int(close.shape[1]),
                 "n_dates": int(close.shape[0]), "start": str(close.index[0].date()),
                 "end": str(close.index[-1].date()), "adjust": "hfq(后复权)",
                 "symbols": list(close.columns), "n_samples": int(X.shape[0]),
                 "n_dates_valid": int(X.index.get_level_values(0).nunique())}

    imp_out = imp.reset_index()
    imp_out.columns = ["feature", "importance_mean", "importance_std", "ridge_coef_std"]
    imp_path = os.path.join(args.out_dir, "feature_importance.csv")
    imp_out.to_csv(imp_path, index=False, float_format="%.8f")

    metrics = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(runtime, 1),
        "env": env,
        "config": config,
        "data": data_info,
        "features": feature_names,
        "cv": {s: {"n_folds": len(f), "folds": [x[0] for x in f],
                   "train_dates_min": int(min(x[1].size for x in f)),
                   "train_dates_max": int(max(x[1].size for x in f)),
                   "horizon": args.horizon,
                   "purge_rule": "train_pos + H < test_first_pos"}
               for s, f in schemes.items()},
        "n_folds": {s: len(f) for s, f in schemes.items()},
        "oos_range": oos_range,
        "n_rebalance": n_rebalance,
        "models": models_out,
        "importance": {"meta": imp_meta,
                       "table": imp_out.round(8).to_dict(orient="records")},
        "diagnostics": {
            "scheme": diag_scheme,
            "factor_ic_range": factor_ic_range,
            "factor_ic_note": factor_ic_note,
            "watch_features": watch,
            "factor_ic": factor_ic.round(8).reset_index().to_dict(orient="records"),
            "fold_stability": stability.round(8).to_dict(orient="records"),
            "robustness": [{k: v for k, v in r.items()} for r in robustness],
        },
        "conclusions": conclusions,
    }
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2, default=float)

    ctx = {"config": config, "data": data_info, "features": feature_names,
           "models": models_out, "importance": imp, "importance_meta": imp_meta,
           "conclusions": conclusions, "top_k": args.top_k, "bottom_k": args.bottom_k,
           "oos_range": oos_range, "n_rebalance": n_rebalance,
           "n_folds": {s: len(f) for s, f in schemes.items()},
           "n_models": len(specs), "n_schemes": len(models_out),
           "factor_ic": factor_ic, "factor_ic_range": factor_ic_range,
           "factor_ic_note": factor_ic_note,
           "fold_stability": stability, "watch_features": watch,
           "robustness": robustness,
           "runtime_seconds": round(runtime, 1), "env": env}
    report_path = os.path.join(args.out_dir, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(render_report(ctx))

    print(f"\n[产出] {report_path}")
    print(f"[产出] {metrics_path}")
    print(f"[产出] {imp_path}")
    print(f"[完成] 总耗时 {runtime:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
