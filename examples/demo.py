"""Kairos ML 演示：合成行情 → 三重障碍打标 → 保序 CV → 逻辑回归 → 绩效评估。

运行： python examples/demo.py

全流程离线、固定随机种子、可复现；所有模型均为 numpy 自研，不依赖 sklearn。

流程
----
1. 合成带正自相关（动量结构）的价格，使方向信号可被学习。
2. 三重障碍法打标：为每个入场时刻设置止盈/止损/垂直轨，得到 {-1,0,+1} 标签
   与该笔「障碍交易」的实际出场收益 tb['ret']（作为样本外真实盈亏口径）。
3. 用 walk-forward 与 purged K-fold 两套保序 CV 滚动训练 numpy 自研逻辑回归。
4. 汇总样本外 hit_rate / precision / 信号 PnL / IC。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import kairos_ml as kml


def make_momentum_prices(n: int = 900, kappa: float = 0.05, rho: float = 0.4,
                         vol: float = 0.011, seed: int = 7) -> pd.Series:
    """合成「均值回复价格 + 动量收益」序列，使方向信号可被学习且障碍标签大致均衡。

    对数价格围绕 log(100) 均值回复（避免单边趋势导致上/下轨失衡），同时收益率带
    正自相关 rho（动量），使过去动量对未来方向有真实预测力：
        r[t] = -kappa·(logp[t-1]-log 100) + rho·r[t-1] + vol·N(0,1)
    """
    rng = np.random.default_rng(seed)
    log100 = np.log(100.0)
    logp = np.zeros(n)
    r = np.zeros(n)
    logp[0] = log100
    for t in range(1, n):
        r[t] = -kappa * (logp[t - 1] - log100) + rho * r[t - 1] + vol * rng.standard_normal()
        logp[t] = logp[t - 1] + r[t]
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(np.exp(logp), index=idx, name="px")


def build_features(prices: pd.Series) -> pd.DataFrame:
    """由价格构造特征矩阵（全部只用截至当时的历史，无未来函数）。"""
    macd = kml.macd(prices, fast=8, slow=21, signal=5)
    return pd.DataFrame({
        "mom3": kml.rolling_momentum(prices, 3),
        "mom10": kml.rolling_momentum(prices, 10),
        "rsi14": kml.rsi(prices, 14),
        "macd_hist": macd["hist"],
        "rv20": kml.realized_volatility(prices, 20),
        "zscore": kml.rolling_mean(prices, 5) / prices - 1.0,
    }, index=prices.index)


def run_cv(splits, X, y, outcome):
    """在给定 CV 切分上做样本外滚动训练，汇总 OOS 预测与真实结果。"""
    cls_all, proba_all, y_all, out_all = [], [], [], []
    for tr, te in splits:
        model = kml.LogisticRegression(C=1.0).fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        cls_all.append((proba >= 0.5).astype(int))
        proba_all.append(proba)
        y_all.append(y.iloc[te].to_numpy())
        out_all.append(outcome.iloc[te].to_numpy())
    return (np.concatenate(cls_all), np.concatenate(proba_all),
            np.concatenate(y_all), np.concatenate(out_all))


def report(name, cls, proba, y_true, outcome):
    """把一组样本外预测汇总成可读的绩效指标并打印。"""
    signal = 2.0 * proba - 1.0               # 概率 -> 连续仓位 [-1, +1]
    pnl = kml.signal_pnl(pd.Series(signal), pd.Series(outcome))
    t, p = kml.t_statistic(pnl)
    print(f"\n【{name}】样本外样本数 = {len(cls)}")
    print(f"  命中率 hit_rate (方向)         : {kml.hit_rate(pd.Series(outcome), pd.Series(signal)):.4f}")
    print(f"  精确率 precision (预测上轨)     : {kml.precision(y_true, cls, pos_label=1):.4f}")
    print(f"  召回率 recall                  : {kml.recall(y_true, cls, pos_label=1):.4f}")
    print(f"  F1                            : {kml.f1(y_true, cls, pos_label=1):.4f}")
    print(f"  信号 PnL 累计 / 单期均值        : {pnl.sum():.4f} / {pnl.mean():.5f}")
    print(f"  信号 PnL t 统计量 (p 值)        : {t:.3f} (p={p:.3f})")
    print(f"  IC (spearman, 概率 vs 障碍收益) : {kml.ic(proba, outcome, method='spearman'):.4f}")


def main():
    print("=" * 64)
    print("Kairos ML 演示：三重障碍打标 + 保序交叉验证 + numpy 自研逻辑回归")
    print("=" * 64)

    prices = make_momentum_prices()
    feat = build_features(prices)

    # ① 三重障碍打标：pt/sl 按波动率倍数，垂直轨 5 期
    vertical = 5
    tb = kml.triple_barrier(prices, pt=1.5, sl=1.5, vertical=vertical, vol_window=20)
    y = (tb["label"] == 1).astype(int)         # 上轨=1，其余=0（分类目标）
    outcome = tb["ret"]                        # 该笔障碍交易的实际出场收益（真实盈亏口径）

    # 对齐特征与标签，丢弃预热期/末尾 NaN
    X = feat.reindex(tb.index)
    mask = X.notna().all(axis=1) & outcome.notna()
    X, y, outcome = X[mask], y[mask], outcome[mask]
    print(f"有效样本 = {len(X)}，特征 = {list(X.columns)}")
    print(f"标签分布 (上轨=1): {int(y.sum())} / {len(y)}；三重障碍 barrier 计数:")
    print(tb["barrier"].value_counts().to_string())

    # ② walk-forward（滚动窗口）样本外评估
    wf = kml.walk_forward_splits(len(X), train_size=250, test_size=50)
    cls, proba, y_true, out_oos = run_cv(wf, X, y, outcome)
    report(f"Walk-Forward（{len(wf)} 折滚动）", cls, proba, y_true, out_oos)

    # ③ purged K 折（purge + embargo，防标签泄漏）样本外评估
    pk = kml.purged_kfold(len(X), n_splits=6, embargo=vertical, label_horizon=vertical)
    cls2, proba2, y_true2, out_oos2 = run_cv(pk, X, y, outcome)
    report(f"Purged K-Fold（{len(pk)} 折, embargo={vertical}）", cls2, proba2, y_true2, out_oos2)

    # ④ 置换重要性：哪个特征对分类真正有用
    full = kml.LogisticRegression(C=1.0).fit(X, y)
    imp = kml.permutation_importance(full, X, y, scorer="accuracy",
                                     n_repeats=8, random_state=0)
    print("\n【置换重要性 accuracy 下降】")
    print(imp.round(4).to_string())

    # ⑤ 顺带展示：梯度提升回归拟合非线性 + R²
    rng = np.random.default_rng(1)
    Xg = rng.standard_normal((400, 2))
    yg = Xg[:, 0] ** 2 + np.sin(2 * Xg[:, 1])
    gbm = kml.GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                        max_depth=3, random_state=0).fit(Xg, yg)
    print("\n【GradientBoostingRegressor 非线性拟合】")
    print(f"  训练 R² = {kml.r2_score(yg, gbm.predict(Xg)):.4f}（常数预测 R²=0）")
    print(f"  scipy 可用（正态 CDF/分位数增强）: {kml.evaluate.has_scipy()}")


if __name__ == "__main__":
    main()
