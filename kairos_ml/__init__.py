"""Kairos ML —— 面向金融机器学习的 numpy 自研工具库。

本机不含 sklearn，故全部模型均用 numpy 从零实现；必需依赖仅 numpy 与 pandas，
scipy 为可选增强（正态 CDF/分位数，缺失时自动回退 numpy 近似）。

模块导览
--------
- :mod:`kairos_ml.features` : 滚动/滞后/截面特征与自研技术指标 (SMA/EMA/RSI/MACD/已实现波动率)。
- :mod:`kairos_ml.labels`   : 远期收益、三重障碍法 (Triple-Barrier)、元标签 (Meta-Labeling)。
- :mod:`kairos_ml.cv`       : walk-forward 切分与 purged/embargo K 折（防信息泄漏）。
- :mod:`kairos_ml.models`   : 岭回归、逻辑回归、梯度提升回归、置换重要性。
- :mod:`kairos_ml.evaluate` : 命中率、精确率/召回率/F1、信号 PnL、IC、VaR 等。
- :mod:`kairos_ml.pipeline` : ``ModelPipeline`` 串联特征构造与模型 fit/predict。
"""
from __future__ import annotations

from . import cv, evaluate, features, labels, models, pipeline
from .cv import purged_kfold, walk_forward_splits
from .evaluate import (
    f1,
    hit_rate,
    ic,
    precision,
    recall,
    signal_pnl,
    t_statistic,
    value_at_risk,
)
from .features import (
    cross_sectional_rank,
    cross_sectional_zscore,
    ema,
    lags,
    macd,
    realized_volatility,
    rolling_mean,
    rolling_momentum,
    rolling_std,
    rsi,
    sma,
)
from .labels import forward_returns, meta_labeling, triple_barrier
from .models import (
    BaseEstimator,
    GradientBoostingRegressor,
    LogisticRegression,
    RidgeRegression,
    accuracy_score,
    neg_mse,
    permutation_importance,
    r2_score,
)
from .pipeline import ModelPipeline

__version__ = "0.1.0"

__all__ = [
    # 子模块
    "features", "labels", "cv", "models", "evaluate", "pipeline",
    # features
    "rolling_mean", "rolling_std", "rolling_momentum", "lags",
    "cross_sectional_rank", "cross_sectional_zscore",
    "sma", "ema", "rsi", "macd", "realized_volatility",
    # labels
    "forward_returns", "triple_barrier", "meta_labeling",
    # cv
    "walk_forward_splits", "purged_kfold",
    # models
    "BaseEstimator", "RidgeRegression", "LogisticRegression",
    "GradientBoostingRegressor", "permutation_importance",
    "r2_score", "neg_mse", "accuracy_score",
    # evaluate
    "hit_rate", "precision", "recall", "f1", "signal_pnl", "ic",
    "t_statistic", "value_at_risk",
    # pipeline
    "ModelPipeline",
    "__version__",
]
