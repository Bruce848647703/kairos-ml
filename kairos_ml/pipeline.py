"""建模流水线：把「特征构造 → 模型 fit/predict」串成一致接口。

:class:`ModelPipeline` 接受一个特征构造器（callable 或 name→callable 映射）与一个
:mod:`kairos_ml.models` 中的模型，提供 ``fit`` / ``transform`` / ``predict``：

- ``transform(raw)``  : 由原始数据（如价格）构造特征矩阵。
- ``fit(raw, y)``     : 构造特征并训练模型（自动按索引对齐、丢弃 NaN 行）。
- ``predict(raw)``    : 构造特征并用已训练模型预测。

特征构造器只用「截至当时」的历史数据（由使用者保证，见 :mod:`kairos_ml.features`），
配合 :mod:`kairos_ml.cv` 的保序切分即可做到无未来函数的滚动建模。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .models import BaseEstimator

FeatureBuilder = Callable[[object], Union[pd.Series, pd.DataFrame]]
BuilderSpec = Union[FeatureBuilder, Dict[str, FeatureBuilder], None]


class ModelPipeline:
    """特征 + 模型的轻量流水线。

    参数
    ----
    model:           实现 ``fit`` / ``predict`` 的模型（见 :mod:`kairos_ml.models`）。
    feature_builder: 特征构造器。可以是：
                     - ``None``                : raw 本身即特征矩阵；
                     - ``callable``            : raw -> DataFrame/Series；
                     - ``dict[name, callable]``: 分别构造并按列拼接。
    feature_columns: 可选，显式指定/固定特征列顺序（transform 后据此重排）。
    """

    def __init__(self, model: BaseEstimator, feature_builder: BuilderSpec = None,
                 feature_columns: Optional[List[str]] = None):
        self.model = model
        self.feature_builder = feature_builder
        self.feature_columns = list(feature_columns) if feature_columns is not None else None

    # ------------------------------------------------------------------
    def _build_features(self, raw) -> pd.DataFrame:
        """由原始数据构造特征 DataFrame。"""
        if self.feature_builder is None:
            X = raw
        elif isinstance(self.feature_builder, dict):
            parts: Dict[str, Union[pd.Series, pd.DataFrame]] = {}
            for name, fn in self.feature_builder.items():
                parts[name] = fn(raw)
            X = _concat_features(parts)
        else:
            X = self.feature_builder(raw)

        if not isinstance(X, pd.DataFrame):
            if isinstance(X, pd.Series):
                X = X.to_frame()
            else:
                X = pd.DataFrame(np.asarray(X))
        if self.feature_columns is not None:
            missing = [c for c in self.feature_columns if c not in X.columns]
            if missing:
                raise KeyError(f"特征缺少列: {missing}")
            X = X[self.feature_columns]
        return X

    # ------------------------------------------------------------------
    def transform(self, raw) -> pd.DataFrame:
        """构造并返回特征矩阵（不训练）。"""
        return self._build_features(raw)

    def fit(self, raw, y) -> "ModelPipeline":
        """构造特征并训练底层模型。

        若 y 为带索引的 ``Series``，会与特征按索引对齐后丢弃 NaN 行，
        从而自然处理特征预热期与标签末尾期的缺失。
        """
        X = self._build_features(raw)
        if self.feature_columns is None:
            self.feature_columns = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, raw) -> np.ndarray:
        """构造特征并用已训练模型预测。"""
        X = self._build_features(raw)
        return self.model.predict(X)

    def fit_predict(self, raw, y) -> np.ndarray:
        """便捷方法：``fit`` 后立即对同一份 raw ``predict``（含训练集内拟合值）。"""
        return self.fit(raw, y).predict(raw)

    def predict_proba(self, raw) -> np.ndarray:
        """分类模型的类别概率（要求底层模型实现 ``predict_proba``）。"""
        X = self._build_features(raw)
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("底层模型不支持 predict_proba")
        return self.model.predict_proba(X)


def _concat_features(parts: Dict[str, Union[pd.Series, pd.DataFrame]]) -> pd.DataFrame:
    """把 name->Series/DataFrame 的特征片段按列拼接，Series 以 name 命名列。"""
    frames: List[pd.DataFrame] = []
    for name, val in parts.items():
        if isinstance(val, pd.Series):
            frames.append(val.rename(name).to_frame())
        elif isinstance(val, pd.DataFrame):
            frames.append(val)
        else:
            frames.append(pd.Series(np.asarray(val), name=name).to_frame())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)
