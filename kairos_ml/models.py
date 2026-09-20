"""numpy 自研机器学习模型。

本机不含 sklearn，故所有模型均用 numpy 从零实现，统一 ``fit`` / ``predict`` 接口：

- :class:`RidgeRegression`         : 岭回归，闭式解 (XᵀX + αI)⁻¹Xᵀy，可选标准化。
- :class:`LogisticRegression`      : 逻辑回归，带 L2 正则的 Newton/IRLS 求解，输出概率。
- :class:`GradientBoostingRegressor`: 梯度提升回归，自研决策桩/浅树集成（MSE 损失）。
- :func:`permutation_importance`   : 置换重要性——打乱某列后度量的下降幅度。

设计要点
--------
- 纯 numpy 线性代数（``solve`` / ``pinv`` / ``lstsq``），无第三方 ML 依赖。
- 统一在设计矩阵层面处理截距与标准化；正则项不惩罚截距。
- ``fit`` 接受 ``DataFrame`` 或 ``ndarray``；传入 ``DataFrame`` 与 ``Series`` 目标时
  会按索引对齐并丢弃含 NaN 的行，天然适配金融特征/标签的稀疏边界。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._util import check_X, check_y


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
def _align_xy(X, y) -> Tuple[np.ndarray, np.ndarray, Optional[List]]:
    """把 (X, y) 对齐、去 NaN，并转成 numpy。返回 (Xv, yv, feature_names)。"""
    if isinstance(X, pd.DataFrame):
        names: Optional[List] = list(X.columns)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            ys = y if isinstance(y, pd.Series) else y.iloc[:, 0]
            joined = X.copy()
            joined["__target__"] = ys.reindex(X.index)
            joined = joined.dropna()
            yv = joined.pop("__target__").to_numpy(dtype="float64")
            Xv = joined.to_numpy(dtype="float64")
            names = list(joined.columns)
        else:
            Xv = X.to_numpy(dtype="float64")
            yv = check_y(y)
            m = min(Xv.shape[0], yv.shape[0])
            Xv, yv = Xv[:m], yv[:m]
            mask = np.isfinite(Xv).all(axis=1) & np.isfinite(yv)
            Xv, yv = Xv[mask], yv[mask]
        return Xv, yv, names

    Xv, names = check_X(X)
    yv = check_y(y)
    m = min(Xv.shape[0], yv.shape[0])
    Xv, yv = Xv[:m], yv[:m]
    mask = np.isfinite(Xv).all(axis=1) & np.isfinite(yv)
    return Xv[mask], yv[mask], names


def _convert_X(X, names: Optional[List]) -> np.ndarray:
    """预测期把 X 转成 numpy，并（若为 DataFrame）对齐训练时的列顺序。"""
    if isinstance(X, pd.DataFrame):
        if names is not None:
            X = X.reindex(columns=names)
        return X.to_numpy(dtype="float64")
    arr = np.asarray(X, dtype="float64")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------
class BaseEstimator(ABC):
    """所有模型的抽象基类，统一 fit/predict 接口。"""

    @abstractmethod
    def fit(self, X, y) -> "BaseEstimator":  # pragma: no cover - 抽象
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):  # pragma: no cover - 抽象
        raise NotImplementedError


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 logistic sigmoid。"""
    out = np.empty_like(z, dtype="float64")
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


# ---------------------------------------------------------------------------
# 岭回归
# ---------------------------------------------------------------------------
class RidgeRegression(BaseEstimator):
    """岭回归（L2 正则线性回归），闭式解 (XᵀX + αI)⁻¹Xᵀy。

    参数
    ----
    alpha:        L2 正则强度（>=0）。α→0 退化为普通最小二乘；α 越大对共线越稳健。
    fit_intercept: 是否拟合截距（截距不被正则惩罚）。
    standardize:  是否对特征做 z-score 标准化以改善条件数（内部记录均值/标准差）。

    属性
    ----
    coef_ : ndarray, 形状 (n_features,)
    intercept_ : float
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True,
                 standardize: bool = False):
        if alpha < 0:
            raise ValueError("alpha 不能为负")
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.standardize = bool(standardize)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.feature_names_in_: Optional[List] = None

    def _scale_fit(self, Xv: np.ndarray) -> np.ndarray:
        self.x_mean_ = Xv.mean(axis=0)
        std = Xv.std(axis=0)
        self.x_std_ = np.where(std < 1e-12, 1.0, std)
        return (Xv - self.x_mean_) / self.x_std_ if self.standardize else Xv

    def _scale_apply(self, Xv: np.ndarray) -> np.ndarray:
        return (Xv - self.x_mean_) / self.x_std_ if self.standardize else Xv

    def fit(self, X, y) -> "RidgeRegression":
        Xv, yv, names = _align_xy(X, y)
        self.feature_names_in_ = names
        p = Xv.shape[1]
        Xs = self._scale_fit(Xv)

        if self.fit_intercept:
            self._xs_mean = Xs.mean(axis=0)
            self._y_mean = float(yv.mean())
            A = Xs - self._xs_mean
            b = yv - self._y_mean
        else:
            self._xs_mean = np.zeros(p)
            self._y_mean = 0.0
            A, b = Xs, yv

        gram = A.T @ A + self.alpha * np.eye(p)
        rhs = A.T @ b
        try:
            w = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:  # 奇异时用伪逆兜底
            w = np.linalg.pinv(gram) @ rhs
        self.coef_ = w
        self.intercept_ = self._y_mean - float(w @ self._xs_mean) if self.fit_intercept else 0.0
        return self

    def predict(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("请先调用 fit()")
        Xv = _convert_X(X, self.feature_names_in_)
        Xs = self._scale_apply(Xv)
        return Xs @ self.coef_ + self.intercept_


# ---------------------------------------------------------------------------
# 逻辑回归
# ---------------------------------------------------------------------------
class LogisticRegression(BaseEstimator):
    """二分类逻辑回归，L2 正则，Newton/IRLS 求解（自研）。

    参数
    ----
    C:             正则强度的倒数（越大正则越弱）；``None`` 或 ``inf`` 表示不正则。
    fit_intercept: 是否拟合截距（不被正则惩罚）。
    standardize:   是否标准化特征（默认 True，改善 Newton 迭代条件数）。
    max_iter:      最大 Newton 迭代次数。
    tol:           收敛阈值（参数更新范数）。
    ridge_floor:   Hessian 对角的最小阻尼，保证可逆、防止完全可分时发散。

    属性
    ----
    coef_ (n_features,), intercept_ (float), classes_ = [0, 1]
    """

    def __init__(self, C: Optional[float] = 1.0, fit_intercept: bool = True,
                 standardize: bool = True, max_iter: int = 200, tol: float = 1e-8,
                 ridge_floor: float = 1e-9):
        self.C = C
        self.fit_intercept = bool(fit_intercept)
        self.standardize = bool(standardize)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.ridge_floor = float(ridge_floor)
        self.classes_ = np.array([0, 1])
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.feature_names_in_: Optional[List] = None

    @property
    def _alpha(self) -> float:
        if self.C is None or not np.isfinite(self.C) or self.C <= 0:
            return 0.0
        return 1.0 / float(self.C)

    def _scale_fit(self, Xv: np.ndarray) -> np.ndarray:
        self.x_mean_ = Xv.mean(axis=0)
        std = Xv.std(axis=0)
        self.x_std_ = np.where(std < 1e-12, 1.0, std)
        return (Xv - self.x_mean_) / self.x_std_ if self.standardize else Xv

    def _scale_apply(self, Xv: np.ndarray) -> np.ndarray:
        return (Xv - self.x_mean_) / self.x_std_ if self.standardize else Xv

    def fit(self, X, y) -> "LogisticRegression":
        Xv, yv, names = _align_xy(X, y)
        self.feature_names_in_ = names
        yv = (yv > 0).astype("float64")  # 允许 {0,1} 或 {-1,+1} 输入
        Xs = self._scale_fit(Xv)

        if self.fit_intercept:
            D = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        else:
            D = Xs
        p = D.shape[1]
        alpha = self._alpha
        reg = np.full(p, alpha)
        if self.fit_intercept:
            reg[-1] = 0.0            # 不惩罚截距
        R = np.diag(reg)

        w = np.zeros(p)
        for _ in range(self.max_iter):
            z = D @ w
            prob = _sigmoid(z)
            wd = np.clip(prob * (1.0 - prob), 1e-9, None)
            grad = D.T @ (prob - yv) + R @ w
            hess = D.T @ (D * wd[:, None]) + R + self.ridge_floor * np.eye(p)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:  # pragma: no cover
                step = np.linalg.pinv(hess) @ grad
            w = w - step
            if np.linalg.norm(step) < self.tol:
                break

        if self.fit_intercept:
            self.coef_ = w[:-1]
            self.intercept_ = float(w[-1])
        else:
            self.coef_ = w
            self.intercept_ = 0.0
        return self

    def decision_function(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("请先调用 fit()")
        Xv = _convert_X(X, self.feature_names_in_)
        Xs = self._scale_apply(Xv)
        return Xs @ self.coef_ + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        """返回形状 (n, 2) 的概率矩阵，列为 [P(y=0), P(y=1)]，每行和为 1。"""
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        """返回 {0, 1} 类别标签（P(y=1) >= 0.5 判为 1）。"""
        p1 = _sigmoid(self.decision_function(X))
        return (p1 >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# 回归决策树（供梯度提升使用）
# ---------------------------------------------------------------------------
class _RegressionTree:
    """极简 CART 回归树（numpy 自研），按最小化 SSE 贪心分裂。

    节点用嵌套 dict 表示：叶 ``{'leaf': True, 'value': v}``；
    内部 ``{'leaf': False, 'feature': j, 'threshold': t, 'left': .., 'right': ..}``。
    """

    def __init__(self, max_depth: int = 1, min_samples_split: int = 2,
                 min_samples_leaf: int = 1):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.root_: Optional[dict] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RegressionTree":
        self.root_ = self._build(X, y, 0)
        return self

    def _leaf(self, y: np.ndarray) -> dict:
        return {"leaf": True, "value": float(y.mean()) if y.size else 0.0}

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        n, p = X.shape
        best_gain = 0.0
        best: Tuple[Optional[int], Optional[float]] = (None, None)
        parent_sse = float(np.sum((y - y.mean()) ** 2))
        min_leaf = self.min_samples_leaf
        for j in range(p):
            col = X[:, j]
            order = np.argsort(col, kind="mergesort")
            xs = col[order]
            ys = y[order]
            csum = np.cumsum(ys)
            csum2 = np.cumsum(ys * ys)
            total = csum[-1]
            total2 = csum2[-1]
            for k in range(min_leaf - 1, n - min_leaf):
                if xs[k] == xs[k + 1]:
                    continue
                nl = k + 1
                nr = n - nl
                sl = csum[k]
                sl2 = csum2[k]
                sr = total - sl
                sr2 = total2 - sl2
                sse = (sl2 - sl * sl / nl) + (sr2 - sr * sr / nr)
                gain = parent_sse - sse
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (j, float((xs[k] + xs[k + 1]) / 2.0))
        return best

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        n = y.shape[0]
        if (depth >= self.max_depth or n < self.min_samples_split
                or n < 2 * self.min_samples_leaf or np.allclose(y, y[0])):
            return self._leaf(y)
        j, thr = self._best_split(X, y)
        if j is None:
            return self._leaf(y)
        left_mask = X[:, j] <= thr
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return self._leaf(y)
        return {
            "leaf": False, "feature": j, "threshold": thr,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[right_mask], y[right_mask], depth + 1),
        }

    def _predict_row(self, x: np.ndarray) -> float:
        node = self.root_
        while node is not None and not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return float(node["value"]) if node is not None else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("请先调用 fit()")
        return np.array([self._predict_row(X[i]) for i in range(X.shape[0])])


class GradientBoostingRegressor(BaseEstimator):
    """梯度提升回归（MSE 损失，自研决策桩/浅树集成）。

    每轮对当前预测的负梯度（MSE 下即残差 y - F）拟合一棵浅树，按学习率累加。
    ``max_depth=1`` 即决策桩 (stump)，只能表达轴对齐的分段常数，但集成后可逼近
    相当复杂的非线性关系。

    参数
    ----
    n_estimators:  提升轮数（树的数量）。
    learning_rate: 学习率（收缩系数），越小越稳健、需要越多树。
    max_depth:     单棵树最大深度（1=决策桩）。
    min_samples_split / min_samples_leaf: 分裂与叶子的最小样本约束。
    subsample:     每轮随机采样的样本比例（<1 为随机梯度提升），默认 1.0。
    random_state:  随机种子（仅当 subsample<1 时用到）。
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 1, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, subsample: float = 1.0,
                 random_state: Optional[int] = None):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.random_state = random_state
        self.trees_: List[_RegressionTree] = []
        self.init_: float = 0.0
        self.feature_names_in_: Optional[List] = None

    def fit(self, X, y) -> "GradientBoostingRegressor":
        Xv, yv, names = _align_xy(X, y)
        self.feature_names_in_ = names
        rng = np.random.default_rng(self.random_state)
        self.init_ = float(yv.mean()) if yv.size else 0.0
        F = np.full(yv.shape[0], self.init_)
        self.trees_ = []
        n = yv.shape[0]
        for _ in range(self.n_estimators):
            residual = yv - F                       # MSE 的负梯度
            if self.subsample < 1.0 and n > 1:
                k = max(1, int(round(self.subsample * n)))
                sel = rng.choice(n, size=k, replace=False)
                Xs, ys, rs = Xv[sel], yv[sel], residual[sel]
            else:
                Xs, rs = Xv, residual
            tree = _RegressionTree(self.max_depth, self.min_samples_split,
                                   self.min_samples_leaf).fit(Xs, rs)
            update = tree.predict(Xv)
            F = F + self.learning_rate * update
            self.trees_.append(tree)
        return self

    def predict(self, X) -> np.ndarray:
        if not self.trees_ and self.init_ == 0.0:
            raise RuntimeError("请先调用 fit()")
        Xv = _convert_X(X, self.feature_names_in_)
        F = np.full(Xv.shape[0], self.init_)
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(Xv)
        return F


# ---------------------------------------------------------------------------
# 评分器与置换重要性
# ---------------------------------------------------------------------------
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """决定系数 R²（越大越好，1 为完美）。"""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    if sst < 1e-12:
        return 0.0
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - sse / sst


def neg_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """负均方误差（越大越好）。"""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return -float(np.mean((y_true - y_pred) ** 2))


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """分类准确率（越大越好）。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


_SCORERS: dict = {
    "r2": r2_score,
    "neg_mse": neg_mse,
    "mse": neg_mse,
    "accuracy": accuracy_score,
}


def _resolve_scorer(scorer: Union[str, Callable]) -> Callable:
    if callable(scorer):
        return scorer
    key = str(scorer).lower()
    if key not in _SCORERS:
        raise ValueError(f"未知 scorer: {scorer!r}，可选 {list(_SCORERS)} 或传入可调用对象")
    return _SCORERS[key]


def permutation_importance(model: BaseEstimator, X, y,
                           scorer: Union[str, Callable] = "r2",
                           n_repeats: int = 5, random_state: Optional[int] = 0
                           ) -> pd.DataFrame:
    """置换重要性：逐列打乱某特征，度量模型评分相对基线的下降幅度。

    对每个特征重复 ``n_repeats`` 次随机置换，重要性 = 基线评分 − 置换后评分，
    下降越多说明该特征越有用。返回按重要性均值降序排列的 ``DataFrame``，
    含 ``importance_mean`` 与 ``importance_std`` 两列。

    参数
    ----
    model:        已 fit 的模型（需实现 predict）。
    X, y:         评估数据（DataFrame 会用列名标注结果）。
    scorer:       'r2' / 'neg_mse' / 'accuracy' 或自定义 ``callable(y_true, y_pred)``。
    n_repeats:    每列置换次数。
    random_state: 随机种子，保证可复现。
    """
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        Xv = X.to_numpy(dtype="float64")
    else:
        Xv, names = check_X(X)
        names = names or [f"x{i}" for i in range(Xv.shape[1])]
    yv = check_y(y)
    score_fn = _resolve_scorer(scorer)
    rng = np.random.default_rng(random_state)

    baseline = score_fn(yv, model.predict(Xv))
    records = []
    for j, col in enumerate(names):
        drops = []
        for _ in range(n_repeats):
            Xp = Xv.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(baseline - score_fn(yv, model.predict(Xp)))
        drops_arr = np.asarray(drops, dtype="float64")
        records.append((col, float(drops_arr.mean()), float(drops_arr.std())))

    df = pd.DataFrame(records, columns=["feature", "importance_mean", "importance_std"])
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df.set_index("feature")
