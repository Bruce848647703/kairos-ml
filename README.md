# Kairos ML

> Kairos 量化系列的机器学习模块 —— 一个**自研、轻量、零重型依赖**的金融机器学习库。

`kairos_ml` 覆盖金融 ML 的完整链路：**特征工程 → 标签构造 → 时序交叉验证 → 模型 → 评估 → 流水线**。
所有模型（岭回归、逻辑回归、梯度提升树）均用 **numpy 从零实现**，本机不含 sklearn 也能跑；
必需依赖仅 `numpy` 与 `pandas`，`scipy` 为可选增强（正态 CDF/分位数，缺失自动回退 numpy 近似）。
核心代码 100% 原创，离线、确定性、可复现。

## 特性
- **特征工程 `features`**：滚动均值/标准差/动量、滞后项、截面 rank/zscore；自研技术指标 SMA/EMA/RSI/MACD/已实现波动率。
- **标签构造 `labels`**：n 期远期收益；自研**三重障碍法** (Triple-Barrier)——按波动率倍数设止盈/止损轨 + 垂直时间轨，返回 {-1,0,+1} 标签与首个触达障碍；**元标签** (Meta-Labeling) 构造「是否值得下注」的二阶标签。
- **时序交叉验证 `cv`**：`walk_forward_splits`（滚动/扩展窗口，严格保序）；自研 `purged_kfold`（**purge + embargo**，剔除与测试标签期重叠的训练样本并在测试后禁运，防信息泄漏）。
- **numpy 自研模型 `models`**：`RidgeRegression`（闭式解 (XᵀX+αI)⁻¹Xᵀy）、`LogisticRegression`（Newton/IRLS + L2）、`GradientBoostingRegressor`（自研决策桩/浅树集成）、`permutation_importance`（置换重要性）。统一 `fit/predict` 接口。
- **评估 `evaluate`**：命中率、精确率/召回率/F1、信号 PnL、IC（秩相关）、t 统计量、参数法 VaR。
- **流水线 `pipeline`**：`ModelPipeline` 把特征构造与模型 fit/predict 串起来，自动按索引对齐、丢弃 NaN 行。
- **防未来函数**：特征与波动率只用「截至当时」的数据；CV 保证训练恒在测试之前且无标签泄漏。

## 安装
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # 或 pip install numpy pandas
pip install -e ".[dev]"     # 需要跑测试时
pip install -e ".[scipy]"   # 可选：启用 scipy 增强正态分布计算
```

## 快速开始
### ① 特征 + 三重障碍打标
```python
import kairos_ml as kml

prices = ...  # 单资产价格 Series
feat = kml.rolling_momentum(prices, 10).to_frame("mom10")
feat["rsi14"] = kml.rsi(prices, 14)

tb = kml.triple_barrier(prices, pt=1.5, sl=1.5, vertical=5, vol_window=20)
y = (tb["label"] == 1).astype(int)      # 上轨=1，其余=0
```

### ② 保序 CV + 逻辑回归 + 评估
```python
X = feat.reindex(tb.index).dropna()
y = y.reindex(X.index)
splits = kml.purged_kfold(len(X), n_splits=6, embargo=5, label_horizon=5)
for tr, te in splits:
    model = kml.LogisticRegression(C=1.0).fit(X.iloc[tr], y.iloc[tr])
    proba = model.predict_proba(X.iloc[te])[:, 1]
    ...
print(kml.ic(proba, outcome), kml.precision(y_true, cls))
```

### ③ 流水线一键串联
```python
pipe = kml.ModelPipeline(kml.LogisticRegression(), feature_builder=lambda p: feat_from(p))
pipe.fit(prices, y)
pred = pipe.predict(prices)
```

完整可运行示例见 [`examples/demo.py`](examples/demo.py)：合成行情 → 三重障碍打标 →
walk-forward / purged CV → 训练逻辑回归 → 打印 hit_rate / precision / 信号 PnL / IC。

## API 概览
| 模块 | 关键对象 | 说明 |
|---|---|---|
| `features` | `rolling_mean/std/momentum` `lags` `cross_sectional_rank/zscore` `sma/ema/rsi/macd/realized_volatility` | 时序与截面特征、自研技术指标 |
| `labels` | `forward_returns` `triple_barrier` `meta_labeling` | 远期收益、三重障碍、元标签 |
| `cv` | `walk_forward_splits` `purged_kfold` | 保序切分、purge+embargo K 折 |
| `models` | `RidgeRegression` `LogisticRegression` `GradientBoostingRegressor` `permutation_importance` | numpy 自研模型与特征重要性 |
| `evaluate` | `hit_rate` `precision/recall/f1` `signal_pnl` `ic` `t_statistic` `value_at_risk` | 金融 ML 评估指标 |
| `pipeline` | `ModelPipeline` | 特征 → 模型的一致 fit/transform/predict |

## 设计要点
- **纯 numpy 线性代数**：岭回归用 `solve`/`pinv` 求闭式解；逻辑回归用带 L2 阻尼的 Newton/IRLS，
  在完全可分数据上也能收敛不发散；梯度提升自研 CART 回归树（默认决策桩），按学习率累加负梯度拟合。
- **三重障碍**：入场波动率 σ 仅用入场前的收益率滚动标准差，上下轨 = 入场价×(1±倍数·σ)，
  在垂直期内扫描**最先触达**的障碍并给出标签与出场收益，天然带止损/止盈/时间三重风控。
- **防泄漏 CV**：`purged_kfold` 依据 `label_horizon` 剔除标签期与测试区相交的训练样本（purge），
  并在测试区之后禁运 `embargo` 期样本，阻断「未来信息」与序列相关带来的双重泄漏。
- **一致接口 + 索引对齐**：模型统一 `fit/predict`；传入 `DataFrame`/`Series` 时按索引对齐并丢弃 NaN，
  自动处理特征预热期与标签末尾期的缺失边界。
- **scipy 可选**：正态 CDF/分位数优先用 scipy，缺失时回退到 numpy 近似（Abramowitz-Stegun / Acklam），
  保证在纯净环境亦可运行。

## 测试
```bash
make test          # 或 python -m pytest -q
```
覆盖：岭回归闭式解与 `lstsq` 一致且共线下更稳定、逻辑回归线性可分高准确率且概率∈[0,1]、
三重障碍在构造路径上的首触障碍正确、purged K 折无重叠且 embargo 生效、walk-forward 训练恒在测试前、
梯度提升拟合非线性优于常数、置换重要性识别有用特征等。

## 项目结构
```
kairos_ml/        核心包（features / labels / cv / models / evaluate / pipeline / _util）
examples/         可运行示例（demo.py）
tests/            pytest 测试
```

## 许可
MIT © 2026 Bruce848647703，见 [LICENSE](LICENSE)。

## 参考与致谢
本项目为**独立原创实现**，未复制任何第三方代码，且**所有模型均以 numpy 自研、不依赖 sklearn**。
设计思路受业界通用的金融机器学习范式启发——三重障碍法 (Triple-Barrier)、元标签 (Meta-Labeling)、
带 purge 与 embargo 的时序交叉验证、walk-forward 滚动验证、置换重要性 (Permutation Importance)、
梯度提升决策树等——在此向开源量化社区致谢。算法与接口均为本仓库自研。
