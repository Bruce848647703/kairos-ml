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
- **真实数据 `realdata`**：`load_close_panel` 把行情 CSV 目录整理成收盘价面板（非正价→NaN→ffill、按上市日对齐、列序确定性），只读、离线。
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

## 真实数据 ML 信号研究
[`examples/real_ml_signal.py`](examples/real_ml_signal.py) 把同一条链路跑在**真实 A 股日线**上：

```bash
python3 examples/real_ml_signal.py --data-dir /path/to/kairos-data/data/ashare
# 产出：research/real_ml/{REPORT.md, metrics.json, feature_importance.csv}
```

- **数据加载**：`kairos_ml.load_close_panel(data_dir)` 读取目录下的 `<symbol>.csv`
  （列 `date,open,high,low,close,volume`）→ `index=交易日, columns=标的` 的收盘价面板；
  非正价→NaN→按列 `ffill`（停牌沿用前值），并按「全体上市日」裁剪。列按文件名排序，
  重复日期保留最后一条，**离线、确定性、只读**（见 [`tests/test_realdata.py`](tests/test_realdata.py)）。
- **特征**（16 个，全部无量纲、只用截至当日的数据）：滚动动量 1/5/10/20/60、已实现波动率 10/20、
  RSI(14)、均线乖离 5/20、EMA 快慢差、归一化 MACD 柱，以及 4 个当日**截面 z-score** 版本。
- **标签**：`forward_returns(H=5)`；`--label barrier` 可切换为 `triple_barrier` 的障碍出场收益。
- **严格样本外**：`walk_forward_splits`（滚动窗口）+ `purged_kfold`（purge/embargo，扩展窗口），
  两套切分都再施加 `train_pos + H < 测试期首日` 的 purge 约束——训练只用**标签已完全实现**的历史样本，
  训练集恒在测试集之前；模型超参数在看结果前固定，不做任何调参/选模。
- **模型**：`RidgeRegression` / `LogisticRegression` / `GradientBoostingRegressor`（均为 numpy 自研）。
- **评估**：hit_rate、precision/recall/F1、pooled IC、**日频截面 IC**（均值/ICIR/t 值）、
  top-K 多头与多空的信号 PnL（`signal_pnl`，每 H 日调仓、各期不重叠）、相对等权基准的超额与 t 值、VaR95。
- **诊断**：单因子「裸 IC」对照、逐折 IC 稳定性（含系数符号 vs 同期实现 IC 的翻转统计）、
  置换重要性（按样本外日频截面 IC 下降度量）、稳健性设定（换目标函数/持有期）。
- **结论写在报告里**：[`research/real_ml/REPORT.md`](research/real_ml/REPORT.md) 是本脚本自动生成的
  研究记录，含各模型样本外指标与**诚实结论**（该样本上样本外预测力微弱/不稳定，不构成投资建议）。

38 只标的 × ~1900 交易日 ≈ 7 万条池化样本，全流程约 2 分钟（梯度提升树按固定步长子采样训练行以控制耗时）。

### 数据声明
- 示例数据为**公开渠道获取的 A 股日线行情**（后复权 hfq 口径，位于同系列仓库
  `kairos-data/data/ashare/`），版权归原作者/数据源所有；本仓库**不分发**行情数据，
  仅通过 `--data-dir` 只读引用。
- hfq 口径下**价格水平被放大、但收益率正确**，本研究只使用收益率/比值/z-score 类特征，不受水平影响。
- 数据不保证准确、完整或及时；样本仅覆盖少数大市值个股与一段特定行情，存在生存偏差。
- 所有回测/研究结果**仅用于研究与学习，不构成任何投资建议**。

## API 概览
| 模块 | 关键对象 | 说明 |
|---|---|---|
| `features` | `rolling_mean/std/momentum` `lags` `cross_sectional_rank/zscore` `sma/ema/rsi/macd/realized_volatility` | 时序与截面特征、自研技术指标 |
| `labels` | `forward_returns` `triple_barrier` `meta_labeling` | 远期收益、三重障碍、元标签 |
| `cv` | `walk_forward_splits` `purged_kfold` | 保序切分、purge+embargo K 折 |
| `models` | `RidgeRegression` `LogisticRegression` `GradientBoostingRegressor` `permutation_importance` | numpy 自研模型与特征重要性 |
| `evaluate` | `hit_rate` `precision/recall/f1` `signal_pnl` `ic` `t_statistic` `value_at_risk` | 金融 ML 评估指标 |
| `pipeline` | `ModelPipeline` | 特征 → 模型的一致 fit/transform/predict |
| `realdata` | `load_close_panel` `list_symbol_files` | 真实行情 CSV 目录 → 收盘价面板（只读、确定性） |

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
梯度提升拟合非线性优于常数、置换重要性识别有用特征、真实数据加载（形状/列序确定性/非正价/停牌 ffill/
上市日裁剪/异常输入，全部用 `tmp_path` 造小 CSV，**不联网**）等。

## 项目结构
```
kairos_ml/          核心包（features / labels / cv / models / evaluate / pipeline / realdata / _util）
examples/           可运行示例（demo.py 合成行情；real_ml_signal.py 真实 A 股数据）
tests/              pytest 测试（含 test_realdata.py 离线数据加载测试）
research/real_ml/   真实数据研究产出（REPORT.md / metrics.json / feature_importance.csv）
```

## 许可
MIT © 2026 Bruce848647703，见 [LICENSE](LICENSE)。

## 参考与致谢
本项目为**独立原创实现**，未复制任何第三方代码，且**所有模型均以 numpy 自研、不依赖 sklearn**。
设计思路受业界通用的金融机器学习范式启发——三重障碍法 (Triple-Barrier)、元标签 (Meta-Labeling)、
带 purge 与 embargo 的时序交叉验证、walk-forward 滚动验证、置换重要性 (Permutation Importance)、
梯度提升决策树等——在此向开源量化社区致谢。算法与接口均为本仓库自研。
