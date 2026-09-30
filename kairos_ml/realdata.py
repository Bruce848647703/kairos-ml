"""真实行情数据加载模块：把「一个目录下的多只标的 CSV」整理成收盘价面板。

面向真实 A 股日线数据（如 ``kairos-data/data/ashare/<symbol>.csv``，列
``date,open,high,low,close,volume``）的最小可用加载器，为
:mod:`kairos_ml.features` / :mod:`kairos_ml.labels` 提供
``index=交易日, columns=标的`` 的价格面板。

设计约定
--------
- **只读**：不写入、不修改任何数据文件；仅 numpy/pandas，离线可用。
- **确定性**：列顺序按 CSV **文件名排序**（与目录枚举顺序、文件系统无关），
  行顺序按日期升序，重复日期保留最后一条；同样输入必得同样输出。
- **异常价格**：非正数（<=0）与缺失值一律置为 NaN，再按列 ``ffill``
  （停牌期间沿用最后一个有效收盘价，收益率因此为 0 而非跳空）。
- **上市日对齐**：``drop_incomplete=True`` 时把面板裁剪到「全体标的都已上市」
  的公共区间 [max(首个有效日), min(末个有效日)]，避免截面样本数随时间漂移。
  注意 ``ffill`` 在裁剪**之前**执行，因此提前退市/停止更新的标的会被前值补齐到
  面板末端，裁剪实际由「最晚上市日」驱动（hfq 面板的常规口径）。

注意：后复权 (hfq) 口径下价格水平被放大，但**收益率正确**，适合特征/标签研究。
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = ["load_close_panel", "list_symbol_files"]


def list_symbol_files(data_dir: str, pattern: str = "*.csv") -> List[str]:
    """列出数据目录下的行情 CSV 路径，**按文件名排序**以保证确定性。

    参数
    ----
    data_dir: 数据目录（只读）。
    pattern:  文件名匹配模式，默认 ``*.csv``。

    返回
    ----
    绝对/相对形式与 ``data_dir`` 一致的文件路径列表（已按 basename 升序）。
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在或不是目录: {data_dir}")
    paths = glob.glob(os.path.join(data_dir, pattern))
    paths = [p for p in paths if os.path.isfile(p)]
    return sorted(paths, key=lambda p: os.path.basename(p))


def _read_one(path: str, column: str) -> pd.Series:
    """读取单个 CSV，返回「日期索引 + 单列价格」的 Series（非正价→NaN）。"""
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "date" not in cols:
        raise ValueError(f"{os.path.basename(path)} 缺少 date 列，实际列: {list(df.columns)}")
    key = str(column).strip().lower()
    if key not in cols:
        raise ValueError(f"{os.path.basename(path)} 缺少 {column} 列，实际列: {list(df.columns)}")

    dates = pd.to_datetime(df[cols["date"]])
    vals = pd.to_numeric(df[cols[column]], errors="coerce").to_numpy(dtype="float64")
    s = pd.Series(vals, index=pd.DatetimeIndex(dates), name=os.path.splitext(os.path.basename(path))[0])
    s = s[~s.index.duplicated(keep="last")]      # 重复日期保留最后一条
    s = s.sort_index()                            # 文件内日期可能乱序
    s[~(s.to_numpy(dtype="float64") > 0)] = np.nan  # 非正价 / NaN 一律视为无效
    return s


def load_close_panel(data_dir: str, drop_incomplete: bool = True,
                     column: str = "close", pattern: str = "*.csv",
                     start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """把目录下的行情 CSV 加载成收盘价面板 ``DataFrame(index=日期, columns=标的)``。

    参数
    ----
    data_dir:        数据目录，内含若干 ``<symbol>.csv``（列需包含 ``date`` 与 ``column``）。
    drop_incomplete: True（默认）时裁剪到「全体标的均已上市」的公共日期区间，
                     即 ``[max(各列首个有效日), min(各列末个有效日)]``；
                     False 时保留全部日期并集，上市前的位置为 NaN。
    column:          取用的价格列名，默认 ``close``（不区分大小写）。
    pattern:         文件名匹配模式，默认 ``*.csv``。
    start, end:      可选的额外日期裁剪（在 ``drop_incomplete`` 之后生效）。

    返回
    ----
    ``DataFrame``，索引为升序去重的 ``DatetimeIndex``（名为 ``date``），
    列为按文件名排序的标的代码（列轴名为 ``symbol``）。

    处理链
    ------
    读取 → 非正价置 NaN → 按列 ffill（停牌沿用前值）→ 可选上市日裁剪。

    异常
    ----
    目录不存在抛 ``FileNotFoundError``；目录下无匹配文件、或任一文件缺列/
    全列无有效价格时抛 ``ValueError``。
    """
    paths = list_symbol_files(data_dir, pattern)
    if not paths:
        raise ValueError(f"目录下未找到匹配 {pattern} 的行情文件: {data_dir}")

    series: Dict[str, pd.Series] = {}
    for p in paths:
        s = _read_one(p, column)
        symbol = str(s.name)
        if symbol in series:
            raise ValueError(f"标的代码重复（同名文件）: {symbol}")
        series[symbol] = s

    # 列顺序 = 文件名排序（确定性）；行索引 = 全体日期并集（升序去重）
    panel = pd.DataFrame({k: series[k] for k in sorted(series)}).sort_index()
    panel.index.name = "date"
    panel.columns.name = "symbol"
    panel = panel.astype("float64").ffill()       # 停牌 / 缺日沿用最后有效价

    if drop_incomplete:
        firsts = [panel[c].first_valid_index() for c in panel.columns]
        lasts = [panel[c].last_valid_index() for c in panel.columns]
        if any(f is None for f in firsts) or any(l is None for l in lasts):
            bad = [c for c in panel.columns
                   if panel[c].first_valid_index() is None or panel[c].last_valid_index() is None]
            raise ValueError(f"以下标的全列无有效价格，无法对齐上市日: {bad}")
        panel = panel.loc[max(firsts):min(lasts)]

    if start is not None or end is not None:
        panel = panel.loc[start:end]

    if panel.empty:
        raise ValueError(f"裁剪后面板为空: data_dir={data_dir}, start={start}, end={end}")
    return panel
