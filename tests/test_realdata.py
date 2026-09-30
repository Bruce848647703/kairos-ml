"""真实数据加载测试：``kairos_ml.realdata.load_close_panel``（全部离线，tmp_path 造小 CSV）。

覆盖：形状与列序确定性、非正价→NaN→ffill、停牌缺日填充、上市日裁剪、
文件内日期乱序/重复、异常输入报错，以及与 features 模块的联动。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos_ml as kml
from kairos_ml.realdata import list_symbol_files, load_close_panel

HEADER = "date,open,high,low,close,volume"


def _write_csv(directory, name: str, rows) -> None:
    """写一个最小行情 CSV：rows 为 (date, close) 序列，其余列由 close 派生。"""
    lines = [HEADER]
    for d, c in rows:
        lines.append(f"{d},{c},{c},{c},{c},1000")
    (directory / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def three_assets(tmp_path):
    """三只标的：a 全程在市；b 晚一天上市且含非正价；c 中间停牌两天。"""
    _write_csv(tmp_path, "a", [("2024-01-01", 10.0), ("2024-01-02", 11.0),
                               ("2024-01-03", 12.0), ("2024-01-04", 13.0),
                               ("2024-01-05", 14.0)])
    _write_csv(tmp_path, "b", [("2024-01-02", 20.0), ("2024-01-03", 0.0),
                               ("2024-01-04", 22.0), ("2024-01-05", 23.0)])
    _write_csv(tmp_path, "c", [("2024-01-01", 30.0), ("2024-01-03", 31.0),
                               ("2024-01-05", 32.0)])
    return tmp_path


# ---------------------------------------------------------------------------
# 基本形状 / 列序 / 索引
# ---------------------------------------------------------------------------
def test_shape_columns_and_index(three_assets):
    panel = load_close_panel(three_assets, drop_incomplete=False)
    assert panel.shape == (5, 3)
    assert list(panel.columns) == ["a", "b", "c"]        # 按文件名排序
    assert panel.columns.name == "symbol"
    assert panel.index.name == "date"
    assert isinstance(panel.index, pd.DatetimeIndex)
    assert panel.index.is_monotonic_increasing
    assert list(panel.index.strftime("%Y-%m-%d")) == [
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


def test_column_order_independent_of_write_order(tmp_path):
    """文件写入顺序（=目录枚举顺序）不影响列顺序，列恒按文件名排序。"""
    rows = [("2024-01-01", 1.0), ("2024-01-02", 2.0)]
    for name in ["zz", "mm", "aa"]:                      # 逆序写入
        _write_csv(tmp_path, name, rows)
    panel = load_close_panel(tmp_path)
    assert list(panel.columns) == ["aa", "mm", "zz"]
    assert list_symbol_files(str(tmp_path)) == [
        str(tmp_path / "aa.csv"), str(tmp_path / "mm.csv"), str(tmp_path / "zz.csv")]


def test_deterministic_repeated_loads(three_assets):
    p1 = load_close_panel(three_assets)
    p2 = load_close_panel(three_assets)
    pd.testing.assert_frame_equal(p1, p2)


# ---------------------------------------------------------------------------
# 非正价 / 停牌 / 缺失
# ---------------------------------------------------------------------------
def test_nonpositive_close_becomes_nan_then_ffill(three_assets):
    panel = load_close_panel(three_assets, drop_incomplete=False)
    assert panel.loc["2024-01-03", "b"] == 20.0          # 0 价 → NaN → 沿用前值
    assert panel.loc["2024-01-02", "b"] == 20.0


def test_suspension_gaps_filled_by_ffill(three_assets):
    """停牌缺日：索引取全体日期并集，缺失位置沿用最后有效价（收益率=0，不跳空）。"""
    panel = load_close_panel(three_assets, drop_incomplete=False)
    assert panel["c"].tolist() == [30.0, 30.0, 31.0, 31.0, 32.0]
    rets = panel["c"].pct_change().dropna()
    assert rets.loc["2024-01-02"] == 0.0                 # 停牌日收益为 0
    assert np.all(np.isfinite(rets.to_numpy()))


def test_pre_listing_stays_nan_without_crop(three_assets):
    panel = load_close_panel(three_assets, drop_incomplete=False)
    assert np.isnan(panel.loc["2024-01-01", "b"])        # 上市前无价，ffill 无从填充
    assert int(panel.isna().sum().sum()) == 1


def test_drop_incomplete_crops_to_common_listing_window(three_assets):
    panel = load_close_panel(three_assets, drop_incomplete=True)
    assert panel.shape == (4, 3)
    assert panel.index[0] == pd.Timestamp("2024-01-02")  # max(各列首个有效日)
    assert panel.index[-1] == pd.Timestamp("2024-01-05")  # min(各列末个有效日)
    assert int(panel.isna().sum().sum()) == 0


def test_crop_is_driven_by_listing_date_not_delisting(tmp_path):
    """裁剪按「全体上市日」（首个有效日的最大值）；ffill 在裁剪之前，故早退市的
    标的会被前值补齐到面板末端，不反向截断尾巴（hfq 面板的常规口径）。"""
    _write_csv(tmp_path, "a", [("2024-01-01", 10.0), ("2024-01-02", 11.0),
                               ("2024-01-03", 12.0)])
    _write_csv(tmp_path, "b", [("2024-01-02", 20.0), ("2024-01-03", 21.0)])
    full = load_close_panel(tmp_path, drop_incomplete=False)
    assert full.shape == (3, 2)
    assert np.isnan(full.loc["2024-01-01", "b"])
    cropped = load_close_panel(tmp_path, drop_incomplete=True)
    assert list(cropped.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert int(cropped.isna().sum().sum()) == 0
    assert cropped["b"].tolist() == [20.0, 21.0]


def test_start_end_slicing_applies_after_crop(three_assets):
    panel = load_close_panel(three_assets, start="2024-01-03", end="2024-01-04")
    assert list(panel.index.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-04"]


# ---------------------------------------------------------------------------
# 文件内容鲁棒性
# ---------------------------------------------------------------------------
def test_unsorted_and_duplicate_dates_in_file(tmp_path):
    """文件内日期乱序会被排序，重复日期保留最后一条。"""
    path = tmp_path / "x.csv"
    path.write_text("\n".join([
        HEADER,
        "2024-01-03,3,3,3,3,1",
        "2024-01-01,1,1,1,1,1",
        "2024-01-02,2,2,2,99,1",     # 重复日期，后写覆盖
        "2024-01-02,2,2,2,2,1",
    ]) + "\n", encoding="utf-8")
    panel = load_close_panel(tmp_path)
    assert list(panel.index.strftime("%Y-%m-%d")) == [
        "2024-01-01", "2024-01-02", "2024-01-03"]
    assert panel["x"].tolist() == [1.0, 2.0, 3.0]


def test_other_price_column_can_be_selected(tmp_path):
    _write_csv(tmp_path, "a", [("2024-01-01", 10.0), ("2024-01-02", 11.0)])
    panel = load_close_panel(tmp_path, column="high")
    assert panel["a"].tolist() == [10.0, 11.0]           # 本 fixture 中 high == close


def test_pattern_filters_files(three_assets):
    (three_assets / "notes.txt").write_text("ignore me", encoding="utf-8")
    _write_csv(three_assets, "d", [("2024-01-01", 1.0), ("2024-01-05", 2.0)])
    panel = load_close_panel(three_assets, drop_incomplete=False, pattern="[ab].csv")
    assert list(panel.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# 异常输入
# ---------------------------------------------------------------------------
def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_close_panel(str(tmp_path / "nope"))


def test_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="未找到"):
        load_close_panel(str(tmp_path))


def test_missing_close_column_raises(tmp_path):
    (tmp_path / "a.csv").write_text("date,vol\n2024-01-01,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="close"):
        load_close_panel(tmp_path)


def test_missing_date_column_raises(tmp_path):
    (tmp_path / "a.csv").write_text("d,close\n2024-01-01,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="date"):
        load_close_panel(tmp_path)


def test_all_invalid_column_raises_on_crop(tmp_path):
    _write_csv(tmp_path, "a", [("2024-01-01", 10.0), ("2024-01-02", 11.0)])
    _write_csv(tmp_path, "b", [("2024-01-01", 0.0), ("2024-01-02", -3.0)])
    with pytest.raises(ValueError, match="无有效价格"):
        load_close_panel(tmp_path, drop_incomplete=True)
    loose = load_close_panel(tmp_path, drop_incomplete=False)
    assert loose["b"].isna().all()


def test_crop_to_empty_window_raises(tmp_path):
    _write_csv(tmp_path, "a", [("2024-01-01", 10.0)])
    with pytest.raises(ValueError):
        load_close_panel(tmp_path, start="2025-01-01")


def test_duplicate_symbol_names_raise(tmp_path):
    """不同扩展名的同名文件会解析出同一标的代码，必须报错而非静默覆盖。"""
    _write_csv(tmp_path, "a", [("2024-01-01", 10.0)])
    (tmp_path / "a.txt").write_text(
        "date,open,high,low,close,volume\n2024-01-01,1,1,1,10,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_close_panel(tmp_path, pattern="*.*")


# ---------------------------------------------------------------------------
# 与 features / labels 的联动（面板可直接喂给本包函数）
# ---------------------------------------------------------------------------
def test_panel_feeds_features_and_labels(three_assets):
    panel = load_close_panel(three_assets, drop_incomplete=False)
    mom = kml.rolling_momentum(panel, 2)
    assert mom.shape == panel.shape
    assert mom.loc["2024-01-03", "a"] == pytest.approx(12.0 / 10.0 - 1.0)
    fret = kml.forward_returns(panel, periods=2)
    assert fret.loc["2024-01-01", "a"] == pytest.approx(12.0 / 10.0 - 1.0)
    assert np.isnan(fret.loc["2024-01-05", "a"])          # 末尾无未来数据
