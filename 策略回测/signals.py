# -*- coding: utf-8 -*-
"""把预警模型的每日信号整理成一张表，供回测用。

两个来源：
  prod  生产数据 data_alt.json（2017-10-18 起，网页上看到的就是这一份）
  long  长面板 ~/us2/raw_long 按同一套权重/门槛重建（2006 起；2017-10 前用于补齐十年窗口与样本外检验）

列：temp_red（红点温度） red（实心红点） red_ad（空心红点） temp_blue（蓝点温度·当日口径）
    blue（实心蓝点） blue_soft（空心蓝点） tdc（TDC 读数） yellow（黄点） vix  repricing
所有列都是 t 日收盘后可知的量，回测里一律 t+1 才能用。
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
PROD_FROM = "2017-10-18"


def prod():
    d = json.load(open(os.path.join(ROOT, "data_alt.json"), encoding="utf-8"))
    s, f = d["series"], d["alerts"]["flags"]
    idx = pd.to_datetime(s["dates"])
    col = lambda v: pd.Series([np.nan if x is None else x for x in v], index=idx, dtype=float)
    return pd.DataFrame({
        "temp_red": col(s["temperature_sell"]), "red": col(f["hot"]).astype(bool),
        "red_ad": col(f["hot_ad"]).astype(bool), "temp_blue": col(s["temperature_fast"]),
        "blue": col(f["cold"]).astype(bool), "blue_soft": col(f["cold_soft"]).astype(bool),
        "tdc": col(s["tdc"]), "yellow": col(f["tdc"]).astype(bool), "vix": col(s["vix"]),
        "repricing": col(s["repricing"]).fillna(0).astype(bool)}).loc[PROD_FROM:]


def long():
    """从只读长面板重算，因子与旗标沿用最优拟合生产入口的规则。"""
    import engine as E, alt_engine as A
    import tdc_signal as TD
    saved_raw = E.RAW
    try:
        E.RAW = os.path.expanduser("~/us2/raw_long")
        # 旧因子缓存早于在册区间口径，每次都从行情重算。
        rawdf, _, spy, _ = E.build_indicators()
        idx = rawdf.index
        dirs = E.direction(spy, idx)
        _, adj, _, _ = E.compose(rawdf, dirs)
        lev_pct, _ = E.leverage_monitor(rawdf)
        lev_pct, _, lev_raw = A.lev_single(idx, lev_pct)
        temp = A.alt_temperature(A.alt_inputs(rawdf, adj, lev_pct, spy, idx))
        if temp is None:
            raise ValueError("长面板红点温度算不出来，检查因子输入")
        V = A.validity(temp, E.load("QQQ")["close"].reindex(idx))

        fast = rawdf.copy()
        for key, daily in {"turnover": "_turnover_daily", "advancing": "_advancing_daily",
                           "leverage": "_leverage_daily"}.items():
            fast[key] = rawdf[daily]
        _, adjf, _, _ = E.compose(fast, dirs, fast=True)
        if lev_raw is not None:
            adjf = A.blue_lev_splice(adjf, dirs, lev_raw["ratio_daily"], fast=True)
        bt = A.blue_temperature(adjf, judge=True)
        if bt is None:
            raise ValueError("长面板蓝点温度算不出来，检查因子输入")
        vix = E.load_vix(idx)
        cold = ((bt < A.BLUE_TH) & (vix >= E.VIX_COLD)).fillna(False)
        if E.PERSIST_COLD > 1:
            cold = (cold.rolling(E.PERSIST_COLD, min_periods=E.PERSIST_COLD)
                    .sum() == E.PERSIST_COLD).fillna(False)
        rp = E.repricing_regime(E.load_real_rate(), E.load_nominal_rate())
        rp = E.align_to(rp.astype(float), idx).fillna(0).astype(bool)

        original_read = TD._read

        def read_unique(raw, sym):
            values = original_read(raw, sym)
            if values is None:
                return None
            return values[~values.index.duplicated(keep="last")]

        TD._read = read_unique
        try:
            T = TD.compute(E.RAW)            # 长面板个别 CSV 有重复日期，去重后再算
        finally:
            TD._read = original_read
        return pd.DataFrame({
            "temp_red": temp, "red": V["hot"], "red_ad": V["hot_ad"], "temp_blue": bt,
            "blue": cold & ~rp, "blue_soft": cold & rp,
            "tdc": T["tdc"].reindex(idx), "yellow": T["red"].reindex(idx).fillna(False).astype(bool),
            "vix": vix, "repricing": rp})
    finally:
        E.RAW = saved_raw


def build():
    P, L = prod(), long()
    P.to_csv(os.path.join(HERE, "_signals_prod.csv")); L.to_csv(os.path.join(HERE, "_signals_long.csv"))
    return P, L


def load(source="splice"):
    """splice：2017-10-18 起用生产信号，之前用长面板；long：全程长面板；prod：只有生产段。"""
    pp, pl = os.path.join(HERE, "_signals_prod.csv"), os.path.join(HERE, "_signals_long.csv")
    if not (os.path.exists(pp) and os.path.exists(pl)):
        build()
    P = pd.read_csv(pp, index_col=0, parse_dates=True).loc[PROD_FROM:]
    L = pd.read_csv(pl, index_col=0, parse_dates=True)
    if source == "prod":
        return P
    if source == "long":
        return L
    return pd.concat([L[L.index < PROD_FROM], P])


if __name__ == "__main__":
    P, L = build()
    ov = P.index.intersection(L.index)
    print(f"生产 {P.index[0].date()}→{P.index[-1].date()}  长面板 {L.index[0].date()}→{L.index[-1].date()}  重叠 {len(ov)} 天")
    for c in ["red", "red_ad", "blue", "blue_soft", "yellow"]:
        a, b = P.loc[ov, c].astype(bool), L.loc[ov, c].astype(bool)
        print(f"{c:10s} 生产 {a.sum():3d}  长面板 {b.sum():3d}  同亮 {(a&b).sum():3d}")
    for c in ["temp_red", "temp_blue", "tdc"]:
        print(f"{c:10s} 相关 {P.loc[ov, c].corr(L.loc[ov, c]):.3f}  平均差 {(P.loc[ov, c]-L.loc[ov, c]).mean():+.2f}")
