# -*- coding: utf-8 -*-
"""红点权重候选对照（2026-09-27）：几个事先定好的组合，不做扫描。

    python3 compare_weights.py          # 长面板因子缓存不存在时先重建（约 1 分钟），再出对照表

候选在跑之前就定死了（见 CANDS），只比较、不挑选——在同样 7 次事件上再扫一遍只会重复过拟合。
每个候选都在训练窗（2017-10-18 起）把门槛对齐到 66 天（与 ALT_W 标定同频），然后看：
  训练窗、A 段（2018-2021）、B 段（2023 起）、2007-2016 留出段（同频门槛）、自适应门槛（2009-07 起）。
数据：~/us2/raw_long 长面板（BRK-B 成交量已校正）；收益按纳指 100 指数之后 30 个交易日。
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import engine as E
import alt_engine as A

RAW_LONG = os.path.expanduser("~/us2/raw_long")
CACHE = os.path.join(HERE, "_long_factors.csv")

CANDS = {
    "C0 现行 ALT_W":   dict(lev_ratio=1.5, lev_int=1.5, top2=2, vix_abs=2, narrow=1, topshare=1),
    "C1 核心四因子等权": dict(lev_ratio=1, lev_int=1, vix_abs=1, narrow=1),
    "C2 五因子等权":    dict(lev_ratio=.5, lev_int=.5, top2=1, vix_abs=1, narrow=1, topshare=1),
    "C3 概念等权":      dict(lev_ratio=.5, lev_int=.5, vix_abs=1, narrow=1, conc=1),   # conc = (TOP2 + 前2%)/2
}
# 熊市途中的下跌段：红点本就不该亮，不计入抓顶
BEAR = {pd.Timestamp(s) for s in ("2008-10-13", "2008-11-04", "2009-02-09", "2022-03-29", "2022-08-15", "2022-12-01")}


def build():
    E.RAW = RAW_LONG
    rawdf, _, spy, _ = E.build_indicators()
    _, adj, _, _ = E.compose(rawdf, E.direction(spy, rawdf.index))
    lev_pct, _ = E.leverage_monitor(rawdf)
    parts = A.alt_inputs(rawdf, adj, lev_pct, spy, rawdf.index)
    X = pd.DataFrame({"temp": A.alt_temperature(parts), **{k: parts[k] for k in A.ALT_W},
                      "lev_ratio": lev_pct["ratio"], "lev_int": lev_pct["intensity"],
                      "top2_raw": rawdf["top2"], "topshare_raw": rawdf["_topshare"]})
    X.to_csv(CACHE)
    return X


def zigzag(s, th):
    up, hi, hi_i, lo, lo_i, segs, pk = True, s.iloc[0], 0, None, None, [], None
    v = s.values
    for i in range(1, len(v)):
        if up:
            if v[i] > hi: hi, hi_i = v[i], i
            elif v[i] <= hi * (1 - th): pk, up, lo, lo_i = hi_i, False, v[i], i
        else:
            if v[i] < lo: lo, lo_i = v[i], i
            elif v[i] >= lo * (1 + th): segs.append((pk, lo_i)); up, hi, hi_i = True, v[i], i
    if not up: segs.append((pk, lo_i))
    return [(s.index[a], s.index[b], v[b] / v[a] - 1) for a, b in segs]


def main():
    X = pd.read_csv(CACHE, index_col=0, parse_dates=True) if os.path.exists(CACHE) else build()
    X["conc"] = (X["top2"] + X["topshare"]) / 2
    ndx = pd.read_csv(os.path.join(HERE, "_tdc_history_long.csv"), parse_dates=["Date"], index_col="Date")["ndx"]
    px = ndx.reindex(X.index).ffill(); f30 = px.shift(-30) / px - 1
    tops = [z for z in zigzag(ndx, 0.11) if z[0] >= pd.Timestamp("2007-07-13") and z[0] not in BEAR]
    ix = X.index
    TR = pd.Series(ix >= "2017-10-18", index=ix)
    HO = pd.Series((ix >= "2007-07-13") & (ix <= "2016-12-31"), index=ix)
    SA = pd.Series((ix >= "2018-01-12") & (ix <= "2021-12-31"), index=ix)
    SB = pd.Series(ix >= "2023-01-03", index=ix)

    def edge(m, win):
        f, b = f30[m & win].dropna(), f30[win].dropna()
        return ((f.mean() - b.mean()) * 100 if len(f) else np.nan), int((m & win).sum()), \
               len(A.events((m & win)))

    def cover(flag, lo, hi):
        sel = [pk for pk, _, _ in tops if pd.Timestamp(lo) <= pk <= pd.Timestamp(hi)]
        got = [pk for pk in sel if flag.iloc[max(0, ix.searchsorted(pk) - 40): ix.searchsorted(pk) + 6].any()]
        return f"{len(got)}/{len(sel)}", got

    print("相对基准＝红点日之后 30 日纳指收益减去同期全部交易日的均值（负＝红点之后跌得更多）\n")
    rows = []
    for nm, w in CANDS.items():
        T = (sum(X[k] * v for k, v in w.items()) / sum(w.values())).where(X["temp"].notna())
        t3 = T.rolling(3).min()
        s = t3[TR].dropna().sort_values(ascending=False)
        th = (s.iloc[65] + s.iloc[66]) / 2
        hot = (t3 > th).fillna(False)
        tr, a, b = edge(hot, TR), edge(hot, SA), edge(hot, SB)
        th2 = t3[HO].dropna().quantile(1 - tr[1] / TR.sum())
        ho = (t3 > th2).fillna(False); h = edge(ho, HO)
        V = A.validity(T, px)
        W = pd.Series(ix >= V["th_ad"].first_valid_index(), index=ix); ad = edge(V["hot_ad"], W)
        c_tr, got_tr = cover(hot, "2017-10-18", "2026-12-31")
        c_ho, _ = cover(ho, "2007-07-13", "2016-12-31")
        c_ad, _ = cover(V["hot_ad"], "2009-07-16", "2026-12-31")
        rows.append((nm, th, tr, a, b, h, ad, c_tr, c_ho, c_ad, got_tr))
    print(f"{'候选':14s}{'门槛':>6s} | {'训练窗':>22s} | {'A段':>8s} {'B段':>8s} | {'07-16留出(同频)':>20s} | {'自适应 09-07起':>20s}")
    for nm, th, tr, a, b, h, ad, c_tr, c_ho, c_ad, _ in rows:
        print(f"{nm:14s}{th:6.1f} | {tr[0]:+6.2f}pp {tr[1]:3d}天 {tr[2]}次 抓{c_tr:>4s} | {a[0]:+6.2f}pp {b[0]:+6.2f}pp"
              f" | {h[0]:+6.2f}pp {h[1]:3d}天 抓{c_ho:>4s} | {ad[0]:+6.2f}pp {ad[1]:3d}天 抓{c_ad:>5s}")
    print("\n训练窗抓到的顶部：")
    for nm, *_, got in rows:
        print(f"  {nm:14s} " + " ".join(f"{t:%Y-%m}" for t in got))


if __name__ == "__main__":
    main()
