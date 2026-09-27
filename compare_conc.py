# -*- coding: utf-8 -*-
"""集中度因子（TOP2、前2%）的读法对照（2026-09-27）。只换这两个因子的读法，其余因子与 ALT_W 权重不动。

担心的两种风险：
  ① 集中度一路上升 → 252 日分位天天是一年新高，读数钉在 90~100，高读数不再只出现在顶部（误报）
  ② 集中度在高位走平 → 分位回到 50 附近，绝对水平再高也显示"不热"（沉默）
候选读法事先定死，只比较不挑选：
  T0 现行：原始值的 252 日分位
  T1 去趋势：原始值相对自己 252 日均值的偏离，再取 252 日分位
  T2 去趋势·长窗口：同一偏离度取 756 日（至少 504 日）分位
  T3 混合：T0 与 T1 各半
    python3 compare_conc.py
"""
import os
import numpy as np, pandas as pd
import engine as E
import alt_engine as A
from compare_weights import CACHE, BEAR, build, zigzag, HERE

W0 = dict(lev_ratio=1.5, lev_int=1.5, top2=2, vix_abs=2, narrow=1, topshare=1)   # 现行 ALT_W


def detr(x):
    return x / x.rolling(252, min_periods=200).mean() - 1


def variants(X):
    out = {}
    for k in ("top2", "topshare"):
        raw = X[f"{k}_raw"]
        out[("T0 现行", k)] = X[k]
        out[("T1 去趋势", k)] = E.rolling_pct(detr(raw))
        out[("T2 去趋势·长窗口", k)] = E.rolling_pct(detr(raw), window=756, min_periods=504)
        out[("T3 混合", k)] = (X[k] + E.rolling_pct(detr(raw))) / 2
    return out


def main():
    X = pd.read_csv(CACHE, index_col=0, parse_dates=True) if os.path.exists(CACHE) else build()
    if "top2_raw" not in X:
        X = build()
    ndx = pd.read_csv(os.path.join(HERE, "_tdc_history_long.csv"), parse_dates=["Date"], index_col="Date")["ndx"]
    px = ndx.reindex(X.index).ffill(); f30 = px.shift(-30) / px - 1
    ix = X.index
    tops = [z for z in zigzag(ndx, 0.11) if z[0] >= pd.Timestamp("2007-07-13") and z[0] not in BEAR]
    TR = pd.Series(ix >= "2017-10-18", index=ix)
    HO = pd.Series((ix >= "2007-07-13") & (ix <= "2016-12-31"), index=ix)
    SA = pd.Series((ix >= "2018-01-12") & (ix <= "2021-12-31"), index=ix)
    SB = pd.Series(ix >= "2023-01-03", index=ix)
    VAR = variants(X)
    names = ["T0 现行", "T1 去趋势", "T2 去趋势·长窗口", "T3 混合"]

    def edge(m, win):
        f, b = f30[m & win].dropna(), f30[win].dropna()
        return ((f.mean() - b.mean()) * 100 if len(f) else np.nan), int((m & win).sum())

    def cover(flag, lo, hi):
        sel = [pk for pk, _, _ in tops if pd.Timestamp(lo) <= pk <= pd.Timestamp(hi)]
        got = [pk for pk in sel if flag.iloc[max(0, ix.searchsorted(pk) - 40): ix.searchsorted(pk) + 6].any()]
        return f"{len(got)}/{len(sel)}", got

    print("一、整体成绩（ALT_W 权重不变，只换集中度读法；门槛在训练窗对齐到 66 天）")
    print(f"{'读法':16s}{'门槛':>6s} | {'训练窗':>18s} | {'A段':>8s} {'B段':>8s} | {'07-16留出':>16s} | {'自适应 09-07起':>18s}")
    got_all = {}
    for nm in names:
        Y = X.copy(); Y["top2"], Y["topshare"] = VAR[(nm, "top2")], VAR[(nm, "topshare")]
        T = (sum(Y[k] * v for k, v in W0.items()) / sum(W0.values())).where(X["temp"].notna() & Y["top2"].notna())
        t3 = T.rolling(3).min()
        s = t3[TR].dropna().sort_values(ascending=False); th = (s.iloc[65] + s.iloc[66]) / 2
        hot = (t3 > th).fillna(False)
        tr, a, b = edge(hot, TR), edge(hot, SA), edge(hot, SB)
        th2 = t3[HO].dropna().quantile(1 - tr[1] / TR.sum()); ho = (t3 > th2).fillna(False); h = edge(ho, HO)
        V = A.validity(T, px)
        Wd = pd.Series(ix >= V["th_ad"].first_valid_index(), index=ix); ad = edge(V["hot_ad"], Wd)
        c_tr, got = cover(hot, "2017-10-18", "2026-12-31"); got_all[nm] = got
        c_ho, _ = cover(ho, "2007-07-13", "2016-12-31"); c_ad, _ = cover(V["hot_ad"], "2009-07-16", "2026-12-31")
        print(f"{nm:16s}{th:6.1f} | {tr[0]:+6.2f}pp {tr[1]:3d}天 抓{c_tr:>4s} | {a[0]:+6.2f}pp {b[0]:+6.2f}pp"
              f" | {h[0]:+6.2f}pp 抓{c_ho:>4s} | {ad[0]:+6.2f}pp 抓{c_ad:>5s}")
    print("\n   训练窗抓到的顶部：")
    for nm in names:
        print(f"   {nm:16s} " + " ".join(f"{t:%Y-%m}" for t in got_all[nm]))

    print("\n二、风险① 趋势期误报：集中度读数（TOP2 与前2% 平均）≥80 的日子，之后 30 日相对基准")
    for lab, win in (("2023 起（AI 行情）", SB), ("2018-2021", SA), ("2010-2016", pd.Series((ix >= "2010") & (ix <= "2016-12-31"), index=ix))):
        row = []
        for nm in names:
            c = (VAR[(nm, "top2")] + VAR[(nm, "topshare")]) / 2
            e, n = edge((c >= 80).fillna(False), win)
            row.append(f"{nm[:2]} {n:4d}天 {e:+6.2f}pp")
        print(f"   {lab:14s} " + "   ".join(row))

    print("\n三、各顶部前 20 个交易日的集中度读数（TOP2 / 前2%）")
    print("   顶部        TOP2原始值 " + "  ".join(f"{nm:>14s}" for nm in names))
    for d in ("2018-01-26", "2018-08-29", "2020-02-19", "2021-11-19", "2024-07-10", "2025-02-19", "2025-10-29", "2026-06-02"):
        i = ix.searchsorted(pd.Timestamp(d)); sl = slice(max(0, i - 20), i + 1)
        cells = [f"{VAR[(nm, 'top2')].iloc[sl].mean():5.0f} / {VAR[(nm, 'topshare')].iloc[sl].mean():3.0f}" for nm in names]
        print(f"   {d}  {X['top2_raw'].iloc[sl].mean():6.1f}%   " + "  ".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
    main()
