# -*- coding: utf-8 -*-
"""红点 VIX 因子的三种写法：现行 / 高于 20 记中性（不否决）/ 不用 VIX。门槛各自在训练窗对齐到 67 天。
两种评判：之后 30 日点对点收益；之后 30 日内最深回撤（从信号日收盘算）。"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(HERE)
import pickle, numpy as np, pandas as pd
B = pickle.load(open("base.pkl", "rb"))
P, qqq, vix = B["parts"], B["qqq"], B["vix"]
FWD = 30
fwd = qqq.shift(-FWD) / qqq - 1
# 之后 30 日内最低收盘相对当日收盘
fmin = pd.concat([qqq.shift(-k) for k in range(1, FWD + 1)], axis=1).min(axis=1) / qqq - 1
fmin[qqq.shift(-FWD).isna()] = np.nan
TR = ("2017-10-18", "2026-09-30"); HO = ("2007-07-01", "2016-12-31")
tops_tr = ["2018-08-29", "2020-02-19", "2020-09-02", "2021-12-27", "2022-08-15", "2024-07-10", "2025-02-19", "2026-06-02"]
tops_ho = ["2007-10-31", "2008-06-05", "2009-02-09", "2010-04-23", "2011-07-26", "2012-04-02", "2012-09-19", "2015-07-20", "2015-12-01"]
idx = P.index
def temp_of(vcol, w=(1, 2, 3, 2, 1)):
    X = pd.concat([P["narrow"], P["top2"], P["lev"], vcol, P["topshare"]], axis=1)
    ww = np.array(w, float); ww = ww / ww.sum()
    t = pd.Series((X.values * ww).sum(axis=1), index=idx)
    return t.where(X.notna().all(axis=1))
def events(flag, gap=30):
    pos = np.where(flag.values)[0]; out = []
    for p in pos:
        if out and p - out[-1][1] <= gap: out[-1][1] = p
        else: out.append([p, p])
    return [(idx[a], idx[b]) for a, b in out]
def caught(flag, tops):
    n = 0; got = []
    for t in tops:
        j = idx.get_loc(pd.Timestamp(t))
        if flag.iloc[max(0, j - 30): j + 6].any(): n += 1; got.append(t[:7])
    return n, got
def th_for(t3, a, b, n=67):
    s = t3[(t3.index >= a) & (t3.index <= b)].dropna().sort_values(ascending=False)
    return (s.iloc[n - 1] + s.iloc[n]) / 2
def report(name, temp):
    t3 = temp.rolling(3, min_periods=3).min()
    th = th_for(t3, *TR)
    for lab, (a, b), tops, thx in [("训练窗", TR, tops_tr, th), ("留出(同频重标)", HO, tops_ho, None)]:
        m = (idx >= a) & (idx <= b)
        if thx is None:
            n_ho = int(round(m.sum() * 67 / 2249)); thx = th_for(t3, a, b, n_ho)
        hot = ((t3 > thx) & m).fillna(False)
        f, g = fwd[hot].dropna(), fmin[hot].dropna()
        base, baseg = fwd[m].mean(), fmin[m].mean()
        ev = events(hot)
        f1 = [fwd.get(e[0]) for e in ev]; f1 = [x for x in f1 if pd.notna(x)]
        g1 = [fmin.get(e[0]) for e in ev]; g1 = [x for x in g1 if pd.notna(x)]
        nc, got = caught(hot, tops)
        print(f"{name:14s} {lab:10s} 门槛 {thx:5.1f} {int(hot.sum()):3d}天/{len(ev):2d}次 | 30日收益 {f.mean()*100:+6.2f}% (比基准 {(f.mean()-base)*100:+6.2f}pp, 为负 {(f<0).mean()*100:3.0f}%) | "
              f"30日内最深 {g.mean()*100:+6.2f}% (比基准 {(g.mean()-baseg)*100:+6.2f}pp) | 按事件首日 收益 {np.mean(f1)*100:+5.2f}% 最深 {np.mean(g1)*100:+6.2f}% | 抓顶 {nc}/{len(tops)} {got}")
    return t3, th
va = P["vix_abs"]
res = {}
res["现行"] = report("现行", temp_of(va))
res["VIX≥20记中性"] = report("VIX高于20记中性", temp_of(va.clip(lower=50)))
res["不用VIX"] = report("不用VIX", temp_of(va * 0 + 50, w=(1, 2, 3, 0, 1)))
# 逐事件对照：现行 vs VIX≥20 记中性
print()
for k in ("现行", "VIX≥20记中性"):
    t3, th = res[k]
    m = (idx >= TR[0])
    hot = ((t3 > th) & m).fillna(False)
    print(k, "门槛", round(th, 2))
    for a, b in events(hot):
        n = int(hot.loc[a:b].sum())
        print(f"   {a.date()} ~ {b.date()}  {n:2d} 天  首日之后30日 {fwd.get(a)*100:+6.2f}%  30日内最深 {fmin.get(a)*100:+6.2f}%  首日VIX {vix.get(a):.1f}")
