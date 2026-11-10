# -*- coding: utf-8 -*-
"""VIX 因子在热端是否在帮倒忙：其余四个因子都很热的日子里，按 VIX 高低分组看之后 30 日 QQQ。"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(HERE)
import pickle, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
B = pickle.load(open("base.pkl", "rb"))
P, temp, qqq, vix = B["parts"], B["temp"], B["qqq"], B["vix"]
W = {"narrow": 1, "top2": 2, "lev": 3, "vix_abs": 2, "topshare": 1}; ws = sum(W.values())
FWD = 30
fwd = qqq.shift(-FWD) / qqq - 1
o4 = (P["narrow"] * 1 + P["top2"] * 2 + P["lev"] * 3 + P["topshare"] * 1) / 7.0
o4_3 = o4.rolling(3, min_periods=3).min()
df = pd.DataFrame({"o4": o4, "o4_3": o4_3, "vix": vix, "va": P["vix_abs"], "fwd": fwd, "temp": temp}).dropna(subset=["o4_3", "vix"])
def ev_count(mask):
    pos = np.where(mask.values)[0]; n = 0; last = -999
    for p in pos:
        if p - last > 30: n += 1
        last = p
    return n
for name, a, b in [("训练窗 2017-10-18~", "2017-10-18", "2026-12-31"), ("留出 2007-07~2016", "2007-07-01", "2016-12-31"), ("全部 2007-07~", "2007-07-01", "2026-12-31")]:
    d = df[(df.index >= a) & (df.index <= b)]
    base = d["fwd"].mean() * 100
    print(f"\n== {name}  基准 {base:+.2f}%  N={len(d)}")
    for lo in (85, 88, 90):
        h = d[d["o4_3"] >= lo]
        for lab, m in [("VIX<15", h["vix"] < 15), ("15~20", (h["vix"] >= 15) & (h["vix"] < 20)), ("VIX>=20", h["vix"] >= 20)]:
            g = h[m]; k = g["fwd"].dropna()
            mask = pd.Series(False, index=d.index); mask.loc[g.index] = True
            print(f"  其余四项连3日>= {lo}  {lab:8s} {len(g):4d} 天/{ev_count(mask):2d} 次  之后30日 {k.mean()*100 if len(k) else float('nan'):+6.2f}%  为负 {(k<0).mean()*100 if len(k) else float('nan'):5.1f}%")
