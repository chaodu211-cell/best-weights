# -*- coding: utf-8 -*-
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(HERE)
import pickle, numpy as np, pandas as pd
B = pickle.load(open("base.pkl", "rb"))
P, temp, qqq, vix = B["parts"], B["temp"], B["qqq"], B["vix"]
idx = temp.index; FWD = 30
fwd = qqq.shift(-FWD) / qqq - 1
fmin = pd.concat([qqq.shift(-k) for k in range(1, FWD + 1)], axis=1).min(axis=1) / qqq - 1
fmin[qqq.shift(-FWD).isna()] = np.nan
t3 = temp.rolling(3, min_periods=3).min()
def events(flag, gap=30):
    pos = np.where(flag.values)[0]; out = []
    for p in pos:
        if out and p - out[-1][1] <= gap: out[-1][1] = p
        else: out.append([p, p])
    return out
print("== 连3日温度分档：之后 30 日收益 / 30 日内最深回撤（括号里是事件数）")
edges = [0, 40, 60, 70, 75, 80, 82.5, 85, 90, 101]
for lab, a, b in (("训练窗 2017-10-18~2026-09", "2017-10-18", "2026-09-30"), ("2007-07~2016", "2007-07-01", "2016-12-31")):
    m = (idx >= a) & (idx <= b)
    print(f" {lab}  基准 收益 {fwd[m].mean()*100:+.2f}%  最深 {fmin[m].mean()*100:+.2f}%")
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = m & (t3 >= lo) & (t3 < hi)
        g = pd.Series(g, index=idx).fillna(False)
        if g.sum() == 0: continue
        print(f"   {lo:5.1f}~{hi:5.1f}  {int(g.sum()):4d} 天（{len(events(g))} 次）  收益 {fwd[g].mean()*100:+6.2f}%  为负 {(fwd[g].dropna()<0).mean()*100:3.0f}%  最深 {fmin[g].mean()*100:+6.2f}%")
print("\n== 其余四项很热（连3日 >= 88）但 VIX >= 20、因而没亮红点的日子（训练窗）")
o4 = (P["narrow"] + 2 * P["top2"] + 3 * P["lev"] + P["topshare"]) / 7.0
o43 = o4.rolling(3, min_periods=3).min()
m = (idx >= "2017-10-18") & (idx <= "2026-09-30")
g = pd.Series(m & (o43 >= 88) & (vix >= 20) & ~(t3 > 82.5), index=idx).fillna(False)
for s, e in events(g, 15):
    d0 = idx[s]
    seg = qqq.iloc[s:s + 31]
    print(f"   {idx[s].date()} ~ {idx[e].date()}  {int(g.values[s:e+1].sum())} 天  VIX {vix.iloc[s]:.1f}  其余四项 {o43.iloc[s]:.1f}  红点温度(连3日) {t3.iloc[s]:.1f}  首日之后30日 {fwd.iloc[s]*100:+.1f}%  30日内最深 {fmin.iloc[s]*100:+.1f}%")
print("\n== 2024-10 ~ 2025-02 每月连3日温度最高值（2025-02 顶之前）")
print((t3["2024-10":"2025-03"].groupby([t3["2024-10":"2025-03"].index.year, t3["2024-10":"2025-03"].index.month]).max().round(1)).to_dict())
