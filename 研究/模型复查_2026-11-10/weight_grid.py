# -*- coding: utf-8 -*-
"""五个因子的权重网格（0~4），门槛各自对齐到训练窗 67 天。看现行权重 (1,2,3,2,1) 在不同评判标准下排第几。
评判：A 按天之后 30 日均值（原标定用的）；B 同 A 但不计 2020-01~03 的日子；C 每个事件等权；D 事件首日；
E 按天 30 日内最深回撤；F 抓到几个顶。只是看「最优」对评判标准有多敏感，不是重新标定。"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(HERE)
import pickle, itertools, numpy as np, pandas as pd
B = pickle.load(open("base.pkl", "rb"))
P, qqq = B["parts"], B["qqq"]
idx = P.index; FWD = 30
fwd = (qqq.shift(-FWD) / qqq - 1).values
fmin = (pd.concat([qqq.shift(-k) for k in range(1, FWD + 1)], axis=1).min(axis=1) / qqq - 1)
fmin[qqq.shift(-FWD).isna()] = np.nan; fmin = fmin.values
m = ((idx >= "2017-10-18") & (idx <= "2026-09-30"))
X = P[["narrow", "top2", "lev", "vix_abs", "topshare"]].values
ok = np.isfinite(X).all(axis=1)
covid = ((idx >= "2020-01-01") & (idx <= "2020-03-31"))
tops = ["2018-08-29", "2020-02-19", "2020-09-02", "2021-12-27", "2022-08-15", "2024-07-10", "2025-02-19", "2026-06-02"]
tw = [(max(0, idx.get_loc(pd.Timestamp(t)) - 30), idx.get_loc(pd.Timestamp(t)) + 6) for t in tops]
base = np.nanmean(fwd[m]); baseg = np.nanmean(fmin[m])
N = 67
def run(w):
    w = np.array(w, float)
    t = X @ (w / w.sum()); t[~ok] = np.nan
    s = pd.Series(t)
    t3 = s.rolling(3, min_periods=3).min().values
    v = np.where(m, t3, np.nan)
    srt = np.sort(v[np.isfinite(v)])[::-1]
    th = (srt[N - 1] + srt[N]) / 2
    hot = np.isfinite(v) & (v > th)
    pos = np.where(hot)[0]
    ev = []
    for p in pos:
        if ev and p - ev[-1][-1] <= 30: ev[-1].append(p)
        else: ev.append([p])
    A = np.nanmean(fwd[hot]) - base
    hb = hot & ~covid
    Bv = np.nanmean(fwd[hb]) - np.nanmean(fwd[m & ~covid]) if hb.sum() >= 10 else np.nan
    C = np.nanmean([np.nanmean(fwd[e]) for e in ev]) - base
    Dv = np.nanmean([fwd[e[0]] for e in ev]) - base
    Ev = np.nanmean(fmin[hot]) - baseg
    F = sum(hot[a:b].any() for a, b in tw)
    return dict(w=tuple(int(x) for x in w), th=th, days=int(hot.sum()), ev=len(ev), A=A * 100, B=Bv * 100, C=C * 100, D=Dv * 100, E=Ev * 100, F=F,
                covid_days=int((hot & covid).sum()))
rows = []
for w in itertools.product(range(5), repeat=5):
    if sum(1 for x in w if x > 0) < 2: continue
    g = np.gcd.reduce([x for x in w if x > 0])
    if g > 1: continue          # 去掉成比例的重复
    rows.append(run(w))
G = pd.DataFrame(rows)
cur = G[G["w"] == (1, 2, 3, 2, 1)].iloc[0]
print("组合数", len(G))
print("现行 (拥挤1, TOP2 2, 杠杆3, VIX 2, 前2% 1)：", {k: (round(float(cur[k]), 2) if k not in ("w",) else cur[k]) for k in ("th", "days", "ev", "A", "B", "C", "D", "E", "F", "covid_days")})
for k, lab in [("A", "A 按天30日（原标定口径）"), ("B", "B 按天，不计2020-01~03"), ("C", "C 事件等权"), ("D", "D 事件首日"), ("E", "E 按天30日内最深回撤")]:
    r = (G[k] < cur[k]).sum() + 1
    print(f"  {lab:24s} 现行 {cur[k]:+6.2f}pp  排第 {r:4d}/{len(G)}（前 {r/len(G)*100:4.1f}%）  网格最好 {G[k].min():+6.2f}  中位 {G[k].median():+6.2f}")
print(f"  F 抓顶 现行 {int(cur['F'])}/8；网格里抓到 >= {int(cur['F'])} 个的占 {(G['F']>=cur['F']).mean()*100:.0f}%，抓到 6 个以上的 {(G['F']>=6).sum()} 组")
print("\n各评判标准下前 5 名的权重（拥挤, TOP2, 杠杆, VIX, 前2%）：")
for k in ("A", "B", "C", "D", "E"):
    top = G.nsmallest(5, k)
    print(" ", k, [(r.w, round(r[k], 2), int(r.F)) for r in top.itertuples() for k in [k]] if False else [(tuple(r.w), round(getattr(r, k), 2), int(r.F)) for r in top.itertuples()])
print("\n各评判标准之间的秩相关（全网格）：")
print(G[["A", "B", "C", "D", "E"]].corr(method="spearman").round(2).to_string())
# A 口径前 5% 的组合，在别的口径下的中位名次
topA = G.nsmallest(int(len(G) * 0.05), "A")
for k in ("B", "C", "D", "E"):
    pr = (G[k].rank(pct=True))[topA.index].median()
    print(f"  A 口径前 5% 的组合在 {k} 口径下的名次中位：前 {pr*100:.0f}%")
print("  A 口径前 5% 的组合里 2020-01~03 的红点天数：中位", topA["covid_days"].median(), "；全网格中位", G["covid_days"].median())
print("  A 口径与「2020-01~03 红点天数」的秩相关：", round(G[["A", "covid_days"]].corr(method="spearman").iloc[0, 1], 2))
G.to_pickle("grid.pkl")
