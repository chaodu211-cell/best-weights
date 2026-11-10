# -*- coding: utf-8 -*-
import os as _os, sys as _sys
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(HERE, "..", ".."))
SP = HERE
import sys, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, REPO); os.chdir(REPO)
import engine as E
B = pickle.load(open(SP + "/base.pkl", "rb")); Q = pickle.load(open(SP + "/pit.pkl", "rb"))
P = B["parts"]; idx = P.index; mask = Q["mask"]; names = list(mask.columns)
stocks = E.load_many(E.STOCKS)
dv = pd.DataFrame({t: stocks[t]["rawclose"] * stocks[t]["volume"] for t in names}).reindex(idx)
V0 = np.where(np.isfinite(dv.values) & (dv.values > 0), dv.values, 0.0)
tot = V0.sum(axis=1)
# 训练窗里「当时还没进指数」的成交额占全篮子的比例
pre = V0 * (~mask.values)
share = pd.Series(pre.sum(axis=1) / tot * 100, index=idx)
print("「当时还没进指数的股票」占模型里全篮子成交额的比例（年均 %）：")
print(share["2006":].groupby(share["2006":].index.year).mean().round(1).to_dict())
m = (idx >= "2017-10-18")
contrib = pd.Series((pre[m].sum(axis=0) / tot[m].sum()) * 100, index=names).sort_values(ascending=False)
mem = pd.read_csv(SP + "/members.csv", parse_dates=["Date added"]); mem["sym"] = mem["Symbol"].str.replace(".", "-", regex=False)
ad = dict(zip(mem["sym"], mem["Date added"]))
print("训练窗里占比最大的 10 只「后来者」（纳入前的成交额 / 全篮子，%）：")
print([(t, round(v, 2), str(ad.get(t, ""))[:10]) for t, v in contrib.head(10).items()])
sec = np.array([E.SECTOR_OF[t] for t in names]); secs = sorted(set(sec))
S = np.stack([(sec == s) for s in secs], axis=1).astype(float)
def rp(x): 
    s = pd.Series(np.asarray(x, float), index=idx); return (s.rolling(252, min_periods=252).rank() - 0.5) / 252 * 100
def temp_of(V):
    sd = V @ S; t = sd.sum(axis=1); srt = np.sort(sd, axis=1)[:, ::-1]
    top2 = np.where(t > 0, (srt[:, 0] + srt[:, 1]) / np.where(t > 0, t, np.nan) * 100, np.nan)
    cnt = (V > 0).sum(axis=1); k = np.maximum(1, np.ceil(E.CROWD_TOP_FRAC * cnt)).astype(int)
    cum = np.cumsum(-np.sort(-V, axis=1), axis=1); top = cum[np.arange(len(V)), np.clip(k, 1, V.shape[1]) - 1]
    ts = np.where((cnt >= 20) & (t > 0), top / np.where(t > 0, t, np.nan) * 100, np.nan)
    W = np.array([1, 2, 3, 2, 1], float); W /= W.sum()
    X = np.column_stack([P["narrow"].values, rp(top2).values, P["lev"].values, P["vix_abs"].values, rp(ts).values])
    return pd.Series(X @ W, index=idx), pd.Series(top2, index=idx), pd.Series(ts, index=idx)
T0, a0, b0 = temp_of(V0)
mk = np.ones(V0.shape); j = names.index("TSLA"); mk[idx < pd.Timestamp("2020-12-21"), j] = 0
T1, a1, b1 = temp_of(V0 * mk)
h0 = (T0.rolling(3).min() > 82.5); h1 = (T1.rolling(3).min() > 82.5)
w = slice("2020-01-02", "2020-02-28")
print(f"只把特斯拉在 2020-12-21 纳入之前拿掉：2020-01~02 红点 {int(h0[w].sum())} 天 → {int(h1[w].sum())} 天；训练窗合计 {int(h0['2017-10-18':'2026-09-30'].sum())} → {int(h1['2017-10-18':'2026-09-30'].sum())} 天")
print(f"   2020-02 上旬「前2%成交额占比」原始值均值 {b0['2020-02-03':'2020-02-14'].mean():.1f}% → {b1['2020-02-03':'2020-02-14'].mean():.1f}%；特斯拉当时占全篮子成交额 {(V0[:, j]/tot)[(idx>='2020-02-03')&(idx<='2020-02-14')].mean()*100:.1f}%")
