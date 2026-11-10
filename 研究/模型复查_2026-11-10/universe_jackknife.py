# -*- coding: utf-8 -*-
"""红点对成分股名单的敏感度：每次随机拿掉 k 只成分股（其余代码、数据、门槛都不动），重算 TOP2 与前2% 两个因子，
看历史上每个红点事件还在不在。只读仓库数据。"""
import os as _os, sys as _sys
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(HERE, "..", ".."))
SP = HERE
import sys, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, REPO); os.chdir(REPO)
import engine as E
B = pickle.load(open(os.path.join(HERE, "base.pkl"), "rb"))
P = B["parts"]; idx = P.index
stocks = E.load_many(E.STOCKS)
names = sorted(stocks)
dv = pd.DataFrame({t: stocks[t]["rawclose"] * stocks[t]["volume"] for t in names}).reindex(idx)
V = np.where(np.isfinite(dv.values) & (dv.values > 0), dv.values, 0.0)
sec = np.array([E.SECTOR_OF[t] for t in names]); secs = sorted(set(sec))
S = np.stack([(sec == s) for s in secs], axis=1).astype(float)          # 股票 × 行业
def rpct(x):
    s = pd.Series(x, index=idx)
    return ((s.rolling(252, min_periods=252).rank() - 0.5) / 252 * 100).values
def factors(keep):
    v = V[:, keep]
    # TOP2：行业成交额占比最大的两个
    sd = v @ S[keep]
    tot = sd.sum(axis=1)
    srt = np.sort(sd, axis=1)[:, ::-1]
    top2 = np.where(tot > 0, (srt[:, 0] + srt[:, 1]) / np.where(tot > 0, tot, np.nan) * 100, np.nan)
    # 前 2%
    cnt = (v > 0).sum(axis=1)
    k = np.maximum(1, np.ceil(E.CROWD_TOP_FRAC * cnt)).astype(int)
    cum = np.cumsum(-np.sort(-v, axis=1), axis=1)
    top = cum[np.arange(len(v)), np.clip(k, 1, v.shape[1]) - 1]
    ts = np.where((cnt >= 20) & (tot > 0), top / np.where(tot > 0, tot, np.nan) * 100, np.nan)
    return rpct(top2), rpct(ts)
W = np.array([1, 2, 3, 2, 1], float); W /= W.sum()
def temp_with(t2, ts):
    X = np.column_stack([P["narrow"].values, t2, P["lev"].values, P["vix_abs"].values, ts])
    t = X @ W
    return pd.Series(t, index=idx)
def hot_of(temp, th=82.5):
    t3 = temp.rolling(3, min_periods=3).min()
    return (t3 > th).fillna(False)
full = np.ones(len(names), bool)
t2, ts = factors(full)
print("复现检查：TOP2 最大差", np.nanmax(np.abs(t2 - P["top2"].values)), " 前2% 最大差", np.nanmax(np.abs(ts - P["topshare"].values)))
base_temp = temp_with(t2, ts)
m = (idx >= "2017-10-18") & (idx <= "2026-09-30")
base_hot = hot_of(base_temp) & m
def events(flag, gap=30):
    pos = np.where(flag.values)[0]; out = []
    for p in pos:
        if out and p - out[-1][1] <= gap: out[-1][1] = p
        else: out.append([p, p])
    return out
bev = events(base_hot)
print("基准：", int(base_hot.sum()), "天 /", len(bev), "次")
t3b = base_temp.rolling(3, min_periods=3).min()
rng = np.random.default_rng(20261110)
for kdrop, reps in ((3, 300), (5, 300), (25, 200)):
    surv = np.zeros(len(bev)); days = []; newev = []; dT = []
    daycount = np.zeros((reps, len(bev)))
    for r in range(reps):
        keep = full.copy(); keep[rng.choice(len(names), kdrop, replace=False)] = False
        a, b = factors(keep)
        tp = temp_with(a, b)
        h = hot_of(tp) & m
        days.append(int(h.sum()))
        dT.append(np.nanmax(np.abs((tp - base_temp)[m].values)))
        for j, (s, e) in enumerate(bev):
            n = int(h.values[s:e + 1].sum())
            daycount[r, j] = n
            surv[j] += n > 0
        # 新冒出来的事件：与任何基准事件相隔 30 日以上
        ne = 0
        for s2, e2 in events(h):
            if all(s2 > e + 30 or e2 < s - 30 for s, e in bev): ne += 1
        newev.append(ne)
    print(f"\n每次随机拿掉 {kdrop} 只（{kdrop/len(names)*100:.1f}%），{reps} 次：红点天数 中位 {np.median(days):.0f}（{np.percentile(days,5):.0f}~{np.percentile(days,95):.0f}），"
          f"温度最大变动 中位 {np.median(dT):.1f} 分；出现基准没有的新事件的比例 {np.mean(np.array(newev)>0)*100:.0f}%")
    for j, (s, e) in enumerate(bev):
        marg = t3b.iloc[s:e + 1].max() - 82.5
        print(f"   {idx[s].date()} ~ {idx[e].date()}  基准 {int(base_hot.values[s:e+1].sum()):2d} 天，最高超出门槛 {marg:4.1f} 分 → 仍然出现 {surv[j]/reps*100:5.1f}%  天数中位 {np.median(daycount[:, j]):.0f}（{np.percentile(daycount[:, j],5):.0f}~{np.percentile(daycount[:, j],95):.0f}）")
