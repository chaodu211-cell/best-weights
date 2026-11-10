# -*- coding: utf-8 -*-
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(HERE)
import pickle, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
B = pickle.load(open("base.pkl", "rb"))
P, temp, qqq, vix = B["parts"], B["temp"], B["qqq"], B["vix"]
W = {"narrow": 1, "top2": 2, "lev": 3, "vix_abs": 2, "topshare": 1}; ws = sum(W.values())
TH = 82.5
t3 = temp.rolling(3, min_periods=3).min()
hot = (t3 > TH).fillna(False)
idx = temp.index
# 下跌顶：收盘是前 63 日最高、之后 63 日内回撤 >= 11%，63 日内候选合并取最高
c = qqq.dropna(); v = c.values
cand = []
for i in range(63, len(v) - 1):
    if v[i] >= v[max(0, i - 63):i + 1].max():
        fut = v[i + 1:i + 64]
        if len(fut) and fut.min() / v[i] - 1 <= -0.11:
            cand.append(i)
tops = []
for i in cand:
    if tops and i - tops[-1] <= 63:
        if v[i] > v[tops[-1]]: tops[-1] = i
    else:
        tops.append(i)
rows = []
for i in tops:
    d = c.index[i]; j = idx.get_loc(d)
    w = slice(max(0, j - 30), j + 6)
    seg = t3.iloc[w]
    if seg.notna().sum() == 0: continue
    dm = seg.idxmax(); k = idx.get_loc(dm)
    # 连 3 日最低温度那三天里，最低的那一天（它才是决定 t3 的那一天）
    tri = temp.iloc[k - 2:k + 1]; dmin = tri.idxmin()
    p = P.loc[dmin]
    fut = v[i + 1:i + 64]
    is_hi252 = v[i] >= v[max(0, i - 251):i + 1].max()
    rows.append(dict(top=d.date(), dd63=round((fut.min() / v[i] - 1) * 100, 1), bearRally=("" if is_hi252 else "熊反"),
                     caught=bool(hot.iloc[w].any()), t3max=round(seg.max(), 1), day=dmin.date(),
                     narrow=round(p["narrow"], 0), top2=round(p["top2"], 0), lev=round(p["lev"], 0),
                     vix_abs=round(p["vix_abs"], 0), topshare=round(p["topshare"], 0), VIX=round(vix.loc[dmin], 1),
                     others4=round((temp.loc[dmin] - W["vix_abs"] / ws * p["vix_abs"]) / (1 - W["vix_abs"] / ws), 1)))
T = pd.DataFrame(rows)
print(T.to_string(index=False))
# 各因子拖了多少分：缺口 = 权重 × (100 − 读数)
print("\n各顶：离 100 分的缺口按因子分摊（分）")
for r in rows:
    gaps = {k: round(W[k] / ws * (100 - r[k]), 1) for k in W}
    print(r["top"], "t3max", r["t3max"], "缺口", gaps, "合计", round(sum(gaps.values()), 1))
