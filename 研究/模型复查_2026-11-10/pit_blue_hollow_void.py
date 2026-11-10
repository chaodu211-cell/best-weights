# -*- coding: utf-8 -*-
import os as _os, sys as _sys
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(HERE, "..", ".."))
SP = HERE
import sys, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, REPO); os.chdir(REPO)
import engine as E, alt_engine as A
B = pickle.load(open(SP + "/base.pkl", "rb")); Q = pickle.load(open(SP + "/pit.pkl", "rb"))
P = B["parts"]; idx = P.index; qqq = B["qqq"]; vix = B["vix"]; adjf = B["adjf"]; bt0 = B["bt"]
mask = Q["mask"]; T0, T1 = Q["T0"], Q["T1"]
FWD = 30; fwd = qqq.shift(-FWD) / qqq - 1; fwd63 = qqq.shift(-63) / qqq - 1
def events(flag, gap=30):
    pos = np.where(flag.values)[0]; out = []
    for p in pos:
        if out and p - out[-1][1] <= gap: out[-1][1] = p
        else: out.append([p, p])
    return out
# —— 红点留出段（同频重标）——
ho = (idx >= "2007-07-01") & (idx <= "2016-12-31")
tops_ho = ["2007-10-31", "2008-06-05", "2009-02-09", "2010-04-23", "2011-07-26", "2012-04-02", "2012-09-19", "2015-07-20", "2015-12-01"]
print("== 红点留出段 2007-07~2016（同频重标）")
for lab, T in (("现行（今天的名单回填）", T0), ("纳入日之前不计", T1)):
    t3 = T.rolling(3, min_periods=3).min(); n = int(round(ho.sum() * 67 / 2249))
    s = t3[ho].dropna().sort_values(ascending=False); th = (s.iloc[n - 1] + s.iloc[n]) / 2
    hot = ((t3 > th) & ho).fillna(False); f = fwd[hot].dropna()
    nc = sum(hot.iloc[max(0, idx.get_loc(pd.Timestamp(t)) - 30): idx.get_loc(pd.Timestamp(t)) + 6].any() for t in tops_ho)
    print(f"  {lab:22s} 门槛 {th:5.1f}  {int(hot.sum())} 天/{len(events(hot))} 次  之后30日 {f.mean()*100:+5.2f}%（比基准 {(f.mean()-fwd[ho].mean())*100:+5.2f}pp）  抓顶 {nc}/9"
          f"   固定 82.5 亮 {int(((t3 > 82.5) & ho).sum())} 天")
# —— 蓝点：TOP2、站上MA20 依赖成分股 ——
stocks = E.load_many(E.STOCKS); names = list(mask.columns)
px = pd.DataFrame({t: stocks[t]["close"] for t in names}).reindex(idx)
dv = pd.DataFrame({t: stocks[t]["rawclose"] * stocks[t]["volume"] for t in names}).reindex(idx)
sec = np.array([E.SECTOR_OF[t] for t in names]); secs = sorted(set(sec))
S = np.stack([(sec == s) for s in secs], axis=1).astype(float)
def rp(s): return (s.rolling(252, min_periods=252).rank() - 0.5) / 252 * 100
def blue(mk):
    V = np.where(np.isfinite(dv.values) & (dv.values > 0), dv.values, 0.0) * mk
    sd = V @ S; tot = sd.sum(axis=1); srt = np.sort(sd, axis=1)[:, ::-1]
    top2 = pd.Series(np.where(tot > 0, (srt[:, 0] + srt[:, 1]) / np.where(tot > 0, tot, np.nan) * 100, np.nan), index=idx)
    ma = px.rolling(E.MA_WIN, min_periods=E.MA_WIN).mean()
    ok = ma.notna().values & (mk > 0)
    above = ((px > ma).values & ok).sum(axis=1) / np.where(ok.sum(axis=1) > 0, ok.sum(axis=1), np.nan) * 100
    ma20 = pd.Series(above, index=idx)
    X = pd.concat([rp(top2), rp(ma20), adjf["leverage"], adjf["erp"].clip(upper=A.BLUE_ERP_CAP)], axis=1)
    return X.mean(axis=1, skipna=False)
b0 = blue(np.ones(mask.shape)); b1 = blue(mask.values.astype(float))
print("\n== 蓝点（实心+空心合计；VIX>=30 且判定温度<21.8）")
print("  复现检查：与管线蓝点温度最大差", round(float((b0 - bt0).abs().max()), 3))
for lab, a, b in (("训练窗 2017-10-18~", "2017-10-18", "2026-12-31"), ("2007-07~2016", "2007-07-01", "2016-12-31")):
    m = (idx >= a) & (idx <= b)
    c0 = ((b0 < A.BLUE_TH) & (vix >= E.VIX_COLD) & m).fillna(False); c1 = ((b1 < A.BLUE_TH) & (vix >= E.VIX_COLD) & m).fillna(False)
    both = int((c0 & c1).sum())
    print(f"  {lab}: 现行 {int(c0.sum())} 天/{len(events(c0, 40))} 次 之后30/63日 {fwd[c0].mean()*100:+.1f}/{fwd63[c0].mean()*100:+.1f}% → 纳入前不计 {int(c1.sum())} 天/{len(events(c1, 40))} 次 {fwd[c1].mean()*100:+.1f}/{fwd63[c1].mean()*100:+.1f}%；重合 {both} 天；温度逐日差平均绝对 {(b1-b0)[m].abs().mean():.2f} 分")
    gone = c0 & ~c1; new = c1 & ~c0
    print("     消失:", [d.strftime("%Y-%m-%d") for d in idx[gone.values]][:12], " 新增:", [d.strftime("%Y-%m-%d") for d in idx[new.values]][:12])
# —— 空心红点：只亮空心（不含实心）的日子 ——
print("\n== 空心红点（自适应门槛），只算「空心亮、实心没亮」的日子")
t3 = T0.rolling(3, min_periods=3).min()
th_ad = t3.shift(1).rolling(A.AD_WIN, min_periods=A.AD_MINP).quantile(1 - A.ALT_DESIGN_RATE)
had = (t3 > th_ad).fillna(False); hs = (t3 > 82.5).fillna(False)
for lab, a, b in (("训练窗 2017-10-18~2026-09", "2017-10-18", "2026-09-30"), ("2009-07~2017-10", "2009-07-01", "2017-10-17")):
    m = (idx >= a) & (idx <= b); only = had & ~hs & m; f = fwd[only].dropna(); base = fwd[m].mean()
    ev = events(only)
    print(f"  {lab}: 只空心 {int(only.sum())} 天/{len(ev)} 次  之后30日 {f.mean()*100:+.2f}%（比基准 {(f.mean()-base)*100:+.2f}pp，为负 {(f<0).mean()*100:.0f}%）  事件首日 {np.mean([fwd.iloc[e[0]] for e in ev])*100:+.2f}%"
          f" | 实心 {int((hs&m).sum())} 天 | 自适应门槛范围 {th_ad[m].min():.1f}~{th_ad[m].max():.1f}")
    yr = pd.Series(only[m].values, index=idx[m]).groupby(idx[m].year).sum()
    print("     逐年只空心天数:", {int(k): int(v) for k, v in yr.items() if v})
# —— 黄点作废之后 ——
print("\n== 黄点「作废」（确认时纳指已跌破 13%）之后纳指走势")
nd = qqq
for d in Q["void0"] + Q["yellow0"]:
    if d not in nd.index: continue
    i = nd.index.get_loc(d); seg = nd.iloc[i + 1:i + 64]
    kind = "作废" if d in Q["void0"] else "黄点"
    print(f"  {kind} {d.date()}  当日距52周高点 {Q['dd'].loc[d]*100:+.1f}%  之后63日内最深 {(seg.min()/nd.iloc[i]-1)*100:+.1f}%  63日后 {(nd.iloc[min(i+63, len(nd)-1)]/nd.iloc[i]-1)*100:+.1f}%")
