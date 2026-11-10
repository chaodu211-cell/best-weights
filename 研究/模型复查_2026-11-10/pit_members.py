# -*- coding: utf-8 -*-
"""后来才进指数的成分股，在纳入之前不该出现在因子里。用维基百科「Date added」把每只股票纳入日之前的数据拿掉，
重算依赖成分股的因子，看红点 / 蓝点 / 黄点变多少。只修「后来者提前出现」，被剔除的旧成分股仍然缺（没有数据）。只读。"""
import os as _os, sys as _sys
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(HERE, "..", ".."))
SP = HERE
import sys, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, REPO); os.chdir(REPO)
import engine as E, alt_engine as A, tdc_signal as TD
B = pickle.load(open(SP + "/base.pkl", "rb"))
P = B["parts"]; idx = P.index; qqq = B["qqq"]; vix = B["vix"]
mem = pd.read_csv(SP + "/members.csv", parse_dates=["Date added"])
mem["sym"] = mem["Symbol"].str.replace(".", "-", regex=False)
added = dict(zip(mem["sym"], mem["Date added"]))
stocks = E.load_many(E.STOCKS); names = sorted(stocks)
miss = [t for t in names if t not in added and t.replace(".", "-") not in added]
print("本地成分股", len(names), "| 维基对不上的", len(miss), miss[:12])
def add_date(t):
    return added.get(t, added.get(t.replace(".", "-"), pd.NaT))
first = {t: stocks[t].index[0] for t in names}
late = [t for t in names if pd.notna(add_date(t)) and add_date(t) > max(first[t], idx[0])]
print("纳入日晚于数据起点的", len(late), "只；其中 2017-10-18 之后纳入", sum(add_date(t) >= pd.Timestamp("2017-10-18") for t in late))
# 掩码：纳入日之前 = 不在样本里
mask = pd.DataFrame(True, index=idx, columns=names)
for t in late: mask.loc[idx < add_date(t), t] = False
print("样本只数：2006-01", int(mask.loc["2006-01-03"].sum()), " 2010-01", int(mask.loc["2010-01-04"].sum()), " 2017-10", int(mask.loc["2017-10-18"].sum()),
      " 2020-01", int(mask.loc["2020-01-02"].sum()), " 2024-01", int(mask.loc["2024-01-02"].sum()), " 现在", int(mask.iloc[-1].sum()))
dv = pd.DataFrame({t: stocks[t]["rawclose"] * stocks[t]["volume"] for t in names}).reindex(idx)
V0 = np.where(np.isfinite(dv.values) & (dv.values > 0), dv.values, 0.0)
sec = np.array([E.SECTOR_OF[t] for t in names]); secs = sorted(set(sec))
S = np.stack([(sec == s) for s in secs], axis=1).astype(float)
def rpct(x):
    s = pd.Series(np.asarray(x, float), index=idx)
    return (s.rolling(252, min_periods=252).rank() - 0.5) / 252 * 100
def conc(V):
    sd = V @ S; tot = sd.sum(axis=1); srt = np.sort(sd, axis=1)[:, ::-1]
    top2 = np.where(tot > 0, (srt[:, 0] + srt[:, 1]) / np.where(tot > 0, tot, np.nan) * 100, np.nan)
    cnt = (V > 0).sum(axis=1); k = np.maximum(1, np.ceil(E.CROWD_TOP_FRAC * cnt)).astype(int)
    cum = np.cumsum(-np.sort(-V, axis=1), axis=1); top = cum[np.arange(len(V)), np.clip(k, 1, V.shape[1]) - 1]
    ts = np.where((cnt >= 20) & (tot > 0), top / np.where(tot > 0, tot, np.nan) * 100, np.nan)
    return pd.Series(top2, index=idx), pd.Series(ts, index=idx)
t2r0, tsr0 = conc(V0); t2r1, tsr1 = conc(V0 * mask.values)
W = np.array([1, 2, 3, 2, 1], float); W /= W.sum()
def temp_with(t2, ts):
    X = np.column_stack([P["narrow"].values, t2.values, P["lev"].values, P["vix_abs"].values, ts.values])
    return pd.Series(X @ W, index=idx)
T0 = temp_with(rpct(t2r0), rpct(tsr0)); T1 = temp_with(rpct(t2r1), rpct(tsr1))
FWD = 30; fwd = qqq.shift(-FWD) / qqq - 1
m = (idx >= "2017-10-18") & (idx <= "2026-09-30")
def events(flag, gap=30):
    pos = np.where(flag.values)[0]; out = []
    for p in pos:
        if out and p - out[-1][1] <= gap: out[-1][1] = p
        else: out.append([p, p])
    return out
def th_n(t3, n=67):
    s = t3[m].dropna().sort_values(ascending=False); return (s.iloc[n - 1] + s.iloc[n]) / 2
tops = ["2018-08-29", "2020-02-19", "2020-09-02", "2021-12-27", "2022-08-15", "2024-07-10", "2025-02-19", "2026-06-02"]
def rep(lab, T, th):
    t3 = T.rolling(3, min_periods=3).min(); hot = ((t3 > th) & m).fillna(False)
    ev = events(hot); f = fwd[hot].dropna(); base = fwd[m].mean()
    nc = sum(hot.iloc[max(0, idx.get_loc(pd.Timestamp(t)) - 30): idx.get_loc(pd.Timestamp(t)) + 6].any() for t in tops)
    print(f"  {lab:26s} 门槛 {th:5.2f}  {int(hot.sum()):3d} 天/{len(ev)} 次  之后30日 {f.mean()*100:+6.2f}%（比基准 {(f.mean()-base)*100:+6.2f}pp，为负 {(f<0).mean()*100:.0f}%）"
          f"  事件首日 {np.mean([fwd.iloc[e[0]] for e in ev])*100:+5.2f}%  抓顶 {nc}/8")
    return hot, ev
print("\n== 红点（训练窗 2017-10-18~2026-09-30）")
print(f"  原始值相关：TOP2 {t2r0[m].corr(t2r1[m]):.3f}  前2% {tsr0[m].corr(tsr1[m]):.3f}；红点温度逐日差：平均绝对 {(T1-T0)[m].abs().mean():.2f} 分，最大 {(T1-T0)[m].abs().max():.1f} 分，差超过 3 分的日子 {((T1-T0)[m].abs()>3).mean()*100:.0f}%")
h0, e0 = rep("现行（今天的名单回填）", T0, 82.5)
h1, e1 = rep("纳入日之前不计·门槛不动", T1, 82.5)
h2, e2 = rep("纳入日之前不计·重标 67 天", T1, th_n(T1.rolling(3, min_periods=3).min()))
print("  逐事件（现行 → 纳入日之前不计，门槛都用 82.5）：")
t31 = T1.rolling(3, min_periods=3).min(); t30 = T0.rolling(3, min_periods=3).min()
for s, e in e0:
    n1 = int(h1.values[max(0, s - 5): e + 6].sum())
    print(f"     {idx[s].date()} ~ {idx[e].date()}  {int(h0.values[s:e+1].sum()):2d} 天 → {n1:2d} 天   连3日温度最高 {t30.iloc[s:e+1].max():.1f} → {t31.iloc[max(0,s-5):e+6].max():.1f}")
for s, e in e1:
    if all(s > b + 30 or e < a - 30 for a, b in e0):
        print(f"     新出现 {idx[s].date()} ~ {idx[e].date()}  {int(h1.values[s:e+1].sum())} 天  首日之后30日 {fwd.iloc[s]*100:+.2f}%")
print("  各顶前「连3日温度最高」：", {t[:7]: (round(t30.iloc[idx.get_loc(pd.Timestamp(t)) - 30: idx.get_loc(pd.Timestamp(t)) + 6].max(), 1), round(t31.iloc[idx.get_loc(pd.Timestamp(t)) - 30: idx.get_loc(pd.Timestamp(t)) + 6].max(), 1)) for t in tops})

# —— 黄点：TDC 内部损伤三项用成分股 ——
px, vx, const, ndx = TD.load()
def tdc_red(cst):
    F = TD.factors(px, vx, cst)
    Pp = F.rolling(TD.WIN, min_periods=TD.MINP).rank(pct=True) * 100
    legs = pd.DataFrame({k: Pp[v].mean(axis=1) for k, v in TD.LEGS.items()}); Sx = legs.mean(axis=1)
    dd = ndx / ndx.rolling(252, min_periods=60).max() - 1
    hot = ((Sx > TD.TH).rolling(TD.PERSIST).sum() == TD.PERSIST).fillna(False)
    armed, fired, voided = True, [], []
    for t, h, v, x in zip(px.index, hot.values, Sx.values, dd.values):
        if not np.isnan(v) and v < TD.REARM: armed = True
        if armed and h and np.isfinite(x):
            if TD.GATE_HI < x <= TD.GATE_LO: armed = False; fired.append(t)
            elif x <= TD.GATE_HI: armed = False; voided.append(t)
    return Sx, fired, voided, dd
cm = const.copy()
for t in late:
    if t in cm.columns: cm.loc[cm.index < add_date(t), t] = np.nan
S0, f0, v0, dd0 = tdc_red(const); S1, f1, v1, _ = tdc_red(cm)
print("\n== 黄点（全历史）")
print("  现行      ：", [d.strftime("%Y-%m-%d") for d in f0], "| 作废", [d.strftime("%Y-%m-%d") for d in v0])
print("  纳入前不计：", [d.strftime("%Y-%m-%d") for d in f1], "| 作废", [d.strftime("%Y-%m-%d") for d in v1])
print(f"  TDC 逐日差（2007 起）：平均绝对 {(S1-S0)['2007':].abs().mean():.2f} 分，最大 {(S1-S0)['2007':].abs().max():.1f} 分")
pickle.dump(dict(late=late, mask=mask, T0=T0, T1=T1, tdc0=S0, tdc1=S1, yellow0=f0, void0=v0, dd=dd0), open(SP + "/pit.pkl", "wb"))
