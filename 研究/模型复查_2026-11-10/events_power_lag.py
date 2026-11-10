# -*- coding: utf-8 -*-
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(HERE)
import pickle, json, io, numpy as np, pandas as pd
B = pickle.load(open("base.pkl", "rb"))
P, temp, qqq, vix = B["parts"], B["temp"], B["qqq"], B["vix"]
idx = temp.index; FWD = 30
fwd = qqq.shift(-FWD) / qqq - 1
t3 = temp.rolling(3, min_periods=3).min()
TR = ("2017-10-18", "2026-09-30")
m = (idx >= TR[0]) & (idx <= TR[1])
hot = ((t3 > 82.5) & m).fillna(False)
def events(flag, gap=30):
    pos = np.where(flag.values)[0]; out = []
    for p in pos:
        if out and p - out[-1][1] <= gap: out[-1][1] = p
        else: out.append([p, p])
    return [(idx[a], idx[b]) for a, b in out]
print("== (a) 训练窗「按天」成绩的来源：每个事件占多少")
base = fwd[m].mean()
tot = fwd[hot].sum(); n = int(hot.sum())
rows = []
for a, b in events(hot):
    f = fwd[hot & (idx >= a) & (idx <= b)]
    rows.append((a.date(), b.date(), len(f), f.mean() * 100, f.sum() / tot * 100, fwd[a] * 100))
    print(f"  {a.date()} ~ {b.date()}  {len(f):2d} 天（占 {len(f)/n*100:4.1f}%）  这些天之后30日均值 {f.mean()*100:+6.2f}%  对「按天均值」的贡献 {f.sum()/tot*100:5.1f}%  首日之后 {fwd[a]*100:+5.2f}%")
print(f"  按天均值 {fwd[hot].mean()*100:+.2f}%（基准 {base*100:+.2f}%）；每个事件等权的均值 {np.mean([r[3] for r in rows]):+.2f}%；按事件首日 {np.mean([r[5] for r in rows]):+.2f}%")
big = [r for r in rows if r[2] >= 15]
print(f"  最长两段合计 {sum(r[2] for r in big)} 天 = {sum(r[2] for r in big)/n*100:.0f}% 的红点天数")
rest = hot.copy()
for r in big: rest &= ~((idx >= pd.Timestamp(r[0])) & (idx <= pd.Timestamp(r[1])))
print(f"  去掉这两段后：{int(rest.sum())} 天，之后30日 {fwd[rest].mean()*100:+.2f}%（比基准 {(fwd[rest].mean()-base)*100:+.2f}pp），为负 {(fwd[rest]<0).mean()*100:.0f}%")

print("\n== (c) 预先登记的检验功效（5 个事件首日、之后 30 日、随机日置换 p<0.10）")
r30 = fwd[m].dropna().values
sd_overlap = r30.std()
print(f"  训练窗 30 日收益：均值 {r30.mean()*100:+.2f}%  标准差 {sd_overlap*100:.2f}%")
rng = np.random.default_rng(1)
def power(edge, n_ev=5, sims=4000, draws=2000):
    hit = 0
    for _ in range(sims):
        # 事件收益 = 随机抽的真实 30 日收益 + edge（保留真实分布形状）
        ev = rng.choice(r30, n_ev, replace=False) + edge
        null = rng.choice(r30, (draws, n_ev)).mean(axis=1)
        p = (null <= ev.mean()).mean()
        hit += p < 0.10
    return hit / sims
for edge in (0.0, -0.02, -0.047, -0.08, -0.11):
    print(f"  真实效果 = 比基准 {edge*100:+.1f}pp → 判「通过」的概率 5 次 {power(edge)*100:4.0f}%   8 次 {power(edge, 8)*100:4.0f}%   12 次 {power(edge, 12)*100:4.0f}%")

print("\n== (d) 流水线发布时点（远端 _信号快照.csv）")
s = pd.read_csv("remote_snapshot.csv")
s = s[s["来源"] == "当日"].copy()
tcol = [c for c in s.columns if "记录" in c or "时间" in c][0]
d = pd.to_datetime(s["日期"]); rec = pd.to_datetime(s[tcol])            # 北京时间
rec_utc = rec - pd.Timedelta(hours=8)
def et(ts):  # 2026 年美国夏令时 3-08 ~ 11-01
    return ts - pd.Timedelta(hours=4 if pd.Timestamp("2026-03-08 07:00") <= ts < pd.Timestamp("2026-11-01 06:00") else 5)
rec_et = rec_utc.map(et)
cal = pd.bdate_range("2026-09-01", "2026-12-31")
nxt = d.map(lambda x: cal[cal > x][0])
open_n = nxt + pd.Timedelta(hours=9, minutes=30); close_n = nxt + pd.Timedelta(hours=16)
lag_h = (rec_et - (d + pd.Timedelta(hours=16))).dt.total_seconds() / 3600
out = pd.DataFrame({"数据日": d.dt.date.values, "发布(美东)": rec_et.dt.strftime("%m-%d %H:%M").values, "距收盘小时": lag_h.round(1).values,
                    "赶上次日开盘": (rec_et < open_n).values, "赶上次日收盘": (rec_et < close_n).values})
print(out.tail(16).to_string(index=False))
print(f"  共 {len(out)} 个交易日：赶上次日开盘 {out['赶上次日开盘'].mean()*100:.0f}%，赶上次日收盘 {out['赶上次日收盘'].mean()*100:.0f}%，距收盘中位 {out['距收盘小时'].median():.1f} 小时")
o2 = out[pd.to_datetime(out["数据日"]) >= "2026-10-20"]
print(f"  10-20 以来 {len(o2)} 天：赶上次日开盘 {o2['赶上次日开盘'].mean()*100:.0f}%，赶上次日收盘 {o2['赶上次日收盘'].mean()*100:.0f}%，距收盘中位 {o2['距收盘小时'].median():.1f} 小时")

print("\n== 当前状态（远端 data_alt.json）")
D = json.load(io.open("remote_data_alt.json", encoding="utf-8"))
S = D["series"]; dt = pd.to_datetime(S["dates"])
nd = pd.Series(S["ndx"], index=dt, dtype=float); tdc = pd.Series(S["tdc"], index=dt, dtype=float)
vx = pd.Series(S["vix"], index=dt, dtype=float); ts = pd.Series(S["temperature_sell"], index=dt, dtype=float)
dd = nd / nd.rolling(252, min_periods=60).max() - 1
print("  as_of", D["as_of"], "generated", D["generated_at"])
print(pd.DataFrame({"QQQ": nd, "距52周高点%": (dd * 100).round(1), "TDC": tdc, "VIX": vx, "红点温度": ts}).tail(8).to_string())
for k in ("hot", "hot_ad", "cold", "cold_soft", "tdc"):
    f = pd.Series(D["alerts"]["flags"][k], index=dt)
    print("  最近一次", k, f[f].index[-1].date() if f.any() else None)
