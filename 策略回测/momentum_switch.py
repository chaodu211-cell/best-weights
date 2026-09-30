# -*- coding: utf-8 -*-
"""纳指 / 半导体"动量二选一"本身的检验（2026-09-28）：不带模型信号、不带杠杆，只看选得好不好。

    python3 策略回测/momentum_switch.py        → _mom_switch.csv，并打印汇总

变体（事先定好，全部跑）：
  信号   ret:L        过去 L 日涨幅，L ∈ 21/42/63/84/126/189/252
         skip:L       过去 L 日涨幅但剔除最近 21 日（经典 12-1 做法），L ∈ 63/126/252
         sharpe:L     L 日涨幅 ÷ L 日波动（风险调整；半导体波动大，这个会偏向纳指），L ∈ 63/126/252
         ratio:L      SOXX/QQQ 价格比在其 L 日均线之上选半导体、之下选纳指，L ∈ 20/50/100/200
         combo        21/63/126/252 日涨幅的平均
  调仓   D 每日 / W 每周第一个交易日 / M 每月第一个交易日（M 与现行一致）
  缓冲   0 / 2% / 5%：对手的动量要超过当前持有的这么多才换（ratio 类是比值偏离均线的幅度）
  现金   abs=1：选中的那只自身动量 ≤ 0 就拿现金（现行做法，拿现金期间每天复查）；abs=0：永远持有其一
执行口径与 bt.py 相同：t 日收盘出信号，t+1 开盘成交，含成本。
"""
import itertools, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bt

PER = {"A 2004-03~2009-12": ("2004-03-01", "2009-12-31"), "B 2010-01~2016-09": ("2009-12-31", "2016-09-22"),
       "C 2016-09~2023-09": ("2016-09-22", "2023-09-22"), "D 2023-09~2026-09": ("2023-09-22", "2026-09-22")}
FULL = ("2004-03-01", "2026-09-22")


def signals(q, s):
    """→ {名称: (score_q, score_s, 自身动量 q, 自身动量 s)}；score 大者胜"""
    out = {}
    rq, rs = q.pct_change(), s.pct_change()
    for L in (21, 42, 63, 84, 126, 189, 252):
        a, b = q / q.shift(L) - 1, s / s.shift(L) - 1
        out[f"ret:{L}"] = (a, b, a, b)
    for L in (63, 126, 252):
        a, b = q.shift(21) / q.shift(L) - 1, s.shift(21) / s.shift(L) - 1
        out[f"skip:{L}"] = (a, b, a, b)
    for L in (63, 126, 252):
        a, b = q / q.shift(L) - 1, s / s.shift(L) - 1
        va, vb = rq.rolling(L).std() * np.sqrt(L), rs.rolling(L).std() * np.sqrt(L)
        out[f"sharpe:{L}"] = (a / va, b / vb, a, b)
    R = s / q
    for L in (20, 50, 100, 200):
        dev = R / R.rolling(L).mean() - 1                       # >0 半导体相对走强
        out[f"ratio:{L}"] = (-dev / 2, dev / 2, q / q.rolling(L).mean() - 1, s / s.rolling(L).mean() - 1)
    a = sum(q / q.shift(L) - 1 for L in (21, 63, 126, 252)) / 4
    b = sum(s / s.shift(L) - 1 for L in (21, 63, 126, 252)) / 4
    out["combo"] = (a, b, a, b)
    return out


def choose(idx, sig, sched, band, absf):
    """逐日决定持有：0=QQQ 1=SOXX -1=现金。只在调仓日（或拿现金时每天）重新判断。"""
    sq, ss, aq, as_ = (x.reindex(idx).values for x in sig)
    if sched == "D":
        day = np.ones(len(idx), bool)
    elif sched == "W":
        wk = idx.to_period("W"); day = np.r_[True, wk[1:] != wk[:-1]]
    else:
        day = np.r_[True, idx.month[1:] != idx.month[:-1]]
    cur = -1
    out = np.full(len(idx), -1)
    for t in range(len(idx)):
        if not (np.isfinite(sq[t]) and np.isfinite(ss[t])):
            continue
        if day[t] or cur == -1:
            if cur == 1:
                pick = 0 if sq[t] > ss[t] + band else 1
            elif cur == 0:
                pick = 1 if ss[t] > sq[t] + band else 0
            else:
                pick = 1 if ss[t] > sq[t] else 0
            own = as_[t] if pick == 1 else aq[t]
            cur = pick if (not absf or own > 0) else -1
        out[t] = cur
    return out


def weights(M, ch, legs):
    W = np.zeros((len(M.idx), len(M.cols)))
    for k, sym in enumerate(legs):
        W[ch == k, M.col_i[sym]] = 1.0
    return W


def run_all(M, legs, periods, full):
    q, s = M.C["QQQ"], M.C["SOXX"]
    S = signals(q, s)
    rows = []
    for (nm, sig), sched, band, absf in itertools.product(S.items(), "DWM", (0.0, 0.02, 0.05), (1, 0)):
        ch = choose(M.idx, sig, sched, band, absf)
        W = weights(M, ch, legs)
        r = dict(signal=nm, sched=sched, band=band, abs=absf)
        nav, tu = bt.simulate(M, W, *full)
        st = bt.stats(nav, tu)
        r.update(full=st["cagr"], full_mdd=st["mdd"], turn=st["turn_yr"],
                 sw=float((np.diff(ch[(M.idx >= full[0]) & (M.idx <= full[1])]) != 0).sum()) / ((pd.Timestamp(full[1]) - pd.Timestamp(full[0])).days / 365.25))
        for p, (a, b) in periods.items():
            r[p] = bt.stats(bt.simulate(M, W, a, b)[0])["cagr"]
        rows.append(r)
    return pd.DataFrame(rows)


def bench(M, legs, periods, full, seed=20260928, n_rand=300):
    q, s = legs
    out = {}
    for nm, W in (("持有纳指", {q: 1.0}), ("持有半导体", {s: 1.0}), ("各半（每日再平衡）", {q: .5, s: .5})):
        W = pd.DataFrame(W, index=M.idx)
        r = {"full": bt.stats(bt.simulate(M, W, *full)[0])["cagr"]}
        for p, (a, b) in periods.items():
            r[p] = bt.stats(bt.simulate(M, W, a, b)[0])["cagr"]
        out[nm] = r
    # 上帝视角：每月初就知道本月谁涨得多（上限，不可实现）
    idx = M.idx
    mf = np.r_[True, idx.month[1:] != idx.month[:-1]]
    mid = np.cumsum(mf)
    fq, fs = M.C[q], M.C[s]
    grp = pd.Series(mid, index=idx)
    endq = fq.groupby(grp).transform("last") / fq.groupby(grp).transform("first")
    ends = fs.groupby(grp).transform("last") / fs.groupby(grp).transform("first")
    ch = np.where(ends.values > endq.values, 1, 0)
    ch = np.r_[ch[1:], ch[-1]]                                 # t 行 = 明天开盘要持有的
    W = weights(M, ch, legs)
    r = {"full": bt.stats(bt.simulate(M, W, *full)[0])["cagr"]}
    for p, (a, b) in periods.items():
        r[p] = bt.stats(bt.simulate(M, W, a, b)[0])["cagr"]
    out["上帝视角（每月选对）"] = r
    # 随机：每月初抛硬币
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_rand):
        coin = rng.integers(0, 2, mid.max() + 1)[mid]
        W = weights(M, coin, legs)
        rs.append(bt.stats(bt.simulate(M, W, *full)[0])["cagr"])
    out["_random"] = np.array(rs)
    return out


if __name__ == "__main__":
    M = bt.Market("2003-01-01", "2026-09-22")
    df = run_all(M, ("QQQ", "SOXX"), PER, FULL)
    B = bench(M, ("QQQ", "SOXX"), PER, FULL)
    df.to_csv(os.path.join(HERE, "_mom_switch.csv"), index=False)
    rnd = B.pop("_random")
    pcols = list(PER)
    print("基准（1 倍，年化）")
    for k, v in B.items():
        print(f"  {k:14s} 全程 {v['full']:6.1%}  " + "  ".join(f"{p[:1]} {v[p]:6.1%}" for p in pcols))
    print(f"  每月抛硬币×{len(rnd)}  全程中位 {np.median(rnd):.1%}  90 分位 {np.percentile(rnd, 90):.1%}  97.5 分位 {np.percentile(rnd, 97.5):.1%}")
    df["pct_rand"] = [(rnd < x).mean() for x in df["full"]]
    half = B["各半（每日再平衡）"]
    df["beat_half"] = sum((df[p] > half[p]).astype(int) for p in pcols)
    cur = df[(df.signal == "ret:63") & (df.sched == "M") & (df.band == 0) & (df["abs"] == 1)].iloc[0]
    df["beat_cur"] = sum((df[p] > cur[p]).astype(int) for p in pcols)
    show = lambda d: print(d[["signal", "sched", "band", "abs", "full", "full_mdd", "sw"] + pcols + ["pct_rand", "beat_half"]]
                           .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n现行（ret:63 每月 无缓冲 abs=1）排第", int((df.full > cur.full).sum()) + 1, "/", len(df))
    show(df[(df.signal == "ret:63") & (df.sched == "M") & (df.band == 0)])
    print("\n全程年化前 15")
    show(df.sort_values("full", ascending=False).head(15))
    print("\n四段都跑赢各半的变体数：", int((df.beat_half == 4).sum()), " 其中全程前 15：")
    show(df[df.beat_half == 4].sort_values("full", ascending=False).head(15))
    print("\n按单一维度平均（全程年化，其他维度取平均）")
    for k in ("sched", "band", "abs"):
        print(" ", k, df.groupby(k)["full"].mean().round(4).to_dict())
    fam = df["signal"].str.split(":").str[0]
    print("  信号族", df.groupby(fam)["full"].mean().round(4).to_dict())
    print("  回看期(ret)", df[fam == "ret"].groupby("signal")["full"].mean().round(4).to_dict())
