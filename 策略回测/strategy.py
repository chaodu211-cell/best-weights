# -*- coding: utf-8 -*-
"""策略层：预警模型的信号 → 每日目标权重。

结构（三层，参数见 DEFAULT）：

① 模型状态机（t 日收盘后判定，t+1 开盘执行）
   红点      最近一次红点（实心；use_ad 时空心也算）之后 red_hold 个交易日内，常态敞口乘 red_mult
   黄点      亮起即锁仓，直到 TDC 回落到 rearm 以下（模型自己的"重新武装"线）
   实心蓝点  之后 blue_hold 个交易日进入"抄底"：买 blue_asset 的杠杆版，敞口 = blue_lev × 分批系数，
             分批系数 = min(1, 近 blue_hold 日蓝点天数 / blue_k)（blue_k=1 即一次满仓）
             blue_release="perm"：蓝点永久解除黄点锁仓；"temp"：抄底期结束后若 TDC 仍 ≥ rearm 则回到锁仓
             （蓝点的设计口径是之后 30 日，本身不宣告趋势结束）
   空心蓝点  soft_mult × 同上（0 = 不理会）
   优先级：蓝点 > 黄点锁仓 > 常态（常态里再叠红点、TDC、熊市三个乘数）

② 常态敞口（以"底层指数的倍数"计，0~lev_max）
   vol_tgt 为 None → lev_max；否则 clip(vol_tgt / 底层 vol_win 日实现波动, 0, lev_max)
   tdc_gate：TDC > tdc_gate 时乘 tdc_mult
   bear_mult：纳指处于模型的"熊市"（收盘 < MA200 且 MA200 低于 21 日前，engine.bear_regime）时乘 bear_mult

③ 选品
   fixed:<底层>  只做这一只底层（如 QQQ）的最高倍数可交易杠杆多头
   rot           在全部 1x 指数/行业 ETF 里按 mom 日动量取前 top 名，持有各自最高倍数的可交易杠杆多头；
                 每月第一个交易日换一次；abs_mom 为真时动量 ≤ 0 的名次改拿现金
   mix:A+B       两只底层各占一半，各自按波动率目标定敞口
   choice        逐日选择由外部给定：p["choice"][t] 为 p["choice_u"] 里的下标，-1 为现金（动量二选一的变体检验用）

④ 做空腿（2026-09-28 加，默认关闭，旧结果不变）
   short_yellow  黄点锁仓期持有 SQQQ 的权重（1/3 ≈ −1 倍纳指）
   short_red     红点清仓期（常态敞口被乘到 0 时）持有 SQQQ 的权重

Prep(allow=...) 可把可交易品种限定在一个清单里（如 SOXL/USD/SOXX/TQQQ/QLD/QQQ/SQQQ），
轮动的候选底层随之只剩清单里有代表的那几只。
"""
import numpy as np, pandas as pd

DEFAULT = dict(
    universe="fixed:QQQ", top=1, mom=126, abs_mom=True,
    lev_max=3.0, vol_tgt=None, vol_win=20,
    red_hold=30, red_mult=0.0, use_ad=False,
    rearm=60.0,
    blue_hold=40, blue_k=1, blue_lev=3.0, blue_asset="QQQ", soft_mult=0.0, blue_release="perm",
    tdc_gate=None, tdc_mult=0.0, bear_mult=1.0,
    short_yellow=0.0, short_red=0.0,
)


class Prep:
    """与参数无关的预计算，全部转成 numpy。"""

    def __init__(self, M, S, allow=None):
        self.M = M
        self.allow = set(allow) if allow else None
        idx = M.idx
        S = S.reindex(idx)
        self.S = S
        b = lambda c: S[c].fillna(False).astype(bool).values
        self.red, self.red_ad = b("red"), b("red_ad")
        self.blue, self.soft, self.yellow = b("blue"), b("blue_soft"), b("yellow")
        self.tdc = S["tdc"].values.astype(float)
        ux = M.ux
        self.unders = list(ux.columns)
        self.ui = {u: i for i, u in enumerate(self.unders)}
        r = ux.pct_change()
        self.vol = {w: (r.rolling(w, min_periods=int(w * .8)).std() * np.sqrt(252)).values for w in (10, 20, 40, 60)}
        self.mom = {m: (ux / ux.shift(m) - 1).values for m in (63, 126, 252)}
        q = M.C["QQQ"]; ma = q.rolling(200, min_periods=200).mean()
        self.bear = ((q < ma) & (ma - ma.shift(21) < 0)).values
        # 每只底层逐日可交易的最高倍数多头：best_col[u_i, t]（-1 = 没有），best_L[u_i, t]
        T = len(idx)
        trad = M.tradable[M.cols].values
        lev = M.lev.reindex(columns=M.cols).values
        self.best_col = -np.ones((len(self.unders), T), dtype=int)
        self.best_L = np.zeros((len(self.unders), T))
        for u in self.unders:
            i = self.ui[u]
            for s in [s for s in M.syms if M.under[s] == u and (self.allow is None or s in self.allow)]:
                c = M.col_i[s]
                L = np.where(trad[:, c], lev[:, c], 0.0)
                better = L > self.best_L[i]
                self.best_L[i] = np.where(better, L, self.best_L[i])
                self.best_col[i] = np.where(better, c, self.best_col[i])
        self.month_first = np.r_[True, idx.month[1:] != idx.month[:-1]]
        self.sq_col = M.col_i["SQQQ"]
        self.sq_ok = trad[:, self.sq_col] & (self.allow is None or "SQQQ" in self.allow)
        self.ok_mom = {m: np.isfinite(v) & (self.best_col.T >= 0) for m, v in self.mom.items()}


def build(P, p, t_from=None):
    """→ (目标权重矩阵 T×列, 状态数组)。t 行在 t+1 开盘执行。"""
    p = {**DEFAULT, **p}
    M = P.M
    T, n = len(M.idx), len(M.cols)
    Wm = np.zeros((T, n))
    state = np.zeros(T, dtype=np.int8)      # 0 常态 1 红点 2 闸门 3 黄点锁仓 4 蓝点
    red = (P.red | P.red_ad) if p["use_ad"] else P.red
    vol = P.vol[p["vol_win"]]
    bh, bk = p["blue_hold"], p["blue_k"]
    ncum_b, ncum_s = np.cumsum(P.blue), np.cumsum(P.soft)
    fixed = p["universe"].startswith("fixed:") or p["universe"].startswith("mix:")
    fu = ([P.ui[p["universe"][6:]]] if p["universe"].startswith("fixed:")
          else [P.ui[x] for x in p["universe"][4:].split("+")] if fixed else None)
    ba = P.ui[p["blue_asset"]]
    last_red = last_blue = last_soft = -10**9
    locked = False
    sel = []
    for t in range(T):
        if red[t]: last_red = t
        if P.blue[t]:
            last_blue = t
            if p["blue_release"] == "perm": locked = False
        if P.soft[t]: last_soft = t
        x = P.tdc[t]
        if P.yellow[t]: locked = True
        elif locked and x == x and x < p["rearm"]: locked = False
        # —— 抄底 ——
        if t - last_blue < bh or (p["soft_mult"] > 0 and t - last_soft < bh):
            lo = t - bh
            nb = ncum_b[t] - (ncum_b[lo] if lo >= 0 else 0)
            ns = ncum_s[t] - (ncum_s[lo] if lo >= 0 else 0)
            E = p["blue_lev"] * min(1.0, (nb + p["soft_mult"] * ns) / bk)
            c, L = P.best_col[ba, t], P.best_L[ba, t]
            if c >= 0 and E > 0:
                Wm[t, c] = min(1.0, E / L)
            state[t] = 4
            continue
        if locked:
            state[t] = 3
            if p["short_yellow"] > 0 and P.sq_ok[t]:
                Wm[t, P.sq_col] = p["short_yellow"]
            continue
        mult = 1.0
        if t - last_red < p["red_hold"]:
            mult *= p["red_mult"]; state[t] = 1
        if p["tdc_gate"] is not None and x == x and x > p["tdc_gate"]:
            mult *= p["tdc_mult"]; state[t] = state[t] or 2
        if P.bear[t]:
            mult *= p["bear_mult"]
            if p["bear_mult"] < 1: state[t] = state[t] or 2
        if mult <= 0:
            if state[t] == 1 and p["short_red"] > 0 and P.sq_ok[t]:
                Wm[t, P.sq_col] = p["short_red"]
            continue
        # —— 选品 ——
        if fixed:
            sel = fu
        elif p["universe"] == "choice":
            ch = p["choice"][t]
            sel = [P.ui[p["choice_u"][ch]]] if ch >= 0 else []
        elif P.month_first[t] or not sel:
            mo = P.mom[p["mom"]][t]
            ok = np.where(P.ok_mom[p["mom"]][t])[0]
            ok = ok[np.argsort(-mo[ok])][:p["top"]]
            sel = [i for i in ok if (mo[i] > 0 or not p["abs_mom"])]
        k = len(fu) if fixed else p["top"]
        for i in sel:
            c, L = P.best_col[i, t], P.best_L[i, t]
            if c < 0: continue
            if p["vol_tgt"] is None:
                E = p["lev_max"]
            else:
                v = vol[t, i]
                E = p["lev_max"] if not (v == v) or v <= 0 else min(p["lev_max"], p["vol_tgt"] / v)
            Wm[t, c] += min(1.0, E * mult / L) / k
    tot = Wm.sum(axis=1, keepdims=True)
    Wm = Wm / np.maximum(tot, 1.0)
    return Wm, state


STATE_NAMES = {0: "常态", 1: "红点减仓", 2: "闸门", 3: "黄点锁仓", 4: "蓝点抄底"}
