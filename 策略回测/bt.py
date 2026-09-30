# -*- coding: utf-8 -*-
"""回测引擎：预警模型信号 → 目标仓位 → 逐日模拟（t 日收盘出信号，t+1 开盘成交）。

口径（所有策略共用，不随参数变）：
  · 价格：Yahoo adjclose（含分红）；开盘价按 adjclose/close 同比例复权。
  · 成交：t 日收盘后才知道 t 日信号（数据管线北京时间次日早上才跑完），t+1 开盘按目标仓位成交。
  · 成本：单边按 60 日均成交额分档 —— ≥5 亿美元 3bp、≥5000 万 8bp、≥1000 万 20bp；
          低于 1000 万美元的标的当天不可交易（UGE、UCC、LTL 这类冷门杠杆 ETF 基本被这条挡掉）。
  · 现金：BIL（1-3 月国库券 ETF，含费率）；BIL 上市前按 ^IRX/252。
  · 杠杆 ETF 只在真实上市之后才可交易，不做回填。
  · 再平衡：目标换标的时必换；同一标的实际权重偏离目标超过 band 才调。
"""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ETF = os.path.join(HERE, "etf")
U = json.load(open(os.path.join(HERE, "universe.json")))

# 实测有效杠杆（逐年回归，见 fetch 后的核对）：这几只在 2020-04 由 3x 降为 2x
DELEV = {"ERX": "2020-04-01", "GUSH": "2020-04-01", "NUGT": "2020-04-01"}
COST_TIERS = ((5e8, 3e-4), (5e7, 8e-4), (1e7, 20e-4))
MIN_ADV = 1e7


def _read(sym):
    d = pd.read_csv(os.path.join(ETF, sym.replace("^", "_") + ".csv"), parse_dates=["date"])
    d = d.drop_duplicates("date", keep="last").set_index("date").sort_index()
    return d


class Market:
    """所有标的对齐到 QQQ 交易日：隔夜收益、日内收益、可交易性、成本、底层与倍数。"""

    def __init__(self, start="2005-01-01", end=None):
        q = _read("QQQ")
        idx = q.index[(q.index >= start) & ((q.index <= end) if end else True)]
        self.idx = idx
        self.syms = [s for s in U if s != "BIL"]
        C, O, DV = {}, {}, {}
        for s in self.syms + ["BIL"]:
            d = _read(s).reindex(idx)
            f = d["adjclose"] / d["close"]
            C[s], O[s] = d["adjclose"], d["open"] * f
            DV[s] = (d["close"] * d["volume"]).rolling(60, min_periods=20).mean()
        self.C, self.O = pd.DataFrame(C), pd.DataFrame(O)
        irx = _read("^IRX")["close"].reindex(idx).ffill() / 100 / 252
        # 隔夜 / 日内收益；缺开盘价时整日记在隔夜
        on = self.O / self.C.shift(1) - 1
        idr = self.C / self.O - 1
        full = self.C / self.C.shift(1) - 1
        bad = on.isna() | idr.isna()
        self.on = on.where(~bad, full).fillna(0.0)
        self.id = idr.where(~bad, 0.0).fillna(0.0)
        cash = full["BIL"].where(full["BIL"].notna(), irx)
        self.on["BIL"], self.id["BIL"] = cash.fillna(0.0), 0.0
        adv = pd.DataFrame(DV)
        # 可交易：有价格、上市满 20 日、前一日 60 日均成交额 ≥ MIN_ADV（只用 t−1 及以前）
        adv_prev = adv.shift(1)
        self.tradable = (self.C.notna() & self.C.shift(20).notna() & (adv_prev >= MIN_ADV))
        self.tradable["BIL"] = True
        cost = pd.DataFrame(COST_TIERS[-1][1], index=idx, columns=adv.columns)
        for th, c in reversed(COST_TIERS[:-1]):
            cost = cost.where(~(adv_prev >= th), c)
        cost["BIL"] = 1e-4
        self.cost = cost
        # 底层与倍数（倍数随时间变的单独处理）
        self.under = {s: U[s][0] for s in self.syms}
        lev = pd.DataFrame({s: float(U[s][1]) for s in self.syms}, index=idx)
        for s, d in DELEV.items():
            lev.loc[lev.index < d, s] = 3.0
        self.lev = lev
        # 底层 1x 的收盘（选品、波动率用）
        self.ux = self.C[[s for s in self.syms if U[s][1] == 1]]
        self.cols = list(self.C.columns)
        self.col_i = {s: i for i, s in enumerate(self.cols)}
        self._on, self._id = self.on[self.cols].values, self.id[self.cols].values
        self._cost = self.cost[self.cols].values

    def pos(self, t):
        return self.idx.get_loc(pd.Timestamp(t)) if not isinstance(t, int) else t


def simulate(mkt, W, start, end, band=0.05):
    """W：目标权重（DataFrame 或与 mkt.cols 对齐的 T×n 矩阵），t 行 = t 日收盘后决定、t+1 开盘执行
    （行和 ≤ 1，余数进 BIL）。返回逐日净值与换手。"""
    idx = mkt.idx
    i0, i1 = idx.searchsorted(pd.Timestamp(start)), idx.searchsorted(pd.Timestamp(end), side="right") - 1
    if isinstance(W, pd.DataFrame):
        W = W.reindex(index=idx, columns=mkt.cols).fillna(0.0).values
    bil = mkt.col_i["BIL"]
    # 只保留窗口里用到的列，加速
    used = np.where((W[max(0, i0 - 1):i1 + 1] > 0).any(axis=0))[0]
    cols = np.unique(np.r_[used[used != bil], bil])
    b = int(np.where(cols == bil)[0][0])
    Wt = W[:, cols].copy()
    Wt[:, b] = 0.0
    Wt[:, b] = 1.0 - Wt.sum(axis=1)
    ON, ID, CO = mkt._on[:, cols], mkt._id[:, cols], mkt._cost[:, cols]
    h = np.zeros(len(cols)); h[b] = 1.0
    V = 1.0
    nav = np.empty(i1 - i0 + 1); nav[0] = 1.0
    turn = np.zeros_like(nav)
    for k, t in enumerate(range(i0 + 1, i1 + 1), start=1):
        h = h * (1.0 + ON[t])                      # 隔夜：t−1 收盘 → t 开盘
        V = h.sum()
        tgt = Wt[t - 1]                            # 开盘按 t−1 收盘时定的目标调仓
        w = h / V
        dw = np.abs(tgt - w)
        if dw.max() > band or ((tgt > 1e-9) != (w > 1e-9)).any():
            V *= 1.0 - (dw * CO[t]).sum()
            h = tgt * V
            turn[k] = dw.sum() / 2
        h = h * (1.0 + ID[t])                      # 日内：t 开盘 → t 收盘
        nav[k] = h.sum()
    out = pd.Series(nav, index=idx[i0:i1 + 1])
    return out, pd.Series(turn, index=out.index)


def stats(nav, turn=None):
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    vol = r.std() * np.sqrt(252)
    sh = (r.mean() * 252) / vol if vol > 0 else np.nan
    out = dict(cagr=cagr, mdd=mdd, vol=vol, sharpe=sh, calmar=cagr / -mdd if mdd < 0 else np.nan,
               mult=nav.iloc[-1] / nav.iloc[0], mdd_date=dd.idxmin())
    if turn is not None:
        out["turn_yr"] = turn.sum() / yrs
    return out


def simulate_fast(mkt, W, start, end, band=0.05):
    """与 simulate 同口径，内层改成纯标量循环（用到的列通常只有 2~4 个，numpy 逐行开销反而大）。
    网格筛选用；最终报告的数字一律用 simulate 在各自窗口重跑。"""
    idx = mkt.idx
    i0, i1 = idx.searchsorted(pd.Timestamp(start)), idx.searchsorted(pd.Timestamp(end), side="right") - 1
    bil = mkt.col_i["BIL"]
    used = np.where((W[max(0, i0 - 1):i1 + 1] > 0).any(axis=0))[0]
    cols = [c for c in used if c != bil] + [bil]
    n = len(cols); b = n - 1
    Wt = W[:, cols].copy(); Wt[:, b] = 0.0; Wt[:, b] = 1.0 - Wt.sum(axis=1)
    Wt = Wt.tolist(); ON = mkt._on[:, cols].tolist(); ID = mkt._id[:, cols].tolist(); CO = mkt._cost[:, cols].tolist()
    h = [0.0] * n; h[b] = 1.0
    nav = [1.0]; turn = [0.0]
    rng = range(n)
    for t in range(i0 + 1, i1 + 1):
        on, tg = ON[t], Wt[t - 1]
        h = [h[j] * (1.0 + on[j]) for j in rng]
        V = sum(h)
        w = [x / V for x in h]
        dw = [abs(tg[j] - w[j]) for j in rng]
        tr = 0.0
        if max(dw) > band or any((tg[j] > 1e-9) != (w[j] > 1e-9) for j in rng):
            co = CO[t]
            V *= 1.0 - sum(dw[j] * co[j] for j in rng)
            h = [tg[j] * V for j in rng]
            tr = sum(dw) / 2
        d = ID[t]
        h = [h[j] * (1.0 + d[j]) for j in rng]
        nav.append(sum(h)); turn.append(tr)
    return pd.Series(nav, index=idx[i0:i1 + 1]), pd.Series(turn, index=idx[i0:i1 + 1])


def window_stats(nav, turn, a, b):
    """从一条连续净值里截出窗口 [a, b] 的统计（窗口起点按当日收盘净值归一）"""
    x = nav[a:b]; x = x / x.iloc[0]
    return stats(x, turn[a:b].iloc[1:])
