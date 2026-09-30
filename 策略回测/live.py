# -*- coding: utf-8 -*-
"""把"新·十年版"策略（纳指 / 半导体动量二选一 + 红黄蓝点）落到每天的目标持仓，供 make_alt_page.py 调用。

    current(d, raw_dir) → {"as_of", "holdings": [{"sym", "weight"}], "cash", "state", "pick", ...}

与回测完全同一份规则代码（strategy.build + momentum_switch.choose），只把数据换成生产的：
  · 信号：页面当次算出的 data_alt.json 结构（红点 hot、黄点 tdc、蓝点 cold / cold_soft、TDC 读数），2017-10-18 起；
    每日信号快照（_信号快照.csv，2026-09-30 起）覆盖的日子改用快照
  · 价格：raw/ 里的 QQQ、SOXX（选品、波动率、熊市闸门）与 TQQQ、SOXL、SQQQ（是否可交易）
t 日收盘后给出的持仓在 t+1 开盘执行，和回测口径一致。

策略参数在 STRATEGY 里写死（来自 _picked_skip.json 的十年窗选定，2026-09-28），不读结果文件，
免得重跑网格时页面上的规则悄悄变了。
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strategy as ST
from momentum_switch import signals as msig, choose

NAME = "纳指 / 半导体动量二选一（新·十年版）"
STRATEGY = dict(lev_max=2.5, vol_tgt=0.3, red_hold=30, red_mult=0.0, use_ad=False, rearm=60.0,
                blue_hold=40, blue_k=10, blue_lev=2.0, blue_release="perm", soft_mult=0.0, blue_asset="SOXX",
                tdc_gate=None, bear_mult=0.5, short_yellow=0.0, short_red=1 / 3)
RULE = ("skip:126", "M", 0.0, 0)          # 每月第一个交易日比 t−126 → t−21 的涨幅，永远持有其一
SYMS = {"QQQ": ("QQQ", 1.0), "SOXX": ("SOXX", 1.0), "TQQQ": ("QQQ", 3.0), "SOXL": ("SOXX", 3.0), "SQQQ": ("QQQ", -3.0)}


def _read(raw, sym):
    df = pd.read_csv(os.path.join(raw, f"{sym}.csv"), header=None, names=["date", "adj", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date", keep="first").set_index("date")["adj"].sort_index()


class _Market:
    """strategy.Prep 需要的最小接口（回测里是 bt.Market）"""

    def __init__(self, raw, end):
        C = pd.DataFrame({s: _read(raw, s) for s in SYMS})
        C = C[C.index <= pd.Timestamp(end)]
        C = C[C["QQQ"].notna()]
        self.idx = C.index
        self.C = C
        self.syms = [s for s in SYMS if s not in ("QQQ", "SOXX")] + ["QQQ", "SOXX"]
        self.cols = list(C.columns)
        self.col_i = {s: i for i, s in enumerate(self.cols)}
        self.under = {s: u for s, (u, _) in SYMS.items()}
        self.ux = C[["QQQ", "SOXX"]]
        self.tradable = C.notna() & C.shift(20).notna()
        self.lev = pd.DataFrame({s: L for s, (_, L) in SYMS.items()}, index=self.idx)


def _signals(d, snap=None):
    """snap：每日信号快照（make_alt_page.snap_signals），它覆盖的日子一律以快照为准——
    快照记的是当时页面上显示的信号；页面上的历史每次都按今天的成分股名单重算，会被改写。"""
    s, f = d["series"], d["alerts"]["flags"]
    idx = pd.to_datetime(s["dates"])
    col = lambda v: pd.Series([np.nan if x is None else x for x in v], index=idx, dtype=float)
    S = pd.DataFrame({"red": col(f["hot"]).astype(bool), "red_ad": col(f["hot_ad"]).astype(bool),
                      "blue": col(f["cold"]).astype(bool), "blue_soft": col(f["cold_soft"]).astype(bool),
                      "yellow": col(f["tdc"]).astype(bool), "tdc": col(s["tdc"])})
    if snap is not None:
        snap = snap.reindex(S.index)
        for c in S.columns:
            m = snap[c].notna() if c in snap else None
            if m is not None and m.any():
                S.loc[m, c] = snap.loc[m, c].astype(float) if c == "tdc" else snap.loc[m, c].astype(bool)
    return S


def history(d, raw, snap=None):
    """整段目标权重（DataFrame，t 行 = t 日收盘后决定）与状态序列"""
    M = _Market(raw, d["as_of"])
    P = ST.Prep(M, _signals(d, snap))
    ch = choose(M.idx, msig(M.C["QQQ"], M.C["SOXX"])[RULE[0]], *RULE[1:])
    W, st = ST.build(P, dict(STRATEGY, universe="choice", choice=ch, choice_u=["QQQ", "SOXX"]))
    W = pd.DataFrame(W, index=M.idx, columns=M.cols)
    return W, pd.Series(st, index=M.idx).map(ST.STATE_NAMES), pd.Series(ch, index=M.idx)


def current(d, raw, snap=None):
    W, st, ch = history(d, raw, snap)
    t = W.index[-1]
    w = W.loc[t]
    hold = [{"sym": s, "weight": round(float(w[s]), 4)} for s in ("SOXL", "TQQQ", "SQQQ", "QQQ", "SOXX") if w[s] > 0.0005]
    return {"name": NAME, "as_of": f"{t:%Y-%m-%d}", "holdings": hold,
            "cash": round(max(0.0, 1.0 - sum(h["weight"] for h in hold)), 4),
            "state": st.iloc[-1], "pick": ["QQQ", "SOXX"][int(ch.iloc[-1])] if ch.iloc[-1] >= 0 else None}
