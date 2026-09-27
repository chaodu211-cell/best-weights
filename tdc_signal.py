# -*- coding: utf-8 -*-
"""趋势下跌预警的计算核心（TDC 与红色预警），供脚本与网页管线共用。

红色预警（网页上是黄点）：
    纳指（QQQ）距 52 周高点回落在 6%~13% 之间，且 TDC > 70 连续 3 个交易日，且探测器处于武装状态。
    报警后立即解除武装——同一轮下跌只报一次。TDC 跌回 60 以下才重新武装。
    作废：武装中 TDC 已连 3 日过 70、纳指却已跌破 13% → 本轮不报，同样解除武装。

TDC = 三条腿等权，腿内三因子等权，每个因子先做 1260 日滚动分位（0~100）：
    内部损伤  成分股站上MA200比例 · 创52周新低比例 · 已跌入熊市的成分股比例
    波动中枢  20日已实现波动 · 20日内跌1%以上的天数 · VIX的20日最低值
    趋势结构  距MA200偏离 · 120日动量 · MA50−MA200
因子筛选与全部反面证据见「趋势下跌确认.md」。
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent

WIN, MINP = 1260, 504          # 分位窗口：5 年，至少 2 年才出数
TH, PERSIST = 70.0, 3          # 报警门槛与连续天数
GATE_LO, GATE_HI = -0.06, -0.13   # 纳指距 52 周高点的回落区间：(−13%, −6%]
#   2026-09-23 闸门从"标普 ≤ −6%"改成"纳指 −6%~−13% + 作废"。标普口径下 14 次预警里有
#   5 次是摸底报警（此后再跌 <3%），其中 2010-07-01 / 2011-08-08 / 2015-08-24 三次报警时
#   纳指已跌 14%~16%——急跌已经走完，TDC 才迟迟确认。实测（长历史，2006 起）：
#       标普 ≤−6%             14次  摸底 5  大跌前(此后再跌≥10%) 5
#       纳指 −6%~−12% +作废   10次  摸底 2  大跌前 5
#       纳指 −6%~−13% +作废   11次  摸底 2  大跌前 5   ← 现行
#       纳指 −6%~−14% +作废   12次  摸底 3  大跌前 5   （放回 2015-08-24）
#   五次大跌前的预警（2007-11-09 / 2018-10-15 / 2020-03-05 / 2022-01-20 / 2025-03-12）
#   时点逐日不变。上沿取 13 不取 12 是为了留余量：好信号里最深的是 2025-03-12 的
#   −11.6%（次日即 −13.3%），坏信号里最浅的是 2010-06-08 的 −12.6%，两边只隔 1pp；
#   −12% 只给 2025 留 0.4pp，−13% 给 1.4pp（2020-03-05 −10.8% 同理从 1.2pp 变 2.2pp），
#   代价是放回 2010-06-08（此后再跌 3.8%、18 日到底，不算摸底）。
#   作废必须以"TDC 已连 3 日过 70"为前提，不能只看跌幅：2020 纳指 02-27 就到了 −13.2%，
#   TDC 03-05 才确认——只看跌幅会把 2020 整轮作废。没有作废这一条，跌穿上沿后
#   探测器仍武装，反弹回区间就在反弹日报警（2011-08-09、2015-08-26），等于把摸底报警挪后两天。
#   上沿能用，靠的是"确认来晚了"，不是"跌得深就是底"：按日看，TDC 已过 70、纳指
#   −12%~−16% 的日子此后中位还要再跌 12%~24%（2008、2022 撑起来的）。
#   浅位的摸底（2014-10、2018-03、2019-05）区间去不掉，它们和 2018-10 一样浅。
#   这是样本内结果：被去掉的三次正是看 2006-2016 那张图时发现的，14 个事件，别把 1pp 的缝当规律。
#
#   —— 以下是改纳指之前的记录（当时闸门按标普算）——
#   2026-09-22 从 −3% 改到 −6%。理由不是"噪音变少"，而是**一个真信号的时点都没动**：
#   2018-10-15 / 2020-03-05 / 2022-01-20 / 2025-03-12 / 2026-03-20 逐日相同，
#   只有 2007 年那两次挪了几天。真下跌在 TDC 过 70 时本来就已经跌了 6.6%~11.6%，
#   而假信号（2015-01、2019-05、2019-08）都卡在 −5% 附近，所以这道线是免费的。
#   实测（长历史，2006 起，≥8% 下跌段为"真"）：
#       闸门 3%  16次 真12 噪4   ← 旧口径
#       闸门 5%  15次 真12 噪3
#       闸门 6%  14次 真12 噪2   ← 现行
#       闸门 7%  13次 真12 噪1   代价：2018 从 −6.6% 推迟到 −10.3%
#       闸门 8%  12次 真12 噪0   代价同上，且 2026 推到只剩 −2.7%
#   7%/8% 的"零噪音"是在 16 个样本上挑出来的，别当真。
REARM = 60.0                   # TDC 跌回这个值以下，探测器重新武装
LEGS = {
    "内部损伤": ["F1_pct_ma200", "F6_nl", "F9_pct_bear"],
    "波动中枢": ["C5_rv20", "C8_bigdown", "C9_vixfloor"],
    "趋势结构": ["A1_ma200dev", "B3_roc120", "A3_ma50_200"],
}


def _read(raw, sym):
    p = Path(raw) / f"{sym}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, header=None, names=["date", "close", "rawclose", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()


def load(raw=None):
    raw = Path(raw) if raw else HERE / "raw"
    px = _read(raw, "SPY")
    v = pd.read_csv(raw / "_vix.csv", header=None, names=["date", "vix"])
    v["date"] = pd.to_datetime(v["date"])
    vix = v.set_index("date")["vix"].sort_index().reindex(px.index).ffill()
    syms = sorted(json.load(open(HERE / "sectors.json")).keys())
    const = pd.DataFrame({s: c for s in syms if (c := _read(raw, s)) is not None}).reindex(px.index)
    ndx = _read(raw, "QQQ")
    if ndx is None:
        raise FileNotFoundError(f"缺 {raw / 'QQQ.csv'}：黄点闸门按纳指回撤算")
    return px, vix, const, ndx.reindex(px.index)


def factors(px, vix, const):
    F = pd.DataFrame(index=px.index)
    r = px.pct_change()
    n = const.notna().sum(axis=1).replace(0, np.nan)
    F["F1_pct_ma200"] = -(const > const.rolling(200).mean()).sum(axis=1) / n * 100
    F["F6_nl"] = (const <= const.rolling(252).min()).sum(axis=1) / n * 100
    F["F9_pct_bear"] = (const / const.rolling(252).max() - 1 <= -0.20).sum(axis=1) / n * 100
    F["C5_rv20"] = r.rolling(20).std() * np.sqrt(252) * 100
    F["C8_bigdown"] = (r <= -0.01).rolling(20).sum()
    F["C9_vixfloor"] = vix.rolling(20).min()
    F["A1_ma200dev"] = -(px / px.rolling(200).mean() - 1)
    F["B3_roc120"] = -(px / px.shift(120) - 1)
    F["A3_ma50_200"] = -(px.rolling(50).mean() / px.rolling(200).mean() - 1)
    return F


def compute(raw=None):
    """→ dict(tdc, legs, dd=纳指回撤（闸门用）, dd_spx, red=报警日, void=作废日, armed=末日是否武装)"""
    px, vix, const, ndx = load(raw)
    F = factors(px, vix, const)
    P = F.rolling(WIN, min_periods=MINP).rank(pct=True) * 100
    legs = pd.DataFrame({k: P[v].mean(axis=1) for k, v in LEGS.items()})
    S = legs.mean(axis=1)
    dd = ndx / ndx.rolling(252, min_periods=60).max() - 1
    dd_spx = px / px.rolling(252, min_periods=60).max() - 1
    hot = ((S > TH).rolling(PERSIST).sum() == PERSIST).fillna(False)

    # 同一轮下跌只报一次：报警即解除武装，等 TDC 跌回 REARM 以下才重新武装。
    # 确认时纳指已跌破 GATE_HI → 这一轮作废（同样解除武装），不等反弹回区间再报。
    armed, fired, voided = True, [], []
    for t, h, v, x in zip(px.index, hot.values, S.values, dd.values):
        if not np.isnan(v) and v < REARM:
            armed = True
        if armed and h and np.isfinite(x):
            if GATE_HI < x <= GATE_LO:
                armed = False
                fired.append(t)
            elif x <= GATE_HI:
                armed = False
                voided.append(t)
    red = pd.Series(False, index=px.index)
    red.loc[fired] = True
    void = pd.Series(False, index=px.index)
    void.loc[voided] = True
    return dict(tdc=S, legs=legs, dd=dd, dd_spx=dd_spx, red=red, void=void, armed=armed,
                factors=F, pct=P)
