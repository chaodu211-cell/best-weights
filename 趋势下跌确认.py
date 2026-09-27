# -*- coding: utf-8 -*-
"""趋势下跌确认分（TDC）——用 raw/ 里已有的数据算，不依赖网络。

和 alt_engine.py 的红点是两件事：
  · 红点  = 左侧。问"顶部有没有形成"，看的是拥挤度 / 杠杆 / VIX，在高点附近报警。
  · TDC   = 右侧。问"这次回撤会不会转成一段趋势性下跌"，看的是市场内部已经造成的
            损伤、波动中枢的位置、以及价格结构，在回撤 3%~10% 的窗口里才有意义。

三条腿等权，每条腿内部三个因子等权，全部先做 1260 日滚动百分位（min 504）：

  内部损伤  F1 成分股站上MA200比例(反号) · F6 创52周新低比例 · F9 已跌入熊市的成分股比例
  波动中枢  C5 20日已实现波动 · C8 20日内跌1%以上的天数 · C9 VIX的20日最低值
  趋势结构  A1 指数距MA200偏离(反号) · B3 120日动量(反号) · A3 MA50/MA200之差(反号)

为什么是这九个：见「趋势下跌确认.md」。三条腿的分工是实测出来的——回撤 0~3% 时
内部损伤最灵，3~7% 时波动中枢接手，>7% 之后只有趋势结构还有判别力。

用法:
    python3 趋势下跌确认.py              # 打印当前分数与分项
    python3 趋势下跌确认.py --history     # 另存 _tdc_history.csv
"""
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

WIN, MINP = 1260, 504          # 百分位窗口：5 年，至少 2 年才出数
LEGS = {
    "内部损伤": ["F1_pct_ma200", "F6_nl", "F9_pct_bear"],
    "波动中枢": ["C5_rv20", "C8_bigdown", "C9_vixfloor"],
    "趋势结构": ["A1_ma200dev", "B3_roc120", "A3_ma50_200"],
}
LEG_W = (1.0, 1.0, 1.0)
GATE = (-0.10, -0.03)          # 闸门：只在指数回撤落在这个区间时承认信号（只管入场）
TH, PERSIST = 70.0, 3          # 门槛与连续天数
EXIT = 65.0                    # 粘滞：进场后只看分数，跌破这个值才解除（闸门不再参与）


def _read(sym):
    p = RAW / f"{sym}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, header=None, names=["date", "close", "rawclose", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()


def load():
    spy = _read("SPY")
    vix = pd.read_csv(RAW / "_vix.csv", header=None, names=["date", "vix"])
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.set_index("date")["vix"].sort_index()
    syms = sorted(json.load(open(HERE / "sectors.json")).keys())
    cols = {s: _read(s) for s in syms}
    const = pd.DataFrame({k: v for k, v in cols.items() if v is not None})
    return spy, vix.reindex(spy.index).ffill(), const.reindex(spy.index)


def factors(px, vix, const):
    F = pd.DataFrame(index=px.index)
    r = px.pct_change()
    n = const.notna().sum(axis=1).replace(0, np.nan)

    a200 = (const > const.rolling(200).mean()).sum(axis=1) / n * 100
    F["F1_pct_ma200"] = -a200
    F["F6_nl"] = (const <= const.rolling(252).min()).sum(axis=1) / n * 100
    F["F9_pct_bear"] = (const / const.rolling(252).max() - 1 <= -0.20).sum(axis=1) / n * 100

    F["C5_rv20"] = r.rolling(20).std() * np.sqrt(252) * 100
    F["C8_bigdown"] = (r <= -0.01).rolling(20).sum()
    F["C9_vixfloor"] = vix.rolling(20).min()

    F["A1_ma200dev"] = -(px / px.rolling(200).mean() - 1)
    F["B3_roc120"] = -(px / px.shift(120) - 1)
    F["A3_ma50_200"] = -(px.rolling(50).mean() / px.rolling(200).mean() - 1)
    return F


def score(F):
    P = F.rolling(WIN, min_periods=MINP).rank(pct=True) * 100
    legs = pd.DataFrame({k: P[v].mean(axis=1) for k, v in LEGS.items()})
    w = np.array(LEG_W) / sum(LEG_W)
    S = sum(wi * legs[k] for wi, k in zip(w, LEGS))
    return S.rename("TDC"), legs, P


def sticky(fire, S):
    """进场：fire（闸门内连续 PERSIST 日 > TH）；出场：分数跌破 EXIT。

    没有这条粘滞，解除权在闸门手里——而闸门的下沿恰恰在跌势加速时关上，
    等于在最坏的时点把人送回市场（实测 49 次解除里有 23 次是这么来的）。
    """
    out = []
    on = False
    for f, v in zip(fire.values, S.values):
        if not on:
            on = bool(f)
        elif not np.isnan(v) and v < EXIT:
            on = False
        out.append(on)
    return pd.Series(out, index=fire.index)


def main():
    px, vix, const = load()
    F = factors(px, vix, const)
    S, legs, P = score(F)
    dd = px / px.cummax() - 1
    gate = (dd >= GATE[0]) & (dd <= GATE[1])
    fire = ((S > TH) & gate).rolling(PERSIST).sum() == PERSIST

    d = S.dropna().index[-1]
    print(f"截至 {d:%Y-%m-%d}   标普距高点 {dd[d]:+.1%}   成分股覆盖 {int(const.loc[d].notna().sum())}")
    print(f"\n  TDC 合成分 {S[d]:5.1f}   （门槛 {TH:.0f}，闸门 回撤{-GATE[1]:.0%}~{-GATE[0]:.0%}）")
    print(f"  " + "   ".join(f"{k} {legs[k][d]:4.1f}" for k in LEGS))
    hold = sticky(fire.fillna(False), S)
    print(f"  闸门{'开' if gate[d] else '关'}  当日触发{'是' if fire[d] else '否'}"
          f"（需连续 {PERSIST} 日）  粘滞状态{'**信号中**' if hold[d] else '空'}"
          f"（进场后分数跌破 {EXIT:.0f} 才解除）")
    if hold[d]:
        k = 0
        for v in hold[:d][::-1]:
            if not v:
                break
            k += 1
        print(f"  本段已持续 {k} 个交易日，起于 {hold[:d].index[-k]:%Y-%m-%d}")
    print("\n  分项（括号内为原始值）:")
    raw_fmt = {"F1_pct_ma200": lambda v: f"站上MA200 {-v:.0f}%", "F6_nl": lambda v: f"创新低 {v:.1f}%",
               "F9_pct_bear": lambda v: f"熊市股 {v:.0f}%", "C5_rv20": lambda v: f"年化波动 {v:.1f}",
               "C8_bigdown": lambda v: f"{v:.0f} 天", "C9_vixfloor": lambda v: f"VIX底 {v:.1f}",
               "A1_ma200dev": lambda v: f"距MA200 {-v:+.1%}", "B3_roc120": lambda v: f"120日 {-v:+.1%}",
               "A3_ma50_200": lambda v: f"MA50-200 {-v:+.1%}"}
    for leg, fs in LEGS.items():
        for f in fs:
            print(f"    {leg}  {f:14s} {P[f][d]:5.1f}   ({raw_fmt[f](F[f][d])})")
    print("\n  近 10 个交易日:")
    tail = pd.DataFrame({"TDC": S, **{k: legs[k] for k in LEGS}, "回撤%": dd * 100}).dropna().tail(10)
    print(tail.round(1).to_string())

    if "--history" in sys.argv:
        out = pd.DataFrame({"spx_proxy": px, "dd": dd, "TDC": S, **{k: legs[k] for k in LEGS},
                            "gate": gate.astype(int), "fire": fire.fillna(False).astype(int),
                            "hold": sticky(fire.fillna(False), S).astype(int)}).dropna()
        out.to_csv(HERE / "_tdc_history.csv")
        print(f"\n已写出 _tdc_history.csv（{len(out)} 行，{out.index[0]:%Y-%m-%d} 起）")


if __name__ == "__main__":
    main()
