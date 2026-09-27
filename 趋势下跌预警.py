# -*- coding: utf-8 -*-
"""趋势下跌预警（网页上的黄点）。计算核心在 tdc_signal.py，本文件只负责打印。

    python3 趋势下跌预警.py            # 今天什么状态
    python3 趋势下跌预警.py --history   # 历次预警 + 写出 _预警历史.csv

和红点的分工：
  红点 = 市场已经不健康（顶部拥挤度），在高位报警。
  黄点 = 下跌已经开始、而且像是趋势性的，在纳指距 52 周高点回落 6%~13% 时报警。

规则（同一轮下跌只报一次）：
  纳指距 52 周高点回落在 6%~13% 之间，且 TDC > 70 连续 3 个交易日，且探测器处于武装状态；
  报警后立即解除武装，直到 TDC 跌回 60 以下才重新武装。
  TDC 确认时纳指已跌破 13% → 本轮作废（确认来晚了，此时报警多半在底部），同样解除武装。

代价与反面证据见「趋势下跌确认.md」，**先读第四节和第十节再用**。
"""
import sys
import pandas as pd
import tdc_signal as T


def main():
    r = T.compute()
    S, legs, dd, red, void = r["tdc"], r["legs"], r["dd"], r["red"], r["void"]
    P, F = r["pct"], r["factors"]
    d = S.dropna().index[-1]
    armed = r["armed"]
    fires, voids = red[red[:d]].index, void[void[:d]].index
    last = fires[-1] if len(fires) else None
    last_void = voids[-1] if len(voids) else None
    lo, hi = -T.GATE_LO, -T.GATE_HI

    print(f"\n{d:%Y-%m-%d}   纳指距 52 周高点 {dd[d]:+.1%}（标普 {r['dd_spx'][d]:+.1%}）"
          f"   TDC {S[d]:.0f}（门槛 {T.TH:.0f}）")
    if red[d]:
        print("\n   🟡 今天触发趋势下跌预警")
    elif void[d]:
        print(f"\n   · 今天本该报警，但纳指已跌破 {hi:.0%}——确认来晚了，本轮作废")
    else:
        x = dd[d]
        where = (f"纳指回落不到 {lo:.0%}，不报警" if x > T.GATE_LO else
                 f"纳指已跌破 {hi:.0%}，此时确认即作废" if x <= T.GATE_HI else
                 f"纳指在 {lo:.0%}~{hi:.0%} 区间内；再要 TDC>{T.TH:.0f} 连续 {T.PERSIST} 日即报警")
        print(f"\n   · 无预警。{where}")
        print(f"     探测器：{'已武装' if armed else '未武装（等 TDC 跌回 %.0f 以下）' % T.REARM}"
              + (f"　上次预警 {last:%Y-%m-%d}" if last is not None else "")
              + (f"　上次作废 {last_void:%Y-%m-%d}" if last_void is not None else ""))
    print("\n   三条腿：" + "   ".join(f"{k} {legs[k][d]:.0f}" for k in T.LEGS))

    raw_fmt = {"F1_pct_ma200": lambda v: f"站上MA200 {-v:.0f}%", "F6_nl": lambda v: f"创新低 {v:.1f}%",
               "F9_pct_bear": lambda v: f"熊市股 {v:.0f}%", "C5_rv20": lambda v: f"年化波动 {v:.1f}",
               "C8_bigdown": lambda v: f"{v:.0f} 天", "C9_vixfloor": lambda v: f"VIX底 {v:.1f}",
               "A1_ma200dev": lambda v: f"距MA200 {-v:+.1%}", "B3_roc120": lambda v: f"120日 {-v:+.1%}",
               "A3_ma50_200": lambda v: f"MA50-200 {-v:+.1%}"}
    print("\n   分项（括号内为原始值）:")
    for leg, fs in T.LEGS.items():
        for f in fs:
            print(f"     {leg}  {f:14s} {P[f][d]:5.1f}   ({raw_fmt[f](F[f][d])})")

    tail = pd.DataFrame({"TDC": S, "纳指回撤%": dd * 100,
                         "预警": red.map({True: "🟡", False: "·"}).where(~void, "作废")}).dropna().tail(8)
    print("\n" + tail.round(1).to_string())

    if "--history" in sys.argv:
        print("\n=== 历次预警 ===")
        print("  ⚠ 2019-2021 那几次的分位窗口不足 5 年（raw/ 只有十年数据，分位要 504 天预热），"
              "\n    读数偏高、不能当实盘证据。长历史见「趋势下跌确认.md」第十节。")
        px = None
        for t in (red | void)[red | void].index:
            tag = "🟡" if red[t] else "✕ 作废"
            print(f"  {tag} {t:%Y-%m-%d}   TDC {S[t]:.0f}   纳指距 52 周高点 {dd[t]:+.1%}")
        out = pd.DataFrame({"dd": dd, "TDC": S, **{k: legs[k] for k in T.LEGS},
                            "red": red.astype(int), "void": void.astype(int)}).dropna()
        out.round(3).to_csv(T.HERE / "_预警历史.csv")
        print(f"\n已写出 _预警历史.csv（{len(out)} 行，{out.index[0]:%Y-%m-%d} 起）")


if __name__ == "__main__":
    main()
