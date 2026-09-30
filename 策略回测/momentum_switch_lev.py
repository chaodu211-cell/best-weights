# -*- coding: utf-8 -*-
"""动量二选一的候选规则放到 3 倍（TQQQ/SOXL 纯切换）和完整七标的策略里再看一遍。

    python3 策略回测/momentum_switch_lev.py
候选是看过 momentum_switch.py 的 1 倍结果之后定的（见 CANDS 注释），所以这里不是独立检验。
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bt, strategy as ST, grid_semi as GS
from momentum_switch import signals, choose, weights

CANDS = {
    "现行 ret63·月·无缓冲·跌则现金": ("ret:63", "M", 0.0, 1),
    "ret63·月·无缓冲·永远持有": ("ret:63", "M", 0.0, 0),
    "ret63·月·2%缓冲·永远持有": ("ret:63", "M", 0.02, 0),
    "ret126·月·2%缓冲·永远持有": ("ret:126", "M", 0.02, 0),
    "skip63·月·2%缓冲·永远持有": ("skip:63", "M", 0.02, 0),       # 1 倍全程第一
    "skip126·月·无缓冲·永远持有": ("skip:126", "M", 0.0, 0),
    "sharpe126·月·无缓冲·永远持有": ("sharpe:126", "M", 0.0, 0),
    "combo·月·2%缓冲·永远持有": ("combo", "M", 0.02, 0),
}
PER3 = {"B 2010-04~2016-09": ("2010-04-01", "2016-09-22"), "C 2016-09~2023-09": ("2016-09-22", "2023-09-22"),
        "D 2023-09~2026-09": ("2023-09-22", "2026-09-22")}
FULL3 = ("2010-04-01", "2026-09-22")

if __name__ == "__main__":
    GS.init(); M, P = GS.G["M"], GS.G["P"]
    S = signals(M.C["QQQ"], M.C["SOXX"])
    ch = {k: choose(M.idx, S[v[0]], v[1], v[2], v[3]) for k, v in CANDS.items()}

    print("=== 3 倍纯切换（TQQQ / SOXL，满仓，不带模型信号）年化")
    rows = []
    for nm, W in [("持有 TQQQ", pd.DataFrame({"TQQQ": 1.0}, index=M.idx)), ("持有 SOXL", pd.DataFrame({"SOXL": 1.0}, index=M.idx)),
                  ("TQQQ/SOXL 各半（每日再平衡）", pd.DataFrame({"TQQQ": .5, "SOXL": .5}, index=M.idx))] + \
                 [(k, weights(M, v, ("TQQQ", "SOXL"))) for k, v in ch.items()]:
        s = bt.stats(bt.simulate(M, W, *FULL3)[0])
        print(f"  {nm:28s} 全程 {s['cagr']:6.1%} (回撤 {s['mdd']:6.1%})  " +
              "  ".join(f"{p[:1]} {bt.stats(bt.simulate(M, W, a, b)[0])['cagr']:6.1%}" for p, (a, b) in PER3.items()))

    print("\n=== 放回完整七标的策略（三年窗挑出的那套，只换选品规则）")
    c = json.load(open(os.path.join(HERE, "_picked_semi.json")))["3y"]["cfg"]
    WINS = {"三年": GS.WIN["3y"], "2023起": ("2022-12-30", "2026-09-22"), "十年": GS.WIN["10y"],
            "之前7年": GS.WIN["pre3y"], "07-16": GS.WIN["oos"]}
    for k, v in ch.items():
        p = GS.expand(dict(c)); p.update(universe="choice", choice=v, choice_u=["QQQ", "SOXX"])
        W, _ = ST.build(P, p)
        out = [bt.stats(bt.simulate(M, W, *w)[0]) for w in WINS.values()]
        print(f"  {k:28s} " + "  ".join(f"{n} {o['cagr']:6.1%}/{o['mdd']:6.1%}" for n, o in zip(WINS, out)))
    W, _ = ST.build(P, GS.expand(dict(c)))
    out = [bt.stats(bt.simulate(M, W, *w)[0]) for w in WINS.values()]
    print(f"  {'（对照）原实现 rot/1/63':28s} " + "  ".join(f"{n} {o['cagr']:6.1%}/{o['mdd']:6.1%}" for n, o in zip(WINS, out)))
