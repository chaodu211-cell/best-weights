# -*- coding: utf-8 -*-
"""七只标的网格，选品规则事先定死为"剔除最近一月的 126 日动量二选一"（2026-09-28）。

    python3 策略回测/grid_skip.py [抽样数，默认 40000]      → _grid_skip.csv
    python3 策略回测/pick.py grid_skip                       → _picked_skip.json
    python3 策略回测/robust_semi.py grid_skip                → _robust_skip.json

选品：每月第一个交易日比较 QQQ 与 SOXX 从 t−126 到 t−21 的涨幅，持有高的那只的最高倍数杠杆版；
两只都跌也不拿现金（避险交给红点/黄点）。这条规则来自 momentum_switch.py 的 324 种变体检验，
定下之后才跑本网格；其余参数空间、抽样数、种子、挑选规则与 grid_semi.py 完全相同，只是去掉了 universe 这一维。
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bt, strategy as ST, signals as SG
import grid_semi as _S
from grid_semi import WIN, SPAN, ALLOW, normalize, sample as _sample
from momentum_switch import signals as msig, choose

OUT = os.path.join(HERE, "_grid_skip.csv")
RULE = ("skip:126", "M", 0.0, 0)
SPACE = dict(_S.SPACE, universe=["skip126"])
G = {}


def init():
    M = bt.Market("2005-01-01", "2026-09-22")
    G["M"], G["P"] = M, ST.Prep(M, SG.load("splice"), allow=ALLOW)
    G["choice"] = choose(M.idx, msig(M.C["QQQ"], M.C["SOXX"])[RULE[0]], *RULE[1:])


def expand(c):
    p = dict(c)
    p.pop("universe", None)
    if "choice" not in G:
        init()
    p.update(universe="choice", choice=G["choice"], choice_u=["QQQ", "SOXX"])
    return p


def evaluate(c):
    M, P = G["M"], G["P"]
    W, st = ST.build(P, expand(c))
    row = dict(c)
    nav, tu = bt.simulate_fast(M, W, SPAN[0], SPAN[1])
    for w, (a, b) in WIN.items():
        s = bt.window_stats(nav, tu, a, b)
        row.update({f"{w}_cagr": s["cagr"], f"{w}_mdd": s["mdd"]})
        if w in ("10y", "3y"):
            row[f"{w}_turn"] = s["turn_yr"]
    return row


def sample(n, seed=20260928):
    # 与 grid_semi 同一种子；universe 固定为 skip126 后去重
    cs, _ = _sample(n, seed)
    seen, out = set(), []
    for c in cs:
        c = dict(c, universe="skip126")
        k = tuple(str(c[x]) for x in SPACE)
        if k not in seen:
            seen.add(k); out.append(c)
    tot = 1
    for v in SPACE.values():
        tot *= len(v)
    return out, tot


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    cs, tot = sample(n)
    print(f"参数空间 {tot:,} 组，抽样后去重 {len(cs):,} 组", flush=True)
    t0 = time.time()
    with Pool(8, initializer=init) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(evaluate, cs, chunksize=50)):
            rows.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1:,}  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"完成 {len(rows):,} 组，{time.time()-t0:.0f}s → {os.path.basename(OUT)}", flush=True)
