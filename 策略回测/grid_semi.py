# -*- coding: utf-8 -*-
"""纳指 + 半导体 七只标的的参数网格（2026-09-28 加）：SOXL、USD、SOXX、TQQQ、QLD、QQQ、SQQQ。

    python3 策略回测/grid_semi.py [抽样数，默认 40000]      → _grid_semi.csv

与 grid.py 同一套口径、同一批窗口，只换了参数空间和可交易品种；挑选仍用 pick.py（传 grid_semi）。
SOXS 不在清单里，所以做空只能用 SQQQ；2010-02 之前没有 SQQQ/TQQQ/SOXL，做空腿空着、多头用 QLD/USD。
"""
import itertools, os, sys, time, warnings, random
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bt, strategy as ST, signals as SG
from grid import WIN, SPAN

ALLOW = ("SOXL", "USD", "SOXX", "TQQQ", "QLD", "QQQ", "SQQQ")
OUT = os.path.join(HERE, "_grid_semi.csv")
SPACE = dict(
    universe=["fixed:QQQ", "fixed:SOXX", "rot/1/63", "rot/1/126", "rot/1/252", "mix:QQQ+SOXX"],
    lev_max=[2.0, 2.5, 3.0],
    vol_tgt=[None, 0.30, 0.40, 0.50, 0.60],
    red_hold=[0, 20, 30, 60], red_mult=[0.0, 0.5], use_ad=[False, True],
    rearm=[50.0, 60.0],
    blue_hold=[20, 40, 60], blue_k=[1, 5, 10], blue_lev=[1.0, 2.0, 3.0],
    blue_release=["perm", "temp"], soft_mult=[0.0, 0.5], blue_asset=["QQQ", "SOXX"],
    tdc_gate=[None, 60.0, 70.0], bear_mult=[1.0, 0.5, 0.0],
    short_yellow=[0.0, 1 / 3, 2 / 3], short_red=[0.0, 1 / 3],
)
G = {}


def init():
    M = bt.Market("2005-01-01", "2026-09-22")
    G["M"], G["P"] = M, ST.Prep(M, SG.load("splice"), allow=ALLOW)


def expand(c):
    p = dict(c)
    u = p.pop("universe")
    if u.startswith("rot/"):
        _, k, m = u.split("/")
        p.update(universe="rot", top=int(k), mom=int(m))
    else:
        p["universe"] = u
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


def normalize(c):
    if c["red_hold"] == 0:
        c["red_mult"], c["use_ad"], c["short_red"] = 0.0, False, 0.0
    if c["red_mult"] > 0:
        c["short_red"] = 0.0          # 只在红点清仓（乘到 0）时才做空
    return c


def sample(n, seed=20260928):
    keys = list(SPACE)
    sizes = [len(SPACE[k]) for k in keys]
    tot = int(np.prod(sizes))
    rnd = random.Random(seed)
    seen, out = set(), []
    while len(out) < n and len(seen) < tot:
        c = normalize({k: rnd.choice(SPACE[k]) for k in keys})
        key = tuple(str(c[k]) for k in keys)
        if key in seen:
            continue
        seen.add(key); out.append(c)
    return out, tot


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    cs, tot = sample(n)
    print(f"参数空间 {tot:,} 组，抽样 {len(cs):,} 组", flush=True)
    t0 = time.time()
    with Pool(8, initializer=init) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(evaluate, cs, chunksize=50)):
            rows.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1:,}  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"完成 {len(rows):,} 组，{time.time()-t0:.0f}s → {os.path.basename(OUT)}", flush=True)
