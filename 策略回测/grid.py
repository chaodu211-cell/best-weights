# -*- coding: utf-8 -*-
"""参数网格（随机抽样）：每组参数同时跑十年窗、三年窗、2007-2016 样本外，结果存 _grid.csv。

    python3 策略回测/grid.py [抽样数，默认 30000]

参数空间在 SPACE 里一次定死；挑选规则见 pick.py，不在这里挑。
"""
import itertools, os, sys, time, warnings, random
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bt, strategy as ST, signals as SG

WIN = {"10y": ("2016-09-22", "2026-09-22"), "3y": ("2023-09-22", "2026-09-22"),
       "pre3y": ("2016-09-22", "2023-09-22"), "oos": ("2007-01-03", "2016-09-22"),
       "h1": ("2016-09-22", "2021-09-22"), "h2": ("2021-09-22", "2026-09-22")}

SPAN = ("2007-01-03", "2026-09-22")
ROT = [f"rot/{k}/{m}" for k in (1, 2, 3) for m in (63, 126, 252)]
SPACE = dict(
    universe=["fixed:QQQ", "fixed:SPY", "fixed:SOXX", "fixed:XLK", "fixed:IWM"] + ROT,
    lev_max=[2.0, 2.5, 3.0],
    vol_tgt=[None, 0.30, 0.40, 0.50, 0.60],
    red_hold=[0, 20, 30, 60], red_mult=[0.0, 0.5], use_ad=[False, True],
    rearm=[50.0, 60.0],
    blue_hold=[20, 40, 60], blue_k=[1, 5, 10], blue_lev=[1.0, 2.0, 3.0],
    blue_release=["perm", "temp"], soft_mult=[0.0, 0.5],
    tdc_gate=[None, 60.0, 70.0], bear_mult=[1.0, 0.5, 0.0],
)
G = {}


def init():
    M = bt.Market("2005-01-01", "2026-09-22")
    G["M"], G["P"] = M, ST.Prep(M, SG.load("splice"))


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
    nav, tu = bt.simulate_fast(M, W, SPAN[0], SPAN[1])       # 一条连续净值，各窗口从中截取
    for w, (a, b) in WIN.items():
        s = bt.window_stats(nav, tu, a, b)
        row.update({f"{w}_cagr": s["cagr"], f"{w}_mdd": s["mdd"]})
        if w in ("10y", "3y"):
            row[f"{w}_turn"] = s["turn_yr"]
    return row


def sample(n, seed=20260928):
    keys = list(SPACE)
    full = list(itertools.product(*[SPACE[k] for k in keys]))
    rnd = random.Random(seed)
    pick = rnd.sample(range(len(full)), min(n, len(full)))
    out = []
    for i in pick:
        c = dict(zip(keys, full[i]))
        if c["red_hold"] == 0:          # 不用红点时，red_mult / use_ad 无意义，归一
            c["red_mult"], c["use_ad"] = 0.0, False
        if c["tdc_gate"] is None:
            pass
        out.append(c)
    # 去重
    seen, uniq = set(), []
    for c in out:
        k = tuple(sorted((a, str(b)) for a, b in c.items()))
        if k not in seen:
            seen.add(k); uniq.append(c)
    return uniq, len(full)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    cs, tot = sample(n)
    print(f"参数空间 {tot:,} 组，抽样 {len(cs):,} 组")
    t0 = time.time()
    with Pool(8, initializer=init) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(evaluate, cs, chunksize=50)):
            rows.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1:,}  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "_grid.csv"), index=False)
    print(f"完成 {len(df):,} 组，{time.time()-t0:.0f}s → _grid.csv")
