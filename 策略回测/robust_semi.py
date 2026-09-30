# -*- coding: utf-8 -*-
"""七只标的策略（_picked_semi.json）的精确重跑与稳健性检验，结果写 _robust_semi.json。

    python3 策略回测/robust_semi.py              # 七标的网格（_picked_semi.json → _robust_semi.json）
    python3 策略回测/robust_semi.py grid_skip    # 选品定死为剔除一月 126 日动量（→ _robust_skip.json）
"""
import importlib, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bt, strategy as ST
import robust
from robust import build, yearly

MOD = sys.argv[1] if len(sys.argv) > 1 else "grid_semi"
GS = importlib.import_module(MOD)
SUF = MOD.split("_", 1)[-1]
robust.GR = GS                     # build() 用对应网格的 expand（grid_skip 要把选品换成外部给定的逐日选择）

WINS = dict(GS.WIN, y23=("2022-12-30", "2026-09-22"))
NAMES = {"10y": "十年 2016-09~2026-09", "3y": "三年 2023-09~2026-09", "y23": "2023-01-01 起", "h1": "十年前半",
         "h2": "十年后半", "pre3y": "三年窗之前 7 年", "oos": "2007-01~2016-09 样本外"}


def dd_top(nav, st, n=3):
    dd = nav / nav.cummax() - 1; d = dd.copy(); out = []
    for _ in range(n):
        lo = d.idxmin()
        if d[lo] > -0.08: break
        pk = nav[:lo].idxmax(); rec = nav[lo:][nav[lo:] >= nav[pk]]
        end = rec.index[0] if len(rec) else nav.index[-1]
        out.append((str(pk.date()), str(lo.date()), round(dd[lo], 4), st.shift(1)[pk:lo].value_counts().to_dict()))
        d[pk:end] = 0
    return out


def main():
    GS.init()
    M, P = GS.G["M"], GS.G["P"]
    picked = json.load(open(os.path.join(HERE, f"_picked_{SUF}.json")))
    out = {}
    for w, b in picked.items():
        c = b["cfg"]
        W, st = build(P, c)
        stS = pd.Series(st, index=M.idx).map(ST.STATE_NAMES)
        print(f"\n=== 按{w}窗口挑出的七标的策略：{json.dumps(c, ensure_ascii=False)}")
        res = {}
        for k, (a, e) in WINS.items():
            nav, tu = bt.simulate(M, W, a, e); s = bt.stats(nav, tu)
            res[k] = dict(cagr=s["cagr"], mdd=s["mdd"], turn=s["turn_yr"], dd=dd_top(nav, stS),
                          yearly=yearly(nav).round(4).to_dict())
            print(f"  {NAMES[k]:18s} {s['cagr']:7.1%} / {s['mdd']:7.1%}   最大回撤段 {res[k]['dd'][:1]}")
        A = {"完整策略": {}, "去掉做空腿": dict(short_yellow=0.0, short_red=0.0), "去掉红点": dict(red_hold=0, short_red=0.0),
             "去掉黄点": dict(_no_yellow=True), "去掉蓝点": dict(_no_blue=True), "去掉熊市闸门": dict(bear_mult=1.0),
             "去掉全部模型信号": dict(red_hold=0, short_red=0.0, tdc_gate=None, bear_mult=1.0, _no_yellow=True, _no_blue=True)}
        abl = {}
        for nm, v in A.items():
            Wa, _ = build(P, {**c, **v})
            abl[nm] = {k: tuple(bt.stats(*bt.simulate(M, Wa, *WINS[k]))[x] for x in ("cagr", "mdd")) for k in ("10y", "3y", "oos")}
            print(f"  {nm:10s} " + "  ".join(f"{k} {abl[nm][k][0]:6.1%}/{abl[nm][k][1]:6.1%}" for k in ("10y", "3y", "oos")))
        a, e = WINS[w]
        C0 = M._cost.copy(); M._cost = C0 * 2
        s2 = bt.stats(*bt.simulate(M, W, a, e)); M._cost = C0
        Wl, _ = build(P, c, lag=1); sl = bt.stats(*bt.simulate(M, Wl, a, e))
        extra = {"成本加倍": (s2["cagr"], s2["mdd"]), "信号晚一天": (sl["cagr"], sl["mdd"])}
        print("  " + "  ".join(f"{k} {v[0]:.1%}/{v[1]:.1%}" for k, v in extra.items()))
        Wd = pd.DataFrame(W, index=M.idx, columns=M.cols)
        use = {}
        for k in ("10y", "3y", "oos"):
            x = Wd.loc[WINS[k][0]:WINS[k][1]]
            use[k] = {s: round(float((x[s] > 0.001).mean()), 3) for s in GS.ALLOW if (x[s] > 0.001).any()}
        print("  持有天数占比:", use)
        state = {k: stS.loc[WINS[k][0]:WINS[k][1]].value_counts(normalize=True).round(3).to_dict() for k in ("10y", "3y")}
        out[w] = dict(cfg=c, res=res, ablation=abl, extra=extra, use=use, state=state, nb=b)
    json.dump(out, open(os.path.join(HERE, f"_robust_{SUF}.json"), "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
