# -*- coding: utf-8 -*-
"""选定策略的稳健性检验：消融（逐个去掉模型信号）、成本加倍、信号晚一天、逐年收益。

    python3 策略回测/robust.py        # 读 _picked.json，结果写 _robust.json
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bt, strategy as ST, signals as SG, grid as GR


def ablations(c):
    """每一项只去掉一个模型组件，其余不动"""
    A = {"完整策略": {}}
    A["去掉红点"] = dict(red_hold=0)
    A["去掉黄点"] = dict(_no_yellow=True)
    A["去掉蓝点"] = dict(_no_blue=True)
    A["去掉 TDC 闸门"] = dict(tdc_gate=None)
    A["去掉熊市闸门"] = dict(bear_mult=1.0)
    A["去掉全部模型信号"] = dict(red_hold=0, tdc_gate=None, bear_mult=1.0, _no_yellow=True, _no_blue=True)
    return {k: {**c, **v} for k, v in A.items()}


def build(P, c, lag=0):
    """_no_yellow / _no_blue：把对应信号清零后再建仓位；lag：整张信号表再晚 lag 天"""
    c = dict(c)
    ny, nb = c.pop("_no_yellow", False), c.pop("_no_blue", False)
    saved = (P.yellow, P.blue, P.soft, P.red, P.red_ad, P.tdc)
    try:
        if ny: P.yellow = np.zeros_like(P.yellow)
        if nb: P.blue, P.soft = np.zeros_like(P.blue), np.zeros_like(P.soft)
        if lag:
            sh = lambda a, fill: np.r_[np.full(lag, fill, dtype=a.dtype), a[:-lag]]
            P.yellow, P.blue, P.soft = sh(P.yellow, False), sh(P.blue, False), sh(P.soft, False)
            P.red, P.red_ad, P.tdc = sh(P.red, False), sh(P.red_ad, False), sh(P.tdc, np.nan)
        return ST.build(P, GR.expand(c))
    finally:
        P.yellow, P.blue, P.soft, P.red, P.red_ad, P.tdc = saved


def yearly(nav):
    y = nav.groupby(nav.index.year).last()
    y0 = pd.concat([pd.Series([nav.iloc[0]], index=[nav.index[0].year - 1]), y])
    return (y0 / y0.shift(1) - 1).dropna()


def main():
    GR.init()
    M, P = GR.G["M"], GR.G["P"]
    picked = json.load(open(os.path.join(HERE, "_picked.json")))
    out = {}
    for w, b in picked.items():
        c = b["cfg"]
        a, e = GR.WIN[w]
        res = {}
        print(f"\n=== {w} 策略 ===")
        for nm, cc in ablations(c).items():
            W, _ = build(P, cc)
            row = {}
            for k, (x, y) in GR.WIN.items():
                s = bt.stats(*bt.simulate(M, W, x, y))
                row[k] = (s["cagr"], s["mdd"])
            res[nm] = row
            print(f"  {nm:12s} " + "  ".join(f"{k} {v[0]:6.1%}/{v[1]:6.1%}" for k, v in row.items() if k in (w, 'oos')))
        # 成本加倍 / 信号晚一天 / 再平衡带宽
        W, st = build(P, c)
        C0 = M._cost.copy()
        M._cost = C0 * 2
        s2 = bt.stats(*bt.simulate(M, W, a, e)); M._cost = C0
        Wl, _ = build(P, c, lag=1)
        sl = bt.stats(*bt.simulate(M, Wl, a, e))
        sb = bt.stats(*bt.simulate(M, W, a, e, band=0.10))
        nav, tu = bt.simulate(M, W, a, e)
        s0 = bt.stats(nav, tu)
        extra = {"基准": (s0["cagr"], s0["mdd"]), "成本加倍": (s2["cagr"], s2["mdd"]),
                 "信号晚一天": (sl["cagr"], sl["mdd"]), "再平衡带宽 10%": (sb["cagr"], sb["mdd"])}
        for k, v in extra.items():
            print(f"  {k:12s} {v[0]:6.1%}/{v[1]:6.1%}")
        out[w] = dict(ablation=res, extra=extra, yearly=yearly(nav).round(4).to_dict(),
                      turn_yr=s0["turn_yr"], state=pd.Series(st, index=M.idx)[a:e].map(ST.STATE_NAMES)
                      .value_counts(normalize=True).round(4).to_dict())
    json.dump(out, open(os.path.join(HERE, "_robust.json"), "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
