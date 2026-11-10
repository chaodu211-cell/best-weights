# -*- coding: utf-8 -*-
"""用仓库里的引擎重算最优拟合的各因子与温度，结果存到本目录的 base.pkl（其余脚本都读它）。不改仓库里其他文件。"""
import os as _os, sys as _sys
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.abspath(_os.path.join(HERE, "..", ".."))
SP = HERE
import sys, os, pickle, time
sys.path.insert(0, REPO)
os.chdir(REPO)
import numpy as np, pandas as pd
import engine as E, alt_engine as A
t0 = time.time()
rawdf, meta, spy, _ = E.build_indicators()
dirs = E.direction(spy, rawdf.index)
_, adj, _, _ = E.compose(rawdf, dirs)
lev_pct, _ = E.leverage_monitor(rawdf)
lev_pct2, lev_single, lev_raw = A.lev_single(rawdf.index, lev_pct)
parts = A.alt_inputs(rawdf, adj, lev_pct2, spy, rawdf.index)
temp = A.alt_temperature(parts)
rf = rawdf.copy()
for k, v in {"turnover": "_turnover_daily", "advancing": "_advancing_daily", "leverage": "_leverage_daily"}.items():
    rf[k] = rawdf[v]
_, adjf, _, _ = E.compose(rf, dirs, fast=True)
if lev_raw is not None:
    adjf = A.blue_lev_splice(adjf, dirs, lev_raw["ratio_daily"], fast=True)
bt = A.blue_temperature(adjf, judge=True)
vix = E.load_vix(rawdf.index)
qqq = E.load("QQQ")["close"].reindex(rawdf.index)
import tdc_signal as TD
t = TD.compute(getattr(E, "RAW", None))
out = dict(rawdf=rawdf, adj=adj, adjf=adjf, lev_pct=lev_pct2, parts=pd.DataFrame(parts), temp=temp, bt=bt,
           vix=vix, qqq=qqq, spy=spy["close"].reindex(rawdf.index), tdc=t["tdc"], tdc_legs=t["legs"], tdc_red=t["red"],
           tdc_dd=t["dd"])
pickle.dump(out, open(os.path.join(HERE, "base.pkl"), "wb"))
print("done %.1fs" % (time.time() - t0), rawdf.index[0].date(), rawdf.index[-1].date(), len(rawdf))
print(pd.DataFrame(parts).dropna().tail(3).round(1))
print("temp last", temp.dropna().tail(3).round(1).to_dict())
