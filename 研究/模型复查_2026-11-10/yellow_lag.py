# -*- coding: utf-8 -*-
"""黄点当日收盘之后，晚 1 / 2 个交易日收盘才成交，纳指这段时间的涨跌。先跑 base.py、pit_members.py。"""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(HERE)
import pickle, numpy as np, pandas as pd
B = pickle.load(open("base.pkl", "rb")); Q = pickle.load(open("pit.pkl", "rb"))
q = B["qqq"]; idx = q.index
def lag(d, k):
    i = idx.get_loc(d); return (q.iloc[i + k] / q.iloc[i] - 1) * 100 if i + k < len(q) else np.nan
r1, r2 = [], []
for d in Q["yellow0"]:
    a, b = lag(d, 1), lag(d, 2); r1.append(a); r2.append(b)
    print(f"  {d.date()}  T→T+1 {a:+5.2f}%   T→T+2 {b:+5.2f}%")
print(f"  平均 T→T+1 {np.mean(r1):+.2f}%  T→T+2 {np.mean(r2):+.2f}%；最差 T→T+2 {np.min(r2):+.2f}%；T+1→T+2 这一天最差 {np.min(np.array(r2)-np.array(r1)):+.2f}%")
