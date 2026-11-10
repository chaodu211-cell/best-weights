# -*- coding: utf-8 -*-
import sys, io, pandas as pd
src = sys.argv[1]; out = sys.argv[2]
tabs = pd.read_html(io.open(src, encoding="utf-8"))
t0 = tabs[0]
print("表数", len(tabs), "| 第一张表列:", list(t0.columns)[:8], "行数", len(t0))
c = t0[["Symbol", "GICS Sector", "Date added"]].copy()
c["Date added"] = pd.to_datetime(c["Date added"].astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})")[0], errors="coerce")
c.to_csv(out + "/members.csv", index=False)
print("有纳入日期", c["Date added"].notna().sum(), "| 2005 年后纳入", (c["Date added"] >= "2005-03-01").sum(), "| 2017-10 后纳入", (c["Date added"] >= "2017-10-18").sum())
t1 = tabs[1]
t1.columns = ["_".join([str(x) for x in col]).strip() if isinstance(col, tuple) else str(col) for col in t1.columns]
print("第二张表列:", list(t1.columns), "行数", len(t1))
t1.to_csv(out + "/changes.csv", index=False)
print(t1.head(4).to_string()); print(t1.tail(3).to_string())
