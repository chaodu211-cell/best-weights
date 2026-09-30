#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""策略回测用的 ETF 日线：指数 / 行业 ETF、对应的全部杠杆 ETF（多空）、现金（BIL）与 13 周国库券利率。

    python3 策略回测/fetch_etf.py          # 约 1 分钟，输出 策略回测/etf/<TICKER>.csv

列：date,open,high,low,close,adjclose,volume（旧到新）。open/high/low 为未复权价，
回测时按 adjclose/close 同比例复权。只用标准库。
"""
import json, os, ssl, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "etf")
UA = "Mozilla/5.0"          # 带完整浏览器串反而被 Yahoo 限流（429），实测只写这一段可以
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{s}?period1=1041379200&period2={p2}&interval=1d&events=div%2Csplit"

# 标的 → (1x 底层, 杠杆倍数)。底层自己记为 1。
UNIVERSE = {
    # —— 指数 ——
    "SPY": ("SPY", 1), "SSO": ("SPY", 2), "UPRO": ("SPY", 3), "SPXL": ("SPY", 3),
    "SH": ("SPY", -1), "SDS": ("SPY", -2), "SPXU": ("SPY", -3), "SPXS": ("SPY", -3),
    "QQQ": ("QQQ", 1), "QLD": ("QQQ", 2), "TQQQ": ("QQQ", 3),
    "PSQ": ("QQQ", -1), "QID": ("QQQ", -2), "SQQQ": ("QQQ", -3),
    "DIA": ("DIA", 1), "DDM": ("DIA", 2), "UDOW": ("DIA", 3), "DXD": ("DIA", -2), "SDOW": ("DIA", -3),
    "IWM": ("IWM", 1), "UWM": ("IWM", 2), "TNA": ("IWM", 3), "URTY": ("IWM", 3),
    "TWM": ("IWM", -2), "TZA": ("IWM", -3), "SRTY": ("IWM", -3),
    "MDY": ("MDY", 1), "MVV": ("MDY", 2), "MIDU": ("MDY", 3),
    # —— 11 个 SPDR 行业 ——
    "XLK": ("XLK", 1), "ROM": ("XLK", 2), "TECL": ("XLK", 3), "TECS": ("XLK", -3),
    "XLF": ("XLF", 1), "UYG": ("XLF", 2), "FAS": ("XLF", 3), "FAZ": ("XLF", -3),
    "XLE": ("XLE", 1), "DIG": ("XLE", 2), "ERX": ("XLE", 2), "ERY": ("XLE", -2),
    "XLV": ("XLV", 1), "RXL": ("XLV", 2), "CURE": ("XLV", 3),
    "XLY": ("XLY", 1), "UCC": ("XLY", 2), "WANT": ("XLY", 3),
    "XLP": ("XLP", 1), "UGE": ("XLP", 2),
    "XLI": ("XLI", 1), "UXI": ("XLI", 2), "DUSL": ("XLI", 3),
    "XLB": ("XLB", 1), "UYM": ("XLB", 2),
    "XLU": ("XLU", 1), "UPW": ("XLU", 2), "UTSL": ("XLU", 3),
    "XLRE": ("XLRE", 1), "URE": ("XLRE", 2), "DRN": ("XLRE", 3), "DRV": ("XLRE", -3),
    "XLC": ("XLC", 1), "LTL": ("XLC", 2),
    # —— 细分行业 ——
    "SOXX": ("SOXX", 1), "USD": ("SOXX", 2), "SOXL": ("SOXX", 3), "SOXS": ("SOXX", -3),
    "XBI": ("XBI", 1), "LABU": ("XBI", 3), "LABD": ("XBI", -3),
    "IBB": ("IBB", 1), "BIB": ("IBB", 2),
    "KRE": ("KRE", 1), "DPST": ("KRE", 3),
    "ITB": ("ITB", 1), "NAIL": ("ITB", 3),
    "XRT": ("XRT", 1), "RETL": ("XRT", 3),
    "IYT": ("IYT", 1), "TPOR": ("IYT", 3),
    "XOP": ("XOP", 1), "GUSH": ("XOP", 2),
    "ITA": ("ITA", 1), "DFEN": ("ITA", 3),
    "GDX": ("GDX", 1), "NUGT": ("GDX", 2),
    "FDN": ("FDN", 1), "WEBL": ("FDN", 3),
    "XPH": ("XPH", 1), "PILL": ("XPH", 3),
    # —— 现金 ——
    "BIL": ("BIL", 1),
}
EXTRA = ["^IRX"]          # 13 周国库券收益率（%），合成杠杆的融资成本


def fetch(sym):
    url = URL.format(s=urllib.parse.quote(sym), p2=int(time.time()))
    for i in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as r:
                j = json.loads(r.read())
            res = j["chart"]["result"][0]
            ts, q = res["timestamp"], res["indicators"]["quote"][0]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or q["close"]
            rows = []
            for k, t in enumerate(ts):
                if q["close"][k] is None or adj[k] is None:
                    continue
                d = time.strftime("%Y-%m-%d", time.gmtime(t + res["meta"]["gmtoffset"]))
                rows.append((d, q["open"][k], q["high"][k], q["low"][k], q["close"][k], adj[k], q["volume"][k] or 0))
            name = sym.replace("^", "_")
            with open(os.path.join(OUT, name + ".csv"), "w") as f:
                f.write("date,open,high,low,close,adjclose,volume\n")
                for r in rows:
                    f.write(",".join(str(x) for x in r) + "\n")
            return sym, len(rows), rows[0][0], rows[-1][0]
        except Exception as e:
            err = e
            time.sleep(1.5 * (i + 1))
    return sym, 0, str(err), ""


if __name__ == "__main__":
    import urllib.parse
    os.makedirs(OUT, exist_ok=True)
    syms = list(UNIVERSE) + EXTRA
    with ThreadPoolExecutor(3) as ex:
        for s, n, a, b in ex.map(fetch, syms):
            print(f"{s:6s} {n:5d}  {a} → {b}")
    json.dump(UNIVERSE, open(os.path.join(HERE, "universe.json"), "w"), indent=1)
