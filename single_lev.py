#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""红点杠杆因子并入「正股成交额前十」的单股杠杆 ETF（2026-09-30 起）。

为什么：杠杆因子（多空比、交易强度）原来只看 15 只指数杠杆 ETF。2022-08 起单股杠杆 ETF 陆续上市，
投机杠杆有一部分换到了这些产品上：单股杠杆占全部杠杆 ETF 成交额 2024 年 20%、2025 年 31%；2025 年只看
指数杠杆时交易强度从 0.26 掉到 0.22，看着像降温，并入后几乎持平（0.33→0.32）。

规则（用户 2026-09-30 定）：
  · 每天按**正股成交额**（未复权收盘 × 成交量）给标普 500 成分股排序，从高到低取前 10 只；
    某只股票当天没有已上市的杠杆 ETF 就顺延到第 11、12…名。同一家公司的多类股（GOOGL/GOOG）只算一次。
  · 这 10 只股票的全部杠杆 ETF（倍数绝对值 > 1：2x、1.5x、1.25x 做多，2x、1.5x 做空；1 倍反向不算杠杆，
    与指数篮子不含 SH/PSQ 一致）的成交额，做多的加进多空比与强度的「做多」，做空的加进「做空」。
  · **FROM 之前的读数一律不动**（仍是只含指数杠杆 ETF 的旧口径）；FROM 起换新口径。新口径的分位与它自己
    同口径的过去 252 日比——不能拿"含单股"的今天去比"不含单股"的过去一年，否则读数会凭空跳到顶。
  · 红点的杠杆因子（多空比 + 交易强度）与蓝点的杠杆多空比（当日口径）同日换成新口径，两处用同一个多空比
    （蓝点最初没换，用户 2026-09-30 当天改为一起换）。

对照表 single_lev_etfs.json：从 stockanalysis 各发行商页面的基金全称解析（"GraniteShares 2x Long NVDA Daily ETF"、
"Direxion Daily TSLA Bull 2X ETF"、"T-Rex 2X Inverse Tesla Daily Target ETF"…），每 30 天重爬一次补上新产品。
本文件的对照表部分只用标准库（fetch_sp500.py 也是）；计算部分才用 pandas。
"""
import html, json, os, re, time
from datetime import date

BASE = os.path.dirname(os.path.abspath(__file__))
MAP_JSON = os.path.join(BASE, "single_lev_etfs.json")
FROM = "2026-09-30"        # 起用日（含）：之前的读数沿用旧口径，不回头改
TOP_N = 10
REFRESH_DAYS = 30
ALIAS = {"GOOG": "GOOGL"}  # 同一家公司的多类股
# 基金名称里用公司名而不是代码的（T-Rex 等），以及 Direxion 的 BRKB
NAME2T = {"NVIDIA": "NVDA", "TESLA": "TSLA", "ALPHABET": "GOOGL", "GOOGLE": "GOOGL", "MICROSOFT": "MSFT",
          "APPLE": "AAPL", "AMAZON": "AMZN", "BROADCOM": "AVGO", "NETFLIX": "NFLX", "PALANTIR": "PLTR", "BRKB": "BRK-B"}
_PAT = [re.compile(r"Direxion Dail+y (?P<u>\S+) (?P<d>Bull|Bear) (?P<x>\d+(?:\.\d+)?)X", re.I),
        re.compile(r"(?P<x>\d+(?:\.\d+)?)X (?P<d>Long|Short|Inverse) (?P<u>.+?) (?:Daily|ETF)", re.I),
        re.compile(r"ProShares (?P<d>UltraShort|UltraPro Short|UltraPro|Ultra) (?P<u>[A-Z][A-Za-z0-9.\-]*)(?: ETF)?$")]
_ROW = re.compile(r'<a href="/etf/([a-z0-9.\-]+)/">[A-Z0-9.\-]+</a>(?:(?!</tr>).)*?<td class="slw[^"]*">([^<]+)</td>', re.S)


def parse_name(name):
    """基金全称 → (标的代码, 倍数)；做空为负。认不出来返回 None"""
    for p in _PAT:
        m = p.search(name)
        if not m:
            continue
        g = m.groupdict()
        d = g["d"].lower()
        x = {"ultra": 2.0, "ultrashort": 2.0, "ultrapro": 3.0, "ultrapro short": 3.0}.get(d) or float(g["x"])
        sgn = -1 if d in ("bear", "short", "inverse", "ultrashort", "ultrapro short") else 1
        u = g["u"].strip().upper()
        u = NAME2T.get(u, u)
        return ALIAS.get(u, u), sgn * x
    return None


def discover(get, universe):
    """爬 stockanalysis 全部发行商页面 → {股票: {"long": [[etf, 倍数]...], "short": [...]}}，只留 universe 里的股票"""
    idx = get("https://stockanalysis.com/etf/provider/")
    provs = sorted(set(re.findall(r'href="/etf/provider/([a-z0-9\-]+)/"', idx)))
    m = {}
    for p in provs:
        try:
            page = get(f"https://stockanalysis.com/etf/provider/{p}/")
        except Exception:
            continue
        for tk, name in _ROW.findall(page):
            r = parse_name(html.unescape(name).strip())
            if not r:
                continue
            u, L = r
            if u not in universe or abs(L) <= 1:
                continue
            side = m.setdefault(u, {"long": [], "short": []})["long" if L > 0 else "short"]
            if tk.upper() not in [e for e, _ in side]:
                side.append([tk.upper(), L])
    return {k: {s: sorted(v[s]) for s in v} for k, v in sorted(m.items())}


def load():
    try:
        return json.load(open(MAP_JSON, encoding="utf-8"))["stocks"]
    except Exception:
        return {}


def etfs(m=None):
    m = load() if m is None else m
    return sorted({e for v in m.values() for s in ("long", "short") for e, _ in v[s]})


def maybe_refresh(get, universe, force=False):
    """对照表缺失、超过 REFRESH_DAYS 天或 force 时重爬。结果不像样（太少、缺头部股票）就保留旧表。→ 说明文字"""
    try:
        old = json.load(open(MAP_JSON, encoding="utf-8"))
        age = (date.today() - date.fromisoformat(old["updated"])).days
    except Exception:
        old, age = None, 10**6
    if not force and age <= REFRESH_DAYS:
        return None
    try:
        m = discover(get, universe)
    except Exception as e:
        return f"单股杠杆 ETF 对照表重爬失败，沿用旧表：{str(e)[:80]}"
    if len(m) < 20 or not {"NVDA", "TSLA", "AAPL"} <= set(m):
        return f"单股杠杆 ETF 对照表重爬结果不完整（{len(m)} 只股票），沿用旧表"
    if old and old.get("stocks") == m:
        old["updated"] = date.today().isoformat()
        json.dump(old, open(MAP_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return "单股杠杆 ETF 对照表重爬：无变化"
    added = sorted(set(etfs(m)) - set(etfs(old["stocks"]) if old else set()))
    json.dump({"updated": date.today().isoformat(), "source": "stockanalysis.com 发行商页面的基金全称",
               "rule": "倍数绝对值 > 1 的单股杠杆/反向 ETF；GOOG 并入 GOOGL", "stocks": m},
              open(MAP_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return f"单股杠杆 ETF 对照表已更新：{len(m)} 只股票、{len(etfs(m))} 只 ETF" + (f"，新增 {added}" if added else "")


def components(idx, stock_dv, etf_dv, m=None, n=TOP_N):
    """每天的「前 n 只正股」及其杠杆 ETF 成交额。

    stock_dv: DataFrame（列 = 成分股，成交额）；etf_dv: DataFrame（列 = 单股杠杆 ETF，成交额），都已对齐到 idx。
    → (long_add, short_add, picks)：两条成交额（没有产品的年份为 0）与每天选中的股票列表。
    ETF 当天缺数据（数据源晚到）时用前一天的成交额补一天，免得最新一天的读数被悄悄压低。
    """
    import numpy as np
    import pandas as pd
    m = load() if m is None else m
    cols = list(etf_dv.columns)
    E = etf_dv.reindex(idx).copy()
    if len(idx) > 1:                                   # 只补最后一天（前一天有、今天没到的）
        E.iloc[-1] = E.iloc[-1].where(E.iloc[-1].notna(), E.iloc[-2])
    L = {k: [e for e, _ in v["long"] if e in cols] for k, v in m.items()}
    S = {k: [e for e, _ in v["short"] if e in cols] for k, v in m.items()}
    long_add = pd.Series(0.0, index=idx)
    short_add = pd.Series(0.0, index=idx)
    picks = pd.Series([[] for _ in idx], index=idx, dtype=object)
    first = E.notna().any(axis=1)
    if not first.any():
        return long_add, short_add, picks
    Ev = E.values
    ci = {c: i for i, c in enumerate(cols)}
    X = stock_dv.reindex(idx)
    names = list(X.columns)
    Xv = X.values
    for t in np.where(first.values)[0]:
        row = Xv[t]
        order = np.argsort(-np.nan_to_num(row, nan=-1.0))
        got, seen, la, sa = [], set(), 0.0, 0.0
        for j in order:
            if not np.isfinite(row[j]) or len(got) >= n:
                break
            k = ALIAS.get(names[j], names[j])
            if k in seen or k not in m:
                continue
            lv = [Ev[t, ci[e]] for e in L[k]]
            sv = [Ev[t, ci[e]] for e in S[k]]
            live = [v for v in lv + sv if np.isfinite(v)]
            if not live:
                continue                              # 这只股票当天还没有已上市的杠杆 ETF → 顺延
            seen.add(k)
            got.append(k)
            la += sum(v for v in lv if np.isfinite(v))
            sa += sum(v for v in sv if np.isfinite(v))
        long_add.iloc[t], short_add.iloc[t], picks.iloc[t] = la, sa, got
    return long_add, short_add, picks


if __name__ == "__main__":
    m = load()
    if not m:
        raise SystemExit("还没有 single_lev_etfs.json")
    print(f"对照表：{len(m)} 只股票、{len(etfs(m))} 只 ETF，更新于 {json.load(open(MAP_JSON))['updated']}")
    for k, v in m.items():
        print(f"  {k:6s} 多 {' '.join(f'{e}({x:g})' for e, x in v['long'])}"
              + (f"  空 {' '.join(f'{e}({x:g})' for e, x in v['short'])}" if v["short"] else ""))
