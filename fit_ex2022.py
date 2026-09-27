# -*- coding: utf-8 -*-
"""剔除 2022 年后重新标定红点/蓝点权重，并画到纳指图上。

口径：
  · 标的 QQQ，判据 = 触发后 30 个交易日平均收益（红点求最小、蓝点求最大）。
  · 样本 = 12 因子齐备（2018-01-12 起）且前瞻收益可知的日子，**日期在 2022 年的不参与
    标定与打分**。因子本身照常按 252 日滚动分位计算（2023 年初的读数仍参照 2022）。
    2021 年底的触发日保留——它们的前瞻收益落在 2022 年初，正是 2021-12 那个顶的信号。
  · 触发频率按设计值：红点 65/2241（alt_engine.ALT_DESIGN_RATE），蓝点 42/2244（BLUE_TH 口径）。
  · 两段式：粗扫 0/1/2 → 入选率 ≥60% 的因子（并列不截断，最多 8 个）细扫 0~4。
  · 红点连 3 日；蓝点当日口径、VIX ≥ 30 为独立闸门、不进权重。
  · 2022 年当作留出年：用剔除 2022 后标出的门槛，回头看它在 2022 的表现。

用法： python3 fit_ex2022.py          # 扫描 + 出图（约 8 分钟）
       python3 fit_ex2022.py plot     # 只重画
"""
import os, sys, json, itertools
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import scan_soxx as S

OUT_JSON = os.path.join(HERE, "_fit_ex2022.json")
OUT_HTML = os.path.join(HERE, "剔除2022对照.html")
RED_RATE, BLUE_RATE = 65 / 2241, 42 / 2244
EXCL = ("2022-01-01", "2022-12-31")
BLUE_ALL = [n for n in S.NAMES if n != "vix_abs"]
CUR_RED  = dict(narrow_gap=2, top2=4, lev_ratio=3, lev_inten=3, vix_abs=4, topshare=2)   # 现行 ALT_W
CUR_BLUE = dict(top2=1, ma20=1, lev_ratio=1, erp=1)                                      # 现行 BLUE_W


def load():
    F = pd.read_csv(S.CACHE, index_col=0, parse_dates=True)
    F["fwd"] = F["qqq"].shift(-S.FWD) / F["qqq"] - 1.0
    F = F.loc[F[S.NAMES].notna().all(axis=1).idxmax():]          # 12 因子齐备起
    F = F[F[S.NAMES].notna().all(axis=1)]
    return F


def masks(F):
    ok = np.isfinite(F["fwd"].values)
    in22 = (F.index >= EXCL[0]) & (F.index <= EXCL[1])
    return ok & ~in22, ok & in22, ok        # 标定样本 / 2022 留出 / 全部


def cols(names, fast):
    return [S.FASTCOL.get(n, n) if fast else n for n in names]


def _persist(c, n):
    if n <= 1: return c
    o = c.copy()
    for k in range(1, n): o[k:] &= c[:-k]
    o[:n-1] = False
    return o


def scan_valid(X, fwd, W, side, levels, target, pn, gate, valid, batch=4096):
    """与 scan_soxx.scan 相同，但温度与连日判定用全序列，只有 valid 的日子计数与打分"""
    N, M = X.shape[0], W.shape[1]
    f = np.where(valid, np.nan_to_num(fwd), 0.0).astype(np.float32)
    nv = valid.sum()
    bs, bd = np.zeros(M, np.float32), np.zeros(M, np.int32)
    for s in range(0, M, batch):
        w = W[:, s:s+batch].astype(np.float32); w = w / w.sum(0, keepdims=True)
        T = X @ w
        Sv = np.sort(T[valid], axis=0)
        bestd = np.full(T.shape[1], 10**9, np.int64)
        for q in levels:
            th = Sv[min(int(q * nv), nv - 1)]
            c = (T > th) if side == "hot" else (T < th)
            if gate is not None: c &= gate[:, None]
            m = _persist(c, pn) & valid[:, None]
            d = m.sum(0).astype(np.int64)
            sc = np.where(d > 0, (f @ m.astype(np.float32)) / np.maximum(d, 1), np.nan)
            take = (np.abs(d - target) < bestd) & (d > 0)
            bestd = np.where(take, np.abs(d - target), bestd)
            bs[s:s+batch] = np.where(take, sc, bs[s:s+batch])
            bd[s:s+batch] = np.where(take, d, bd[s:s+batch])
    return bs, bd


def temp_of(F, w, fast):
    v = np.array(list(w.values()), float); v = v / v.sum()
    return pd.Series(F[cols(list(w), fast)].values @ v, index=F.index)


def calibrate(t, rate, side, gate, pn, valid):
    target = rate * valid.sum(); v = t.values
    lo, hi = np.nanmin(v), np.nanmax(v)
    for _ in range(60):
        mid = (lo + hi) / 2
        c = (v > mid) if side == "hot" else (v < mid)
        if gate is not None: c = c & gate
        n = (_persist(c, pn) & valid).sum()
        if side == "hot": lo, hi = (mid, hi) if n > target else (lo, mid)
        else:             lo, hi = (lo, mid) if n > target else (mid, hi)
    return round((lo + hi) / 2, 1)


def trig(t, th, side, gate, pn):
    c = (t.values > th) if side == "hot" else (t.values < th)
    if gate is not None: c = c & gate
    return _persist(c, pn)


def fit(F, side):
    fast = side == "cold"
    names = S.NAMES if side == "hot" else BLUE_ALL
    valid, _, _ = masks(F)
    gate = (F["vix"].values >= 30) if side == "cold" else None
    rate = RED_RATE if side == "hot" else BLUE_RATE
    pn = 3 if side == "hot" else 1
    tgt = int(round(rate * valid.sum()))
    lv = np.arange(0.90, 0.9651, 0.005) if side == "hot" else np.arange(0.02, 0.42, 0.005)
    X = F[cols(names, fast)].values.astype(np.float32); fwd = F["fwd"].values
    W = np.array(list(itertools.product([0, 1, 2], repeat=len(names))), np.int8).T
    W = W[:, W.sum(0) > 0]
    bs, bd = scan_valid(X, fwd, W.astype(np.float32), side, lv, tgt, pn, gate, valid)
    ok = (bd >= tgt*0.7) & (bd <= tgt*1.4)
    sc = np.where(ok, bs, np.inf if side == "hot" else -np.inf)
    o = np.argsort(sc if side == "hot" else -sc)[:2000]
    rate_in = {n: float((W[k, o] > 0).mean()) for k, n in enumerate(names)}
    order = sorted(names, key=lambda n: -rate_in[n])
    core = [n for n in order if rate_in[n] >= 0.60]
    if len(core) > 8:                      # 并列不截断：取到第 8 名的入选率为止，但总数最多 8
        cut = rate_in[core[7]]; core = [n for n in core if rate_in[n] >= cut][:8]
    if len(core) < 4: core = order[:4]
    X2 = F[cols(core, fast)].values.astype(np.float32)
    W2 = np.array(list(itertools.product(range(5), repeat=len(core))), np.int8).T
    W2 = W2[:, W2.sum(0) > 0]
    lv2 = np.arange(0.90, 0.9651, 0.0025) if side == "hot" else np.arange(0.02, 0.42, 0.005)
    b2, d2 = scan_valid(X2, fwd, W2.astype(np.float32), side, lv2, tgt, pn, gate, valid)
    ok2 = (d2 >= tgt*0.7) & (d2 <= tgt*1.4)
    s2 = np.where(ok2, b2, np.inf if side == "hot" else -np.inf)
    i = int(np.argmin(s2) if side == "hot" else np.argmax(s2))
    best = {n: float(v) for n, v in zip(core, W2[:, i]) if v > 0}
    print("[%s] 粗扫 %d 组合、细扫 %d 组合（%s）→ %s" % (
        "红点" if side == "hot" else "蓝点", W.shape[1], W2.shape[1], "/".join(S.CN[n] for n in core),
        " ".join("%s%g" % (S.CN[k], v) for k, v in best.items())))
    return best, {S.CN[n]: round(rate_in[n], 2) for n in order}, [S.CN[n] for n in core]


def tops(px):
    c = px.values; n = len(c); cand = []
    for i in range(63, n - 1):
        if c[i] >= np.nanmax(c[i-63:i+1]) and np.nanmin(c[i:min(n, i+64)]) / c[i] - 1 <= -0.11:
            cand.append(i)
    out = []
    for i in cand:
        if out and i - out[-1] <= 63:
            if c[i] > c[out[-1]]: out[-1] = i
        else: out.append(i)
    return out


def evaluate(F, w, side):
    fast = side == "cold"
    valid, hold, allv = masks(F)
    gate = (F["vix"].values >= 30) if side == "cold" else None
    rate = RED_RATE if side == "hot" else BLUE_RATE
    pn = 3 if side == "hot" else 1
    t = temp_of(F, w, fast)
    th = calibrate(t, rate, side, gate, pn, valid)
    m = trig(t, th, side, gate, pn)
    fwd = F["fwd"].values
    def st(sel):
        mm = m & sel
        if mm.sum() == 0: return dict(n=0)
        i = np.flatnonzero(mm); ev = 1 + int((np.diff(i) > S.FWD).sum())
        return dict(n=int(mm.sum()), ev=ev, mean=float(np.nanmean(fwd[mm])*100),
                    neg=float(np.nanmean(fwd[mm] < 0)*100), base=float(np.nanmean(fwd[sel])*100))
    out = dict(w=w, th=th, dates=[str(d.date()) for d in F.index[m]],
               train=st(valid), hold22=st(hold), all=st(allv))
    if side == "hot":
        tp = tops(F["qqq"]); rows = []
        for p in tp:
            hit = np.flatnonzero(m[max(0, p-30):p+6])
            rows.append(dict(top=str(F.index[p].date()), hit=bool(len(hit)),
                             first=str(F.index[max(0, p-30) + hit[0]].date()) if len(hit) else None))
        out["tops"] = rows
    return out


def run():
    F = load()
    valid, hold, _ = masks(F)
    print("标定样本 %s → %s，剔除 2022 后 %d 天；2022 留出 %d 天" %
          (F.index[0].date(), F.index[valid][-1].date(), valid.sum(), hold.sum()))
    res = {}
    for side in ("hot", "cold"):
        best, rate_in, core = fit(F, side)
        cur = CUR_RED if side == "hot" else CUR_BLUE
        res[side] = dict(opt=evaluate(F, best, side), cur=evaluate(F, cur, side),
                         rate_in=rate_in, core=core)
    json.dump(res, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)
    return res


def report(res):
    for side, nm in (("hot", "红点"), ("cold", "蓝点")):
        print("\n" + "=" * 100)
        print("【%s】粗扫入选率：%s" % (nm, "  ".join("%s %.2f" % kv for kv in res[side]["rate_in"].items())))
        for k, lab in (("opt", "剔除2022最优"), ("cur", "现行权重")):
            r = res[side][k]; s = sum(r["w"].values())
            print("  %-12s %s  门槛 %.1f" % (lab, " / ".join("%s %.0f%%" % (S.CN[n], v/s*100) for n, v in r["w"].items()), r["th"]))
            for seg, sl in (("train", "标定样本(无2022)"), ("hold22", "2022 留出年"), ("all", "全部(含2022)")):
                x = r[seg]
                if x["n"] == 0: print("      %-16s 0 天" % sl); continue
                print("      %-16s %3d天/%2d事件  后30日 %+6.2f%%  为负 %3.0f%%  基准 %+5.2f%%  边际 %+6.2fpp" % (
                    sl, x["n"], x["ev"], x["mean"], x["neg"], x["base"], x["mean"] - x["base"]))
            if "tops" in r:
                print("      顶：" + "  ".join("%s%s" % (t["top"][:7], "✓" if t["hit"] else "✗") for t in r["tops"]))


# ---------------- 画图 ----------------
def chart(res):
    F = pd.read_csv(S.CACHE, index_col=0, parse_dates=True)
    px = F["qqq"].dropna(); dates = list(px.index); vals = px.values
    t0, t1 = dates[0].value, dates[-1].value
    ly, hy = np.log(vals.min()*0.93), np.log(vals.max()*1.07)
    Wp, Hp, ML, MR, MT, MB = 1280, 400, 58, 18, 30, 26
    X = lambda d: ML + (d.value - t0)/(t1 - t0)*(Wp - ML - MR)
    Y = lambda v: MT + (hy - np.log(v))/(hy - ly)*(Hp - MT - MB)
    pos = {d: i for i, d in enumerate(dates)}
    line = " ".join("%s%.1f,%.1f" % ("M" if i == 0 else "L", X(d), Y(v)) for i, (d, v) in enumerate(zip(dates, vals)))
    x0, x1 = X(pd.Timestamp(EXCL[0])), X(pd.Timestamp(EXCL[1]))
    grid = ""
    for yr in range(2017, 2027):
        d = pd.Timestamp("%d-01-01" % yr)
        if dates[0] <= d <= dates[-1]:
            grid += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="g"/><text x="%.1f" y="%d" class="ax" text-anchor="middle">%d</text>' % (
                X(d), MT, X(d), Hp-MB, X(d), Hp-10, yr)
    for v in (120, 200, 300, 450, 700):
        if vals.min()*0.93 <= v <= vals.max()*1.07:
            grid += '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="g2"/><text x="%d" y="%.1f" class="ax" text-anchor="end">%d</text>' % (
                ML, Y(v), Wp-MR, Y(v), ML-6, Y(v)+3, v)
    def fmt(x):
        if x["n"] == 0: return "0 天"
        return "%d天/%d事件 · 后30日 %+.2f%%（边际 %+.2fpp）· 为负 %.0f%%" % (x["n"], x["ev"], x["mean"], x["mean"]-x["base"], x["neg"])
    wtxt = lambda w: " · ".join("%s %.0f%%" % (S.CN[k], v/sum(w.values())*100) for k, v in w.items())
    panels = []
    for key, title in (("opt", "剔除 2022 后的最优权重"), ("cur", "现行权重（对照）")):
        dots = ""
        for side, color, dy, nm in (("hot", "var(--red)", -9, "红点"), ("cold", "var(--blue)", 9, "蓝点")):
            for ds in res[side][key]["dates"]:
                d = pd.Timestamp(ds)
                if d in pos:
                    v = vals[pos[d]]
                    dots += '<circle cx="%.1f" cy="%.1f" r="3.2" fill="%s" fill-opacity=".88"><title>%s %s · QQQ %.1f</title></circle>' % (
                        X(d), Y(v)+dy, color, ds, nm, v)
        tops_txt = ""
        if "tops" in res["hot"][key]:
            tops_txt = " ".join('<span class="%s">%s%s</span>' % ("hit" if t["hit"] else "miss", t["top"][:7], "✓" if t["hit"] else "✗")
                                for t in res["hot"][key]["tops"])
        r, b = res["hot"][key], res["cold"][key]
        panels.append(f'''
<section>
  <h2>{title}</h2>
  <svg viewBox="0 0 {Wp} {Hp}" class="chart" role="img" aria-label="纳斯达克走势与{title}的红蓝点">
    <rect x="{x0:.1f}" y="{MT}" width="{x1-x0:.1f}" height="{Hp-MT-MB}" class="excl"/>
    <text x="{(x0+x1)/2:.1f}" y="{MT-9}" class="ax" text-anchor="middle">2022 剔除（留出年）</text>
    {grid}
    <path d="{line}" class="px"/>
    {dots}
  </svg>
  <div class="meta">
    <div><b class="r">红点</b> {wtxt(r["w"])}　门槛 {r["th"]:.1f}（连 3 日）</div>
    <div class="ind">标定样本 {fmt(r["train"])}</div>
    <div class="ind">2022 留出 {fmt(r["hold22"])}</div>
    <div class="ind">下跌顶 {tops_txt}</div>
    <div><b class="b">蓝点</b> {wtxt(b["w"])}　门槛 {b["th"]:.1f}　且 VIX ≥ 30（独立闸门）</div>
    <div class="ind">标定样本 {fmt(b["train"])}</div>
    <div class="ind">2022 留出 {fmt(b["hold22"])}</div>
  </div>
</section>''')
    html = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>剔除2022对照</title>
<style>
:root{{--bg:#fff;--fg:#111827;--mut:#6B7280;--g:#EEF0F3;--px:#111827;--ex:rgba(17,24,39,.06);--red:#CE5A4E;--blue:#3D7FB8}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0B0F16;--fg:#E8EAED;--mut:#9AA3AF;--g:#1C222C;--px:#E8EAED;--ex:rgba(232,234,237,.07);--red:#E07366;--blue:#5B9BD5}}}}
:root[data-theme=dark]{{--bg:#0B0F16;--fg:#E8EAED;--mut:#9AA3AF;--g:#1C222C;--px:#E8EAED;--ex:rgba(232,234,237,.07);--red:#E07366;--blue:#5B9BD5}}
body{{background:var(--bg);color:var(--fg);margin:0 auto;padding:22px 16px 40px;max-width:1340px;
 font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
h1{{font-size:19px;margin:0 0 4px}} h2{{font-size:15px;margin:26px 0 4px;font-weight:600}}
.chart{{width:100%;height:auto;display:block}} .ax{{font-size:10.5px;fill:var(--mut)}}
.g{{stroke:var(--g)}} .g2{{stroke:var(--g)}} .px{{fill:none;stroke:var(--px);stroke-width:1.1;stroke-opacity:.6}}
.excl{{fill:var(--ex)}}
.meta{{font-size:12.8px}} .ind{{color:var(--mut);padding-left:2.6em}}
.r{{color:var(--red)}} .b{{color:var(--blue)}} .hit{{color:var(--fg)}} .miss{{color:var(--mut);text-decoration:line-through}}
.note{{color:var(--mut);font-size:12.5px;margin-top:6px}}
</style></head><body>
<h1>剔除 2022 年重新标定 · 纳斯达克</h1>
<div class="note">标的 QQQ，判据为触发后 30 个交易日的平均收益。2022 年的交易日不参与标定与打分，
用标出来的门槛回头看它在 2022 年的表现（留出年）。分位窗口 252 日；红点连 3 日，蓝点当日口径且 VIX ≥ 30。
触发频率按设计值（红点 2.90%、蓝点 1.87%）。「下跌顶」= 前 63 日最高且之后 63 日内回撤 ≥ 11%，
顶前 30 日至顶后 5 日内有红点记为抓到。</div>
{"".join(panels)}
<div class="note">红点画在价格线上方、蓝点在下方，仅为避让。悬停看日期。蓝点未区分实际利率重估期（页面上的空心蓝点），这里一律实心。</div>
</body></html>'''
    open(OUT_HTML, "w", encoding="utf-8").write(html)
    print("\n图 →", OUT_HTML)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        res = json.load(open(OUT_JSON))
    else:
        res = run()
    report(res); chart(res)
