# -*- coding: utf-8 -*-
"""分窗口标定 + 全周期回放：
   拟合标的 = 纳斯达克 QQQ（后 30 个交易日）
   窗口A 2018-01-12~2021-12-31（用户要的 2017-10 起，但 12 因子齐备从 2018-01-12 才开始）
   窗口B 2022-01-03~至今
在各自窗口内扫出红点/蓝点的最优权重与门槛，然后把**这套固定的规则**铺到整个十年上，
画到纳斯达克（QQQ）走势图上。窗口内是样本内，窗口外就是真正的样本外——两张图的
对比就是"这套权重能不能出窗口"的答案。

用法： python3 fit_windows.py            # 扫描 + 出图（约 6 分钟）
       python3 fit_windows.py plot       # 用上次扫出的权重只重画图（秒级）
"""
import os, sys, json, itertools
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import scan_soxx as S

OUT_JSON = os.path.join(HERE, "_fit_windows.json")
OUT_HTML = os.path.join(HERE, "窗口对照.html")

TARGET = "qqq"      # 拟合标的：纳斯达克（QQQ）。对照口径 SOXX。
OTHER  = "soxx"

def load_target():
    """从因子缓存重建矩阵，前瞻收益取 TARGET。"""
    F = pd.read_csv(S.CACHE, index_col=0, parse_dates=True)
    F["fwd"]   = F[TARGET].shift(-S.FWD) / F[TARGET] - 1.0
    F["fwd_o"] = F[OTHER].shift(-S.FWD) / F[OTHER] - 1.0
    return F.dropna(subset=S.NAMES + ["fwd", "vix"])


WINDOWS = [("A", "2018-01-12", "2021-12-31", "窗口A：2018-01 ~ 2021-12"),
           ("B", "2023-01-03", "2099-01-01", "窗口B：2023-01 ~ 至今")]
RATE_HOT, RATE_COLD = 66/2151, 30/2151      # 触发密度沿用全样本标定（红 3.07% / 蓝 1.40%）


def fit_one(F, lo, hi, side):
    """在 [lo,hi] 窗口内跑 粗扫 → 定形状 → 细扫，返回 (权重dict, 门槛, 诊断)"""
    w = F.loc[lo:hi]
    gate = (w["vix"].values >= S.VIX_TH) if side == "cold" else None
    fast = (side == "cold")
    cols_all = S.NAMES if side == "hot" else S.BLUE_ALL
    tgt = int(round(len(w) * (RATE_HOT if side == "hot" else RATE_COLD)))
    lv = S.LV_HOT if side == "hot" else S.LV_COLD
    pn = 3 if side == "hot" else 1

    X = S._grid(cols_all, w, fast); fwd = w["fwd"].values
    W = np.array(list(itertools.product([0, 1, 2], repeat=len(cols_all))), np.int8).T
    W = W[:, W.sum(0) > 0]
    bs, bd, _ = S.scan(X, fwd, W.astype(np.float32), side, lv, tgt, pn, gate)
    ok = (bd >= tgt*0.7) & (bd <= tgt*1.4)
    sc = np.where(ok, bs, np.inf if side == "hot" else -np.inf)
    o = np.argsort(sc if side == "hot" else -sc)[:2000]
    rate = {c: (W[k, o] > 0).mean() for k, c in enumerate(cols_all)}
    core = [c for c in cols_all if rate[c] >= 0.60]
    core = sorted(core, key=lambda c: -rate[c])[:6] or sorted(cols_all, key=lambda c: -rate[c])[:4]

    X2 = S._grid(core, w, fast)
    W2 = np.array(list(itertools.product(range(5), repeat=len(core))), np.int8).T
    W2 = W2[:, W2.sum(0) > 0]
    b2, d2, _ = S.scan(X2, fwd, W2.astype(np.float32), side, lv, tgt, pn, gate)
    ok2 = (d2 >= tgt*0.7) & (d2 <= tgt*1.4)
    s2 = np.where(ok2, b2, np.inf if side == "hot" else -np.inf)
    i = int(np.argmin(s2) if side == "hot" else np.argmax(s2))
    best = {c: float(v) for c, v in zip(core, W2[:, i]) if v > 0}

    t = S.temp_of(w, best, fast)
    th, n = S.calibrate(t, tgt, side, gate, pn)
    return best, th, {"入选率": {S.CN[c]: round(rate[c], 2) for c in cols_all},
                      "细扫因子": [S.CN[c] for c in core], "窗口内": float(b2[i]),
                      "目标天数": tgt, "实际天数": int(n)}


def apply_all(F, w, th, side):
    """把固定规则铺到全周期，返回触发布尔序列"""
    fast = (side == "cold")
    t = S.temp_of(F, w, fast)
    gate = (F["vix"].values >= S.VIX_TH) if side == "cold" else None
    m = S._mask(t, th, side, gate, 3 if side == "hot" else 1)
    return pd.Series(m & t.notna().values, index=F.index), t


def stats(F, m, lo, hi):
    fwd_q = F["fwd"].values          # 拟合标的：QQQ
    fwd_s = F["fwd_o"].values        # 对照：SOXX
    inw = (F.index >= lo) & (F.index <= hi)
    out = {}
    for tag, sel in (("窗口内", m.values & inw), ("窗口外", m.values & ~inw)):
        if sel.sum() == 0:
            out[tag] = None; continue
        out[tag] = {"天": int(sel.sum()),
                    "QQQ": float(np.nanmean(fwd_q[sel])), "QQQ为负": float(np.nanmean(fwd_q[sel] < 0)),
                    "SOXX": float(np.nanmean(fwd_s[sel]))}
    out["基准"] = {"QQQ": float(np.nanmean(fwd_q)), "SOXX": float(np.nanmean(fwd_s))}
    return out


def run(only=None):
    """only='A'/'B' 时只重标那一个窗口，另一个沿用 _fit_windows.json 里的旧结果。"""
    F = load_target()
    res = json.load(open(OUT_JSON)) if (only and os.path.exists(OUT_JSON)) else {}
    for key, lo, hi, label in WINDOWS:
        if only and key != only:
            continue
        res[key] = {"label": label, "lo": lo, "hi": hi}
        for side in ("hot", "cold"):
            w, th, diag = fit_one(F, lo, hi, side)
            m, t = apply_all(F, w, th, side)
            res[key][side] = {"w": w, "th": th, "diag": diag,
                              "dates": [str(d.date()) for d in F.index[m.values]],
                              "stats": stats(F, m, lo, hi)}
            print("[%s %s] 权重 %s  门槛 %.1f  全周期 %d 天" %
                  (key, "红点" if side == "hot" else "蓝点",
                   " ".join("%s%g" % (S.CN[k], v) for k, v in w.items()), th, m.sum()))
            print("     ", json.dumps(res[key][side]["stats"], ensure_ascii=False))
    json.dump(res, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)
    return res


# ---------------- 画图：内联 SVG，无依赖 ----------------
def chart(res):
    F = pd.read_csv(S.CACHE, index_col=0, parse_dates=True)
    px = F["qqq"].dropna()
    dates = list(px.index); vals = px.values
    t0, t1 = dates[0].value, dates[-1].value
    lo_y, hi_y = np.log(vals.min()*0.93), np.log(vals.max()*1.07)
    Wpx, Hpx, ML, MR, MT, MB = 1280, 392, 58, 18, 30, 26
    def X(d): return ML + (d.value - t0)/(t1 - t0) * (Wpx - ML - MR)
    def Y(v): return MT + (hi_y - np.log(v))/(hi_y - lo_y) * (Hpx - MT - MB)
    pos = {d: i for i, d in enumerate(dates)}

    panels = []
    for key, lo, hi, label in WINDOWS:
        r = res[key]
        line = " ".join("%s%.1f,%.1f" % ("M" if i == 0 else "L", X(d), Y(v))
                        for i, (d, v) in enumerate(zip(dates, vals)))
        x0, x1 = X(pd.Timestamp(lo)), X(min(pd.Timestamp(hi), dates[-1]))
        dots = []
        for side, color, dy in (("hot", "#CE5A4E", -9), ("cold", "#3D7FB8", 9)):
            for ds in r[side]["dates"]:
                d = pd.Timestamp(ds)
                if d not in pos: continue
                v = vals[pos[d]]
                dots.append('<circle cx="%.1f" cy="%.1f" r="3.1" fill="%s" fill-opacity=".85"><title>%s %s  QQQ %.1f</title></circle>'
                            % (X(d), Y(v)+dy, color, ds, "红点" if side == "hot" else "蓝点", v))
        ticks = ""
        for yr in range(2017, 2027):
            d = pd.Timestamp("%d-01-01" % yr)
            if d < dates[0] or d > dates[-1]: continue
            ticks += ('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#E5E7EB"/>'
                      '<text x="%.1f" y="%d" class="ax" text-anchor="middle">%d</text>'
                      % (X(d), MT, X(d), Hpx-MB, X(d), Hpx-10, yr))
        yt = ""
        for v in (120, 200, 300, 450, 700):
            if not (vals.min()*0.93 <= v <= vals.max()*1.07): continue
            yt += ('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#F1F2F4"/>'
                   '<text x="%d" y="%.1f" class="ax" text-anchor="end">%d</text>'
                   % (ML, Y(v), Wpx-MR, Y(v), ML-6, Y(v)+3, v))
        st = r["hot"]["stats"]; sc = r["cold"]["stats"]
        def fmt(s, k):
            if s[k] is None: return "—"
            return "%d天 QQQ %+.1f%% / SOXX %+.1f%%" % (s[k]["天"], s[k]["QQQ"]*100, s[k]["SOXX"]*100)
        wtxt = lambda d: " · ".join("%s %.0f%%" % (S.CN[k], v/sum(d.values())*100) for k, v in d.items())
        panels.append(f'''
<section>
  <h2>{label}<span class="sub">阴影 = 标定窗口（样本内），其余为样本外</span></h2>
  <svg viewBox="0 0 {Wpx} {Hpx}" class="chart">
    <rect x="{x0:.1f}" y="{MT}" width="{x1-x0:.1f}" height="{Hpx-MT-MB}" fill="#111827" fill-opacity=".055"/>
    <line x1="{x0:.1f}" y1="{MT}" x2="{x0:.1f}" y2="{Hpx-MB}" stroke="#9AA3AF" stroke-dasharray="3 3"/>
    <line x1="{x1:.1f}" y1="{MT}" x2="{x1:.1f}" y2="{Hpx-MB}" stroke="#9AA3AF" stroke-dasharray="3 3"/>
    <text x="{(x0+x1)/2:.1f}" y="{MT-9}" class="ax" text-anchor="middle">样本内（标定窗口）</text>
    {yt}{ticks}
    <path d="{line}" fill="none" stroke="#111827" stroke-width="1.1" stroke-opacity=".62"/>
    {''.join(dots)}
  </svg>
  <div class="meta">
    <div><b class="r">红点</b> {wtxt(r['hot']['w'])}　门槛 {r['hot']['th']:.1f}（连 3 日）</div>
    <div class="ind">窗口内 {fmt(st,'窗口内')}　｜　窗口外 {fmt(st,'窗口外')}</div>
    <div><b class="b">蓝点</b> {wtxt(r['cold']['w'])}　门槛 {r['cold']['th']:.1f}　且 VIX ≥ 30（独立闸门）</div>
    <div class="ind">窗口内 {fmt(sc,'窗口内')}　｜　窗口外 {fmt(sc,'窗口外')}</div>
  </div>
</section>''')

    base_q = res["A"]["hot"]["stats"]["基准"]["QQQ"]*100
    base_s = res["A"]["hot"]["stats"]["基准"]["SOXX"]*100
    html = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>窗口对照</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fff;--fg:#111827;--mut:#6B7280;--line:#E5E7EB}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0B0F16;--fg:#E8EAED;--mut:#9AA3AF;--line:#232A35}}
 .chart path{{stroke:#E8EAED!important}} .chart text{{fill:#9AA3AF}}}}
body{{background:var(--bg);color:var(--fg);margin:0;padding:22px 16px 40px;
 font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:1340px;margin:0 auto}}
h1{{font-size:19px;margin:0 0 4px}} h2{{font-size:15px;margin:26px 0 6px;font-weight:600}}
.sub{{color:var(--mut);font-weight:400;font-size:12.5px;margin-left:10px}}
.chart{{width:100%;height:auto;display:block}} .ax{{font-size:10.5px;fill:#9AA3AF}}
.meta{{font-size:12.8px;color:var(--fg);margin-top:2px}} .ind{{color:var(--mut);padding-left:2.6em;margin-bottom:5px}}
.r{{color:#CE5A4E}} .b{{color:#3D7FB8}} .note{{color:var(--mut);font-size:12.5px;margin-top:8px}}
</style></head><body>
<h1>分窗口标定 · 铺到十年纳斯达克</h1>
<div class="note">拟合标的 <b>纳斯达克 QQQ</b>，判据为触发后 30 个交易日的平均收益；SOXX 作对照口径。
全样本基准：QQQ {base_q:+.2f}% / SOXX {base_s:+.2f}%。
12 因子齐备自 2018-01-12 起，之前无温度读数，故 2016-09~2018-01 一段没有点。</div>
{''.join(panels)}
<div class="note">红点画在价格线上方、蓝点在下方，仅为避让，位置不含额外信息。鼠标悬停看日期。</div>
</body></html>'''
    open(OUT_HTML, "w").write(html)
    print("图已写入", OUT_HTML)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else ""
    if a == "plot":
        chart(json.load(open(OUT_JSON)))
    elif a in ("A", "B"):
        chart(run(a))
    else:
        chart(run())
