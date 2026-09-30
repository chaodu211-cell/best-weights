# -*- coding: utf-8 -*-
"""生成 策略回测/策略回测报告.html：读 _picked.json / _robust.json / _grid.csv，所有数字用精确模拟重跑。

    python3 策略回测/report.py
"""
import json, os, sys, warnings, html
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bt, strategy as ST, grid as GR, svg as SV
from report_text import TEXT, rules_text

OUT = os.path.join(HERE, "策略回测报告.html")
pct = lambda v, d=1: "—" if v is None or not np.isfinite(v) else f"{v * 100:+.{d}f}%".replace("+-", "−").replace("-", "−")
pct0 = lambda v, d=1: "—" if v is None or not np.isfinite(v) else f"{v * 100:.{d}f}%".replace("-", "−")


def cls_ok(cagr, mdd):
    return "ok" if (cagr >= 0.5 and mdd >= -0.30) else "no"


def main():
    GR.init()
    M, P = GR.G["M"], GR.G["P"]
    picked = json.load(open(os.path.join(HERE, "_picked.json")))
    rob = json.load(open(os.path.join(HERE, "_robust.json")))
    grid = pd.read_csv(os.path.join(HERE, "_grid.csv"))
    S = P.S
    sec = []
    for w, title in (("10y", "十年策略"), ("3y", "三年策略")):
        c = picked[w]["cfg"]
        a, e = GR.WIN[w]
        W, st = ST.build(P, GR.expand(c))
        nav, tu = bt.simulate(M, W, a, e)
        s = bt.stats(nav, tu)
        bm = {}
        for b in ("QQQ", "TQQQ", "SOXL"):
            bn, _ = bt.simulate(M, pd.DataFrame({b: 1.0}, index=M.idx), a, e)
            bm[b] = (bn, bt.stats(bn))
        stS = pd.Series(st, index=M.idx).map(ST.STATE_NAMES)
        ix = nav.index
        marks = {"m-red": [d for d in ix if S["red"].get(d, False) is True or S["red"].get(d, 0) == 1],
                 "m-blue": [d for d in ix if bool(S["blue"].get(d, False))],
                 "m-yel": [d for d in ix if bool(S["yellow"].get(d, False))]}
        eq = SV.equity([("策略", nav, "s-main"), ("TQQQ", bm["TQQQ"][0], "s-tqqq"), ("QQQ", bm["QQQ"][0], "s-qqq")],
                       state=stS[a:e], marks=marks, label=f"{title}净值（对数刻度）")
        dd = SV.drawdown([("策略", nav, "s-main"), ("TQQQ", bm["TQQQ"][0], "s-tqqq"), ("QQQ", bm["QQQ"][0], "s-qqq")])
        # 持仓明细：实际用到的标的与时间占比
        Wd = pd.DataFrame(W, index=M.idx, columns=M.cols)[a:e]
        used = (Wd > 0).mean().sort_values(ascending=False)
        used = used[used > 0.005]
        # 逐年
        yr = rob[w]["yearly"]
        byr = {b: bm[b][0].groupby(bm[b][0].index.year).last() for b in ("QQQ", "TQQQ")}
        sec.append(dict(w=w, title=title, cfg=c, s=s, bm={k: v[1] for k, v in bm.items()}, eq=eq, dd=dd,
                        used=used, yr=yr, rob=rob[w], row=picked[w]["row"], nb=picked[w], byr=byr, nav=nav))
    # 两套策略放到每个窗口（精确模拟）
    cross = {}
    for x in sec:
        W, _ = ST.build(P, GR.expand(x["cfg"]))
        x["W"] = W
        cross[x["w"]] = {k: bt.stats(*bt.simulate(M, W, a, e)) for k, (a, e) in GR.WIN.items()}
    bmx = {b: {k: bt.stats(*bt.simulate(M, pd.DataFrame({b: 1.0}, index=M.idx), a, e)) for k, (a, e) in GR.WIN.items()}
           for b in ("QQQ", "TQQQ")}
    # 做空腿补充：锁仓/减仓期改持 SQQQ（上市前 QID）
    ci = M.col_i
    short = []
    for x in sec:
        W0, st = ST.build(P, GR.expand(x["cfg"]))
        for nm, states, wt in (("现金（选定策略）", [], 0), ("黄点锁仓期 → SQQQ 1/3", [3], 1 / 3),
                               ("黄点锁仓期 → SQQQ 2/3", [3], 2 / 3), ("红点期 → SQQQ 1/3", [1], 1 / 3),
                               ("红点与黄点期 → SQQQ 1/3", [1, 3], 1 / 3)):
            W = W0.copy(); m = np.isin(st, states)
            sq, qd = M.tradable["SQQQ"].values, M.tradable["QID"].values
            W[m & sq, ci["SQQQ"]] = wt; W[m & ~sq & qd, ci["QID"]] = min(1, wt * 1.5)
            r = [bt.stats(*bt.simulate(M, W, *GR.WIN[k])) for k in (x["w"], "oos")]
            short.append((x["title"], nm, r[0]["cagr"], r[0]["mdd"], r[1]["cagr"], r[1]["mdd"]))
    render(sec, grid, cross, bmx, short)


WNAME = {"10y": "十年 2016-09~2026-09", "3y": "三年 2023-09~2026-09", "pre3y": "三年窗之前 7 年 2016-09~2023-09",
         "h1": "十年前半 2016-09~2021-09", "h2": "十年后半 2021-09~2026-09", "oos": "2007-01~2016-09（样本外）"}


def render(sec, grid, cross, bmx, short):
    h = []
    A = h.append
    A(HEAD)
    A('<main class="wrap">')
    A(f'<header class="top"><h1>预警模型杠杆策略</h1><p class="lede">{TEXT["lede"]}</p>'
      f'<div class="chips"><span class="chip">数据截至 2026-09-22</span><span class="chip">t 日信号 · t+1 开盘成交</span>'
      f'<span class="chip">含交易成本</span><span class="chip">目标：年化 ≥ 50%，最大回撤 ≤ 30%</span></div></header>')
    # —— 结论卡 ——
    A('<section class="verdict">')
    for x in sec:
        s = x["s"]; r = x["row"]
        ok = cls_ok(s["cagr"], s["mdd"])
        cx = cross[x["w"]]
        oos_c, oos_m = cx["oos"]["cagr"], cx["oos"]["mdd"]
        other = "3y" if x["w"] == "10y" else "pre3y"
        lab = "放到三年窗" if x["w"] == "10y" else "之前 7 年（2016-09~2023-09）"
        extra = (f'<div class="vrow"><span>{lab}</span>'
                 f'<b class="mono">{pct(cx[other]["cagr"])}</b><b class="mono">{pct0(cx[other]["mdd"])}</b></div>')
        A(f'<article class="vcard {ok}"><h2>{x["title"]}</h2><p class="win">{GR.WIN[x["w"]][0]} → {GR.WIN[x["w"]][1]}</p>'
          f'<div class="vhead"><span></span><span>年化</span><span>最大回撤</span></div>'
          f'<div class="vrow main"><span>目标窗口（样本内）</span><b class="mono">{pct(s["cagr"])}</b><b class="mono">{pct0(s["mdd"])}</b></div>'
          f'{extra}'
          f'<div class="vrow"><span>2007-01~2016-09（样本外）</span><b class="mono">{pct(oos_c)}</b><b class="mono">{pct0(oos_m)}</b></div>'
          f'<div class="vrow bm"><span>同期 TQQQ 持有</span><b class="mono">{pct(x["bm"]["TQQQ"]["cagr"])}</b><b class="mono">{pct0(x["bm"]["TQQQ"]["mdd"])}</b></div>'
          f'<div class="vrow bm"><span>同期 QQQ 持有</span><b class="mono">{pct(x["bm"]["QQQ"]["cagr"])}</b><b class="mono">{pct0(x["bm"]["QQQ"]["mdd"])}</b></div>'
          f'<p class="vnote">{TEXT["verdict_" + x["w"]]}</p></article>')
    A('</section>')
    A(f'<section class="warn"><h2>先读这一段</h2>{TEXT["warn"]}</section>')
    # —— 每个策略 ——
    for x in sec:
        A(f'<section class="strat" id="s{x["w"]}"><h2>{x["title"]} <span class="mut">· {GR.WIN[x["w"]][0]} → {GR.WIN[x["w"]][1]}</span></h2>')
        A(f'<div class="rules">{rules_text(x["cfg"])}</div>')
        A('<div class="legend"><span><i class="k s-main"></i>策略</span><span><i class="k s-tqqq"></i>TQQQ 持有</span>'
          '<span><i class="k s-qqq"></i>QQQ 持有</span><span><i class="dot m-red"></i>红点</span><span><i class="dot m-yel"></i>黄点</span>'
          '<span><i class="dot m-blue"></i>蓝点</span><span><i class="sw band-r"></i>红点减仓期</span><span><i class="sw band-y"></i>黄点锁仓期</span>'
          '<span><i class="sw band-b"></i>蓝点抄底期</span><span><i class="sw band-g"></i>闸门关闭</span></div>')
        A(f'<div class="chart-box">{x["eq"]}</div>')
        A(f'<p class="cap">净值（对数刻度，起点 = 1）。</p><div class="chart-box">{x["dd"]}</div><p class="cap">回撤。虚线为 −30% 上限。</p>')
        # 逐年
        A('<div class="cols"><div><h3>逐年收益</h3><div class="tbl"><table><thead><tr><th>年份</th><th>策略</th><th>TQQQ</th><th>QQQ</th></tr></thead><tbody>')
        byr = x["byr"]
        for y, v in x["yr"].items():
            y = int(y)
            def yret(ser):
                s = ser[ser.index <= y]; p = ser[ser.index < y]
                return s.iloc[-1] / (p.iloc[-1] if len(p) else 1.0) - 1
            A(f'<tr><td>{y}{"*" if y in (x["nav"].index[0].year, x["nav"].index[-1].year) else ""}</td>'
              f'<td class="mono">{pct(v)}</td><td class="mono mut">{pct(yret(byr["TQQQ"]))}</td><td class="mono mut">{pct(yret(byr["QQQ"]))}</td></tr>')
        A('</tbody></table></div><p class="cap">* 首尾两年不满一年。</p></div>')
        # 状态与持仓
        A('<div><h3>时间都花在哪</h3><div class="tbl"><table><tbody>')
        for k, v in sorted(x["rob"]["state"].items(), key=lambda kv: -kv[1]):
            A(f'<tr><td>{k}</td><td class="mono">{v * 100:.1f}%</td></tr>')
        A('</tbody></table></div><h3>实际持有过的标的</h3><div class="tbl"><table><tbody>')
        for k, v in x["used"].items():
            if k == "BIL":
                k = "BIL（现金）"
            A(f'<tr><td>{k}</td><td class="mono">{v * 100:.1f}% 的交易日</td></tr>')
        A(f'</tbody></table></div><p class="cap">年换手 {x["rob"]["turn_yr"]:.1f} 倍（单边）。</p></div></div>')
        # 消融
        A('<h3>每个模型信号贡献了多少（只去掉一项，其余不动）</h3><div class="tbl"><table><thead><tr><th></th>'
          f'<th>目标窗口 年化</th><th>最大回撤</th><th>2007-16 样本外 年化</th><th>最大回撤</th></tr></thead><tbody>')
        for k, v in x["rob"]["ablation"].items():
            a = v[x["w"]]; o = v["oos"]
            A(f'<tr class="{"hl" if k == "完整策略" else ""}"><td>{k}</td><td class="mono {cls_ok(*a)}">{pct(a[0])}</td><td class="mono">{pct0(a[1])}</td>'
              f'<td class="mono">{pct(o[0])}</td><td class="mono">{pct0(o[1])}</td></tr>')
        A(f'</tbody></table></div><p class="cap">{TEXT["abl_" + x["w"]]}</p>')
        # 稳健性
        A('<h3>稳健性</h3><div class="tbl"><table><thead><tr><th></th><th>年化</th><th>最大回撤</th></tr></thead><tbody>')
        for k, v in x["rob"]["extra"].items():
            A(f'<tr><td>{k}</td><td class="mono {cls_ok(*v)}">{pct(v[0])}</td><td class="mono">{pct0(v[1])}</td></tr>')
        nb = x["nb"]
        A(f'<tr><td>参数邻居（每个参数单独挪一格，共 {len(nb["nb"])} 组）仍达标的比例</td><td class="mono" colspan="2">{nb["nb_feas"] * 100:.0f}%'
          f'　中位年化 {pct(nb["nb_cagr_med"])}　最差回撤 {pct0(nb["nb_worst_mdd"])}</td></tr>')
        A('</tbody></table></div>')
        A(f'<details><summary>参数邻居明细</summary><div class="tbl"><table><thead><tr><th>挪动的参数</th><th>新取值</th><th>年化</th><th>最大回撤</th></tr></thead><tbody>')
        for k, v, cg, md in nb["nb"]:
            A(f'<tr><td>{k}</td><td class="mono">{v}</td><td class="mono {cls_ok(cg, md)}">{pct(cg)}</td><td class="mono">{pct0(md)}</td></tr>')
        A('</tbody></table></div></details>')
        A('</section>')
    # —— 交叉表 ——
    A('<section><h2>两套策略放到每个窗口</h2><div class="tbl"><table><thead><tr><th>窗口</th>')
    for x in sec:
        A(f'<th>{x["title"]}</th>')
    A('<th>TQQQ 持有</th><th>QQQ 持有</th></tr></thead><tbody>')
    for k in ("10y", "3y", "h1", "h2", "pre3y", "oos"):
        A(f'<tr><td>{WNAME[k]}</td>')
        for x in sec:
            v = cross[x["w"]][k]
            tag = " ★" if k == x["w"] else ""
            A(f'<td class="mono {cls_ok(v["cagr"], v["mdd"])}">{pct(v["cagr"])} / {pct0(v["mdd"])}{tag}</td>')
        for b in ("TQQQ", "QQQ"):
            v = bmx[b][k]
            A(f'<td class="mono mut">{pct(v["cagr"])} / {pct0(v["mdd"])}</td>')
        A('</tr>')
    A('</tbody></table></div><p class="cap">每格为 年化 / 最大回撤；绿色 = 两个约束都满足；★ = 该策略的挑选窗口。'
      '2007-2016 期间 TQQQ 2010-02 才上市，之前的格子里用 QLD。</p></section>')
    # —— 做空腿 ——
    A('<section><h2>补充：反向 ETF 做空腿</h2><div class="tbl"><table><thead><tr><th>策略</th><th>锁仓 / 减仓期持有</th>'
      '<th>目标窗口 年化</th><th>最大回撤</th><th>2007-16 年化</th><th>最大回撤</th></tr></thead><tbody>')
    for t, nm, a1, a2, b1, b2 in short:
        A(f'<tr><td>{t}</td><td>{nm}</td><td class="mono {cls_ok(a1, a2)}">{pct(a1)}</td><td class="mono">{pct0(a2)}</td>'
          f'<td class="mono">{pct(b1)}</td><td class="mono">{pct0(b2)}</td></tr>')
    A(f'</tbody></table></div><p class="cap">{TEXT["short"]}</p></section>')
    # —— 网格总体 ——
    A(f'<section><h2>在多大的搜索里挑出来的</h2>{TEXT["grid"](grid)}</section>')
    A(f'<section><h2>口径</h2>{TEXT["method"]}</section>')
    A('</main>')
    open(OUT, "w", encoding="utf-8").write("".join(h))
    print("→", OUT)


HEAD = open(os.path.join(HERE, "report_head.html"), encoding="utf-8").read() if os.path.exists(os.path.join(HERE, "report_head.html")) else ""

if __name__ == "__main__":
    main()
