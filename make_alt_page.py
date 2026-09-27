#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 dashboard_alt.html：替代口径红点 + 原样蓝点，去掉黑框与温度走势图。

做法是**复用生产管线**：先让 engine.main() 产出标准结构，再把红点那一部分换掉。
这样分项面板、杠杆面板、实际利率图、宏观状态行全部照旧，不用再抄一遍接线——
oos_check.py 里手抄 pipeline() 的教训（"存在与 main() 走样的风险"）不必重演。

与生产页面的差别，全部在本文件的 patch() 里：
  · 红点温度换成 alt_engine 的权重（来历与局限见该文件顶部）
  · 黑框预警整条摘掉
  · 分项里补上 VIX（绝对刻度）与 前2%成交额个股占比
  · 蓝点、实际利率重估期分级、杠杆面板全部不动
模板 alt_tpl.html 是 dash_tpl.html 的派生：只删了温度走势那一节，区间滑块挪到纳指图上方。

用法：
    python3 make_alt_page.py                  # 用 raw/（与生产同源）
    python3 make_alt_page.py --raw raw_long   # 用拼接长面板，历史更长
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
H = 30   # 前瞻窗口：2026-09-18 起红点的检验周期改为 30 个交易日（此前仓库各处是 63）
# 展示与评估标的：2026-09-18 曾改用费城半导体 SOXX，2026-09-20 按用户要求改回纳指。
# 权重本来就是以 QQQ 为标的标定的（见 alt_engine 顶部），标的与标定口径现在一致。
TARGET = "QQQ"
TARGET_LABEL = "纳斯达克100 · QQQ"


PREREG_CSV = os.path.join(BASE, "_红点预登记.csv")
PREREG_COLS = ["信号", "首日", "温度", "门槛", "登记时间"]


def prereg(V, temp, A, log):
    """预先登记：起算日之后首日出现的红点事件追加写进 _红点预登记.csv，再按事先定好的标准评判。

    只追加、不改写——登记表记的是"当时看到了什么"，之后权重或数据怎么变都不回头改它。
    log=False（非生产数据，如 --raw raw_long）时只读不写。
    """
    old = (pd.read_csv(PREREG_CSV, dtype={"信号": str}, parse_dates=["首日"])
           if os.path.exists(PREREG_CSV) else pd.DataFrame(columns=PREREG_COLS))
    start, now = pd.Timestamp(A.PREREG_FROM), pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    sigs = (("固定门槛", V["hot"], lambda t: A.ALT_TH), ("自适应门槛", V["hot_ad"], lambda t: V["th_ad"].get(t)))
    new = []
    for key, flag, th in sigs:
        have = pd.to_datetime(old.loc[old["信号"] == key, "首日"])
        for a, _ in A.events(flag):
            if a < start:
                continue
            # 数据修订可能让同一事件的首日挪一两天：45 个自然日内已登记的算同一次
            if len(have) and ((have - a).abs() <= pd.Timedelta(days=45)).any():
                continue
            new.append({"信号": key, "首日": a, "温度": round(float(temp.get(a)), 1),
                        "门槛": round(float(th(a)), 1), "登记时间": now})
    if log:
        if not os.path.exists(PREREG_CSV):
            pd.DataFrame(columns=PREREG_COLS).to_csv(PREREG_CSV, index=False, encoding="utf-8")
        if new:
            nd = pd.DataFrame(new, columns=PREREG_COLS)
            nd["首日"] = nd["首日"].dt.strftime("%Y-%m-%d")
            nd.to_csv(PREREG_CSV, mode="a", header=False, index=False, encoding="utf-8")
            print(f"  预先登记：新增 {len(new)} 条 → {os.path.basename(PREREG_CSV)}")
        nd_all = pd.DataFrame(new, columns=PREREG_COLS)
        rows = nd_all if old.empty else (old if nd_all.empty else pd.concat([old, nd_all], ignore_index=True))
    else:
        rows = old
    fwd = V["fwd"]
    out_rows = []
    for r in rows.itertuples(index=False):
        t = pd.Timestamp(r[1])
        f = fwd.get(t, np.nan)
        out_rows.append({"sig": r[0], "date": f"{t:%Y-%m-%d}", "temp": float(r[2]), "th": float(r[3]),
                         "logged": str(r[4]), "fwd": (None if pd.isna(f) else round(float(f) * 100, 2))})
    judge = {k: A.prereg_judge([pd.Timestamp(x) for x in rows.loc[rows["信号"] == k, "首日"]], fwd)
             for k, _, _ in sigs}
    return {"from": A.PREREG_FROM, "min_events": A.PREREG_MIN_EVENTS, "p_cut": A.PREREG_P,
            "draws": A.PREREG_DRAWS, "rows": out_rows, "judge": judge}


def _runs(pos, gap=1):
    out = []
    for i in pos:
        if out and i - out[-1][-1] <= gap:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def patch(d, E, A, log=False):
    """把标准结构改成替代口径。所有差异集中在这里。

    做三件事：① 换成替代红点的温度与旗标、摘掉黑框；② 展示与评估标的按 TARGET 设定
    （现为纳指 QQQ）；③ **按实际参与计算的因子重建分项面板**——生产结构里的 panel 是按生产权重
    拼的，直接沿用会出现"页面写着某因子、实际权重是 0"这类对不上的情况。
    """
    dates = pd.to_datetime(d["series"]["dates"])
    num = lambda v, n=1: (None if v is None or not np.isfinite(v) else round(float(v), n))
    ser = lambda s: [num(v) for v in s.reindex(dates).values]

    # —— 重算替代红点温度（再取一次原始面板；比抄一份接线安全）——
    rawdf, meta, spy, _ = E.build_indicators()
    dirs = E.direction(spy, rawdf.index)
    _, adj, _, _ = E.compose(rawdf, dirs)
    lev_pct, _ = E.leverage_monitor(rawdf)
    # 传 lev_pct 而不是 lev_temp：杠杆温度的内部比例由 alt_engine.LEV_INNER 决定（1:1），
    # 不走 leverage.py 的 LEV_W（3:1）。实测这一处就值 2.65pp。
    parts = A.alt_inputs(rawdf, adj, lev_pct, spy, rawdf.index)
    temp_alt = A.alt_temperature(parts)
    if temp_alt is None:
        sys.exit(f"替代红点温度算不出来：分项缺失（检查 {TARGET} / VIX / 杠杆ETF 是否齐全）")
    ta = temp_alt.reindex(dates)
    # 蓝点用的当日口径分项（换手率/上涨占比/杠杆多空比取未平滑值），与生产完全一致
    rf = rawdf.copy()
    for k, v in {"turnover": "_turnover_daily", "advancing": "_advancing_daily",
                 "leverage": "_leverage_daily"}.items():
        rf[k] = rawdf[v]
    _, adjf, _, _ = E.compose(rf, dirs, fast=True)

    d["temperature_sell"] = num(ta.iloc[-1])
    d["series"]["temperature_sell"] = ser(temp_alt)
    d["sell_threshold"] = A.ALT_TH
    d["sell_weights"] = {k: v for k, v in A.ALT_W.items()}
    d["alt"] = {"legs": list(A.ALT_LEGS), "lev_inner": list(A.LEV_INNER),
                "vix_anchors": list(A.VIX_ANCHORS), "persist": A.ALT_PERSIST,
                "fwd": H, "target": TARGET}

    # —— ② 展示与评估标的（TARGET，现为纳指 QQQ）——
    tgt = E.load(TARGET)
    if tgt is None:
        sys.exit(f"缺 raw/{TARGET}.csv，先抓一次行情再跑")
    d["series"]["ndx"] = [None if not np.isfinite(v) else round(float(v), 2)
                          for v in tgt["close"].reindex(dates).values]
    d["ndx_label"] = TARGET_LABEL

    # —— 红点旗标重算；黑框整条摘掉；蓝点原样保留 ——
    hot = (ta > A.ALT_TH)
    if A.ALT_PERSIST > 1:
        hot = (hot.rolling(A.ALT_PERSIST, min_periods=A.ALT_PERSIST).sum() == A.ALT_PERSIST)
    d["alerts"]["flags"]["hot"] = [bool(x) for x in hot.fillna(False).values]
    for box in (d["alerts"].get(k) for k in ("flags", "counts", "persist")):
        if isinstance(box, dict):      # persist 在生产结构里是个整数，不是字典
            box.pop("crowd", None)
    d["alerts"]["rules"] = [r for r in d["alerts"]["rules"] if r["key"] != "crowd"]

    # —— 蓝点也换权重（BLUE_W），闸门 VIX >= VIX_COLD 不动 ——
    bt_fast = A.blue_temperature(adjf)          # 判定用：当日未平滑口径
    bt_show = A.blue_temperature(adj)           # 展示用：平滑口径
    if bt_fast is None or bt_show is None:
        sys.exit("蓝点温度算不出来：BLUE_W 里的分项缺失")
    vix_all = E.load_vix(dates)
    cold_all = ((bt_fast.reindex(dates) < A.BLUE_TH)
                & (vix_all.reindex(dates) >= E.VIX_COLD)).fillna(False)
    if E.PERSIST_COLD > 1:
        cold_all = (cold_all.rolling(E.PERSIST_COLD, min_periods=E.PERSIST_COLD)
                    .sum() == E.PERSIST_COLD).fillna(False)
    rp = pd.Series(d["series"].get("repricing") or [False]*len(dates), index=dates).astype(bool)
    d["alerts"]["flags"]["cold"] = [bool(x) for x in (cold_all & ~rp).values]
    d["alerts"]["flags"]["cold_soft"] = [bool(x) for x in (cold_all & rp).values]
    d["temperature_fast"] = num(bt_fast.reindex(dates).iloc[-1])
    d["series"]["temperature_fast"] = ser(bt_fast)
    d["temperature"] = num(bt_show.reindex(dates).iloc[-1])
    d["series"]["temperature"] = ser(bt_show)
    d["regime"] = E.regime(d["temperature"])
    d["blue_weights"] = dict(A.BLUE_W)
    d["blue_threshold"] = A.BLUE_TH

    # —— 趋势下跌预警（黄点）：与红/蓝点并列的第三种信号，计算核心在 tdc_signal.py ——
    try:
        import tdc_signal as TD
        t = TD.compute(getattr(E, "RAW", None))
        d["series"]["tdc"] = ser(t["tdc"])
        for k in TD.LEGS:
            d["series"]["tdc_" + {"内部损伤": "dmg", "波动中枢": "vol", "趋势结构": "trend"}[k]] = \
                ser(t["legs"][k])
        d["alerts"]["flags"]["tdc"] = [bool(v) for v in t["red"].reindex(dates).fillna(False)]
        d["alerts"]["rules"].append({
            "key": "tdc", "name": "趋势下跌预警", "mark": "dot", "color": "#E8C24A",
            "persist": TD.PERSIST,
            "desc": (f"纳指距 52 周高点回落在 {-TD.GATE_LO:.0%}~{-TD.GATE_HI:.0%} 之间，且 TDC &gt; "
                     f"{TD.TH:g} 连续 {TD.PERSIST} 个交易日，且探测器处于武装状态。"
                     f"<b>同一轮下跌只报一次</b>——报警后立即解除武装，直到 TDC 跌回 "
                     f"{TD.REARM:g} 以下才重新武装。"
                     f"TDC 确认时纳指已跌破 {-TD.GATE_HI:.0%} 则<b>本轮作废</b>：确认来得太晚，"
                     f"这时报警多半落在底部（2010-07、2011-08、2015-08 三次都是）。"
                     f"TDC = 三条腿等权（内部损伤 / 波动中枢 / 趋势结构），每条腿三个因子等权，"
                     f"每个因子先做 {TD.WIN} 日滚动分位。"
                     f"它<b>不预告顶部</b>：五次大跌的顶部当天 TDC 是 32/29/45/46/43。"
                     f"完整因子筛选、反面证据与代价见「趋势下跌确认.md」。")})
    except Exception as exc:                      # 缺数据时整块跳过，不要让页面挂掉
        print(f"  ! 趋势下跌预警未计算：{exc}")

    # —— 红点有效性检验（2026-09-24）：自适应门槛（空心红点）、近 2 年触发频率、滚动 IC、预先登记 ——
    # 全部在全历史上算（温度与标的都用 rawdf 的索引），再截到展示窗口；来历见 alt_engine 的常量注释。
    V = A.validity(temp_alt, tgt["close"].reindex(temp_alt.index))
    red_rule = next(r for r in d["alerts"]["rules"] if r["key"] == "hot")
    d["alerts"]["flags"]["hot_ad"] = [bool(x) for x in V["hot_ad"].reindex(dates).fillna(False).values]
    d["alerts"]["rules"].append({
        "key": "hot_ad", "name": "空心红点（自适应门槛）", "mark": "dot", "hollow": True,
        "color": red_rule["color"], "persist": A.ALT_PERSIST,
        "desc": (f"同一个红点温度，门槛换成<b>自适应</b>的：取前一日及以前 {A.AD_WIN // 252} 年"
                 f"（至少 {A.AD_MINP // 252} 年）连 3 日温度的第 {100 - A.ALT_DESIGN_RATE * 100:.1f} 分位，"
                 f"让触发频率保持在设计值 {A.ALT_DESIGN_RATE:.1%} 附近，只用当时已有的数据。"
                 f"与实心红点<b>并列显示、不替换</b>：两者同时成立时画实心。"
                 f"用途是防止温度基数整体漂移后固定门槛整段失声——2007-2016 长面板上 81.7 十年只亮 2 天，"
                 f"自适应门槛亮 64 天、后 30 日比基准低 1.8pp，下跌顶多抓到 2010-04、2012-04、2015-07 三个；"
                 f"代价是 2017 后比固定门槛弱约 2.5pp。")})
    d["series"]["th_adaptive"] = ser(V["th_ad"])
    d["series"]["red_rate_2y"] = [num(v, 2) for v in V["rate"].reindex(dates).values]
    d["series"]["red_ic"] = [num(v, 3) for v in V["ic"].reindex(dates).values]
    icm = V["ic_m"]
    ic_end = None
    if len(icm):
        j = temp_alt.index.get_loc(icm.index[-1]) - A.FWD
        ic_end = f"{temp_alt.index[j]:%Y-%m-%d}"
    lastv = lambda s_: (lambda z: None if z.empty else z.iloc[-1])(s_.reindex(dates).dropna())
    d["validity"] = {
        "design_rate": round(A.ALT_DESIGN_RATE * 100, 2), "rate_win": A.RATE_WIN,
        "rate_now": num(lastv(V["rate"]), 2), "th_fixed": A.ALT_TH, "th_ad_now": num(lastv(V["th_ad"])),
        "ad_win": A.AD_WIN, "ad_minp": A.AD_MINP, "ic_win": A.IC_WIN, "fwd": A.FWD,
        "ic_now": num(lastv(V["ic"]), 3), "ic_month": (f"{icm.index[-1]:%Y-%m}" if len(icm) else None),
        "ic_window_end": ic_end,
        "ic_hist": [{"m": f"{t:%Y-%m}", "v": round(float(v), 3)} for t, v in icm.items()],
        "prereg": prereg(V, temp_alt, A, log),
    }

    d["alerts"]["counts"] = {k: int(sum(v)) for k, v in d["alerts"]["flags"].items()}

    # —— 规则说明：把每个因子的权重写清楚 ——
    tot = sum(A.ALT_W.values())
    pc = {k: v/tot*100 for k, v in A.ALT_W.items()}
    wtxt = "、".join(f"{A.LABELS[k]} {pc[k]:.0f}%" for k in
                     sorted(A.ALT_W, key=lambda k: -A.ALT_W[k]))
    btot = sum(A.BLUE_W.values())
    bpc = {k: v/btot*100 for k, v in A.BLUE_W.items()}
    btxt = "、".join(f"{A.BLUE_LABELS[k]} {bpc[k]:.0f}%" for k in
                    sorted(A.BLUE_W, key=lambda k: -A.BLUE_W[k]))
    li, lj = A.LEV_INNER
    for r in d["alerts"]["rules"]:
        if r["key"] == "hot":
            r["name"] = "红点预警（替代权重）"
            r["desc"] = (
                f"替代红点温度 > {A.ALT_TH:g}，连续 {A.ALT_PERSIST} 个交易日。"
                f"温度 = {wtxt}（合计 100%）。"
                f"其中<b>杠杆温度</b>内部按「多空比 : 交易强度 = {li:g} : {lj:g}」合成"
                f"（不是 leverage.py 的 3:1，实测这一处值 2.65pp）；"
                f"<b>VIX</b>用固定锚点 {'/'.join(f'{x:g}' for x in A.VIX_ANCHORS)} 映射，不是滚动分位；"
                f"<b>市值加权跑赢等权</b>保留「只在上涨市成立、下跌市记中性 50」的语义。"
                f"权重按前瞻 {H} 个交易日、以 {TARGET} 为标的标定。"
                f"站上MA20、换手率、宽度恶化、贴近峰值四项扫出来的权重都是 0，未参与。")
        elif r["key"] == "cold":
            r["name"] = "蓝点预警（替代权重）"
            r["desc"] = (
                f"蓝点温度（当日口径）< {A.BLUE_TH:g} 且 VIX ≥ {E.VIX_COLD}，"
                f"{'当日成立即触发' if E.PERSIST_COLD <= 1 else f'连续 {E.PERSIST_COLD} 个交易日'}。"
                f"温度 = {btxt}（<b>四项等权</b>，各 1/4）。"
                f"其中杠杆多空比取<b>当日未平滑值</b>（另三项本就是当日值）。"
                f"<b>VIX ≥ {E.VIX_COLD} 是独立闸门，不进温度</b>——它是外生的绝对刻度，"
                f"实测只用这道闸门就有 +6.2pp 边际，温度筛选再加约 5.6pp。"
                f"相比原来的六项等权，<b>删掉了换手率与上涨个股占比</b>："
                f"前者恐慌与狂热都会暴增、方向不可知，后者在 VIX ≥ {E.VIX_COLD} 的日子里"
                f"几乎必然极低、等于常数，两者都不含底部信息。"
                f"留下的四项覆盖定位（杠杆多空比、站上MA20、TOP2抱团）与估值（ERP）两个维度。"
                f"扫描的局部最优是把多空比加到 50%，只再值 1.07pp 且与全网格证据矛盾"
                f"（多空比单因子秩相关 −0.041，ERP 是 −0.336），故未采纳。"
                f"口径按前瞻 {H} 个交易日、以 {TARGET} 为标的检验。"
                f"【只在训练窗验证过：42 天只有 6~7 个独立事件，标准误约 4.1pp】")

    # —— ③ 按实际计算重建分项面板 ——
    raw_pct = lambda c: (rawdf[c].reindex(dates) if c in rawdf.columns else None)
    vix_s = E.load_vix(dates)
    rows = [
        ("narrow", "市值加权跑赢等权", "指数靠大票撑着的程度",
         f"红点 {pc['narrow']:.0f}%", parts["narrow"],
         (lambda: (lambda g: "—" if g is None else f"{g:+.2f}%")(num(raw_pct('_narrow_gap').iloc[-1], 2)))(),
         f"SPY 近 {E.NT_WIN} 日相对等权组合的超额；下跌市记中性 50"),
        ("top2", "TOP2行业成交额占比", "资金抱团度",
         f"红点 {pc['top2']:.0f}% · 蓝点 {bpc['top2']:.0f}%", adj["top2"],
         f"{num(rawdf['top2'].reindex(dates).iloc[-1])}%", "当日占比（红蓝两条温度共用）"),
        ("lev", "杠杆温度", "加杠杆的方向与强度",
         f"红点 {pc['lev']:.0f}% · 蓝点 {bpc['leverage']:.0f}%（仅多空比）", parts["lev"],
         f"多空比 {num(lev_pct['ratio'].reindex(dates).iloc[-1]):.0f} · "
         f"强度 {num(lev_pct['intensity'].reindex(dates).iloc[-1]):.0f}",
         f"两项分位按 {li:g}:{lj:g} 合成；蓝点只用多空比的当日口径"),
        ("vix_abs", "VIX（绝对刻度）", "恐慌／麻木",
         f"红点 {pc['vix_abs']:.0f}%", parts["vix_abs"],
         f"{num(vix_s.iloc[-1])}",
         f"100 − 按锚点 {'/'.join(f'{x:g}' for x in A.VIX_ANCHORS)} 分段线性映射；VIX 越低越热"),
        ("topshare", "前2%成交额个股占比", "成交额挤进头部多少只",
         f"红点 {pc['topshare']:.0f}%", parts["topshare"],
         f"{num(rawdf['_topshare'].reindex(dates).iloc[-1])}%",
         f"成交额最大的前 {E.CROWD_TOP_FRAC:.0%} 只成分股占全篮子的比例，252 日滚动分位"),
        ("ma20", "站上MA20占比", "中期趋势宽度", f"蓝点 {bpc['ma20']:.0f}%", adjf["ma20"],
         f"{num(rawdf['ma20'].reindex(dates).iloc[-1])}%", "当日占比"),
        ("erp", "风险溢价 ERP", "估值性价比（反向）", f"蓝点 {bpc['erp']:.0f}%", adjf["erp"],
         f"{num(rawdf['erp'].reindex(dates).iloc[-1], 2)}%", "E/P − 10年期美债；分位已反向"),
        ("vix_gate", "VIX 收盘", "蓝点的独立闸门", f"蓝点闸门 ≥{E.VIX_COLD}", None,
         f"{num(vix_s.iloc[-1])}", "不进任何温度，只作为蓝点的必要条件"),
    ]
    # 逐日原始值：页面上"查看某一天"要用。格式与上面 value 列逐字一致，所以直接下发字符串，
    # 不在前端再写一遍格式化（同一条规则写两份迟早走样）。
    def fmt(s, f):
        if s is None:
            return None
        return [None if not np.isfinite(v) else f(float(v)) for v in s.reindex(dates).values]
    lr = lev_pct["ratio"].reindex(dates).values
    lt = lev_pct["intensity"].reindex(dates).values
    raw_by_key = {
        "narrow":   fmt(raw_pct("_narrow_gap"), lambda v: f"{v:+.2f}%"),
        "top2":     fmt(rawdf["top2"], lambda v: f"{v:.1f}%"),
        "lev":      [None if not (np.isfinite(a) and np.isfinite(b))
                     else f"多空比 {a:.0f} · 强度 {b:.0f}" for a, b in zip(lr, lt)],
        "vix_abs":  fmt(vix_s, lambda v: f"{v:.1f}"),
        "topshare": fmt(rawdf["_topshare"], lambda v: f"{v:.1f}%"),
        "ma20":     fmt(rawdf["ma20"], lambda v: f"{v:.1f}%"),
        "erp":      fmt(rawdf["erp"], lambda v: f"{v:.2f}%"),
        "vix_gate": fmt(vix_s, lambda v: f"{v:.1f}"),
    }

    panel = []
    for key, name, desc, tag, s, val, sub in rows:
        panel.append({"key": key, "name": name, "desc": desc, "tag": tag,
                      "value": val, "sub": sub,
                      "pct": (None if s is None else num(s.reindex(dates).iloc[-1])),
                      "series": ([] if s is None else ser(s)),
                      "raw_series": raw_by_key.get(key)})
    d["panel"] = panel

    # 杠杆面板的权重标注要跟实际合成一致
    lm = d.get("leverage_monitor")
    if isinstance(lm, dict):
        lm["weights"] = {"ratio": li, "intensity": lj}
        lm["temperature"] = num(parts["lev"].reindex(dates).iloc[-1])
        for c in lm.get("components", []):
            c["weight"] = li if c.get("key") == "ratio" else lj
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="原始数据目录（默认用 engine 的 raw/）")
    ap.add_argument("--out", default="dashboard_alt.html")
    ap.add_argument("--data-out", default="data_alt.json")
    a = ap.parse_args()

    import engine as E
    import alt_engine as A
    if a.raw:
        E.RAW = a.raw if os.path.isabs(a.raw) else os.path.join(BASE, a.raw)
        if not os.path.isdir(E.RAW):
            sys.exit(f"目录不存在：{E.RAW}")
        print(f"原始数据目录：{E.RAW}")

    out_json = os.path.join(BASE, a.data_out)
    E.OUT = out_json
    E.main()                       # 产出标准结构（不碰 data.json，OUT 已改向）
    d = json.load(open(out_json, encoding="utf-8"))
    d = patch(d, E, A, log=(a.raw is None))   # 只有生产数据才写预先登记表
    json.dump(d, open(out_json, "w", encoding="utf-8"), ensure_ascii=False)

    tpl = open(os.path.join(BASE, "alt_tpl.html"), encoding="utf-8").read()
    html = tpl.replace("__DATA__", json.dumps(d, ensure_ascii=False))
    open(os.path.join(BASE, a.out), "w", encoding="utf-8").write(html)

    # —— 回执：信号统计，拿来跟生产页对比 ——
    px = pd.Series(d["series"]["ndx"], index=pd.to_datetime(d["series"]["dates"]), dtype=float)
    fwd = px.shift(-H) / px - 1
    base = fwd.dropna()
    print(f"\n写入 {a.out}（{len(html)} 字节）与 {a.data_out}")
    print(f"覆盖 {d['series']['dates'][0]} ~ {d['series']['dates'][-1]}"
          f"（{len(d['series']['dates'])} 个交易日）")
    print(f"截止日：替代红点温度 {d['temperature_sell']}（门槛 {d['sell_threshold']:g}）  "
          f"蓝点当日温度 {d.get('temperature_fast')}  "
          f"重估期 {'是' if (d['series'].get('repricing') or [False])[-1] else '否'}")
    print(f"基准 后{H}日 {base.mean()*100:+.2f}% / {(base < 0).mean()*100:.0f}%为负")
    v = d.get("validity") or {}
    if v:
        pj = v["prereg"]["judge"]
        print(f"有效性：近 2 年触发 {v['rate_now']}%（设计 {v['design_rate']}%）  "
              f"自适应门槛 {v['th_ad_now']}（固定 {v['th_fixed']:g}）  "
              f"滚动 IC {v['ic_now']}（{v['ic_month']} 月末，窗口止于 {v['ic_window_end']}）  "
              f"预先登记 自 {v['prereg']['from']}："
              + "；".join(f"{k} {j['n']} 次/{j['verdict']}" for k, j in pj.items()))
    for k, fl in d["alerts"]["flags"].items():
        pos = np.where(np.array(fl))[0]
        if not len(pos):
            print(f"  {k:10s}   0 天"); continue
        f = fwd.iloc[pos].dropna()
        print(f"  {k:10s} {len(pos):3d}天 / {len(_runs(pos)):2d}段 / {len(_runs(pos, H))} 事件   "
              f"后{H}日 {f.mean()*100:+.2f}% / {(f < 0).mean()*100:.0f}%为负 / 最差 {f.min()*100:+.1f}%")


if __name__ == "__main__":
    main()
