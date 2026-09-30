# -*- coding: utf-8 -*-
"""报告用的内联 SVG：对数净值图（含状态色带、信号点）、回撤图。颜色全部走 CSS 变量，深浅主题通用。"""
import numpy as np, pandas as pd

W, H = 1080, 360
L, R, T, B = 56, 92, 18, 30


def _x(idx, a, b):
    t0, t1 = idx[0].value, idx[-1].value
    return lambda d: L + (pd.Timestamp(d).value - t0) / (t1 - t0) * (W - L - R)


def _years(idx, X, y0, y1):
    out = []
    step = 1 if (idx[-1] - idx[0]).days < 5 * 366 else 1
    for y in range(idx[0].year + 1, idx[-1].year + 1, step):
        d = pd.Timestamp(f"{y}-01-01")
        if d < idx[0] or d > idx[-1]:
            continue
        x = X(d)
        out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{y1 + 16}" class="ax" text-anchor="middle">{y}</text>')
    if (idx[-1] - idx[0]).days < 4 * 366:      # 短窗口再补季度刻度
        for d in pd.date_range(idx[0], idx[-1], freq="QS"):
            if d.month != 1:
                x = X(d)
                out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" class="grid faint"/>')
    return "".join(out)


def equity(series, state=None, marks=None, label="", height=H):
    """series: [(名称, 净值 Series, css 类)]，第一条是主策略；state: 状态 Series（中文名）；
    marks: {css 类: [日期...]} 画在主策略曲线上"""
    idx = series[0][1].index
    X = _x(idx, None, None)
    lo = min(s.min() for _, s, _ in series); hi = max(s.max() for _, s, _ in series)
    lg0, lg1 = np.log10(lo * 0.92), np.log10(hi * 1.08)
    y0, y1 = T, height - B
    Y = lambda v: y1 - (np.log10(v) - lg0) / (lg1 - lg0) * (y1 - y0)
    parts = [f'<svg viewBox="0 0 {W} {height}" class="chart" role="img" aria-label="{label}">']
    # 状态色带
    if state is not None:
        st = state.reindex(idx)
        cls = {"黄点锁仓": "band-y", "红点减仓": "band-r", "蓝点抄底": "band-b", "闸门": "band-g"}
        run_s, run_c = None, None
        for d, v in list(st.items()) + [(None, None)]:
            c = cls.get(v)
            if c != run_c:
                if run_c:
                    xa, xb = X(run_s), X(d if d is not None else idx[-1])
                    parts.append(f'<rect x="{xa:.1f}" y="{y0}" width="{max(xb - xa, 1):.1f}" height="{y1 - y0}" class="{run_c}"/>')
                run_s, run_c = d, c
    parts.append(_years(idx, X, y0, y1))
    # 纵轴：对数刻度，取 1-2-5 序列
    ticks = []
    for e in range(int(np.floor(lg0)), int(np.ceil(lg1)) + 1):
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lg0 <= np.log10(v) <= lg1:
                ticks.append(v)
    for v in ticks:
        y = Y(v)
        t = f"{v:g}×" if v >= 1 else f"{v:.1f}×"
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" class="grid faint"/>'
                     f'<text x="{L - 6}" y="{y + 3.5:.1f}" class="ax" text-anchor="end">{t}</text>')
    # 曲线（主策略最后画，压在上面）
    for nm, s, c in list(reversed(series)):
        s = s.reindex(idx).ffill()
        step = max(1, len(s) // 900)
        pts = " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in list(s.items())[::step] + [(s.index[-1], s.iloc[-1])])
        parts.append(f'<polyline points="{pts}" class="ln {c}"/>')
    # 末端标签（防重叠：按 y 排序后推开）
    ends = sorted([(Y(s.iloc[-1]), nm, c, s.iloc[-1]) for nm, s, c in series])
    last = -99
    for y, nm, c, v in ends:
        y = max(y, last + 13); last = y
        parts.append(f'<text x="{W - R + 6}" y="{y + 3.5:.1f}" class="lab {c}">{nm} {v:,.1f}×</text>')
    # 信号点
    if marks:
        main = series[0][1]
        for c, ds in marks.items():
            for d in ds:
                if d in main.index:
                    parts.append(f'<circle cx="{X(d):.1f}" cy="{Y(main[d]):.1f}" r="3.2" class="{c}"/>')
    parts.append("</svg>")
    return "".join(parts)


def drawdown(series, height=150):
    idx = series[0][1].index
    X = _x(idx, None, None)
    dds = [(nm, s / s.cummax() - 1, c) for nm, s, c in series]
    lo = min(-0.30, min(d.min() for _, d, _ in dds)) * 1.05
    y0, y1 = 10, height - B
    Y = lambda v: y0 + (v / lo) * (y1 - y0)
    parts = [f'<svg viewBox="0 0 {W} {height}" class="chart" role="img" aria-label="回撤">']
    parts.append(_years(idx, X, y0, y1))
    for v in np.arange(0, lo, -0.2 if lo < -0.6 else -0.1):
        y = Y(v)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" class="grid faint"/>'
                     f'<text x="{L - 6}" y="{y + 3.5:.1f}" class="ax" text-anchor="end">{v:.0%}</text>')
    y = Y(-0.30)
    parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" class="limit"/>'
                 f'<text x="{W - R + 6}" y="{y + 3.5:.1f}" class="ax lim">−30% 上限</text>')
    for nm, d, c in reversed(dds):
        step = max(1, len(d) // 900)
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in list(d.items())[::step])
        parts.append(f'<polyline points="{L:.1f},{Y(0):.1f} {pts} {X(d.index[-1]):.1f},{Y(0):.1f}" class="dd {c}"/>')
    parts.append("</svg>")
    return "".join(parts)
