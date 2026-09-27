# -*- coding: utf-8 -*-
"""滚动分位窗口（engine.WINDOW，现为 252 日）的最优拟合分析。

为什么单独做：2026-09-23 在长面板上查明，替代红点在 2007-2016 失效的机制就是这个参数——
252 日滚动分位把原始值的**趋势方向**变成读数高低（TOP2、杠杆交易强度的原始值 2010-16
下降、2017 后上升，温度基数差 6.7 分）。所以窗口长度不能只在训练窗里挑，必须看样本外。

口径：
  · 数据用 ~/us2/raw_long（2004 起），训练窗 = 标定窗口 2017-10-18 起，样本外 = 之前。
  · 受窗口影响的因子：红点 4 个（市值跑赢等权、TOP2、杠杆温度两个因子、前2%占比；
    VIX 是绝对刻度不受影响），蓝点 4 个（TOP2、站上MA20、杠杆多空比当日值、ERP）。
  · 权重固定为现行 ALT_W / BLUE_W，只动窗口；门槛每个窗口在训练窗里按设计频率重标。
  · 两种预热口径：strict = 要求满窗口（engine 现行做法，长窗口起算晚）；
    minp252 = 最少 252 日、最多回看 W 日（所有窗口同一天起算，样本外能覆盖 2007-2011）。

用法：python3 scan_window.py build   # 在长面板上构建原始序列并缓存（约 1 分钟）
      python3 scan_window.py         # 扫窗口并出表
"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
LONG = os.path.expanduser("~/us2/raw_long")
CACHE = os.path.join(HERE, "_window_raw.csv")
TRAIN_FROM = "2017-10-18"          # 与 ALT_DESIGN_RATE 的标定窗口同起点
FWD = 30


def build():
    os.chdir(HERE)
    import engine as E, alt_engine as A
    E.RAW = LONG
    rawdf, meta, spy, lev_info = E.build_indicators()
    idx = rawdf.index
    R = pd.DataFrame(index=idx)
    # 红点的原始量（进分位之前）
    R["top2"] = rawdf["top2"]
    R["topshare"] = rawdf["_topshare"]
    R["narrow_gap"] = rawdf["_narrow_gap"]
    R["lev_ratio"] = rawdf["leverage"]              # 5 日均（红点用平滑口径）
    R["lev_inten"] = rawdf["_lev_intensity"]
    # 蓝点的原始量（当日口径）
    R["ma20"] = rawdf["ma20"]
    R["lev_ratio_d"] = rawdf["_leverage_daily"]
    R["erp"] = rawdf["erp"]
    # 外生
    R["spy"] = spy["close"].reindex(idx)
    R["qqq"] = E.load("QQQ")["close"].reindex(idx)
    R["vix"] = E.load_vix(idx)
    # 校验用：alt_engine 自己算出来的 252 日红点温度
    dirs = E.direction(spy, idx)
    pct, adj, _, _ = E.compose(rawdf, dirs)
    lev_pct, _ = E.leverage_monitor(rawdf)
    R["_alt_temp_252"] = A.alt_temperature(A.alt_inputs(rawdf, adj, lev_pct, spy, idx))
    R.to_csv(CACHE)
    print("缓存 →", CACHE, len(R), "行", idx[0].date(), "→", idx[-1].date())


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "build":
    build(); sys.exit()


# ---------------- 分位与合成 ----------------
import engine as E
from scipy.stats import spearmanr

ALT_W  = {"narrow": 1.0, "top2": 2.0, "lev": 3.0, "vix_abs": 2.0, "topshare": 1.0}
BLUE_W = {"top2": 1.0, "ma20": 1.0, "leverage": 1.0, "erp": 1.0}
RED_RATE  = 65 / 2241          # alt_engine.ALT_DESIGN_RATE
BLUE_RATE = 42 / 2244          # alt_engine.BLUE_TH 的对齐口径（生产数据 42 天）
WINDOWS = [63, 126, 189, 252, 378, 504, 756, 1008, 1260, "exp"]
_cache = {}


def pctl(R, col, W, variant):
    key = (col, W, variant)
    if key not in _cache:
        s = R[col]
        if W == "exp":
            _cache[key] = E.expanding_pct(s, min_periods=252)
        else:
            mp = W if (variant == "strict" or W < 252) else 252
            _cache[key] = E.rolling_pct(s, window=W, min_periods=mp)
    return _cache[key]


def red_parts(R, win, variant):
    """win: 一个窗口（全体因子同用），或 {因子: 窗口} 的字典（逐因子）"""
    g = (lambda f: win.get(f, 252)) if isinstance(win, dict) else (lambda f: win)
    sc = pctl(R, "narrow_gap", g("narrow"), variant)
    up = R["spy"].pct_change(E.NT_WIN) > 0
    P = {"narrow": sc.where(up, 50.0).where(sc.notna()),
         "top2": pctl(R, "top2", g("top2"), variant),
         "lev": (pctl(R, "lev_ratio", g("lev"), variant) + pctl(R, "lev_inten", g("lev"), variant)) / 2,
         "vix_abs": 100.0 - E.abs_map(R["vix"], (10.0, 20.0, 40.0)),
         "topshare": pctl(R, "topshare", g("topshare"), variant)}
    return P


def combine(P, Wt):
    X = pd.concat([P[k] for k in Wt], axis=1)
    v = np.array(list(Wt.values()), float); v = v / v.sum()
    return pd.Series((X.values * v).sum(axis=1), index=X.index).where(X.notna().all(axis=1))


def blue_temp(R, W, variant):
    P = {"top2": pctl(R, "top2", W, variant), "ma20": pctl(R, "ma20", W, variant),
         "leverage": pctl(R, "lev_ratio_d", W, variant),
         "erp": 100.0 - pctl(R, "erp", W, variant)}
    return combine(P, BLUE_W)


def load():
    R = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    R["fwd"] = R["qqq"].shift(-FWD) / R["qqq"] - 1.0
    return R


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "check":
    R = load()
    t = combine(red_parts(R, 252, "strict"), ALT_W)
    ref = R["_alt_temp_252"]
    m = t.notna() & ref.notna()
    d = (t[m] - ref[m]).abs()
    print("重建 vs alt_engine（252 日）：共同有效 %d 天，最大偏差 %.2e，起算 %s" %
          (m.sum(), d.max(), t.first_valid_index().date()))
    sys.exit()


# ---------------- 评估 ----------------
def persist(c, n):
    o = c.copy()
    for k in range(1, n): o[k:] &= c[:-k]
    o[:n-1] = False
    return o


def calibrate(t, rate, side, gate=None, pn=3):
    """二分找门槛，使触发天数最接近 rate × 有效天数"""
    v = t.values; ok = np.isfinite(v)
    target = rate * ok.sum()
    def n_of(th):
        c = (v > th) if side == "hot" else (v < th)
        c = c & ok
        if gate is not None: c = c & gate
        return persist(c, pn).sum()
    lo, hi = np.nanmin(v), np.nanmax(v)
    for _ in range(50):
        mid = (lo + hi) / 2
        n = n_of(mid)
        if side == "hot":
            (lo, hi) = (mid, hi) if n > target else (lo, mid)
        else:
            (lo, hi) = (lo, mid) if n > target else (mid, hi)
    return round((lo + hi) / 2, 2)


def mask(t, th, side, gate=None, pn=3):
    v = t.values
    c = ((v > th) if side == "hot" else (v < th)) & np.isfinite(v)
    if gate is not None: c = c & gate
    return persist(c, pn)


def events(m):
    i = np.flatnonzero(m)
    return 0 if len(i) == 0 else 1 + int((np.diff(i) > FWD).sum())


def tops(px):
    """下跌顶：当日是前 63 日最高、之后 63 日内最大回撤 ≥ 11%；63 日内的候选合并取最高点"""
    c = px.values; n = len(c); cand = []
    for i in range(63, n - 1):
        if c[i] >= np.nanmax(c[i-63:i+1]) and np.nanmin(c[i:min(n, i+64)]) / c[i] - 1 <= -0.11:
            cand.append(i)
    out = []
    for i in cand:
        if out and i - out[-1] <= 63:
            if c[i] > c[out[-1]]: out[-1] = i
        else:
            out.append(i)
    return out


def caught(m, tp):
    return sum(1 for p in tp if m[max(0, p-30):p+6].any())


def stat(fwd, m, base):
    if m.sum() == 0: return dict(n=0, mean=np.nan, neg=np.nan, mg=np.nan, ev=0)
    return dict(n=int(m.sum()), mean=np.nanmean(fwd[m])*100, neg=np.nanmean(fwd[m] < 0)*100,
                mg=(np.nanmean(fwd[m]) - base)*100, ev=events(m))


def scan_red(R, variant, wins=WINDOWS, label=None):
    temps = {W: combine(red_parts(R, W, variant), ALT_W) for W in wins}
    start = max(t.first_valid_index() for t in temps.values())
    S = R.loc[start:]; fwd = S["fwd"].values
    tr = (S.index >= TRAIN_FROM) & np.isfinite(fwd)
    oo = (S.index < TRAIN_FROM) & np.isfinite(fwd)
    b_tr, b_oo = np.nanmean(fwd[tr]), np.nanmean(fwd[oo])
    tp_oo = [i for i in tops(S["qqq"]) if oo[i]]
    tp_tr = [i for i in tops(S["qqq"]) if tr[i]]
    rows = []
    for W, t in temps.items():
        t = t.loc[start:]
        th = calibrate(t[tr], RED_RATE, "hot")
        m_all = mask(t, th, "hot")
        m_tr, m_oo = m_all & tr, m_all & oo
        th_oo = calibrate(t[oo], RED_RATE, "hot")
        m_oo2 = mask(t, th_oo, "hot") & oo
        ic_tr = spearmanr(t.values[tr], fwd[tr], nan_policy="omit").correlation
        ic_oo = spearmanr(t.values[oo], fwd[oo], nan_policy="omit").correlation
        rows.append(dict(W=W, th=th, tr=stat(fwd, m_tr, b_tr), oo=stat(fwd, m_oo, b_oo),
                         oo2=stat(fwd, m_oo2, b_oo), th_oo=th_oo,
                         shift=np.nanmean(t.values[oo]) - np.nanmean(t.values[tr]),
                         ic_tr=ic_tr, ic_oo=ic_oo, c_tr=caught(m_tr, tp_tr),
                         c_oo=caught(m_oo, tp_oo), c_oo2=caught(m_oo2, tp_oo),
                         aug18=int(m_all[(S.index >= "2018-08-20") & (S.index <= "2018-09-05")].sum())))
    return rows, start, dict(n_tr=int(tr.sum()), n_oo=int(oo.sum()), b_tr=b_tr*100, b_oo=b_oo*100,
                             tops_oo=len(tp_oo), tops_tr=len(tp_tr),
                             tops_oo_dates=[str(S.index[i].date()) for i in tp_oo])


def print_red(rows, start, meta, title):
    print("=" * 118)
    print("【%s】样本外 %s → 2017-10-17（%d 天，基准 %+.2f%%，下跌顶 %d 个：%s）" %
          (title, start.date(), meta["n_oo"], meta["b_oo"], meta["tops_oo"], " ".join(d[:7] for d in meta["tops_oo_dates"])))
    print("        训练窗 2017-10-18 起（%d 天，基准 %+.2f%%，下跌顶 %d 个）；门槛在训练窗按设计频率 2.90%% 重标" %
          (meta["n_tr"], meta["b_tr"], meta["tops_tr"]))
    print("%6s │%6s %-14s %8s %4s %5s │%-18s %8s %4s │%8s %4s │%7s %7s %7s" % (
        "窗口", "门槛", "训练窗 天/事件", "边际", "顶", "18-08",
        "样本外·固定门槛 天(频率)", "边际", "顶", "同频重标", "顶", "基数差", "IC训练", "IC样本外"))
    for r in rows:
        a, o, o2 = r["tr"], r["oo"], r["oo2"]
        print("%6s │%6.1f %-14s %+7.2fpp %2d  %4s │%-18s %+7.2fpp %2d  │%+7.2fpp %2d  │%+7.1f %+7.3f %+7.3f" % (
            r["W"], r["th"], "%d/%d" % (a["n"], a["ev"]), a["mg"], r["c_tr"], "%d天" % r["aug18"] if r["aug18"] else "—",
            "%d (%.2f%%)" % (o["n"], o["n"] / meta["n_oo"] * 100), o["mg"] if o["n"] else np.nan, r["c_oo"],
            o2["mg"], r["c_oo2"], r["shift"], r["ic_tr"], r["ic_oo"]))


if __name__ == "__main__" and (len(sys.argv) == 1 or sys.argv[1] == "red"):
    R = load()
    for v, title in (("minp252", "口径 minp252：最少 252 日、最多回看 W 日，全部窗口 2007-07 同日起算"),
                     ("strict",  "口径 strict：要求满窗口（engine 现行），长窗口起算晚，样本外被截短")):
        rows, start, meta = scan_red(R, v)
        print_red(rows, start, meta, title)
        json.dump({"rows": rows, "meta": meta, "start": str(start.date())},
                  open(os.path.join(HERE, "_window_red_%s.json" % v), "w"), ensure_ascii=False, default=float, indent=1)


# ---------------- 自适应门槛（空心红点的规则，无前视）----------------
AD_WIN, AD_MINP = 756, 504


def adaptive_mask(t):
    t3 = t.rolling(3, min_periods=3).min()
    th = t3.shift(1).rolling(AD_WIN, min_periods=AD_MINP).quantile(1 - RED_RATE)
    return (t3 > th).fillna(False).values, th


def ev_stats(fwd, m):
    """事件级：每个事件（间隔 > FWD 日算新事件）取其触发日前瞻收益的均值"""
    i = np.flatnonzero(m & np.isfinite(fwd))
    if len(i) == 0: return np.nan, np.nan, 0
    grp, cur = [], [i[0]]
    for a, b in zip(i[:-1], i[1:]):
        if b - a > FWD: grp.append(cur); cur = []
        cur.append(b)
    grp.append(cur)
    e = np.array([fwd[g].mean() for g in grp]) * 100
    se = e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else np.nan
    return e.mean(), se, len(e)


def scan_adaptive(R, variant, wins=WINDOWS):
    temps = {W: combine(red_parts(R, W, variant), ALT_W) for W in wins}
    ads = {W: adaptive_mask(t) for W, t in temps.items()}
    start = max(th.first_valid_index() for _, th in ads.values())
    S = R.loc[start:]; fwd = S["fwd"].values
    tr = (S.index >= TRAIN_FROM) & np.isfinite(fwd)
    oo = (S.index < TRAIN_FROM) & np.isfinite(fwd)
    b_tr, b_oo = np.nanmean(fwd[tr]), np.nanmean(fwd[oo])
    tp = tops(S["qqq"]); tp_oo = [i for i in tp if oo[i]]; tp_tr = [i for i in tp if tr[i]]
    print("=" * 108)
    print("【自适应门槛·%s】样本外 %s → 2017-10-17（%d 天，基准 %+.2f%%，下跌顶 %d 个：%s）" %
          (variant, start.date(), oo.sum(), b_oo*100, len(tp_oo), " ".join(str(S.index[i].date())[:7] for i in tp_oo)))
    print("         训练窗 2017-10-18 起（%d 天，基准 %+.2f%%，下跌顶 %d 个）" % (tr.sum(), b_tr*100, len(tp_tr)))
    print("%6s │%-10s %8s %5s %-16s %3s │%-10s %8s %5s %-16s %3s" % (
        "窗口", "样本外 天", "边际", "为负", "事件级±SE", "顶", "训练窗 天", "边际", "为负", "事件级±SE", "顶"))
    out = []
    for W in wins:
        m = ads[W][0][-len(S):]
        mo, mt = m & oo, m & tr
        eo, so, no = ev_stats(fwd - b_oo, mo)
        et, st, nt = ev_stats(fwd - b_tr, mt)
        r = dict(W=W, oo_n=int(mo.sum()), oo_mg=(np.nanmean(fwd[mo]) - b_oo)*100, oo_neg=np.nanmean(fwd[mo] < 0)*100,
                 oo_ev=eo, oo_se=so, oo_k=no, oo_top=caught(mo, tp_oo),
                 tr_n=int(mt.sum()), tr_mg=(np.nanmean(fwd[mt]) - b_tr)*100, tr_neg=np.nanmean(fwd[mt] < 0)*100,
                 tr_ev=et, tr_se=st, tr_k=nt, tr_top=caught(mt, tp_tr))
        out.append(r)
        print("%6s │%-10s %+7.2fpp %4.0f%% %-16s %3d │%-10s %+7.2fpp %4.0f%% %-16s %3d" % (
            W, "%d/%d事件" % (r["oo_n"], no), r["oo_mg"], r["oo_neg"], "%+.2f±%.2f" % (eo, so), r["oo_top"],
            "%d/%d事件" % (r["tr_n"], nt), r["tr_mg"], r["tr_neg"], "%+.2f±%.2f" % (et, st), r["tr_top"]))
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "adaptive":
    R = load()
    res = {v: scan_adaptive(R, v) for v in ("minp252", "strict")}
    json.dump(res, open(os.path.join(HERE, "_window_adaptive.json"), "w"), ensure_ascii=False, default=float, indent=1)


# ---------------- 每个窗口各自重扫权重（消除"权重是在 252 上扫的"主场优势）----------------
def refit_per_window(R, variant="minp252", wins=WINDOWS):
    import itertools
    from scan_soxx import scan as bscan
    keys = ["narrow", "top2", "lev", "vix_abs", "topshare"]
    temps0 = {W: red_parts(R, W, variant) for W in wins}
    start = max(combine(P, ALT_W).first_valid_index() for P in temps0.values())
    S = R.loc[start:]; fwd = S["fwd"].values
    tr = (S.index >= TRAIN_FROM) & np.isfinite(fwd)
    oo = (S.index < TRAIN_FROM) & np.isfinite(fwd)
    b_tr, b_oo = np.nanmean(fwd[tr]), np.nanmean(fwd[oo])
    Wg = np.array(list(itertools.product(range(5), repeat=5)), np.float32).T
    Wg = Wg[:, Wg.sum(0) > 0]
    tgt = int(round(RED_RATE * tr.sum()))
    print("=" * 112)
    print("【每个窗口各自重扫权重·%s】训练窗 %d 天、目标 %d 天；5 个因子 × 0~4 = %d 组合" %
          (variant, tr.sum(), tgt, Wg.shape[1]))
    print("%6s │%-40s %8s │%-12s %8s │%-12s %8s %s" % (
        "窗口", "训练窗最优权重（跑赢/TOP2/杠杆/VIX/前2%）", "训练边际", "样本外固定", "边际", "样本外自适应", "边际", "事件级±SE"))
    out = []
    for W in wins:
        P = {k: v.loc[start:] for k, v in temps0[W].items()}
        X = np.column_stack([P[k].values for k in keys]).astype(np.float32)
        okr = tr & np.isfinite(X).all(1)
        bs, bd, _ = bscan(X[okr], fwd[okr], Wg, "hot", np.arange(0.90, 0.99, 0.0025), tgt, 3)
        ok = (bd >= tgt*0.7) & (bd <= tgt*1.4)
        i = int(np.argmin(np.where(ok, bs, np.inf)))
        w = dict(zip(keys, Wg[:, i].tolist()))
        t = combine(P, {k: v for k, v in w.items() if v > 0})
        th = calibrate(t[tr], RED_RATE, "hot")
        mf = mask(t, th, "hot")
        ma, _ = adaptive_mask(t)
        ma = ma & oo
        e, se, k = ev_stats(fwd - b_oo, ma)
        r = dict(W=W, w=w, th=th, tr_mg=(np.nanmean(fwd[mf & tr]) - b_tr)*100,
                 oo_fix_n=int((mf & oo).sum()), oo_fix_mg=(np.nanmean(fwd[mf & oo]) - b_oo)*100 if (mf & oo).sum() else np.nan,
                 oo_ad_n=int(ma.sum()), oo_ad_mg=(np.nanmean(fwd[ma]) - b_oo)*100, oo_ev=e, oo_se=se)
        out.append(r)
        s = sum(w.values())
        print("%6s │%-40s %+7.2fpp │%-12s %+7.2fpp │%-12s %+7.2fpp %+.2f±%.2f" % (
            W, " / ".join("%2.0f%%" % (w[k]/s*100) for k in keys), r["tr_mg"],
            "%d天" % r["oo_fix_n"], r["oo_fix_mg"], "%d天/%d事件" % (r["oo_ad_n"], k), r["oo_ad_mg"], e, se))
    return out


def scan_blue(R, variant="minp252", wins=WINDOWS):
    temps = {W: blue_temp(R, W, variant) for W in wins}
    start = max(t.first_valid_index() for t in temps.values())
    S = R.loc[start:]; fwd = S["fwd"].values
    gate = (S["vix"].values >= 30)
    tr = (S.index >= TRAIN_FROM) & np.isfinite(fwd)
    oo = (S.index < TRAIN_FROM) & np.isfinite(fwd)
    b_tr, b_oo = np.nanmean(fwd[tr]), np.nanmean(fwd[oo])
    g_tr, g_oo = np.nanmean(fwd[gate & tr]), np.nanmean(fwd[gate & oo])
    print("=" * 112)
    print("【蓝点·%s】样本外 %s → 2017-10-17（%d 天，基准 %+.2f%%；VIX≥30 共 %d 天、只用闸门 %+.2f%%）" %
          (variant, start.date(), oo.sum(), b_oo*100, (gate & oo).sum(), g_oo*100))
    print("         训练窗（%d 天，基准 %+.2f%%；VIX≥30 共 %d 天、只用闸门 %+.2f%%）；门槛在训练窗按 42/2244 重标" %
          (tr.sum(), b_tr*100, (gate & tr).sum(), g_tr*100))
    print("%6s │%6s │%-10s %8s %5s │%-10s %8s %5s %-14s │%7s" % (
        "窗口", "门槛", "训练 天/事件", "后30日", "为负", "样本外 天/事件", "后30日", "为负", "事件级±SE", "基数差"))
    out = []
    for W, t in temps.items():
        t = t.loc[start:]
        th = calibrate(t[tr], BLUE_RATE, "cold", gate[tr], pn=1)
        m = mask(t, th, "cold", gate, pn=1)
        mt, mo = m & tr, m & oo
        e, se, k = ev_stats(fwd, mo)
        r = dict(W=W, th=th, tr_n=int(mt.sum()), tr_ev=events(mt), tr=np.nanmean(fwd[mt])*100, tr_neg=np.nanmean(fwd[mt] < 0)*100,
                 oo_n=int(mo.sum()), oo_ev=k, oo=np.nanmean(fwd[mo])*100 if mo.sum() else np.nan,
                 oo_neg=np.nanmean(fwd[mo] < 0)*100 if mo.sum() else np.nan, oo_e=e, oo_se=se,
                 shift=np.nanmean(t.values[oo]) - np.nanmean(t.values[tr]))
        out.append(r)
        print("%6s │%6.1f │%-10s %+7.2f%% %4.0f%% │%-10s %+7.2f%% %4.0f%% %-14s │%+7.1f" % (
            W, th, "%d/%d" % (r["tr_n"], r["tr_ev"]), r["tr"], r["tr_neg"],
            "%d/%d" % (r["oo_n"], k), r["oo"], r["oo_neg"], "%+.2f±%.2f" % (e, se), r["shift"]))
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "more":
    R = load()
    res = {"refit": refit_per_window(R), "blue_minp": scan_blue(R, "minp252"), "blue_strict": scan_blue(R, "strict")}
    json.dump(res, open(os.path.join(HERE, "_window_more.json"), "w"), ensure_ascii=False, default=float, indent=1)
