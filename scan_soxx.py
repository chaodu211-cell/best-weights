# -*- coding: utf-8 -*-
"""把 12 个最小独立因子对 **SOXX** 做最优拟合：红点（卖）与蓝点（买）各扫一套权重。

口径与 alt_engine 的重标完全对齐，只换了拟合标的（QQQ → SOXX）：
  · 判据是**真实规则**而不是秩相关：红点 = 温度 > 门槛连 3 日；蓝点 = 温度 < 门槛且 VIX ≥ 30。
  · 门槛不是自由参数：每组权重都把门槛对齐到同一个触发天数（红点 66 天 / 蓝点 30 天），
    否则权重一变分布就变，比的就不是权重而是松紧。
  · 目标函数 = 触发日之后 30 个交易日 SOXX 的平均收益（红点求最小，蓝点求最大）。

**VIX ≥ 30 是蓝点的独立判别**，不进权重向量：它是外生的绝对刻度（恐慌到位与否），
不是标定出来的东西。蓝点扫描因此只扫 11 个因子，VIX 那一格以硬闸门形式挂在外面。

十二个因子（全部 0-100、同向，高 = 热/贪婪，与 engine 共用一把尺子）：
    engine.ORDER 六项  换手率 / TOP2抱团 / 上涨个股占比 / 站上MA20 / 杠杆多空比 / 风险溢价ERP
    杠杆第二条腿      杠杆ETF交易强度
    上涨拥挤度三条腿   市值跑赢等权 / 宽度恶化 / 贴近峰值
    另两项            VIX绝对刻度（锚点 10/20/40）/ 前2%成交额个股占比

用法：
    python3 scan_soxx.py factors    # 只重建因子矩阵（约 10s，缓存到 _soxx_factors.csv）
    python3 scan_soxx.py red        # 红点：3^12 粗扫 + 细扫
    python3 scan_soxx.py blue       # 蓝点：3^11 粗扫 + 细扫
    python3 scan_soxx.py oos        # 分半互测（权重到底有没有可迁移性）
    python3 scan_soxx.py final      # 候选权重的完整对照表
    python3 scan_soxx.py all
"""
import os, sys, time, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "_soxx_factors.csv")

FWD    = 30     # 前瞻窗口（交易日），与 alt_engine.FWD 一致
TGT_H  = 66     # 红点的触发天数预算（与生产红点同频）
TGT_C  = 30     # 蓝点的触发天数预算（现行蓝点在生产数据上是 29~42 天量级）
VIX_TH = 30     # 蓝点的独立闸门

NAMES = ["turnover","top2","advancing","ma20","lev_ratio","erp","lev_inten",
         "narrow_gap","brd_worse","near_peak","vix_abs","topshare"]
CN = {"turnover":"换手率","top2":"TOP2抱团","advancing":"上涨个股占比","ma20":"站上MA20",
      "lev_ratio":"杠杆多空比","erp":"风险溢价ERP","lev_inten":"杠杆交易强度",
      "narrow_gap":"市值跑赢等权","brd_worse":"宽度恶化","near_peak":"贴近峰值",
      "vix_abs":"VIX绝对刻度","topshare":"前2%成交额占比"}
# 蓝点判定走**当日口径**（换手率/上涨占比/杠杆两腿取未平滑值），与生产蓝点一致
FASTCOL = {"turnover":"turnover_f","advancing":"advancing_f",
           "lev_ratio":"lev_ratio_f","lev_inten":"lev_inten_f"}


# ---------------- 因子矩阵 ----------------
def build_factors():
    os.chdir(HERE)
    import engine as E
    rawdf, meta, spy, lev_info = E.build_indicators()
    idx = rawdf.index
    dirs = E.direction(spy, idx)
    pct, adj, _, _ = E.compose(rawdf, dirs)
    lev_pct, _ = E.leverage_monitor(rawdf)
    vix = E.load_vix(idx)

    F = pd.DataFrame(index=idx)
    for k, c in (("turnover","turnover"),("top2","top2"),("advancing","advancing"),
                 ("ma20","ma20"),("lev_ratio","leverage"),("erp","erp")):
        F[k] = adj[c]
    F["lev_inten"]  = lev_pct["intensity"]
    # 上涨拥挤度拆成三条独立腿（不带"只在上涨市成立"的门闸，门闸最后再加回；
    # 扫描要的是最小独立因子，语义门闸是落地时的事）
    F["narrow_gap"] = E.rolling_pct(rawdf["_narrow_gap"])
    F["brd_worse"]  = E.rolling_pct(-rawdf["_breadth_chg"])
    F["near_peak"]  = ((rawdf["_ndx_dd"] + E.NT_DD) / E.NT_DD * 100.0).clip(0, 100)
    F["vix_abs"]    = 100.0 - E.abs_map(vix, (10.0, 20.0, 40.0))
    F["topshare"]   = E.rolling_pct(rawdf["_topshare"])

    rf = rawdf.copy()
    for k, v in {"turnover":"_turnover_daily","advancing":"_advancing_daily",
                 "leverage":"_leverage_daily"}.items():
        rf[k] = rawdf[v]
    _, adjf, _, _ = E.compose(rf, dirs, fast=True)
    F["turnover_f"], F["advancing_f"], F["lev_ratio_f"] = adjf["turnover"], adjf["advancing"], adjf["leverage"]
    F["lev_inten_f"] = E.rolling_pct(rawdf["_lev_intensity_daily"])

    soxx, qqq = E.load("SOXX"), E.load("QQQ")
    F["soxx"] = soxx["close"].reindex(idx)
    F["qqq"]  = qqq["close"].reindex(idx)
    F["vix"]  = vix
    F.to_csv(CACHE)
    print("因子矩阵已写入", CACHE, len(F), "行")
    return F


def load():
    if not os.path.exists(CACHE):
        build_factors()
    F = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    F["fwd"] = F["soxx"].shift(-FWD) / F["soxx"] - 1.0
    return F.dropna(subset=NAMES + ["fwd", "vix"])


# ---------------- 批量扫描 ----------------
def _persist3(c):
    o = c.copy(); o[2:] &= c[1:-1] & c[:-2]; o[:2] = False; return o


def scan(X, fwd, W, side, levels, target, persist_n=3, gate=None, batch=4096):
    """X:(N,K) 因子矩阵, W:(K,M) 未归一权重。每列在 levels 网格里挑触发天数最接近
    target 的门槛——这就是"门槛不是自由参数"的实现。返回 (score, days, q)。"""
    N = X.shape[0]; M = W.shape[1]
    f32 = fwd.astype(np.float32)
    bs, bd, bq = np.zeros(M, np.float32), np.zeros(M, np.int32), np.zeros(M, np.float32)
    ki = [int(q * N) for q in levels]
    for s in range(0, M, batch):
        w = W[:, s:s+batch].astype(np.float32); w = w / w.sum(axis=0, keepdims=True)
        T = X @ w
        S = np.sort(T, axis=0)
        bestd = np.full(T.shape[1], 10**9, np.int64)
        for q, k in zip(levels, ki):
            th = S[k]
            c = (T > th) if side == "hot" else (T < th)
            if gate is not None: c &= gate[:, None]
            m = _persist3(c) if persist_n >= 3 else c
            days = m.sum(0).astype(np.int64)
            sc = np.where(days > 0, (f32 @ m.astype(np.float32)) / np.maximum(days, 1), np.nan)
            take = (np.abs(days - target) < bestd) & (days > 0)
            bestd = np.where(take, np.abs(days - target), bestd)
            bs[s:s+batch] = np.where(take, sc, bs[s:s+batch])
            bd[s:s+batch] = np.where(take, days, bd[s:s+batch])
            bq[s:s+batch] = np.where(take, q, bq[s:s+batch])
    return bs, bd, bq


def _grid(cols, F, fast):
    return F[[FASTCOL.get(c, c) if fast else c for c in cols]].values.astype(np.float32)


def sweep(F, cols, side, target, persist_n, gate, steps, levels, fast=False, top=15, title=""):
    X = _grid(cols, F, fast); fwd = F["fwd"].values
    W = np.array(list(itertools.product(steps, repeat=len(cols))), np.int8).T
    W = W[:, W.sum(0) > 0]
    t0 = time.time()
    bs, bd, bq = scan(X, fwd, W.astype(np.float32), side, levels, target, persist_n, gate)
    ok = (bd >= target * 0.7) & (bd <= target * 1.4)
    sc = np.where(ok, bs, np.inf if side == "hot" else -np.inf)
    order = np.argsort(sc if side == "hot" else -sc)
    print("=== %s ===  %d 组合 / %.0fs" % (title, W.shape[1], time.time() - t0))
    for i in order[:top]:
        nw = W[:, i] / W[:, i].sum() * 100
        print("  %+.2f%% %3d天 q=%.3f  " % (bs[i]*100, bd[i], bq[i]) +
              " ".join("%s %.0f%%" % (CN[c], v) for c, v in zip(cols, nw) if v > 0))
    o2 = order[:2000]
    print("  —— 前 2000 名的因子入选率 ——")
    for k, c in enumerate(cols):
        print("    %-12s 均权 %.2f  出现 %.0f%%" % (CN[c], W[k, o2].mean(), (W[k, o2] > 0).mean()*100))
    return W, bs, bd, bq, order


# ---------------- 单组权重的完整体检 ----------------
def temp_of(F, w, fast=False):
    cols = [FASTCOL.get(k, k) if fast else k for k in w]
    v = np.array(list(w.values()), float); v = v / v.sum()
    return pd.Series(F[cols].values @ v, index=F.index)


def _mask(t, th, side, gate, persist_n):
    c = (t.values > th) if side == "hot" else (t.values < th)
    if gate is not None: c = c & gate
    if persist_n > 1:
        o = c.copy()
        for k in range(1, persist_n): o[k:] &= c[:-k]
        o[:persist_n-1] = False
        c = o
    return c


def calibrate(t, target, side, gate, persist_n):
    lo, hi = (0.80, 0.999) if side == "hot" else (0.005, 0.30)
    best = None
    for q in np.arange(lo, hi, 0.0005):
        th = float(np.quantile(t.values, q))
        n = int(_mask(t, th, side, gate, persist_n).sum())
        if best is None or abs(n - target) < best[0]: best = (abs(n - target), th, n)
    return best[1], best[2]


def report(F, name, w, side, target, persist_n=3, gate=None, fast=False, dates=False):
    t = temp_of(F, w, fast)
    th, n = calibrate(t, target, side, gate, persist_n)
    m = _mask(t, th, side, gate, persist_n)
    fwd = F["fwd"].values; base = fwd.mean(); h = len(F) // 2
    i = np.flatnonzero(m); ev = []
    if len(i):
        cur = [i[0]]
        for a, b in zip(i[:-1], i[1:]):
            if b - a > FWD: ev.append(cur); cur = []
            cur.append(b)
        ev.append(cur)
    evm = [fwd[g].mean() for g in ev]
    m1 = m.copy(); m1[h:] = False
    m2 = m.copy(); m2[:h] = False
    print("── %s ──" % name)
    print("  权重 " + " / ".join("%s %.0f%%" % (CN[k], v/sum(w.values())*100) for k, v in w.items()))
    print("  门槛 %.1f（温度 %s，%s）  %d天 / %d事件" %
          (th, ">" if side == "hot" else "<",
           "连%d日" % persist_n if persist_n > 1 else "当日即触发", n, len(ev)))
    print("  后%d日 SOXX %+.2f%% / 为负 %.0f%% / 边际 %+.2fpp（基准 %+.2f%%）" %
          (FWD, fwd[m].mean()*100, (fwd[m] < 0).mean()*100, (fwd[m].mean()-base)*100, base*100))
    print("  逐事件 " + " ".join("%+.1f%%" % (x*100) for x in evm) +
          ("   事件级均值 %+.2f%% ± %.2fpp" % (np.mean(evm)*100,
           np.std(evm, ddof=1)/np.sqrt(len(evm))*100) if len(evm) > 1 else ""))
    print("  前半 %d天 %+.2f%%（基准%+.2f%%） ｜ 后半 %d天 %+.2f%%（基准%+.2f%%）" %
          (m1.sum(), fwd[m1].mean()*100, fwd[:h].mean()*100,
           m2.sum(), fwd[m2].mean()*100, fwd[h:].mean()*100))
    if dates: print("  触发日 " + ", ".join(str(x.date()) for x in F.index[m]))
    print()
    return th


# ---------------- 各阶段 ----------------
LV_HOT  = np.arange(0.90, 0.9651, 0.0025)
LV_COLD = np.arange(0.02, 0.42, 0.0025)
RED_CORE  = ["top2","lev_ratio","erp","lev_inten","narrow_gap","vix_abs"]
BLUE_CORE = ["top2","ma20","lev_ratio","erp"]
BLUE_ALL  = [n for n in NAMES if n != "vix_abs"]     # VIX 是独立闸门，不进权重

# 候选权重：扫描最优、推荐落地版、以及作为对照的现行生产权重
RED_CANDIDATES = [
    ("红点·扫描最优(6因子)",   dict(top2=3, lev_ratio=4, erp=1, lev_inten=3, narrow_gap=4, vix_abs=3)),
    ("红点·推荐(现行alt−前2%+ERP)", dict(narrow_gap=2, top2=4, lev_ratio=3, lev_inten=3, vix_abs=4, erp=1)),
    ("红点·现行alt(QQQ标定)",  dict(narrow_gap=2, top2=4, lev_ratio=3, lev_inten=3, vix_abs=4, topshare=2)),
]
BLUE_CANDIDATES = [
    ("蓝点·扫描最优(4因子)",   dict(top2=1, ma20=3, lev_ratio=3, erp=1)),
    ("蓝点·现行BLUE_W",       dict(top2=1, ma20=1, lev_ratio=3, erp=1)),
    ("蓝点·六项等权(老口径)",  dict(turnover=1, top2=1, advancing=1, ma20=1, lev_ratio=1, erp=1)),
]


def main(stage):
    if stage == "factors":
        build_factors(); return
    F = load(); gate = (F["vix"].values >= VIX_TH)
    print("样本 %s → %s  %d 天   SOXX 后%d日基准 %+.2f%% / 为负 %.0f%%\n" %
          (F.index[0].date(), F.index[-1].date(), len(F), FWD,
           F["fwd"].mean()*100, (F["fwd"] < 0).mean()*100))
    if stage in ("red", "all"):
        sweep(F, NAMES, "hot", TGT_H, 3, None, [0,1,2], LV_HOT, title="红点粗扫 3^12")
        sweep(F, RED_CORE, "hot", TGT_H, 3, None, range(5), LV_HOT, title="红点细扫 0~4")
    if stage in ("blue", "all"):
        sweep(F, BLUE_ALL, "cold", TGT_C, 1, gate, [0,1,2], LV_COLD, fast=True, title="蓝点粗扫 3^11（VIX≥30 为独立闸门）")
        sweep(F, BLUE_CORE, "cold", TGT_C, 1, gate, range(5), LV_COLD, fast=True, title="蓝点细扫 0~4")
    if stage in ("oos", "all"):
        oos(F, gate)
    if stage in ("final", "all"):
        fwd = F["fwd"].values
        print("只用 VIX≥%d 闸门：%d天 %+.2f%% / 边际 %+.2fpp\n" %
              (VIX_TH, gate.sum(), fwd[gate].mean()*100, (fwd[gate].mean()-fwd.mean())*100))
        for n, w in RED_CANDIDATES:  report(F, n, w, "hot", TGT_H, 3, None, False, dates=True)
        for n, w in BLUE_CANDIDATES: report(F, n, w, "cold", TGT_C, 1, gate, True, dates=True)


def oos(F, gate):
    """分半互测：在一半上选出的最优权重，拿到另一半去，跟"随便挑一组权重"比。
    这是整份分析里唯一能回答"权重值不值得标"的检验。"""
    fwd = F["fwd"].values; h = len(F)//2
    for tag, cols, side, tgt, pn, g, fast, lv in (
            ("红点", NAMES,    "hot",  33, 3, None, False, np.arange(0.88, 0.975, 0.005)),
            ("蓝点", BLUE_ALL, "cold", 15, 1, gate, True,  np.arange(0.02, 0.45, 0.005))):
        X = _grid(cols, F, fast)
        W = np.array(list(itertools.product([0,1,2], repeat=len(cols))), np.int8).T
        W = W[:, W.sum(0) > 0].astype(np.float32)
        out = []
        for sl in (slice(0, h), slice(h, len(F))):
            bs, bd, _ = scan(X[sl], fwd[sl], W, side, lv, tgt, pn, None if g is None else g[sl])
            ok = (bd >= tgt*0.55) & (bd <= tgt*1.7)
            out.append(np.where(ok, bs, np.inf if side == "hot" else -np.inf))
        a, b = out; b1, b2 = fwd[:h].mean(), fwd[h:].mean()
        sgn = 1 if side == "hot" else -1
        print("【%s 分半互测】前半基准 %+.2f%% / 后半基准 %+.2f%%" % (tag, b1*100, b2*100))
        for src, dst, bd_ in ((a, b, b2), (b, a, b1)):
            lbl = "前半→后半" if src is a else "后半→前半"
            med = np.nanmedian(dst[np.isfinite(dst)])
            for k in (1, 10, 50, 200):
                o = np.argsort(sgn*src)[:k]
                print("  %s 前%3d名 → 边际 %+.2fpp （该半全部组合的中位边际 %+.2fpp）" %
                      (lbl, k, (np.nanmean(dst[o])-bd_)*100, (med-bd_)*100))
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "final")
