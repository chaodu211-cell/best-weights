# -*- coding: utf-8 -*-
"""替代口径的红点：与 engine.SELL_W 不同的一套权重，单独成页，不影响生产流水线。

—— 这套权重是怎么来的（2026-09-18 重标）——
在训练窗（2017-10 至今）上把**12 个最小独立因子**当自由参数扫，判据是**真实规则**
（温度 > 门槛连 3 日，门槛对齐到 66 天）触发后 **30 个交易日** QQQ 的平均收益。
两段式：先用 0/1/2 三档扫全部 12 个因子（531441 个组合）定形状，再对入选的
6 个因子用 0~4 细扫（14833 个组合）。最优解：

    市值跑赢等权 11% · TOP2抱团 22% · 杠杆多空比 17% · 杠杆交易强度 17%
    · VIX绝对刻度 22% · 前2%成交额个股占比 11%

落到本文件里时**没有把因子拆开**（保住上涨拥挤度"只在上涨市成立、下跌市填 50"
那道语义门闸），而是把扫描结果折进两个复合量的内部比例：
  · 上涨拥挤度 = 市值跑赢等权 1 : 宽度恶化 0 : 贴近峰值 0（后两个扫出来就是 0）
  · 杠杆温度   = 多空比 1 : 交易强度 1  ←  **不是 leverage.py 的 3:1**
实测门闸加不加**逐日完全相同**（拥挤度只占 11% 权重，下跌市那些天离门槛太远，
门闸从没在触发边缘生效过），所以保留它零成本。

实测（raw_long，训练窗 2017-10 起，前瞻 30 日，基准 +2.46%／32%为负）：
    本口径（杠杆1:1）  门槛 81.7  65天/12段/7事件  -9.16%／94%为负  边际 -11.62pp
    同权重但杠杆3:1    门槛 82.9  66天/14段/9事件  -6.51%／74%为负  边际  -8.98pp
    上一版（63日标定） 门槛 83.0  67天/18段/8事件  -5.33%／73%为负  边际  -7.79pp
也就是说，光把杠杆内部从 3:1 改成 1:1 就值 2.65pp——扫描里杠杆交易强度的权重
秩相关是 -0.475（12 个因子里最强），多空比只有 -0.017（几乎无影响），
而 leverage.py 把 75% 的权重给了多空比，方向正好反过来。

【必读】这套权重**只在训练窗上标定过**，是用户明确要求"只考虑训练窗"下的产物。
留出集（2011-2017）没有参与筛选，也没有为它背书；同类扫描里训练窗前 50 名在
留出集上的边际是 +0.04pp（≈零）。别把上面的数字当成前瞻预期。
完整方法与反面证据见 `过拟合风险评估.md` 第七节。

与生产红点（engine.SELL_W）的差别：
  · 删掉 站上MA20、换手率（扫描里这两格加权重只会变差：秩相关 +0.317 / -0.020）
  · 删掉 上涨拥挤度里的 宽度恶化 与 贴近峰值 两项
  · 加入 VIX 绝对刻度 与 前2%成交额个股占比
  · 杠杆温度权重 2/9 → 3/9，且内部比例 3:1 → 1:1
"""
import numpy as np
import pandas as pd
import engine as E

# 检验前瞻窗口：30 个交易日（2026-09-18 起，此前仓库各处用的是 63）
FWD = 30

# 上涨拥挤度内部：市值跑赢等权 : 宽度恶化 : 贴近峰值
ALT_LEGS = (1.0, 0.0, 0.0)
# 杠杆温度内部：多空比 : 交易强度（leverage.py 锁死的是 3:1，这里按扫描结果取 1:1）
LEV_INNER = (1.0, 1.0)
# 五个格子的权重（归一后 = 拥挤度11% / TOP2 22% / 杠杆33% / VIX 22% / 前2%占比11%）
ALT_W = {"narrow": 1.0, "top2": 2.0, "lev": 3.0, "vix_abs": 2.0, "topshare": 1.0}
# 门槛：训练窗内对齐到 66 天（与生产红点同频）。加门闸与不加门闸的最优门槛都是 81.7。
ALT_TH = 81.7
ALT_PERSIST = 3
# VIX 绝对刻度的锚点：10→最热(100)，20→中性(50)，40→最冷(0)。
# 用固定锚点而不是滚动分位是实测结论；30 日口径下 VIX 是 12 个因子里秩相关最强的
# 一个（-0.177），而 63 日口径下它几乎无用（-0.074）——它量的本来就是眼前的恐慌。
VIX_ANCHORS = (10.0, 20.0, 40.0)

LABELS = {"narrow": "市值加权跑赢等权", "top2": "TOP2抱团", "lev": "杠杆温度",
          "vix_abs": "VIX绝对刻度", "topshare": "前2%成交额占比"}


def alt_inputs(rawdf, adj, lev_pct, spy, idx):
    """返回替代红点的五个分项（都是 0-100，与主口径同尺）。

    lev_pct: engine.leverage_monitor 返回的第一项（含 ratio / intensity 两列分位）。
    注意这里**不用**它返回的 lev_temp——那个是按 leverage.py 的 3:1 合成的。
    """
    cw = spy["close"].reindex(idx) if spy is not None else None
    nq = E.load("QQQ")
    nq = nq["close"].reindex(idx) if nq is not None else cw
    out = {}

    # 上涨拥挤度：内部只剩"市值跑赢等权"，但保留上涨市门闸与下跌市填 50 的语义
    if cw is not None and nq is not None:
        px = pd.DataFrame({t: d["close"] for t, d in E.load_many(E.STOCKS).items()}).reindex(idx)
        eq = (1.0 + px.pct_change().mean(axis=1).fillna(0)).cumprod()
        gap = (cw / cw.shift(E.NT_WIN)) / (eq / eq.shift(E.NT_WIN)) - 1.0
        dd = (nq / nq.cummax() - 1.0) * 100.0
        near = ((dd + E.NT_DD) / E.NT_DD * 100.0).clip(0, 100)
        brd = E.rolling_pct(-(rawdf["ma20"] - rawdf["ma20"].shift(E.NT_WIN)))
        w1, w2, w3 = ALT_LEGS
        sc = (w1 * E.rolling_pct(gap) + w2 * brd + w3 * near) / (w1 + w2 + w3)
        out["narrow"] = sc.where(cw.pct_change(E.NT_WIN) > 0, 50.0).where(sc.notna())

    out["top2"] = adj.get("top2")
    if lev_pct is not None and {"ratio", "intensity"} <= set(lev_pct.columns):
        a, b = LEV_INNER
        out["lev"] = (a * lev_pct["ratio"] + b * lev_pct["intensity"]) / (a + b)
    vix = E.load_vix(idx)
    if vix is not None:
        out["vix_abs"] = 100.0 - E.abs_map(vix, VIX_ANCHORS)   # VIX 低 = 麻木 = 热
    if "_topshare" in rawdf.columns:
        out["topshare"] = E.rolling_pct(rawdf["_topshare"])
    return out


def alt_temperature(parts):
    """按 ALT_W 加权合成。任一分项缺失则当日无读数——与 compose_sell 同样的态度：
    宁可不出，也不用半套输入产出一个看似正常的数。"""
    keys = [k for k in ALT_W if k in parts and parts[k] is not None]
    if len(keys) != len(ALT_W):
        return None
    X = pd.concat([parts[k] for k in keys], axis=1)
    w = np.array([ALT_W[k] for k in keys], dtype=float)
    w = w / w.sum()
    t = pd.Series((X.values * w).sum(axis=1), index=X.index)
    return t.where(X.notna().all(axis=1))


# ---------- 蓝点：四项等权（2026-09-18） ----------
# 原本是六项等权（换手率/TOP2/上涨占比/站上MA20/杠杆多空比/ERP 各 1/6）。
# 2026-09-18 在训练窗上把 9 个候选因子 × 权重 0~3 扫了 261121 个组合，判据是真实规则
# （温度 < 门槛 且 VIX >= 30）触发后 30 日收益。扫描的最优是
# 「TOP2 1 / 站上MA20 1 / 杠杆多空比 3 / ERP 1」（多空比占 50%），但拆开看，改进的
# 四分之三来自**删掉两个因子**，而不是给多空比加权（SOXX 后 30 日，门槛每步重标到 42 天）：
#     六项等权（原）          +12.53% / 38%为负
#     只删换手率与上涨占比      +16.11% / 29%为负   ← +3.58pp，本文件采用这一步
#     再把多空比加到 3         +17.18% / 27%为负   ← 只再值 +1.07pp
#     对照：加 ERP 而不是多空比  +12.70% / 33%为负
# 那 +1.07pp 说不清：多空比单独用在 VIX>=30 的日子里，对后 30 日收益的秩相关只有
# -0.041（近零），而 ERP 是 -0.336；全网格上 ERP 的权重秩相关 +0.764 压倒性第一，
# 多空比只有 +0.158。"局部最优给多空比 50%"与"网格整体说该加 ERP"互相矛盾，
# 更像是把 2020-03 的入场点往后推了几天的局部拟合。故只取有机制解释的那一步。
#
# 为什么删这两项（这是有机制的部分）：
#   · 换手率——恐慌与狂热都会让成交额暴增，方向不可知；DIR_HALF 又让低于常态的部分
#     一律记 50，在暴跌里经常直接失声。
#   · 上涨个股占比——VIX >= 30 的日子里它几乎必然极低，等于常数，区分不出
#     "恐慌到位"与"恐慌刚开始"。
# 留下的四项覆盖两个维度：定位（杠杆多空比、站上MA20、TOP2抱团）与估值（ERP）。
#
# 【VIX >= 30 仍是独立闸门，不进温度】它是外生的绝对刻度，不是标出来的。
# 实测只用这道闸门就有 +6.2pp 边际，温度筛选再加约 5.6pp。
#
# 【必读】删因子这一步同样只在训练窗上验证过。按独立事件算（42 天只有 6~7 个事件），
# 标准误约 4.1pp，+3.58pp 仍在噪声量级内。留着这段话，别把它当前瞻预期。
BLUE_W = {"top2": 1.0, "ma20": 1.0, "leverage": 1.0, "erp": 1.0}   # 各 25%
BLUE_TH = 21.8   # 对齐到现行蓝点在生产数据上的触发天数（42 天）后重标；原六项等权口径是 25
BLUE_LABELS = {"top2": "TOP2行业成交额占比", "ma20": "站上MA20占比",
               "leverage": "杠杆资金多空比", "erp": "风险溢价ERP"}


def blue_temperature(adj_like):
    """按 BLUE_W 合成蓝点温度。传当日口径（adjf）得到判定用的那条，
    传平滑口径（adj）得到页面展示的那条。任一分项缺失则当日无读数。"""
    ks = [k for k in BLUE_W if k in adj_like.columns]
    if len(ks) != len(BLUE_W):
        return None
    X = pd.concat([adj_like[k] for k in ks], axis=1)
    w = np.array([BLUE_W[k] for k in ks], dtype=float)
    w = w / w.sum()
    t = pd.Series((X.values * w).sum(axis=1), index=X.index)
    return t.where(X.notna().all(axis=1))
