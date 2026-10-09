# -*- coding: utf-8 -*-
"""最优拟合口径的红点：与 engine.SELL_W 不同的一套权重，单独成页，不影响生产流水线。

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
import json
import os
import numpy as np
import pandas as pd
import engine as E

# 检验前瞻窗口：30 个交易日（2026-09-18 起，此前仓库各处用的是 63）
FWD = 30

# 上涨拥挤度内部：市值跑赢等权 : 宽度恶化 : 贴近峰值
ALT_LEGS = (1.0, 0.0, 0.0)
# 「市值跑赢等权」的等权一侧：真实等权 ETF 的复权价（2026-10-01 起）。原先用今天的 503 只成分股自建、
# 回填全部历史，有幸存者偏差（2005-2026 年化 16.4%，RSP 只有 10.0%），成分股每次调整还会改写历史读数。
# 同口径（63 日、下跌市记 50）下这个因子读数 ≥90 之后 30 日 QQQ 的热端边际（2008-16 / 2017-26）：
# 自建 −1.17 / −1.13pp，RSP −2.61 / −2.34pp，官方 ^SP500EW −2.37 / −2.62pp。
# 取 RSP 而不是 ^SP500EW：效果相同，RSP 2003 年起就有、与 SPY 同为复权（含分红）口径，且走同一个数据源。
EW_SYM = "RSP"
RAW_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
# 杠杆温度内部：多空比 : 交易强度（leverage.py 锁死的是 3:1，这里按扫描结果取 1:1）
# 2026-09-30 起两项的原始量并入「正股成交额前十」的单股杠杆 ETF，之前的读数不动——见 lev_single()、single_lev.py；
# 蓝点用的杠杆多空比同日起也换成同一口径（blue_lev_splice）
LEV_INNER = (1.0, 1.0)
# 五个格子的权重（归一后 = 拥挤度11% / TOP2 22% / 杠杆33% / VIX 22% / 前2%占比11%）
#
# —— 2026-09-20 曾把抱团类减半，2026-09-21 已全部回退，原因记在这里 ——
# 减半版（TOP2 13%、前2% 7%、门槛 81.6）的出发点是："若 AI 集中行情延续，抱团水平
# 变成常态就不再含信息"。2023 年后抱团的秩相关确实塌到 -0.002±0.189，看着像失效。
# 回退的三条理由，按说服力排序：
#
# 1. **它砍掉的正是这轮行情最可能的收尾形态。** 2018-08-27~30 是全样本里最典型的
#    极致抱团顶（TOP2 93~99.8、前2% 81~98.2 同时顶格，触发后 30 日 QQQ -6.8/-9.0/
#    -6.3%）。减半后 08-29 温度 81.5，比门槛低 **0.1**，三日连续链从中间断开，整段消失。
#    若真如假设那样"集中度继续升高"，这类顶只会更可能出现——在需要探头的前夜拆探头。
# 2. **33% 是扫描出来的峰值，不是随手落的点。** 内核固定、抱团按 TOP2:前2%=2:1 从 0
#    扫到 53%（门槛每档重标到 66 天）：0% -5.47% / 11% -6.49% / 20% -7.68% /
#    27% -8.77% / **33% -9.02%** / 38% -8.86% / 50% -7.93%。两侧都下滑，峰在 33%。
#    而且抱团要 ≥27% 才开始捞到 2018-08，≥33% 才完整捞到。
# 3. **"抱团失效"的证据只有 1.4σ**（2025-2026 段 -0.002±0.189 vs 2023-2024 的
#    -0.311±0.109，该段只相当于 13 个互不重叠的 30 日区间）。同一批数据同样支持另一种
#    读法：集中度一路在升但尚未兑现，所以统计上看不出预测力——那不等于指标坏了。
#    佐证：2025-2026 段 TOP2 的离散度是全样本最高的 32.2，不是被钉住，是波动了没兑现。
#
# 两个附带结论（2026-09-21 实测，别再重试）：
#   · **CROWD_TOP_FRAC = 0.02 是最优**。前 0.5/1/2/3/5/10% 六档，门槛各自重标到 66 天：
#     -7.54 / -8.13 / **-9.02** / -8.80 / -8.56 / -8.79%。1%~10% 是一片平台，2% 是峰。
#   · **抱团两项都不要平滑**。前2% 取 MA5：秩相关从 -0.084 改善到 -0.106，实际成绩却从
#     -9.02% 掉到 -7.69%，2018-08 从 4 天削到 2 天；六个百分比档位无一例外。TOP2 取
#     MA3/5/10 同样：秩相关 -0.134/-0.142/-0.154 一路变好，成绩 -7.23/-7.36/-7.42%
#     一路变差。抱团见顶是**尖峰事件**，平滑正好削在信号最需要分辨力的地方。
#     这是"别拿秩相关去挑闸门因子"（见 engine.py）的又一个实例。
#
# 什么情况下才重新考虑减半：抱团的秩相关在**独立区间数 ≥20** 的样本上仍然接近零
# （即再观察 6~9 个月），并且那期间出现过至少一次抱团顶而红点没响。
# 完整论证与这次往返的全过程见 `权重改动记录.md`。
ALT_W = {"narrow": 1.0, "top2": 2.0, "lev": 3.0, "vix_abs": 2.0, "topshare": 1.0}
# 门槛：训练窗内对齐到 66 天（与生产红点同频）。加门闸与不加门闸的最优门槛都是 81.7。
# 改权重必须同时重标门槛——2026-09-20 减半时重标到 81.6，回退时一并改回 81.7。
# 2026-10-01 等权一侧换成 RSP（EW_SYM）后重标，训练窗 2017-10-18~2026-09-30：连 3 日温度第 65~67 高并列 82.569，
# 恰好 66 天取不到，取最接近的 82.5（67 天/8 段，与换源前 81.7 的 67 天同频）；82.6 只剩 64 天。
ALT_TH = 82.5
ALT_PERSIST = 3

# —— 红点有效性检验（2026-09-24 加，只加显示与一种并列信号，不改上面的红点）——
# 起因：放到 2007-2016 长面板上，81.7 十年只亮 2 天。原因是四个滚动分位因子把原始值的
# **趋势方向**变成读数高低：TOP2、杠杆交易强度的原始值 2010-16 下降、2017 后上升，
# 温度基数因此差 6.7 分，因子间相关 0.02 对 0.21。固定门槛在另一个时代就够不着。
# 设计触发频率：标定窗口生产数据上红点占交易日的比例。它是"设计值"，定下后不随数据更新，只在重标门槛时
# 跟着重定，拿来和实际触发频率比。2026-09-18 标定（2017-10-18~2026-09-18）为 65/2241；
# 2026-10-01 换 RSP 重标（~2026-09-30）为 67/2249。
ALT_DESIGN_RATE = 67 / 2249
# 近 2 年实际触发频率：慢诊断，回答"红点还适不适合当下这个时代"，不预测近期会不会亮。
# （试过"过去一年温度第 97 分位 < 81.7 就判亮不了"：之后 63 日内仍有 10% 会亮，
#   2016-12 以来 9 次红点里 5 次恰恰从这个状态里冒出来——单看一年说明不了什么。）
RATE_WIN = 504
# 自适应门槛（空心红点）：前一日及以前 3 年（至少 2 年）连 3 日温度的同频分位，无前视。
# 长面板实测：2010-16 固定门槛只亮 2 天，自适应 64 天 / 相对基准 −1.83pp，下跌顶多抓 3 个
# （2010-04、2012-04、2015-07）；2017 后 −8.19pp（固定 −10.68pp）。代价是训练窗里弱一些。
AD_WIN, AD_MINP = 756, 504
# 滚动 IC：温度与之后 FWD 日收益的 Spearman 相关，窗口 252 日，月末更新。t 日只能用到
# t−FWD 为止的样本（之后的收益那天还不知道）。负值＝温度高之后跌。长面板上 2010-16
# 平均 −0.20、2018-26 平均 −0.30；它量的是整个分布（含"温度低之后涨"那一端），不等于顶部预警质量。
IC_WIN = 252
# 热端表现（2026-10-01 起替代网页上的滚动 IC）：IC 量的是整个分布的单调关系，红点却只用最热的一小段。
# 2026-09 的 IC 只有 −0.08，拆开看是冷端失灵——温度 <40 的日子之后 64% 还在跌（2026-02~03 一边跌一边冷），
# 热端照常（>81.7 的 9 天之后 89% 为负）。所以网页改看热端：过去 HOT_WIN 日里温度 ≥ HOT_LVL 的日子，
# 之后 FWD 日收益减同期全部日子的均值（pp）；窗口与 IC 相同（止于 t−FWD，月末更新），热端少于 HOT_MIN 天不出数。
# 70 取的是分档收益开始明显走弱的位置（2018-26：60-70 +2.7%、70-81.7 +1.1%、>81.7 −4.8%），没有按结果挑。
# 参照：生产数据 2018-26 年 94% 的月份为负、中位 −2.4pp；长面板 2008-16 年 68% 为负、中位 −0.9pp，
# 但那段 41% 的月份热端不足 10 天。
HOT_LVL, HOT_WIN, HOT_MIN = 70.0, 252, 10
# 预先登记：起算日之后首日出现的红点事件写进 _红点预登记.csv（只追加、不改写），
# 按事先定好的标准评判——这是唯一不受标定过拟合影响的检验。标准写在 说明.md。
PREREG_FROM = "2026-09-25"
PREREG_MIN_EVENTS, PREREG_P, PREREG_DRAWS, PREREG_SEED = 5, 0.10, 10000, 20260924

# —— 集中度状态提示（2026-09-28 加，只作提示，不改红点）——
# 担心的风险：TOP2、前2% 都是 252 日滚动分位，只和过去一年比——集中度一路上升时读数天天顶格，
# 高读数不再只出现在顶部；在高位走平后读数回到 50 附近，绝对水平再高也显示"不热"。
# 试过把读法改成"去趋势后再取分位"（compare_conc.py，2026-09-27）：没有稳定改善，T1 还丢了 2018-08；
# AI 期集中度 ≥80 的日子之后 30 日仍比基准低 1.8pp（误报没有真的出现）；唯一沉默的 2025-02 顶是
# 集中度真实下降（TOP2 57%→50%），任何读法都读 24~29。所以算法不改，改为把状态摆到页面上。
# 状态只看原始值本身，门槛是描述性的，没有按收益调过：
#   上升占比 = 近一年里原始值高于自己过去一年中位数的天数占比（≈ 读数 > 50 的天数占比）
#   水平分位 = 当前原始值在全部可用历史里的分位（生产数据自 2016-09，约 10 年；至少 2 年才出数）
CONC_UP_HI, CONC_UP_LO = 65.0, 35.0   # 上升占比 ≥65 → 趋势上升期；≤35 → 回落期
CONC_LVL_HI = 80.0                    # 其余情况下水平分位 ≥80 → 高位走平期，否则正常
CONC_READ_HI = 80.0                   # 当天读数 ≥80 算"冒尖"。2020-02、2021-11 两个红点顶部都是"高位走平 + 冒尖"
# VIX 绝对刻度的锚点：10→最热(100)，20→中性(50)，40→最冷(0)。
# 用固定锚点而不是滚动分位是实测结论；30 日口径下 VIX 是 12 个因子里秩相关最强的
# 一个（-0.177），而 63 日口径下它几乎无用（-0.074）——它量的本来就是眼前的恐慌。
VIX_ANCHORS = (10.0, 20.0, 40.0)

LABELS = {"narrow": "市值加权跑赢等权", "top2": "TOP2抱团", "lev": "杠杆温度",
          "vix_abs": "VIX绝对刻度", "topshare": "前2%成交额占比"}


def lev_single(idx, lev_pct):
    """红点杠杆因子并入「正股成交额前十」的单股杠杆 ETF（2026-09-30 起，规则与来历见 single_lev.py）。

    lev_pct: engine.leverage_monitor 的分位（旧口径，只含 15 只指数杠杆 ETF）。
    → (拼接后的 lev_pct, 页面用的说明 dict 或 None, 新口径原始量 dict（ratio/ratio_daily/intensity）或 None)。single_lev.FROM 之前的读数原样保留；
    之后换成新口径的分位——新口径的原始量与它自己同口径的过去 252 日比，不和旧口径混比。
    多空比、交易强度的公式与 leverage.letf_components 完全相同，只是做多/做空两边各加上单股杠杆 ETF 的成交额；
    单股部分为 0 的年份，新旧原始量逐日相同。蓝点的杠杆多空比用返回的原始量另行拼接（blue_lev_splice）。
    """
    import leverage as LV
    import single_lev as SL
    m = SL.load()
    if not m:
        return lev_pct, None, None
    got = E.load_many(SL.etfs(m))
    if not got:
        return lev_pct, None, None
    dvl = LV._sum_dv(E.load_many(E.LEV_LONG), idx)
    dvs = LV._sum_dv(E.load_many(E.LEV_SHORT), idx)
    base = {t: E.load(t) for t in LV.BASE_ETFS}
    dvb = LV._sum_dv({t: d for t, d in base.items() if d is not None}, idx)
    if dvl is None or dvs is None or dvb is None:
        return lev_pct, None, None
    sdv = pd.DataFrame({t: LV.dollar_volume(d) for t, d in E.load_many(E.STOCKS).items()}).reindex(idx)
    edv = pd.DataFrame({t: LV.dollar_volume(d) for t, d in got.items()}).reindex(idx)
    la, sa, picks = SL.components(idx, sdv, edv, m)
    L, S = dvl + la, dvs + sa
    ratio_daily = L / (L + S).replace(0, np.nan) * 100.0
    ratio = ratio_daily.rolling(LV.LEV_SMOOTH, min_periods=1).mean()
    inten = ((L + S) / dvb.replace(0, np.nan) * 100.0).rolling(LV.LEV_SMOOTH, min_periods=1).mean()
    new = pd.DataFrame({"ratio": E.rolling_pct(ratio), "intensity": E.rolling_pct(inten)}, index=idx)
    on = idx >= pd.Timestamp(SL.FROM)
    out = lev_pct.copy()
    for c in ("ratio", "intensity"):
        if c in out.columns:
            out.loc[on, c] = new.loc[on, c]
    t = idx[-1]
    tot = float((L + S).loc[t])
    info = {"from": SL.FROM, "top_n": SL.TOP_N, "active": bool(on[-1]), "picks": list(picks.loc[t]),
            "share": (None if not tot else round(float((la + sa).loc[t]) / tot * 100, 1)),
            "n_etfs": len(got), "n_map": len(SL.etfs(m)), "map_updated": json.load(open(SL.MAP_JSON, encoding="utf-8"))["updated"],
            # 新旧两种口径在同一天的读数，供核对"切换处没有跳变"
            "old_now": {c: (None if c not in lev_pct.columns or not np.isfinite(lev_pct[c].loc[t]) else round(float(lev_pct[c].loc[t]), 1)) for c in ("ratio", "intensity")},
            "new_now": {c: (None if not np.isfinite(new[c].loc[t]) else round(float(new[c].loc[t]), 1)) for c in ("ratio", "intensity")}}
    return out, info, {"ratio": ratio, "ratio_daily": ratio_daily, "intensity": inten}


def blue_lev_splice(adj_like, dirs, ratio, fast):
    """蓝点的杠杆多空比同样自 single_lev.FROM 起换新口径（2026-09-30 用户定：红点、蓝点用同一个多空比）。

    adj_like: engine.compose 的方向修正后分位（fast=True 是判定用的当日口径 adjf，False 是展示用的平滑口径 adj）；
    ratio: 对应口径的新多空比原始量（当日值或 5 日均）。分位与方向修正走 engine.compose 同一条路，只换这一列的输入；
    FROM 之前原样保留。实测（2022-08 单股产品上市以来 31 个 VIX ≥ 30 的日子）：蓝点一天不变，温度差最大 2.3 分。
    """
    import single_lev as SL
    if "leverage" not in adj_like.columns:
        return adj_like
    _, adj_new, _, _ = E.compose(pd.DataFrame({"leverage": ratio}), dirs, fast=fast)
    out = adj_like.copy()
    on = out.index >= pd.Timestamp(SL.FROM)
    out.loc[on, "leverage"] = adj_new["leverage"].reindex(out.index)[on]
    return out


def equal_weight(idx):
    """等权一侧（EW_SYM 的复权收盘）。分析脚本用的长面板（~/us2/raw_long）里没有它，就从本仓库 raw/ 取——
    它是外部 ETF，与用哪套成分股面板无关。取不到返回 None。"""
    d = E.load(EW_SYM)
    if d is None and os.path.abspath(E.RAW) != RAW_HERE:
        saved, E.RAW = E.RAW, RAW_HERE
        try:
            d = E.load(EW_SYM)
        finally:
            E.RAW = saved
    return d["close"].reindex(idx) if d is not None else None


def cap_vs_equal(cw, idx):
    """市值跑赢等权：cw（SPY 复权收盘，与 idx 同索引）近 NT_WIN 日相对等权一侧的超额，小数。等权缺失返回 None。"""
    eq = equal_weight(idx)
    return None if eq is None else (cw / cw.shift(E.NT_WIN)) / (eq / eq.shift(E.NT_WIN)) - 1.0


def alt_inputs(rawdf, adj, lev_pct, spy, idx):
    """返回最优拟合红点的五个分项（都是 0-100，与主口径同尺）。

    lev_pct: engine.leverage_monitor 返回的第一项（含 ratio / intensity 两列分位）。
    注意这里**不用**它返回的 lev_temp——那个是按 leverage.py 的 3:1 合成的。
    """
    cw = spy["close"].reindex(idx) if spy is not None else None
    nq = E.load("QQQ")
    nq = nq["close"].reindex(idx) if nq is not None else cw
    out = {}

    # 上涨拥挤度：内部只剩"市值跑赢等权"，但保留上涨市门闸与下跌市填 50 的语义
    gap = cap_vs_equal(cw, idx) if cw is not None else None
    if gap is not None and nq is not None:
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
# —— 判定用的蓝点温度里，ERP 只计便宜一侧（2026-10-09）——
# ERP 反向分位高于 BLUE_ERP_CAP 一律记 BLUE_ERP_CAP（中性），只能把温度往冷拉，不能一票否决。
# 起因：分位 ≥ 87.2 时另外三项全是 0 也到不了 21.8，蓝点在数学上亮不了。这种状态占全部交易日约三分之一
# （正确 ERP 口径下训练窗 32%、样本外 27%），而它恰好常出现在底部附近：「贵」来自慢变量追上来——
# 2022-06、2022-09/10 是 10 年期利率一年涨了 2.5 个百分点（重估期 2022-06-13 已退出，空心蓝点管不到），
# 2009-03 是滚动盈利崩塌。那几次另外三项平均只有 3~11。
# 实测（门槛 21.8 不动，实心蓝点之后 QQQ 30 / 63 日，正确 ERP 口径）：
#     训练窗 2017-10~  32 天/5 次 +17.9/+28.3%、63 日 0% 为负 → 39 天/6 次 +16.2/+23.6%、8% 为负
#     样本外 2007-16   66 天/5 次 +2.3/+4.4%、27% 为负     → 76 天/5 次 +4.4/+7.9%、24% 为负
# 现有蓝点一个不少；新增的要求另外三项均值 < 12.4（= (21.8×4 − 50) ÷ 3）。训练窗新增 2022-06-14/16/21（30 日 +12~16%）、
# 2022-09-26/27/30 与 10-12（63 日 −2.3/−3.7/−0.1/+6.5%，纳指 12 月回探），样本外新增 2009-02-23~03-09 大底（63 日 +21~43%）。
# 代价要认：训练窗里「贵」的一侧有信息（VIX ≥ 30 且另三项极冷时，ERP 便宜之后 63 日 +27%、贵只有 +3%）；
# 样本外正好反过来（便宜 +3.7%、贵 +22.9%）。两个时代反号，所以不让它否决。
# 上限 70~100 训练窗不变、40~60 结果相近；取 50 是因为它是中性点（与 DIR_HALF「低于常态不表态」同一思路），不是按结果挑的。
# 试过更差的：重估期内不计 ERP（实心不变，修不到 2022-10）、利率急升期不计 ERP（样本外 +1.2/+3.5%）、
# 只在利率急升期封顶（样本外抓不到 2009）、全程不计 ERP（训练窗 +13.6/+16.2%）、ERP > 50 时弃权（两段都更差）。
# 只用于判定；页面展示的蓝点温度（平滑口径）不封顶。
BLUE_ERP_CAP = 50.0


def blue_temperature(adj_like, judge=True):
    """按 BLUE_W 合成蓝点温度。传当日口径（adjf）、judge=True 得到判定用的那条（ERP 封顶 BLUE_ERP_CAP），
    传平滑口径（adj）、judge=False 得到页面展示的那条（不封顶）。任一分项缺失则当日无读数。"""
    ks = [k for k in BLUE_W if k in adj_like.columns]
    if len(ks) != len(BLUE_W):
        return None
    X = pd.concat([adj_like[k].clip(upper=BLUE_ERP_CAP) if (judge and k == "erp") else adj_like[k] for k in ks], axis=1)
    w = np.array([BLUE_W[k] for k in ks], dtype=float)
    w = w / w.sum()
    t = pd.Series((X.values * w).sum(axis=1), index=X.index)
    return t.where(X.notna().all(axis=1))


# ───────────────────────── 红点有效性检验 ─────────────────────────

def validity(temp, px):
    """红点有效性检验要用的几条序列，全部只用当时已有的数据。

    temp: 最优拟合红点温度（全历史）；px: 标的收盘（与 temp 同索引）。
    返回 dict：t3（连 3 日温度）、hot（固定门槛红点）、th_ad（自适应门槛）、hot_ad（自适应红点）、
    rate（近 2 年实际触发频率 %）、ic（滚动 IC，月末更新后前向填充）、ic_m（只含月末那几个点）、
    fwd（之后 FWD 日收益）。
    """
    t3 = temp.rolling(ALT_PERSIST, min_periods=ALT_PERSIST).min()
    hot = (t3 > ALT_TH).fillna(False)
    th_ad = t3.shift(1).rolling(AD_WIN, min_periods=AD_MINP).quantile(1 - ALT_DESIGN_RATE)
    hot_ad = (t3 > th_ad).fillna(False)
    rate = hot.astype(float).where(t3.notna()).rolling(RATE_WIN, min_periods=RATE_WIN).mean() * 100
    fwd = px.shift(-FWD) / px - 1
    ic = pd.Series(np.nan, index=temp.index)
    ends = temp.index.to_series().groupby([temp.index.year, temp.index.month]).max()
    for t in ends:
        i = temp.index.get_loc(t) - FWD          # 窗口止于 t−FWD：那天的前瞻收益在 t 日刚好可知
        if i < IC_WIN - 1:
            continue
        a, b = temp.iloc[i - IC_WIN + 1: i + 1], fwd.iloc[i - IC_WIN + 1: i + 1]
        m = a.notna() & b.notna()
        if m.sum() >= IC_WIN * 0.8:
            ic.loc[t] = a[m].corr(b[m], method="spearman")
    return dict(t3=t3, hot=hot, th_ad=th_ad, hot_ad=hot_ad, rate=rate, ic=ic.ffill(), ic_m=ic.dropna(), fwd=fwd)


def events(flag, gap=FWD):
    """把旗标切成事件：相隔不超过 gap 个交易日的算同一次。返回 [(首日, 末日), ...]。"""
    pos = np.where(flag.fillna(False).values)[0]
    out = []
    for p in pos:
        if out and p - out[-1][1] <= gap:
            out[-1][1] = p
        else:
            out.append([p, p])
    return [(flag.index[a], flag.index[b]) for a, b in out]


def prereg_judge(firsts, fwd):
    """按预先登记的标准评判一类信号。

    firsts: 该信号登记在案的事件首日；fwd: 之后 FWD 日收益。
    标准：前瞻收益已可知的事件 ≥ PREREG_MIN_EVENTS 次后，把同样个数的日子随机放在起算日之后
    （前瞻收益已可知的交易日里无放回抽取，PREREG_DRAWS 次），p = 随机均值 ≤ 实际均值的比例；
    p < PREREG_P 判"通过"，否则"未通过"。事件不够时是"累积中"。
    """
    pool = fwd[(fwd.index >= pd.Timestamp(PREREG_FROM))].dropna()
    got = [float(fwd.get(t)) for t in firsts if pd.notna(fwd.get(t, np.nan))]
    res = {"n": len(firsts), "n_known": len(got), "need": PREREG_MIN_EVENTS,
           "mean": (float(np.mean(got)) if got else None),
           "base": (float(pool.mean()) if len(pool) else None), "p": None, "verdict": "累积中"}
    if len(got) >= PREREG_MIN_EVENTS and len(pool) > len(got):
        rng = np.random.default_rng(PREREG_SEED)
        v = pool.values
        sims = np.array([rng.choice(v, len(got), replace=False).mean() for _ in range(PREREG_DRAWS)])
        p = float((sims <= np.mean(got)).mean())
        res.update(p=p, verdict="通过" if p < PREREG_P else "未通过")
    return res


def conc_state(raw):
    """集中度原始值（TOP2 或前2%，单位 %）→ (上升占比 %, 水平分位)。只用当时已有的数据。"""
    above = (raw > raw.rolling(252, min_periods=200).median()).astype(float).where(raw.notna())
    up = above.rolling(252, min_periods=200).mean() * 100
    lvl = E.expanding_pct(raw, min_periods=504)
    return up, lvl


def conc_label(up, lvl, reading):
    """状态 + 当天读数 → {key, name, msg}。只作页面提示，文案里的数字由调用方填。"""
    if up is None or not np.isfinite(up):
        return {"key": "na", "name": "数据不足", "msg": "近一年数据不足，无法判断"}
    hot = reading is not None and np.isfinite(reading) and reading >= CONC_READ_HI
    if up >= CONC_UP_HI:
        return {"key": "up", "name": "趋势上升期",
                "msg": ("读数偏高，但近一年多数日子都在创新高，高读数有一部分是趋势带来的；"
                        "单看它意义有限，要看其他因子是否同时变热" if hot else
                        "集中度在上升通道里，眼下没有冒尖")}
    if up <= CONC_UP_LO:
        return {"key": "down", "name": "回落期",
                "msg": ("集中度整体在下降，但眼下又冒尖" if hot else
                        "集中度整体在下降，读数偏低多半是真实降温")}
    if lvl is not None and np.isfinite(lvl) and lvl >= CONC_LVL_HI:
        return {"key": "plateau", "name": "高位走平期",
                "msg": ("原始值在历史高位、近一年没有明显趋势，眼下又冒尖——"
                        "2020-02、2021-11 两个红点顶部都是这种形态" if hot else
                        "原始值仍在历史高位，读数不高只是因为没比过去一年更高——不代表不拥挤")}
    return {"key": "normal", "name": "正常", "msg": "没有明显趋势、也不在历史高位，读数可以直接看"}


def hot_end(temp, px):
    """热端表现，月末更新（当月按最新一日）。返回以月末日期为索引的 DataFrame：
    n（热端天数）、edge（热端之后 FWD 日均值减全部日子均值，pp；热端 < HOT_MIN 天时为 NaN）、
    hot_mean / hot_neg、all_mean / all_neg（% ）、end（样本止于哪天）。只用当时已有的数据。"""
    fwd = px.shift(-FWD) / px - 1
    ix = temp.index
    ends = ix.to_series().groupby([ix.year, ix.month]).max()
    rows = {}
    for t in ends:
        i = ix.get_loc(t) - FWD               # 窗口止于 t−FWD：那天之后的收益在 t 日刚好可知
        if i < HOT_WIN - 1:
            continue
        a, b = temp.iloc[i - HOT_WIN + 1: i + 1], fwd.iloc[i - HOT_WIN + 1: i + 1]
        m = a.notna() & b.notna()
        if m.sum() < HOT_WIN * 0.8:
            continue
        h = m & (a >= HOT_LVL)
        n = int(h.sum())
        rows[t] = {"n": n,
                   "edge": (b[h].mean() - b[m].mean()) * 100 if n >= HOT_MIN else np.nan,
                   "hot_mean": b[h].mean() * 100 if n else np.nan,
                   "hot_neg": (b[h] < 0).mean() * 100 if n else np.nan,
                   "all_mean": b[m].mean() * 100, "all_neg": (b[m] < 0).mean() * 100, "end": ix[i]}
    return pd.DataFrame.from_dict(rows, orient="index")
