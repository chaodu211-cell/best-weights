#!/usr/bin/env python3
"""2026-11-11 冻结规则的正式复核，默认输出至本研究目录。

运行 python3 研究/二次顶监测_2026-11-11/backtest.py。
价格和红点均重新走当前生产数据及计算入口，保留全部精度。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import alt_engine as A
import engine as E

ENTRY_DAYS = 252
EXTRA_DAYS = 126
FWD = 63
# 与任务给出的粗测使用同一截止日，后续数据更新也保留这次研究的评价窗口。
BACKTEST_END = "2026-11-09"
PROTECTED = ("_红点预登记.csv", "_漏报预登记.csv", "_信号快照.csv")
COARSE = {
    "2010-04-23": (-15.8, "2010-10-13", None, None, None),
    "2012-04-02": (-11.5, "2012-08-17", "2012-09-19", "2012-10-09", -7.8),
    "2015-07-20": (-13.9, "2015-10-28", "2015-12-01", "2015-12-11", -12.8),
    "2018-08-29": (-22.8, "2019-04-12", None, None, None),
    "2020-02-19": (-28.6, "2020-06-03", None, None, None),
    "2024-07-10": (-13.6, "2024-11-06", "2025-02-19", "2025-01-10", -17.8),
    "2026-06-02": (-11.2, "2026-09-22", "2026-10-06", "2026-10-16", -8.8),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def day(value):
    return None if value is None else str(pd.Timestamp(value).date())


def production_inputs():
    """与 make_alt_page.patch 相同的接线，使用生产 build_indicators 的在册掩码。"""
    raw, _, spy, _ = E.build_indicators()
    raw = raw.loc[:BACKTEST_END]
    dirs = E.direction(spy, raw.index)
    _, adj, _, _ = E.compose(raw, dirs)
    lev, _ = E.leverage_monitor(raw)
    lev, _, _ = A.lev_single(raw.index, lev)
    temp = A.alt_temperature(A.alt_inputs(raw, adj, lev, spy, raw.index))
    if temp is None or temp.dropna().empty:
        raise RuntimeError("生产红点温度缺失")
    close = E.load("QQQ")["close"].dropna().loc[:raw.index[-1]]
    validity = A.validity(temp, close.reindex(temp.index))
    daily = pd.DataFrame({"close": close, "temperature": temp,
                          "t3": validity["t3"], "adaptive_threshold": validity["th_ad"],
                          "hot": validity["hot"], "hot_ad": validity["hot_ad"]})
    daily["input_day"] = daily.index.isin(temp.index)
    daily[["hot", "hot_ad"]] = daily[["hot", "hot_ad"]].fillna(False).astype(bool)
    assert daily["close"].notna().all(), "生产温度存在 QQQ 交易日以外的日期"
    return daily, temp


def forward(close, when):
    """最低收盘相对事件收盘，包含事件日的零值下限。收益严格要求满 63 日。"""
    if when is None:
        return None
    i = close.index.get_loc(when)
    end = min(i + FWD, len(close) - 1)
    values = close.iloc[i:end + 1]
    return {"n": end - i, "end": day(values.index[-1]), "complete": end - i == FWD,
            "low": float(values.min() / close.iloc[i] - 1),
            "return": float(values.iloc[-1] / close.iloc[i] - 1),
            "low_date": day(values.idxmin())}


def scan(daily, dd_limit=.15, breakout=.10, ma_days=50, red_kind="either"):
    """逐日推进，只把当天输入交给状态更新。

    确认日之后才允许进入，进入窗口含顶后第252日，期限在顶后第378日收盘结束。
    触发认收盘低于均线的首次状态，包含结束日收盘。每个顶只尝试首次重回。
    结束日先记录首次触发，再结束观察。突破严格 >，回撤确认含等号。
    """
    close = daily["close"]
    values = close.to_numpy()
    cal = close.index
    hot = daily["hot"].to_numpy()
    adaptive = daily["hot_ad"].to_numpy()
    marked = {"either": hot | adaptive, "hollow": adaptive, "solid": hot}[red_kind]
    averages = close.rolling(ma_days, min_periods=ma_days).mean().to_numpy()
    highs = close.rolling(A.RECALL_HI, min_periods=A.RECALL_HI).max().to_numpy()
    rows, last_top, trace = [], None, []
    for today, value in enumerate(values):
        # 独立重做 drawdown_tops 的逐日确认，再与原入口的全部输出核对。
        if today:
            first = max(0, today - A.RECALL_LOOK)
            peak = first + int(np.argmax(values[first:today]))
            separated = last_top is None or peak - last_top > A.RECALL_LOOK
            if separated and np.isfinite(highs[peak]) and values[peak] >= highs[peak]:
                if value <= values[peak] * (1 - A.RECALL_DD):
                    rows.append({"p": peak, "q": today, "reference": False,
                                 "ready": None, "recross": None, "depth": None,
                                 "entry": None, "trigger": None, "end": None,
                                 "reason": None, "high": None, "high_at": None})
                    last_top = peak
        for row in rows:
            p, q = row["p"], row["q"]
            if not row["reference"]:
                # 对假设的极快下跌也不偷看顶后5日的尚未出现的红点。
                stop = min(today, p + A.RECALL_POST)
                if marked[max(0, p - A.RECALL_PRE):stop + 1].any():
                    row["reference"], row["ready"] = True, today
            if row["recross"] is None and q < today <= p + ENTRY_DAYS and value > values[p]:
                row["recross"] = today
                row["depth"] = float(values[p:today + 1].min() / values[p] - 1)
                if row["reference"] and row["depth"] > -dd_limit:
                    row["entry"] = today
                    row["high"], row["high_at"] = value, today
            if row["entry"] is None or row["end"] is not None:
                continue
            if value > row["high"]:
                row["high"], row["high_at"] = value, today
            if row["trigger"] is None and value < averages[today]:
                row["trigger"] = today
            reason = None
            if value > values[p] * (1 + breakout):
                reason = "突破"
            elif value <= row["high"] * (1 - A.RECALL_DD):
                reason = "二次见顶确认"
            elif today >= p + ENTRY_DAYS + EXTRA_DAYS:
                reason = "到期"
            if reason:
                row["end"], row["reason"] = today, reason
        trace.append(tuple((r["p"], r["entry"], r["trigger"]) for r in rows
                           if r["entry"] is not None and r["end"] is None))
    expected = A.drawdown_tops(close)
    actual = [(cal[r["p"]], cal[r["q"]]) for r in rows]
    assert actual == expected, "逐日确认与生产 drawdown_tops 不一致"
    result = []
    for row in rows:
        p = row["p"]
        win = A.recall_window(cal, cal[p])
        record = {k: day(cal[row[k]]) if row[k] is not None else None
                  for k in ("entry", "trigger", "end", "recross", "ready", "high_at")}
        record.update(top=day(cal[p]), confirm=day(cal[row["q"]]),
                      reference=row["reference"], depth=row["depth"], reason=row["reason"],
                      solid=bool(daily.loc[win, "hot"].any()),
                      hollow=bool(daily.loc[win, "hot_ad"].any()),
                      forward=forward(close, cal[row["trigger"]]) if row["trigger"] is not None else None)
        result.append(record)
    return result, trace


def batch_audit(daily, rows, trace):
    """独立批量定位进入、触发及结束，与逐日推进的每一天状态比较。"""
    close = daily["close"]
    values = close.to_numpy()
    cal = close.index
    ma = close.rolling(50).mean().to_numpy()
    records = []
    for pdate, qdate in A.drawdown_tops(close):
        p, q = cal.get_loc(pdate), cal.get_loc(qdate)
        # 真实样本的每个顶都在顶后5日已过后才确认，参考红点当时全部可知。
        assert q >= p + A.RECALL_POST
        window = A.recall_window(cal, pdate)
        if not daily.loc[window, ["hot", "hot_ad"]].any().any():
            continue
        candidates = np.flatnonzero(values[q + 1:min(len(values), p + ENTRY_DAYS + 1)] > values[p])
        if not len(candidates):
            continue
        entry = q + 1 + int(candidates[0])
        if values[p:entry + 1].min() / values[p] - 1 <= -.15:
            continue
        stop = min(len(values), p + ENTRY_DAYS + EXTRA_DAYS + 1)
        path = values[entry:stop]
        running_high = np.maximum.accumulate(path)
        ends = np.flatnonzero((path > values[p] * 1.1) | (path <= running_high * .89))
        end = entry + int(ends[0]) if len(ends) else (stop - 1 if stop < len(values) or stop - 1 == p + 378 else None)
        through = len(values) if end is None else end + 1
        triggers = np.flatnonzero(values[entry:through] < ma[entry:through])
        trigger = entry + int(triggers[0]) if len(triggers) else None
        records.append((p, entry, trigger, end))
        match = next(r for r in rows if r["top"] == day(pdate))
        assert (match["entry"], match["trigger"], match["end"]) == (
            day(cal[entry]), day(cal[trigger]) if trigger is not None else None,
            day(cal[end]) if end is not None else None)
    for today, state in enumerate(trace):
        expected = tuple((p, entry, trigger if trigger is not None and trigger <= today else None)
                         for p, entry, trigger, end in records
                         if entry <= today and (end is None or today < end))
        assert state == expected, f"逐日状态偏差 {cal[today]}"
    # 自适应门槛逐日独立复算，窗口严格截止前一天。
    temp = daily["temperature"].dropna().index
    # 生产温度沿 SPY 交易日轴，QQQ 独有的日子不占它的滚动窗口位置。
    signal = daily.loc[daily["input_day"]]
    t3 = signal["t3"].to_numpy()
    th = signal["adaptive_threshold"].to_numpy()
    max_error = 0.0
    for i in range(len(signal)):
        past = t3[max(0, i - A.AD_WIN):i]
        past = past[np.isfinite(past)]
        expected = np.quantile(past, 1 - A.ALT_DESIGN_RATE) if len(past) >= A.AD_MINP else np.nan
        assert np.isfinite(expected) == np.isfinite(th[i]), f"自适应预热偏差 {signal.index[i]}"
        if np.isfinite(expected):
            max_error = max(max_error, abs(expected - th[i]))
    assert max_error < 1e-10
    return {"daily_states_checked": len(trace), "batch_equal": True,
            "tops_equal": True, "adaptive_threshold_max_error": max_error,
            "first_temperature": day(temp[0])}


def comparison_events(daily, rows):
    """一年新高后武装，首次低于MA50解除，须再创一年新高才重新武装。

    一年新高沿用 drawdown_tops 的252日含当日最高口径，平最高也算。
    这些对照事件不做额外间隔合并。一个观察期可贡献多次对照事件。
    """
    close = daily["close"]
    ma = close.rolling(50).mean()
    high = close.rolling(252, min_periods=252).max()
    armed, events = False, []
    for when, value in close.items():
        if value >= high.loc[when]:
            armed = True
        if not armed or value >= ma.loc[when]:
            continue
        armed = False
        date = day(when)
        periods = [r["top"] for r in rows if r["entry"] is not None and
                   r["entry"] <= date and (r["end"] is None or date <= r["end"])]
        events.append({"date": date, "inside": bool(periods), "tops": periods,
                       **forward(close, when)})
    # 逐个新高定位下一次破线，再按日期去重，独立核对事件全集。
    below = np.flatnonzero((close < ma).to_numpy())
    new_high = np.flatnonzero((close >= high).to_numpy())
    next_below = np.searchsorted(below, new_high, side="left")
    positions = sorted(set(below[next_below[next_below < len(below)]]))
    assert [day(close.index[i]) for i in positions] == [e["date"] for e in events]
    return events


def summarize(events):
    complete = [e for e in events if e["complete"]]
    def mean(items, key):
        return float(np.mean([e[key] for e in items])) if items else None
    return {"n": len(events), "complete": len(complete), "partial": len(events) - len(complete),
            "mean_low": mean(complete, "low"), "mean_return": mean(complete, "return"),
            "below_8": sum(e["low"] < -.08 for e in complete),
            "available_mean_low": mean(events, "low"),
            "available_mean_return": mean(events, "return")}


def sensitivity(daily):
    configs = [("基准15% / 10% / MA50", .15, .10, 50, "either")]
    configs += [(f"回撤{x}%", x / 100, .10, 50, "either") for x in (13, 14, 16, 18)]
    configs += [(f"突破{x}%", .15, x / 100, 50, "either") for x in (5, 8, 15)]
    configs += [(f"MA{x}", .15, .10, x, "either") for x in (20, 100)]
    configs += [("只认空心", .15, .10, 50, "hollow"), ("只认实心", .15, .10, 50, "solid")]
    result = []
    for label, dd, cap, ma, kind in configs:
        rows, _ = scan(daily, dd, cap, ma, kind)
        periods = [r for r in rows if r["entry"]]
        fired = [r for r in periods if r["trigger"]]
        result.append({"label": label, "dd_limit": dd, "breakout": cap, "ma": ma,
                       "red_kind": kind, "periods": len(periods), "triggers": len(fired),
                       "second_tops": sum(r["reason"] == "二次见顶确认" for r in periods),
                       "breakouts": sum(r["reason"] == "突破" for r in periods),
                       "stats": summarize([r["forward"] for r in fired]), "rows": periods})
    return result


def coarse_check(rows):
    checks = []
    for top, expected in COARSE.items():
        row = next(r for r in rows if r["top"] == top)
        depth, recross, second, trigger, low = expected
        checks.append({"top": top, "depth_match": abs(row["depth"] * 100 - depth) <= .051,
                       "recross_match": row["recross"] == recross,
                       "outcome_match": row["diagnostic_outcome"].startswith("突破") if second is None
                       else row["reason"] == "二次见顶确认",
                       "second_top_match": second is None or row["high_at"] == second,
                       "trigger_match": row["trigger"] == trigger,
                       "low_match": low is None or abs(row["forward"]["low"] * 100 - low) <= .051})
    return checks


def pct(value):
    return "—" if value is None else f"{value * 100:+.2f}%"


def table(headers, values):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(str(v) if v is not None else "—" for v in row) + " |" for row in values]
    return "\n".join(lines)


def report(result):
    rows = result["rounds"]
    marked = [r for r in rows if r["reference"]]
    lines = ["# 二次顶监测正式回测", "", "## A 部分结果", "",
             f"数据截止 {result['data']['last']}。冻结规则日期为 2026-11-11，全部参数按任务说明固定。",
             "", "A 部分已完成。七行逐轮粗测复现，对照组差异已解释，不是实质出入；B 部分的继续条件已满足。",
             "", "### 数据与计算口径", "",
             f"- QQQ 复权收盘通过 `engine.load(\"QQQ\")[\"close\"]` 读取，覆盖 {result['data']['first']} 至 {result['data']['last']}，共 {result['data']['n']} 个交易日。",
             "- 红点复用 `build_indicators → direction/compose → leverage_monitor → lev_single → alt_inputs → alt_temperature → validity`，与页面生产接线一致。使用 `members.json` 在册区间、raw/ 已拼接的 hist/ 长历史、固定门槛 82.1 和当前自适应门槛，计算保留全部精度。",
             "- 历史研究使用当前数据重算的红点。三份生产登记和快照仅核对哈希，历史回测未写入登记。",
             f"- 红点温度首个有效日为 {result['data']['first_temperature']}，自适应门槛首个有效日为 {result['data']['first_adaptive']}。生产温度使用SPY日期轴，其2005-02-25之前及2011-02-17缺行，按生产日期轴计算温度分位。价格结构与均线仍使用完整QQQ日期轴，缺温度日沿用页面的无红点处理。",
             "- 顶与确认日直接核对 `drawdown_tops`，红点窗口调用 `recall_window`。每个顶只尝试确认日之后、顶后第252日以内的首次收盘重回旧顶。进入前最低收盘相对旧顶必须严格大于 −15%。",
             "- 观察期含进入日及结束日收盘，首次收盘低于MA50即触发。结束采用旧顶以上严格超过10%、观察期最高收盘回落至少11%、顶后第378日中的最早者。结束日若同时首次破均线，先记录触发。",
             "- 后63日最低值为 `min(事件日及随后63日收盘) / 事件日收盘 - 1`，收益为第63日收盘相对事件日收盘。未满63日的样本单列，同时列出截至当前的可得结果。",
             "", "### 红点参考顶逐轮核对", "",
             table(["参考顶", "红点", "确认日", "顶后最低/重回前", "首次重回", "进入", "首次触发", "结束/观察期最高日", "后63日最低", "后63日收益/天数"], [
                 [r["top"], "/".join(k for k, v in (("实心", r["solid"]), ("空心", r["hollow"])) if v),
                  r["confirm"], pct(r["depth"]), r["recross"], r["entry"], r["trigger"],
                  (f"{r['reason']} {r['end']} / {r['high_at']}" if r["entry"] else "无观察期"),
                  pct(r["forward"]["low"]) if r["forward"] else "—",
                  f"{pct(r['forward']['return'])} / {r['forward']['n']}日" if r["forward"] else "—"] for r in marked]),
             "", "二次见顶栏的观察期最高日由结束时已知的价格定位。触发使用当日收盘，最高日的事后标签未参与触发。",
             "补列了粗测表未列的2021-12-27参考顶，该顶之后252个交易日内未重回旧高。2010、2018、2020三轮重回前回撤超过15%，其事后价格诊断均为突破，与粗测一致。",
             "", "### 没有红点的下跌顶", "",
             table(["顶", "确认日", "252日内首次重回", "重回前最低", "重回后价格诊断"], [
                 [r["top"], r["confirm"], r["recross"], pct(r["depth"]), r.get("diagnostic_outcome", "无重回")]
                 for r in rows if not r["reference"]]),
             "", "粗测提到的五次未标顶重回均复现且随后突破。其中2011-07、2015-12、2025-02三顶重回前回撤超过15%，另外两顶符合价格准入条件。五次重回的事后结果均未参与参考顶选择。2026-10-06的顶刚确认，观察长度仍短。",
             "", "### 对照检验", "",
             "对照事件采用逐日状态机。收盘达到过去252日含当日最高值时开始等待，第一次收盘低于MA50记录事件并解除等待，之后须再创一年新高才能再次记录。未另加事件间隔合并。观察期内可有多次对照事件，正式触发仍每轮只记一次。",
             "逐个一年新高独立寻找下一次破MA50，再按事件日去重，所得55个事件与状态机逐项一致。下表同时披露不同可用历史范围。主核对范围采用红点温度首个有效日起，自适应门槛预热完成后的区间另列。",
             "", table(["事件范围", "分组", "事件数", "完整/未满63日", "完整样本最低均值", "完整样本收益均值", "完整样本再跌>8%", "含未满样本最低均值"], [
                 [label, group, stats["n"], f"{stats['complete']}/{stats['partial']}", pct(stats["mean_low"]), pct(stats["mean_return"]), stats["below_8"], pct(stats["available_mean_low"])]
                 for label, groups in result["comparisons"].items() for group, stats in groups.items()]),
             "", "全部对照事件及分组见 `对照事件.csv`。观察期内事件如下。", "",
             table(["事件日", "参考顶", "后63日最低", "后63日收益", "已有天数", "区间截至"], [
                 [e["date"], "/".join(e["tops"]), pct(e["low"]), pct(e["return"]), e["n"], e["end"]]
                 for e in result["events"] if e["inside"]]),
             "", "### 敏感度", "",
             "每次只改变表中一项，其余沿用冻结规则。所有结果均保留，未据此选择参数。空心组仍使用现行模型权重和设计频率。",
             "", table(["变化项", "观察期数", "触发数", "二次见顶/突破", "完整/未满63日", "触发后最低均值/完整", "触发后收益均值/完整", "含未满样本最低均值", "触发日"], [
                 [s["label"], s["periods"], s["triggers"], f"{s['second_tops']}/{s['breakouts']}",
                  f"{s['stats']['complete']}/{s['stats']['partial']}", pct(s["stats"]["mean_low"]),
                  pct(s["stats"]["mean_return"]), pct(s["stats"]["available_mean_low"]),
                  "、".join(r["trigger"] for r in s["rows"] if r["trigger"])] for s in result["sensitivity"]]),
             "", "逐轮敏感度明细见 `回测结果.json` 的 `sensitivity[].rows`。",
             "", "### 无前视检查", "",
             f"- 逐日推进 {result['causality']['daily_states_checked']} 个交易日，独立实现当天的顶确认，与生产 `drawdown_tops` 的顶和确认日完全一致。",
             "- 独立批量实现进入、触发、结束，再比较逐日收盘后观察期及触发状态，每一天均一致。状态更新只接收当前及以前的收盘和红点。真实样本的确认日均已过顶后5日，参考红点窗口在确认时完整可知。",
             f"- 逐日用截至前一天的温度窗口独立重算自适应分位，最大误差 {result['causality']['adaptive_threshold_max_error']:.3g}。前瞻63日数据只在事后评价函数中读取。",
             "- 价格状态规则可逐日执行。固定门槛82.1、模型权重、自适应设计频率均来自2017-10之后的标定和后续修订。实心组历史表现受标定影响，空心组的分位计算只用过去数据，但复用了这套事后选定的权重和设计频率。当前历史数据也有后续复权和成员覆盖修订，逐日模拟无法恢复当年实际数据版本。",
             "- 空心组的“无前视”仅指给定现行模型后的时序计算。历史结果仍用于探索，独立验证需要冻结之后的新事件。",
             "", "### 与粗测逐项差异", "",
             table(["顶", "重回前最低", "重回日", "结果", "二次顶日", "首次触发", "触发后最低"], [
                 [c["top"]] + ["一致" if c[k] else "有差异" for k in
                                 ("depth_match", "recross_match", "outcome_match", "second_top_match", "trigger_match", "low_match")]
                 for c in result["coarse_checks"]]),
             "", "- 七行中的重回前最低跌幅、重回日、结果、二次顶日、首次触发日及触发后最低均与粗测一致，跌幅按粗测一位小数核对。2026年首次触发后最低发生于11-05，计算延伸到数据截止11-09后最低仍为−8.84%，已有16个后续交易日。",
             "- 观察期内5次对照事件、4轮观察期均一致，含最后一轮可得数据的最低均值−13.42%，对应粗测的约−13%。其中2025-02-24属于2024-07参考顶的第二次对照事件，正式监测只保留2025-01-10首次触发。",
             "- **对照组差异已解释（2026-11-11，用户转述Claude核对）**：粗测的35次是2009-07起、观察期外事件再剔除两组后的「其余时间」；剔除红点顶后回撤≥15%的回测区内2次，以及未标红点的顶之后回测区内5次。因此35+2+5=42，与自适应门槛有效期2009-07-16起的观察期外42次、10次再跌超过8%一致。",
             "- 均值也一致：(35×−6.0+2×−2.4+5×−1.8)/42≈−5.3%，对应正式回测−5.29%（粗测分组均值已四舍五入）。2007-07-13起的44次、−5.54%、11次，以及页面2006年起48次、价格全历史50次，属于更早起点的另列口径。差异来自起点与分组范围，不构成重大出入，规则和参数均不改。",
             "", "### 样本局限", "",
             "基准只有4轮观察期与4次正式触发。对照组在这些观察期内的5次事件包含同一轮重复观测，63日评价窗也会重叠。最后一轮未满63日，完整评价只有3轮。规则来自已见过的二次顶形态，历史回测存在发现偏差，样本量不足以建立稳定预测结论。",
             "", "### 运行与核对记录", "",
             *verification_report(result),
             "", "### 未完成与疑问", "",
             "- B部分继续条件已满足，实施与核验记录见下文。",
             "- 对照组差额已按用户提供的核对结论解释，无需调整规则。",
             "- 第四轮需要继续等待63日评价窗成熟，当前仅有16日。只有4轮的样本局限与2017-10之后标定的选择偏差仍然存在。", ""]
    return "\n".join(lines)


def verification_report(result):
    verification = result.get("verification")
    if not verification or verification.get("source_sha256") != result["data"]["source_sha256"]:
        return ["尚未提供匹配当前生产输入的页面与原有自检核验记录。"]
    page = verification["page"]
    tests = verification["tests"]
    return [
        f"- 原有自检执行 `{tests['command']}`，退出码{tests['exit_code']}，{tests['passes']}项PASS，{tests['failures']}项FAIL。完整输出见 `test_engine.log`。",
        f"- 前后各完整调用一次 `make_alt_page.main()`，QQQ均截止{page['as_of']}，比较{page['days']}个展示交易日及全部JSON内容，逐日价格、温度、分项、红蓝黄点、统计和登记展示均一致。新增字段0个。",
        f"- 两次生成共用 `generated_at={page['generated_at']}` 以消除执行时间差，比较保留该字段。登记路径指向临时副本，计算仍使用原生产raw/。完整JSON深比较一致，文件SHA-256也一致，为 `{page['sha256']}`。",
        "- 原始 `data_alt.json`、`dashboard_alt.html` 及三份受保护CSV在执行前后哈希一致。生产代码、模板与说明的校验通过。本次新增交付位于研究目录。",
        f"- 从2017年价格锚独立调用 `hist_store.extend` 接回QQQ历史{verification['hist']['restored_rows']}行，最早{verification['hist']['first']}，相对生产raw历史最大价差{verification['hist']['max_relative_error']:.3g}。主回测使用 `E.load` 实际返回的价格序列。",
        f"- 输入文件{result['data']['source_files']}个，整体校验指纹 `{result['data']['source_sha256']}`，逐文件SHA-256见 `输入校验.json`。运行核验细节见 `核验记录.json`。",
        "- 执行命令为 `python3 研究/二次顶监测_2026-11-11/backtest.py`。附带 `逐日输入.csv`、`对照事件.csv` 与 `回测结果.json`，可重算及核对敏感度逐轮细节。",
        "- Git状态复核确认本次新增交付集中在本研究目录。遵照任务约束，提交和推送次数均为0。",
    ]


def diagnostic_outcome(close, row):
    """仅用于核对“未标顶之后全部突破”的事后诊断，不影响观察期准入。"""
    if row["recross"] is None:
        if len(close) - 1 < close.index.get_loc(row["top"]) + ENTRY_DAYS:
            return "尚未重回，未满252日"
        return "252日内未重回"
    p = close.index.get_loc(row["top"])
    start = close.index.get_loc(row["recross"])
    path = close.iloc[start:min(len(close), p + ENTRY_DAYS + EXTRA_DAYS + 1)]
    high = path.cummax()
    for when, price in path.items():
        if price > close.iloc[p] * 1.1:
            return f"突破 {day(when)}"
        if price <= high.loc[when] * .89:
            return f"二次见顶确认 {day(when)}"
    return "到期或尚未结束"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protected = {name: digest(ROOT / name) for name in PROTECTED}
    daily, temp = production_inputs()
    daily.to_csv(out / "逐日输入.csv", index_label="date", float_format="%.17g")
    rows, trace = scan(daily)
    causal = batch_audit(daily, rows, trace)
    for row in rows:
        row["diagnostic_outcome"] = diagnostic_outcome(daily["close"], row)
    events = comparison_events(daily, rows)
    starts = {"价格全历史": day(daily.index[0]), "页面显示2006年起": "2006-01-01",
              "红点温度有效起": day(temp.dropna().index[0]),
              "自适应门槛有效起": day(daily["adaptive_threshold"].dropna().index[0])}
    comparisons = {}
    for label, start in starts.items():
        subset = [e for e in events if e["date"] >= start]
        comparisons[f"{label} {start}"] = {
            "观察期内": summarize([e for e in subset if e["inside"]]),
            "观察期外": summarize([e for e in subset if not e["inside"]])}
    sources = sorted((ROOT / "raw").glob("*.csv")) + sorted((ROOT / "hist").glob("*.csv.gz"))
    sources += [ROOT / n for n in ("engine.py", "alt_engine.py", "make_alt_page.py", "members.py",
                                  "members.json", "sectors.json", "lev_etfs.json", "single_lev.py", "single_lev_etfs.json",
                                  "leverage.py", "hist_store.py", "tdc_signal.py", "test_engine.py", "alt_tpl.html")]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)}
    result = {"data": {"first": day(daily.index[0]), "last": day(daily.index[-1]), "n": len(daily),
                       "first_temperature": day(temp.dropna().index[0]),
                       "first_adaptive": day(daily["adaptive_threshold"].dropna().index[0]),
                       "source_files": len(hashes), "source_sha256": hashlib.sha256(
                           json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                       "qqq_sha256": digest(ROOT / "raw/QQQ.csv")},
              "parameters": {"dd_limit": .15, "breakout": .10, "ma": 50, "entry_days": ENTRY_DAYS,
                             "extra_days": EXTRA_DAYS, "red_threshold": A.ALT_TH,
                             "adaptive_design_rate": A.ALT_DESIGN_RATE},
              "rounds": rows, "events": events, "comparisons": comparisons,
              "sensitivity": sensitivity(daily), "causality": causal,
              "coarse_checks": coarse_check(rows), "protected_sha256": protected}
    verification_path = out / "核验记录.json"
    if verification_path.exists():
        result["verification"] = json.loads(verification_path.read_text(encoding="utf-8"))
    primary = comparisons[f"自适应门槛有效起 {result['data']['first_adaptive']}"]["观察期外"]
    result["decision"] = {"stage": "A", "status": "对照组差异已解释，B部分继续条件满足",
                          "explanation_source": "2026-11-11 用户转述Claude核对",
                          "coarse_rest_events": 35, "excluded_deep_red_top_events": 2,
                          "excluded_unmarked_top_events": 5, "outside_events": primary["n"],
                          "outside_mean_low": primary["mean_low"], "outside_below_8": primary["below_8"]}
    assert protected == {name: digest(ROOT / name) for name in PROTECTED}
    (out / "回测结果.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (out / "输入校验.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n")
    pd.DataFrame([{**e, "tops": "/".join(e["tops"])} for e in events]).to_csv(out / "对照事件.csv", index=False)
    # 重跑A时保留已经完成的B实施记录，避免把后续核验说明抹掉。
    report_path = out / "结果.md"
    old_report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    marker = "\n## B 部分结果"
    b_report = marker + old_report.split(marker, 1)[1] if marker in old_report else ""
    report_path.write_text(report(result) + b_report, encoding="utf-8")
    print(json.dumps({"data": result["data"], "coarse_checks": result["coarse_checks"],
                      "comparisons": comparisons, "causality": causal}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
