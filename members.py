# -*- coding: utf-8 -*-
"""成分股的在册区间（2026-11-10 加）：每只股票只在它当时确实在标普 500 里的日子进样本。

为什么要有：以前 sectors.json 只有「今天的名单」，引擎把这 503 只当成一直都在指数里，回填全部历史。两个后果：
  ① 后来者提前出现。今天的名单里 127 只是 2017-10-18（训练窗起点）之后才纳入的。特斯拉 2020-12-21 才纳入，
     2020 年 2 月上旬却占了模型里全篮子成交额的 11.8%，把「前 2% 成交额占比」的原始值从 26.7% 抬到 34.4%；
     只拿掉它，2020-01~02 那段红点就从 26 天变成 11 天。
  ② 名单每调整一次，历史读数整段改写（新成员带着全部历史进来，被剔除的整段消失）。
按在册区间取样本后：新成员只从纳入日算起，被剔除的保留到剔除日，过去的读数不再随名单变动。

members.json（进 git，由 fetch_sp500.py 维护，只增不改）：
    {代码: {"sector": 中文行业, "spans": [[纳入日, 剔除日], ...]}}
  纳入日为 null = 数据起点之前就在；剔除日为 null = 现在仍在。日期是「第一个在册日」与「第一个不在册日」。
  · 纳入日取自维基百科成分股表的 Date added，**第一次记下之后不再跟着维基百科改**（免得那边一处编辑改写历史）。
  · 某只股票不在当天名单里了 → 把它最后一段的剔除日记成当天；行业与历史行情都留着。以后重新纳入 → 新开一段。
  · 局限：2026-11-10 以前就被剔除的旧成分股没有行情数据，补不回来；这份台账只管得住「后来者」和今后的调整。

只用标准库，fetch_sp500.py（不依赖 pandas）与 engine.py 共用。
"""
import json
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "members.json")
MIN_NAMES = 450     # 维基百科解析出来的名单少于这个数就不动台账（多半是页面结构变了或被人改坏）


def load(path=None):
    p = path or PATH
    if not os.path.exists(p):
        return {}
    return json.load(open(p, encoding="utf-8"))


def save(led, path=None):
    with open(path or PATH, "w", encoding="utf-8") as fh:
        json.dump({k: led[k] for k in sorted(led)}, fh, ensure_ascii=False, indent=0)
        fh.write("\n")


def update(led, uni, added, today):
    """按当天的名单更新台账，返回 (台账, 说明列表)。

    uni: {代码: 行业}（当天维基百科名单）；added: {代码: 'YYYY-MM-DD' 或 None}；today: 当天日期（第一个不在册日用它）。
    """
    notes = []
    if len(uni) < MIN_NAMES:
        return led, [f"名单只有 {len(uni)} 只（< {MIN_NAMES}），成分股台账本次不动"]
    led = {k: {"sector": v["sector"], "spans": [list(s) for s in v["spans"]]} for k, v in led.items()}
    for s, sec in uni.items():
        if s not in led:
            led[s] = {"sector": sec, "spans": [[added.get(s), None]]}
            notes.append(f"新成员 {s}（纳入日 {added.get(s) or '未知，按一直在册'}）")
            continue
        led[s]["sector"] = sec
        last = led[s]["spans"][-1]
        if last[1] is not None:                       # 之前被记为剔除，现在又在名单里
            a = added.get(s)
            if a is None or a <= last[1]:             # 维基百科的纳入日不晚于我们记的剔除日：当成名单抖动，接回去
                last[1] = None
                notes.append(f"{s} 回到名单，接回原区间")
            else:
                led[s]["spans"].append([a, None])
                notes.append(f"{s} 重新纳入（{a}）")
    for s, v in led.items():
        if s not in uni and v["spans"][-1][1] is None:
            v["spans"][-1][1] = today
            notes.append(f"{s} 不在名单里了，剔除日记为 {today}")
    return led, notes


def active(led, sym, dates):
    """dates（pandas DatetimeIndex）里哪些日子 sym 在册 → 布尔 numpy 数组。台账里没有这只票 = 一直在册。"""
    import numpy as np
    import pandas as pd
    v = led.get(sym)
    if not v:
        return np.ones(len(dates), dtype=bool)
    ok = np.zeros(len(dates), dtype=bool)
    for a, b in v["spans"]:
        m = np.ones(len(dates), dtype=bool)
        if a:
            m &= dates >= pd.Timestamp(a)
        if b:
            m &= dates < pd.Timestamp(b)
        ok |= m
    return ok
