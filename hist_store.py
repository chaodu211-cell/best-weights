#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长历史：raw/ 只增不减 + hist/ 存档（2026-09-30 加）。仅用标准库（fetch_sp500.py 也是）。

为什么要有：数据源（stockanalysis）只给 10 年。以前每 30 天全量重下时直接覆盖本地文件，起点跟着往后挪：
  · TDC（5 年分位窗口）、自适应门槛（3 年）在起点之后几年回看期不满，页面上 2017-2022 的读数是"残缺窗口"算的；
  · 起点每挪一次，这些历史点就变一次（实测：起点后移一年，2022-01-20 的黄点推迟到 01-24）。
现在：
  · raw/ 只增不减：全量重下后，本地比新数据更早的行按重叠区间的复权比例缩放后接回去（extend 第①步）；
  · hist/ 存"快要滑出数据源窗口"的月份，GitHub 上 raw/ 只在 Actions 缓存里，缓存丢了就从这里恢复（extend 第②步）。
    2016-09-22 以前的部分来自 ~/us2/raw_long（stooq 长历史，已按 stockanalysis 刻度缩放），2026-09-30 一次性导入。
实测补足回看期后：红点、蓝点温度一天不变；TDC 在 2022-09 以前、自适应门槛在 2021-01 以前变成完整窗口的读数，
黄点新增 2018-03-27、2020-03-05（tdc_signal.py 里"五次大跌前预警"之一），2018-10-10 挪到 10-12。

存档格式：hist/YYYY-MM.csv.gz，每行 sym,date,next,fwd,rawclose,volume
  fwd = 该标的下一交易日 next 的复权收盘 ÷ 当日复权收盘 − 1。
  存涨跌幅不存复权价：分红、拆股会让数据源把整段历史的复权价按比例改写，存下来的价格过一阵就和新数据不在一个刻度上；
  涨跌幅不随之后的复权改变，拼接时从已知的最早一天沿 next 往回推，就能还原出与当前刻度一致的复权价。
  原始收盘价与成交量（engine 只用二者之积＝成交额）本来就不随复权变。
每个月份文件只写一次，之后不改。

    python3 hist_store.py          # 看存档覆盖情况
"""
import csv, gzip, io, os, sys
from datetime import date, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(BASE, "hist")
SOURCE_YEARS = 10     # 数据源窗口
AHEAD_DAYS = 180      # 月末早于"今天 − 10 年 + 180 天"就存档：留半年余量，流水线停几个月、缓存丢了也不会漏
MIN_OVERLAP = 20      # 本地旧行与新数据至少重叠这么多天才接
MAX_SPREAD = 0.01     # 重叠区间复权比例的离散度上限：超过说明不是同一只票（代码被别的公司沿用）或数据有问题
HEADER = ["sym", "date", "next", "fwd", "rawclose", "volume"]

_arch = None


def _fmt(x):
    return f"{x:.7g}"


def load():
    """→ {sym: {date: (next, fwd, rawclose, volume)}}，整个进程只读一次"""
    global _arch
    if _arch is None:
        _arch = {}
        if os.path.isdir(HIST):
            for f in sorted(os.listdir(HIST)):
                if not f.endswith(".csv.gz"):
                    continue
                with gzip.open(os.path.join(HIST, f), "rt", encoding="utf-8", newline="") as fh:
                    rd = csv.reader(fh)
                    next(rd, None)
                    for sym, d, nxt, fwd, raw, vol in rd:
                        _arch.setdefault(sym, {})[d] = (nxt, float(fwd), raw, vol)
    return _arch


def extend(sym, rows, local=None):
    """rows：即将写进 raw/<sym>.csv 的行（新到旧，[date, adjclose, close, volume]），往前接历史。→ (rows, 接上的行数)

    ① local（本地 raw/ 原有的行）里比 rows 更早的部分：用最早 60 个重叠日的复权比例缩放后接上；
       重叠不足或比例不稳（不是同一只票）就不接。
    ② hist/ 存档里比 rows 更早的部分：从已知最早一天沿 next 往回推复权价；链断了（某天对不上）就停在那里。
    """
    if not rows:
        return rows, 0
    n0, first = len(rows), rows[-1][0]
    if local:
        older = [r for r in local if r[0] < first]
        if older:
            adj = {r[0]: float(r[1]) for r in rows}
            ks = [adj[r[0]] / float(r[1]) for r in sorted(local) if r[0] in adj and float(r[1]) > 0][:60]
            if len(ks) >= MIN_OVERLAP and max(ks) / min(ks) - 1 <= MAX_SPREAD:
                k = sorted(ks)[len(ks) // 2]
                rows = rows + [[r[0], _fmt(float(r[1]) * k)] + r[2:] for r in older]
                first = rows[-1][0]
    arch = load().get(sym)
    if arch:
        adj = {r[0]: float(r[1]) for r in rows}
        add = []
        for d in sorted((d for d in arch if d < first), reverse=True):
            nxt, fwd, raw, vol = arch[d]
            if nxt not in adj:
                break
            adj[d] = adj[nxt] / (1.0 + fwd)
            add.append([d, _fmt(adj[d]), raw, vol])
        rows = rows + add
    return rows, len(rows) - n0


def _month_end(m):
    y, mo = int(m[:4]), int(m[5:7])
    nxt = date(y + (mo == 12), mo % 12 + 1, 1)
    return (nxt - timedelta(days=1)).isoformat()


def archive(raw_dir, today=None):
    """把 raw_dir 里"快要滑出数据源窗口"的完整月份写进 hist/（已存在的月份不动）。→ 新写的月份列表"""
    today = today or date.today()
    cut = (date(today.year - SOURCE_YEARS, today.month, 1) + timedelta(days=AHEAD_DAYS)).isoformat()
    have = set(f[:7] for f in os.listdir(HIST)) if os.path.isdir(HIST) else set()
    ends = {}
    out = {}
    for f in sorted(os.listdir(raw_dir)):
        if f.startswith("_") or not f.endswith(".csv"):
            continue
        rows = []
        with open(os.path.join(raw_dir, f)) as fh:
            for line in fh:
                p = line.strip().split(",")
                if len(p) == 4 and p[0][:4].isdigit():
                    rows.append(p)
        rows.sort()
        for i in range(len(rows) - 1):
            d, a = rows[i][0], float(rows[i][1])
            m = d[:7]
            if m in have or ends.setdefault(m, _month_end(m)) >= cut or a <= 0:
                continue
            nxt = rows[i + 1]
            out.setdefault(m, []).append([f[:-4], d, nxt[0], f"{float(nxt[1]) / a - 1:.8g}", rows[i][2], rows[i][3]])
    os.makedirs(HIST, exist_ok=True)
    for m, lines in sorted(out.items()):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(HEADER)
        w.writerows(lines)
        # mtime=0：同样的内容压出同样的字节，重跑不会产生无意义的 git 改动
        with open(os.path.join(HIST, f"{m}.csv.gz"), "wb") as fh:
            fh.write(gzip.compress(buf.getvalue().encode("utf-8"), 9, mtime=0))
    global _arch
    _arch = None
    return sorted(out)


if __name__ == "__main__":
    A = load()
    fs = sorted(f for f in os.listdir(HIST) if f.endswith(".csv.gz")) if os.path.isdir(HIST) else []
    if not fs:
        sys.exit("hist/ 还没有存档")
    size = sum(os.path.getsize(os.path.join(HIST, f)) for f in fs)
    print(f"hist/：{len(fs)} 个月份（{fs[0][:7]} ~ {fs[-1][:7]}），{len(A)} 个标的，"
          f"{sum(len(v) for v in A.values()):,} 行，{size / 1e6:.1f} MB")
