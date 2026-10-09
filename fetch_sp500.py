#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标普500全成分股 10 年日线拉取 —— 仅用标准库，无需 pip install。

数据源：
    en.wikipedia.org        成分股名单 + GICS 行业
    stockanalysis.com/api   10 年日线（免 key，返回含拆股/分红复权价）
    www.multpl.com          标普500 TTM 每股收益（月频，用于 ERP）与 CPI-U（把它给的实际 EPS 换回名义值）

用法：
    python3 fetch_sp500.py --check      # 连通性自检
    python3 fetch_sp500.py              # 全量拉取
    python3 fetch_sp500.py --limit 30   # 只拉前 30 只（快速验证）

输出：raw/<TICKER>.csv（date,adjclose,close,volume，新到旧）、sectors.json
"""
import argparse, json, os, re, ssl, sys, threading, time
import urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import hist_store as H   # raw/ 只增不减 + hist/ 长历史存档（2026-09-30）
import single_lev as SL  # 红点杠杆因子并入的单股杠杆 ETF（2026-09-30）
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HIST = "https://stockanalysis.com/api/symbol/{kind}/{sym}/history?range={rng}&period=Day"

# 盘中半成品闸门（2026-10-06 加）：美股交易时段里，数据源会把「今天」当成一根日线返回——收盘价是实时价、
# 成交量只有开盘以来那一段。北京时间 21:30 ~ 次日 06:00 之间手动补跑，这根半成品就会被当成最新一天：
# 2026-10-06 22:12 那次补跑，开盘 40 分钟的数据把红点温度算成 86.4（前一天 61.0），还写进了 _信号快照.csv。
# 规则：美东当天 CLOSE_HOUR_ET 点之前，日期 ≥ 美东「今天」的行一律不要；定时运行在美东早上 6~8 点，不受影响。
CLOSE_HOUR_ET = 18


def complete_cutoff(now=None):
    """→ 'YYYY-MM-DD'：只保留日期严格早于它的日线。"""
    now = now or datetime.now(ZoneInfo("America/New_York"))
    d = now.date() if now.hour < CLOSE_HOUR_ET else now.date() + timedelta(days=1)
    return d.isoformat()


CUTOFF = complete_cutoff()

# 数据源改了代码、维基百科（或我们的文件名）还是旧代码：{旧代码: 数据源的新代码}（2026-10-07 加）。
# 文件名、sectors.json、hist/ 存档一律沿用旧代码，只有向数据源要数据时换成新代码；维基百科跟进改名之后，
# constituents() 也把新代码折回旧代码，所以 raw/ 与 hist/ 里的长历史一直接得上。
# 起因：2026-10-06 stockanalysis 把 Paramount Skydance 从 PSKY 改成 SKYD（访问 PSKY 返回 400、页面 301 到 SKYD），
# 维基百科仍写 PSKY。PSKY 连续拉取失败 → 被踢出 sectors.json → 2005 年以来整段历史少了这只票，TOP2、前2% 等
# 横截面因子全部重算：2026-05-29 红点温度 81.5 → 82.8 跨过门槛，近 2 年触发频率 0.60% → 1.19%。
# 没写进表里的改名，fetch_one 会顺着数据源页面的 301 自动识别（核对过与本地旧行是同一只票才用），并提示补进来。
RENAMED = {"PSKY": "SKYD"}
MOVED = {}      # 本次运行自动识别出的改名 {旧代码: 新代码}

GICS_CN = {
    "Information Technology": "信息技术", "Communication Services": "通信服务",
    "Consumer Discretionary": "可选消费", "Financials": "金融", "Health Care": "医疗保健",
    "Industrials": "工业", "Consumer Staples": "日常消费", "Energy": "能源",
    "Utilities": "公用事业", "Real Estate": "房地产", "Materials": "原材料",
}
# 行业ETF（行业成交额集中度）、SPY/QQQ（指数展示/ERP）、
# 杠杆ETF多空两篮（杠杆资金多空比）——15 只必须全部每日刷新，
# 少拉任何一只，该分项就会用陈旧数据继续算，且平滑窗口会把缺口悄悄补上。
LEV_LONG  = ["TQQQ", "UPRO", "SPXL", "SSO", "QLD", "TNA", "SOXL", "FAS", "TECL", "UDOW"]
LEV_SHORT = ["SQQQ", "SPXS", "SDS", "TZA", "SOXS"]
# RSP：最优拟合红点「市值跑赢等权」的等权一侧（2026-10-01 起；2017-03 以前由 hist/RSP_backfill_2004-2017.csv.gz 补回）
ETFS = (["XLK", "XLC", "XLY", "XLF", "XLV", "XLI", "XLP", "XLE", "XLU", "XLRE", "XLB",
         "SPY", "QQQ", "SOXX", "RSP"] + LEV_LONG + LEV_SHORT)   # SOXX：最优拟合页面的展示与评估标的

CTX = ssl.create_default_context()
_lock = threading.Lock()
_last = [0.0]
MIN_GAP = 0.06          # 全局最小请求间隔，避免给对方站点造成压力


def _pace():
    with _lock:
        wait = MIN_GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def get(url, timeout=40, retries=4, pace=True):
    last = None
    for i in range(retries):
        if pace:
            _pace()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 403, 404):
                raise
            time.sleep(1.0 + 1.5 * i)
        except Exception as e:
            last = e
            time.sleep(1.0 + 1.5 * i)
    raise last


def constituents():
    """Wikipedia 成分股表 → {代码: 中文行业}"""
    html = get(WIKI, pace=False)
    m = re.search(r'id="constituents"[^>]*>\s*<tbody[^>]*>(.*?)</tbody>', html, re.S)
    if not m:
        raise SystemExit("Wikipedia 表格结构变化，无法解析")
    out = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        if len(cells) < 4:
            continue
        def clean(x):
            x = re.sub(r"<[^>]+>", "", x)
            return re.sub(r"&amp;", "&", x).replace("&#160;", " ").strip()
        sym, sector = clean(cells[0]), clean(cells[2])
        if re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", sym):
            out[sym.replace(".", "-")] = GICS_CN.get(sector, sector)
    for old, new in RENAMED.items():          # 维基百科跟进改名后，折回旧代码，见 RENAMED
        if new in out and old not in out:
            out[old] = out.pop(new)
    return out


def _moved(sym, is_etf):
    """数据源的个股/ETF 页面 301 到了另一个代码 → 新代码；没改名或取不到 → None"""
    url = f"https://stockanalysis.com/{'etf' if is_etf else 'stocks'}/{sym.lower()}/"
    try:
        _pace()
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=20, context=CTX) as r:
            m = re.search(r"/(?:stocks|etf)/([a-z0-9.\-]+)/?$", urllib.parse.urlparse(r.geturl()).path)
    except Exception:
        return None
    new = m.group(1).upper().replace(".", "-") if m else None
    return new if new and new != sym.upper() else None


def _same_stock(local, rows):
    """改名前后是不是同一只票：与本地旧行重叠的日子里，收盘价（第 3 列，不含分红复权）中位偏差 < 1%"""
    lo = {r[0]: float(r[2]) for r in local}
    dev = sorted(abs(float(r[2]) / lo[r[0]] - 1) for r in rows if lo.get(r[0], 0) > 0)
    return len(dev) >= 20 and dev[len(dev) // 2] < 0.01


def fetch_one(sym, is_etf, rng, src=None):
    """→ (sym, rows, err)；rows = [[date, adjclose, close, volume], ...] 新到旧
    stockanalysis.com 对双类别股（BRK-B、BF-B…）用点号而非连字符，两种写法都试。
    src：向数据源要数据时用的代码（数据源改过名时与 sym 不同，见 RENAMED）；返回的 sym 始终是文件名用的旧代码。"""
    err = None
    src = src or RENAMED.get(sym, sym)
    syms = [src] if "-" not in src else [src, src.replace("-", ".")]
    for s_ in syms:
        for kind in (["e", "s"] if is_etf else ["s", "e"]):
            try:
                j = json.loads(get(HIST.format(kind=kind, sym=s_.lower(), rng=rng)))
                arr = j.get("data") or []
                if not arr:
                    err = "empty"
                    continue
                rows = []
                for r in arr:
                    try:
                        d = str(r["t"])[:10]
                        c, a, v = float(r["c"]), float(r.get("a", r["c"])), float(r["v"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    rows.append([d, f"{a:g}", f"{c:g}", f"{v:.0f}"])
                rows = [r for r in rows if r[0] < CUTOFF]      # 盘中半成品闸门，见 CUTOFF
                rows.sort(reverse=True)
                if rows:
                    return sym, rows, None
                err = "no valid rows"
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
            except Exception as e:
                err = str(e)[:70]
    # 拉不到：看数据源是不是改了代码。本地有旧行的，必须核对是同一只票才用，防止代码被别的公司接手
    if src == RENAMED.get(sym, sym):
        new = _moved(src, is_etf)
        if new:
            s_, rows, err2 = fetch_one(sym, is_etf, rng, src=new)
            local = _read_local(sym)
            if rows and (local is None or _same_stock(local, rows)):
                MOVED[sym] = new
                return sym, rows, None
            err = f"数据源改名为 {new}，但与本地旧行对不上，未采用" if rows else f"数据源改名为 {new}，仍取不到（{err2}）"
    return sym, None, err


# ---------- 增量更新 ----------
# 全量口径每只票要下 10 年（约 253KB）×531 只 ≈ 134MB，在跨境链路上就是十分钟。
# 增量只下最近 INC_RANGE（约 13KB），流量降到 1/20；代价是必须自己处理复权重算：
# 拆股和分红会让数据源把**整段历史**的复权价按比例改写，此时本地旧行与新行不在同一
# 刻度上，直接拼接会在接缝处造出一根凭空的涨跌幅。故每次都比对重叠区间，一旦对不上
# 就对这只票退回全量重拉——每天真正需要重拉的只有当天除权除息的那几十只。
INC_RANGE = "6M"        # 数据源只认 10Y/5Y/1Y/6M/3M，其余值会被当成 1Y
INC_TOL = 5e-4          # 重叠区间复权价的相对容差；超过即判定发生了复权重算
INC_MIN_ROWS = 200      # 本地文件太短（新上市/上次没拉全）就别增量了，直接全量
FULL_EVERY_DAYS = 30    # 距上次全量超过这么多天，自动强制全量一次，防止误差长期累积


def _read_local(sym):
    """读本地 raw/<SYM>.csv → [[date, adjclose, close, volume], ...] 新到旧；没有则 None"""
    p = os.path.join(RAW, f"{sym}.csv")
    if not os.path.exists(p):
        return None
    rows = []
    with open(p) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) == 4 and parts[0][:4].isdigit():
                rows.append(parts)
    return rows or None


def merge_rows(local, fresh):
    """把新拉的短区间并回本地长历史。

    → (合并后的行, 是否需要全量重拉)
    重叠日期的复权价对不上 = 数据源改写了历史刻度，本地旧行作废，必须全量重拉。
    """
    lo = {r[0]: r for r in local}
    overlap = 0
    for r in fresh:
        old_r = lo.get(r[0])
        if old_r is None:
            continue
        overlap += 1
        try:
            a_new, a_old = float(r[1]), float(old_r[1])
        except ValueError:
            continue
        if a_old > 0 and abs(a_new - a_old) / a_old > INC_TOL:
            return None, True                      # 复权刻度变了
    if overlap == 0:
        return None, True                          # 完全没重叠：本地太旧，短区间接不上
    lo.update({r[0]: r for r in fresh})
    return sorted(lo.values(), reverse=True), False


def fetch_one_incremental(sym, is_etf):
    """先试增量；本地缺失/太短/复权刻度变了，就退回全量。→ (sym, rows, err, 是否走了全量)"""
    local = _read_local(sym)
    if local is None or len(local) < INC_MIN_ROWS:
        s, rows, err = fetch_one(sym, is_etf, "10Y")
        return s, rows, err, True
    s, fresh, err = fetch_one(sym, is_etf, INC_RANGE)
    if not fresh:
        return sym, None, err, False
    merged, need_full = merge_rows(local, fresh)
    if need_full:
        s, rows, err = fetch_one(sym, is_etf, "10Y")
        return s, rows, err, True
    return sym, merged, None, False


MULTPL_EPS = "https://www.multpl.com/s-p-500-earnings/table/by-month"
MULTPL_CPI = "https://www.multpl.com/cpi/table/by-month"


def _multpl_table(html):
    """multpl 的月表 → [[YYYY-MM-DD, 值], ...] 新到旧"""
    MON = {m: i + 1 for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
    rows = []
    for mo, dd, yy, val in re.findall(
            r"<td[^>]*>\s*([A-Z][a-z]{2})\s+(\d{1,2}),\s*(\d{4})\s*</td>\s*"
            r"<td[^>]*>(?:\s|&#x2002;|&nbsp;)*(-?[\d.]+)", html):
        rows.append([f"{yy}-{MON[mo]:02d}-{int(dd):02d}", val])
    rows.sort(reverse=True)
    return rows


def fetch_eps():
    """multpl 标普500 TTM 每股收益（月频）→ (实际值行, 名义值行, 说明)。

    multpl 这张表是**实际** EPS：页面注明 "inflation adjusted, constant <月>, <年> dollars"，每月按最新 CPI 整段重新折算。
    ERP 拿它去除名义价格，越早的 E/P 越被抬高，所以要先换回名义值（2026-10-09 起）：
        名义 = 实际 × 当月 CPI ÷ 基准月 CPI
    CPI 用同一网站的 CPI-U（未季调）月表，与 FRED CPIAUCNS 1362 个月逐月一致；比 CPI 表更新的月份用最新一个月的 CPI。
    换算失败时第二项为 None，调用方保留已有的名义文件。"""
    html = get(MULTPL_EPS, pace=False)
    real = _multpl_table(html)
    if not real:
        raise ValueError("EPS 表解析为空")
    try:
        m = re.search(r"constant\s+([A-Z][a-z]+),?\s+(\d{4})\s+dollars", re.sub(r"<[^>]+>", " ", html))   # "constant" 是个链接
        if not m:
            raise ValueError("页面上找不到 constant <月>, <年> dollars")
        ref = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%B %Y").strftime("%Y-%m")
        cpi = {d[:7]: float(v) for d, v in _multpl_table(get(MULTPL_CPI, pace=False))}
        if ref not in cpi:
            raise ValueError(f"CPI 表里没有基准月 {ref}")
        last = max(cpi)
        nom = [[d, f"{float(v) * cpi.get(d[:7], cpi[last] if d[:7] > last else float('nan')) / cpi[ref]:.4f}"]
               for d, v in real]
        nom = [r for r in nom if r[1] != "nan"]
        return real, nom, f"基准 {ref} 美元，CPI 到 {last}"
    except Exception as e:
        return real, None, f"名义换算失败（{e}）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", default="10Y", help="1Y / 5Y / 10Y")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--incremental", action="store_true",
                    help=f"只下最近 {INC_RANGE} 并合并进本地历史；发生复权重算的个股自动退回全量")
    a = ap.parse_args()

    if a.check:
        print("连通性自检：")
        for nm, u in [("en.wikipedia.org", WIKI),
                      ("stockanalysis.com", HIST.format(kind="s", sym="aapl", rng="1Y")),
                      ("www.multpl.com", MULTPL_EPS), ("www.multpl.com（CPI）", MULTPL_CPI)]:
            try:
                t = get(u, timeout=20, retries=1, pace=False)
                print(f"  ✅ {nm}  ({len(t)} 字节)")
            except Exception as e:
                print(f"  ❌ {nm}  {e}")
        return

    os.makedirs(RAW, exist_ok=True)
    print("① 取标普500成分股名单…")
    uni = constituents()
    print(f"   {len(uni)} 只，行业 {len(set(uni.values()))} 个")
    if a.limit:
        uni = dict(list(uni.items())[:a.limit])

    # 单股杠杆 ETF 对照表：超过 30 天就重爬发行商页面补新产品（失败或结果不完整时沿用旧表）
    if not a.limit:
        note = SL.maybe_refresh(get, set(uni))
        if note:
            print("   " + note)
    targets = [(s, False) for s in uni] + [(s, True) for s in ETFS if s not in uni]
    targets += [(s, True) for s in SL.etfs() if s not in uni and s not in ETFS]

    # 距上次全量太久就强制全量一次：增量每次只校验重叠区间，长期跑下去总有边角情况
    # （长时间停机、数据源补历史、个别票停牌）积累不到，定期整体重下一次最省心。
    state_p = os.path.join(RAW, "_fetch_state.json")
    inc = a.incremental
    if inc:
        try:
            last = json.load(open(state_p))["last_full"]
            age = (time.time() - time.mktime(time.strptime(last, "%Y-%m-%d"))) / 86400
            if age > FULL_EVERY_DAYS:
                print(f"   距上次全量 {age:.0f} 天（> {FULL_EVERY_DAYS}），本次强制全量")
                inc = False
        except Exception:
            print("   没有全量记录，本次先走全量")
            inc = False

    mode = f"增量 {INC_RANGE}（自动补全量）" if inc else f"全量 {a.range}"
    print(f"② 拉取 {len(targets)} 个标的 × {mode} 日线，并发 {a.workers}…")
    print(f"   只用 {CUTOFF} 之前的日线（美东当天 {CLOSE_HOUR_ET}:00 之前不收当天那根，防盘中半成品）")
    ok, fail, t0 = [], [], time.time()
    fulls = ext = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        if inc:
            futs = {ex.submit(fetch_one_incremental, s, e): s for s, e in targets}
        else:
            futs = {ex.submit(fetch_one, s, e, a.range): s for s, e in targets}
        for n, f in enumerate(as_completed(futs), 1):
            res = f.result()
            if inc:
                sym, rows, err, was_full = res
                fulls += was_full
            else:
                sym, rows, err = res
            if rows:
                rows = [r for r in rows if r[0] < CUTOFF]      # 增量合并会带回本地旧的半成品行，写盘前再筛一次
            if rows:
                # 数据源只给 10 年：写盘前把本地更早的行、hist/ 存档接回去，起点不再随全量重下往后挪
                rows, n_ext = H.extend(sym, rows, _read_local(sym))
                ext += n_ext > 0
                with open(os.path.join(RAW, f"{sym}.csv"), "w") as fh:
                    fh.write("\n".join(",".join(r) for r in rows) + "\n")
                ok.append((sym, len(rows)))
            else:
                fail.append((sym, err))
            if n % 50 == 0 or n == len(targets):
                extra = f"  其中全量重拉 {fulls}" if inc else ""
                print(f"   {n}/{len(targets)}  成功 {len(ok)}  失败 {len(fail)}{extra}  {time.time()-t0:.0f}s")
    if not inc and not a.limit:
        json.dump({"last_full": time.strftime("%Y-%m-%d")}, open(state_p, "w"))
    if ext:
        print(f"   {ext} 个标的往前接上了本地旧行或 hist/ 存档（数据源只给 10 年）")
    if not a.limit:
        new_m = H.archive(RAW)
        if new_m:
            print(f"   hist/ 新存档 {len(new_m)} 个月份：{new_m[0]} ~ {new_m[-1]}")

    # 当天没拉到、但 raw/（缓存）里有历史的成分股照样留在名单里，只是最近几天没有新行（2026-10-07 加）。
    # 以前直接踢出 sectors.json：一次网络抖动或数据源改名，引擎就把这只票从 2005 年以来的整段历史里拿掉，
    # TOP2、前2% 这些横截面因子的历史读数跟着全部改写（PSKY 的教训，见 RENAMED）。
    # 留下来的代价只在没有新行的那几天：那几天按其余成分股算，和新上市之前的日子一样。
    okset = {s for s, _ in ok}
    stale = {}
    for s, _ in fail:
        loc = _read_local(s) if s in uni else None
        if loc:
            stale[s] = max(r[0] for r in loc)
    sectors = {s: v for s, v in uni.items() if s in okset or s in stale}
    json.dump(sectors, open(os.path.join(BASE, "sectors.json"), "w"), ensure_ascii=False, indent=1)
    # 给流水线的 status.json 与运行摘要用：哪些标的没拉到、哪些成分股在沿用旧行、哪些是自动识别的改名
    json.dump({"cutoff": CUTOFF, "failed": sorted(s for s, _ in fail), "stale": stale, "moved": MOVED},
              open(os.path.join(RAW, "_fetch_report.json"), "w"), ensure_ascii=False)

    print("③ 取标普500 TTM 每股收益…")
    try:
        eps, eps_nom, note = fetch_eps()
        with open(os.path.join(RAW, "_sp500_eps.csv"), "w") as fh:
            fh.write("\n".join(",".join(r) for r in eps) + "\n")
        print(f"   {len(eps)} 个月  {eps[-1][0]} → {eps[0][0]}")
        if eps_nom:
            with open(os.path.join(RAW, "_sp500_eps_nominal.csv"), "w") as fh:
                fh.write("\n".join(",".join(r) for r in eps_nom) + "\n")
            print(f"   名义值（ERP 用）：{note}")
        else:
            print(f"   ! {note}，沿用已有 _sp500_eps_nominal.csv")
    except Exception as e:
        print(f"   ! 失败（{e}），沿用已有 _sp500_eps.csv 与 _sp500_eps_nominal.csv")

    lens = sorted(n for _, n in ok)
    tag = f"（增量，其中 {fulls} 只因复权重算或本地过短走了全量）" if inc else "（全量）"
    print(f"\n完成{tag}：{len(ok)} 成功 / {len(fail)} 失败，用时 {time.time()-t0:.0f}s")
    print(f"  行数中位 {lens[len(lens)//2] if lens else 0}，最少 {lens[0] if lens else 0}")
    by = {}
    for v in sectors.values():
        by[v] = by.get(v, 0) + 1
    print("  行业分布: " + "  ".join(f"{k}{v}" for k, v in sorted(by.items(), key=lambda x: -x[1])))
    if fail:
        print(f"  失败: {[s for s, _ in fail][:12]}{' …' if len(fail) > 12 else ''}")
        for s, e in fail[:12]:
            print(f"    {s}: {e}")
    if stale:
        print(f"  ! {len(stale)} 只成分股今天没拉到，沿用 raw/ 里的旧行、仍留在样本里（不然整段历史都会少这只票）："
              + "、".join(f"{s}（最后一行 {d}）" for s, d in sorted(stale.items())))
    if MOVED:
        print("  ! 数据源改了代码（本次自动识别，已核对是同一只票）：" + "、".join(f"{o} → {n}" for o, n in MOVED.items())
              + "。请把它们写进 fetch_sp500.py 的 RENAMED，免得每天先失败一轮再识别")
    print("\n下一步：python3 engine.py")


if __name__ == "__main__":
    main()
