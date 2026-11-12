"""VIX 月度期货 S41（第 4 月减第 1 月）与蓝点后的解除日。"""

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen


# Cboe 月度结算价按研究口径拼接。历史只保存倒挂起点、确认解除的三个正值、
# 期末待确认的正值和覆盖边界。未完成的中途正值序列不影响解除日。
# https://www.cboe.com/markets/us/futures/market-statistics/historical-data/futures/
# https://www.cboe.com/markets/us/futures/market-statistics/settlement/futures/daily/
HISTORY = Path(__file__).with_name("vix_term_history.csv")
LIST_URL = "https://www.cboe.com/us/futures/market_statistics/historical_data/product/list/VX/"
CONTRACT_URL = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{}.csv"


def download(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=15) as response:
        return response.read().decode("utf-8-sig")


def contract_settlements(expiry):
    try:
        rows = csv.DictReader(download(CONTRACT_URL.format(expiry)).splitlines())
        result = {}
        for row in rows:
            value = float(row["Settle"] or 0)
            if isfinite(value) and value > 0:
                result[row["Trade Date"]] = value
        return result
    except (OSError, ValueError, KeyError) as exc:
        print(f"  ! Cboe VX {expiry} 结算价暂缺：{exc}")
        return {}


def fetch_spreads(days):
    listing = json.loads(download(LIST_URL))
    expiries = sorted({item["expire_date"] for year in listing.values() for item in year
                       if item.get("futures_root") == "VX" and item.get("duration_type") == "M"})
    by_day = {}
    for day in days:
        start = bisect_right(expiries, day)  # 剔除当天到期合约
        front = expiries[start:start + 4]
        months = [x.year * 12 + x.month for x in map(date.fromisoformat, front)]
        if len(months) == 4 and months == list(range(months[0], months[0] + 4)):
            by_day[day] = front

    needed = sorted({expiry for front in by_day.values() for expiry in front})
    with ThreadPoolExecutor(max_workers=8) as pool:
        contracts = dict(zip(needed, pool.map(contract_settlements, needed)))
    return {day: round(contracts[front[3]][day] - contracts[front[0]][day], 6)
            for day, front in by_day.items() if all(day in contracts[expiry] for expiry in front)}


def load_spreads(dates, raw_dir, refresh=False):
    with HISTORY.open(encoding="utf-8") as f:
        spreads = {r["date"]: float(r["S41"]) for r in csv.DictReader(f)}
    history_end = max(spreads)
    cache_path = Path(raw_dir) / "_vix_s41.json"
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        cached = {}
    except (OSError, ValueError) as exc:
        print(f"  ! 读取 VIX 期限结构缓存失败：{exc}")
        cached = {}
    days = [day for day in dates if day > history_end and day not in cached]
    if refresh and days:
        try:
            updates = fetch_spreads(days)
            if updates:
                cached.update(updates)
                with TemporaryDirectory(dir=cache_path.parent) as directory:
                    tmp = Path(directory) / cache_path.name
                    tmp.write_text(json.dumps(cached, sort_keys=True), encoding="utf-8")
                    tmp.replace(cache_path)
        except (OSError, ValueError, KeyError) as exc:
            print(f"  ! 刷新 VIX 期限结构失败，沿用已知结算价：{exc}")
    return cached | spreads, history_end


def release_markers(dates, blue, spreads):
    """每轮倒挂最多标一次，配对最近一个仍在 126 个交易日内的实心蓝点。"""
    events = {}
    inverted, positive, last_blue = False, 0, None
    for i, (day, is_blue) in enumerate(zip(dates, blue)):
        value = spreads.get(day)
        if value is not None:
            if not inverted:
                inverted = value < 0
            else:
                positive = positive + 1 if value > 0 else 0
                if positive == 3:
                    if last_blue is not None and i - last_blue <= 126:
                        events[day] = {"blue_date": dates[last_blue], "wait": i - last_blue,
                                       "s41": value}
                    inverted, positive, last_blue = False, 0, None
        if inverted and is_blue:
            last_blue = i
    return events


def chart_data(dates, blue, raw_dir, refresh=False):
    spreads, history_end = load_spreads(dates, raw_dir, refresh)
    return {"through": next((day for day in reversed(dates) if day in spreads or day <= history_end), None),
            "events": release_markers(dates, blue, spreads)}
