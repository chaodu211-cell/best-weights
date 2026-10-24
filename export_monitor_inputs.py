#!/usr/bin/env python3
"""Publish validated price inputs used by the dual-model monitor.

Standard library only. Reads existing raw/*.csv, never downloads or modifies them.
Raw CSV layout is date, adjusted close, unadjusted close, volume (no header).
A health failure is serialized; it is never replaced with zero or stale fake data.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path

SYMBOLS = ("QQQ", "SOXX", "TQQQ", "SOXL", "SQQQ", "QLD", "PSQ", "USD")
REQUIRED = frozenset(("QQQ", "SOXX", "TQQQ", "SOXL", "SQQQ"))
MIN_ADV = 10_000_000.0
MIN_HISTORY = 280
MAX_BYTES = 6 * 1024 * 1024


def valid_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("date must have YYYY-MM-DD format")
    if date.fromisoformat(value).isoformat() != value:
        raise ValueError("invalid calendar date")
    return value


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("price must be finite and positive")
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("volume must be finite and nonnegative")
    return number


def read_asset(path, as_of):
    issues, values, observed_dates = [], {}, []
    if not path.is_file():
        return values, None, [{"code": "missing_file", "message": path.name + " is unavailable"}]
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for line, row in enumerate(csv.reader(handle), 1):
                if not row or all(not cell.strip() for cell in row):
                    continue
                if line == 1 and row[0].strip().lower() == "date":
                    continue
                try:
                    if len(row) != 4:
                        raise ValueError("expected exactly date,adjclose,close,volume")
                    day = valid_date(row[0].strip())
                    adjusted, close, volume = positive(row[1]), positive(row[2]), nonnegative(row[3])
                    observed_dates.append(day)
                    if day > as_of:
                        continue
                    point = {"date": day, "adjclose": adjusted, "close": close, "volume": volume}
                    if day in values:
                        if values[day] != point:
                            issues.append({"code": "conflicting_duplicate", "message": f"{day}: conflicting duplicate; first raw row retained"})
                        continue
                    values[day] = point
                except (ValueError, OverflowError) as error:
                    issues.append({"code": "invalid_row", "message": f"line {line}: {error}"})
    except (OSError, UnicodeError, csv.Error) as error:
        issues.append({"code": "unreadable_file", "message": str(error)})
    if not values:
        issues.append({"code": "no_valid_rows", "message": "no valid observations on or before model as_of"})
    elif max(values) != as_of:
        issues.append({"code": "stale_asset", "message": f"latest valid date {max(values)} differs from model as_of {as_of}"})
    return values, max(observed_dates) if observed_dates else None, issues


def build_monitor(raw_dir, as_of, source_commit=None, generated_at=None):
    as_of = valid_date(as_of)
    raw_dir = Path(raw_dir)
    if source_commit is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", source_commit):
        raise ValueError("sourceCommit must be a full 40-character Git SHA")
    generated_at = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    assets, all_values, all_issues = {}, {}, []
    for symbol in SYMBOLS:
        values, source_last, issues = read_asset(raw_dir / (symbol + ".csv"), as_of)
        all_values[symbol] = values
        days = sorted(values)
        assets[symbol] = {
            "firstDate": days[0] if days else None,
            "lastDate": days[-1] if days else None,
            "sourceLastDate": source_last,
            "rowCount": len(days),
            "required": symbol in REQUIRED,
            "health": {"ok": not issues, "issues": issues},
            "rows": [values[day] for day in days],
        }
        for issue in issues:
            all_issues.append({"asset": symbol, **issue, "severity": "error" if symbol in REQUIRED else "warning"})
    calendar = sorted(all_values["QQQ"])
    if len(calendar) < MIN_HISTORY:
        all_issues.append({"asset": "QQQ", "code": "insufficient_history",
                           "message": f"need at least {MIN_HISTORY} QQQ observations; got {len(calendar)}", "severity": "error"})
    # Match bt.Market exactly on the QQQ calendar:
    # C.notna & C.shift(20).notna & rolling(close*volume,60,min_periods=20).mean().shift(1)>=1e7.
    windows = {symbol: deque(maxlen=60) for symbol in SYMBOLS}
    previous_adv = {symbol: None for symbol in SYMBOLS}
    rows = []
    recent = set(calendar[-MIN_HISTORY:])
    for symbol in REQUIRED:
        missing = sorted(day for day in recent if day not in all_values[symbol])
        if missing:
            item = {"code": "missing_recent_prices", "message": f"{len(missing)} missing QQQ-calendar observations in trailing {MIN_HISTORY} rows"}
            assets[symbol]["health"]["ok"] = False
            assets[symbol]["health"]["issues"].append(item)
            all_issues.append({"asset": symbol, **item, "severity": "error"})
    for index, day in enumerate(calendar):
        prices, tradable = {}, {}
        for symbol in SYMBOLS:
            values = all_values[symbol]
            point = values.get(day)
            if point is not None:
                prices[symbol] = point["adjclose"]
            mature = index >= 20 and calendar[index - 20] in values
            adv = previous_adv[symbol]
            tradable[symbol] = bool(point is not None and mature and adv is not None and adv >= MIN_ADV)
            windows[symbol].append(None if point is None else point["close"] * point["volume"])
            observed = [number for number in windows[symbol] if number is not None]
            previous_adv[symbol] = sum(observed) / len(observed) if len(observed) >= 20 else None
        rows.append({"date": day, "prices": prices, "tradable": tradable})
    if source_commit is None:
        all_issues.append({"asset": None, "code": "missing_source_commit", "message": "source Git commit is unknown", "severity": "warning"})
    return {
        "schemaVersion": 1,
        "as_of": as_of,
        "generated_at": generated_at,
        "sourceCommit": source_commit,
        "priceBasis": "adjusted",
        "assets": assets,
        "rows": rows,
        "health": {"ok": not any(issue["severity"] == "error" for issue in all_issues), "issues": all_issues},
    }


def write_monitor(payload, output):
    """Atomic output; serialize no NaN/Infinity and enforce the public size budget."""
    output = Path(output)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_BYTES:
        # prices/tradable in the aligned rows remain the authoritative contract.
        # The duplicated raw rows are diagnostic only, so omit those if necessary.
        for block in payload["assets"].values():
            block.pop("rows", None)
        payload["rawRowsIncluded"] = False
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    else:
        payload["rawRowsIncluded"] = True
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise ValueError("monitor artifact exceeds the 6 MiB size budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".monitor-inputs-", dir=str(output.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return len(encoded)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="raw", help="existing model raw CSV directory")
    parser.add_argument("--data", default="data_alt.json", help="model JSON supplying as_of")
    parser.add_argument("--output", default="_site/monitor-inputs.json")
    parser.add_argument("--source-commit", default=os.environ.get("GITHUB_SHA") or None)
    args = parser.parse_args(argv)
    try:
        with open(args.data, encoding="utf-8") as handle:
            as_of = json.load(handle)["as_of"]
        payload = build_monitor(args.raw, as_of, args.source_commit)
        size = write_monitor(payload, args.output)
        if not payload["health"]["ok"]:
            print("::warning::monitor-inputs.json contains incomplete inputs; monitor must keep its last verified portfolio")
        for issue in payload["health"]["issues"]:
            print("::warning::" + (issue["asset"] + ": " if issue["asset"] else "") + issue["message"])
        print(f"monitor-inputs.json: as_of={as_of}, rows={len(payload['rows'])}, bytes={size}, healthy={payload['health']['ok']}")
        return 0
    except (OSError, KeyError, ValueError, TypeError) as error:
        print(f"::warning::Price monitor export failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

