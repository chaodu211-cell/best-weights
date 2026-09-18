#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""替代权重版的每日刷新：拉行情 → 重算 → 渲染 dashboard_alt.html → 报新触发的预警。

与 us2 仓库里的 daily_refresh.py 是同一条流水线，只有第三、四步不同：
那边跑 engine.py + build.py 出 dashboard.html（生产权重、含黑框、含温度图），
这里跑 make_alt_page.py 出 dashboard_alt.html（替代权重、无黑框、无温度图）。

用法：
    python3 alt_refresh.py           全量：每只票重下十年（约 134MB），慢但绝对干净
    python3 alt_refresh.py --fast    增量：只下最近半年再并回本地（约 7MB），日常用这个

增量会自己校验复权刻度，遇到拆股/分红重算的个股当次自动退回全量；
距上次全量超过 30 天也会整体全量一次。所以一直用 --fast 是安全的。
"""
import json, os, subprocess, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, label):
    print(f"\n=== {label} ===")
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, cwd=BASE, capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"{label} 失败（退出码 {r.returncode}），已终止流水线")
    print(f"({time.time()-t0:.0f}s)")


def main():
    fast = "--fast" in sys.argv
    run(["fetch_sp500.py"] + (["--incremental"] if fast else []),
        "① 拉取标普500量价 + EPS" + ("（增量）" if fast else "（全量）"))
    run(["fetch_vix_dgs10.py"], "② 刷新 VIX / 10年期美债 / TIPS实际利率")
    run(["make_alt_page.py"], "③ 重算并渲染 dashboard_alt.html")

    print("\n=== ④ 检查新触发预警 ===")
    d = json.load(open(os.path.join(BASE, "data_alt.json"), encoding="utf-8"))
    al, as_of = d["alerts"], d["as_of"]

    # 判重：同一交易日重跑（周末、当天已刷过）不再报一次，与生产流水线同样的处理
    state_p = os.path.join(BASE, "_alt_state.json")
    prev = None
    if os.path.exists(state_p):
        try:
            prev = json.load(open(state_p, encoding="utf-8")).get("last_as_of")
        except Exception:
            prev = None
    is_new_day = (prev != as_of)

    newly = []
    if is_new_day:
        for rule in al["rules"]:
            f = al["flags"].get(rule["key"]) or []
            if f and f[-1] and not (len(f) > 1 and f[-2]):
                newly.append(rule)
    json.dump({"last_as_of": as_of}, open(state_p, "w", encoding="utf-8"))

    print(f"截至 {as_of}：替代红点温度 {d['temperature_sell']}（门槛 {d['sell_threshold']:g}）  "
          f"蓝点温度(当日口径) {d.get('temperature_fast')}"
          + ("" if is_new_day else "   [与上次运行同一交易日，跳过重复提醒判定]"))
    if not is_new_day:
        print("NEWLY_TRIGGERED:none")
    elif newly:
        print("NEWLY_TRIGGERED:" + ",".join(r["key"] for r in newly))
        for r in newly:
            print(f"  ⚠️ 新触发 {r['name']}：{r['desc'][:120]}…")
    else:
        print("NEWLY_TRIGGERED:none")
        print("  无新触发预警")


if __name__ == "__main__":
    main()
