# 走势图所需的早期利率输入

`treasury_nominal_2004-2014.csv` 与 `treasury_real_2004-2014.csv` 分别保存美国财政部的10年期名义与TIPS实际收益率，格式为无表头 `YYYY-MM-DD,value`，新到旧排序。每份2753条，覆盖2004-01-02至2014-12-31；单位为百分比。

数据来自现有 `fetch_vix_dgs10.py` 的 `_curve_year` 接口：

- [财政部名义收益率CSV（2004年示例）](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2004/all?type=daily_treasury_yield_curve&field_tdr_date_value=2004)
- [财政部实际收益率CSV（2004年示例）](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2004/all?type=daily_treasury_real_yield_curve&field_tdr_date_value=2004)

2026-10-26重新读取官方2004、2008、2011、2014年CSV核对：两类利率分别1001条观察逐日一致。

用途是恢复发布容器缺失的历史输入。发布缓存原来只包含2015年起的利率，导致更早的ERP及蓝点温度为空，缺测被当成了没有触发。刷新时存档只补缺失日期，已有同日数据优先；缺失年份仍会从官方补取。模型权重、阈值与信号规则不变。
