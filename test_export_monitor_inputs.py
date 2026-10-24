"""Meaningful contract and no-lookahead checks for the public price export."""
import csv
import importlib.util
import json
import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("exporter", Path(__file__).with_name("export_monitor_inputs.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
COMMIT = "a" * 40


def business_dates(count):
    dates, current = [], date(2024, 1, 2)
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current.isoformat())
        current += timedelta(days=1)
    return dates


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.days = business_dates(310)
        for symbol in M.REQUIRED:
            self.write(symbol, [(day, 90.5, 100, 1_000_000) for day in self.days])

    def tearDown(self):
        self.temp.cleanup()

    def write(self, symbol, rows):
        with (self.raw / (symbol + ".csv")).open("w", newline="") as handle:
            csv.writer(handle).writerows(reversed(rows))

    def build(self, as_of=None):
        return M.build_monitor(self.raw, as_of or self.days[-1], COMMIT, "2026-10-25T00:00:00Z")

    def test_sorted_full_history_and_adjusted_price_contract(self):
        payload = self.build()
        self.assertTrue(payload["health"]["ok"])
        self.assertEqual([row["date"] for row in payload["rows"]], self.days)
        self.assertEqual(payload["rows"][-1]["prices"]["QQQ"], 90.5)
        self.assertEqual(payload["assets"]["QQQ"]["rows"][0]["close"], 100)
        self.assertEqual(payload["assets"]["QQQ"]["rowCount"], 310)
        self.assertEqual(payload["priceBasis"], "adjusted")

    def test_future_rows_are_excluded_without_losing_actual_source_date(self):
        payload = self.build(self.days[-2])
        self.assertEqual(payload["as_of"], self.days[-2])
        self.assertEqual(payload["rows"][-1]["date"], self.days[-2])
        self.assertEqual(payload["assets"]["QQQ"]["sourceLastDate"], self.days[-1])
        self.assertEqual(payload["assets"]["QQQ"]["lastDate"], self.days[-2])

    def test_missing_optional_assets_do_not_impersonate_zero_prices(self):
        payload = self.build()
        self.assertTrue(payload["health"]["ok"])
        self.assertFalse(payload["assets"]["USD"]["health"]["ok"])
        self.assertNotIn("USD", payload["rows"][-1]["prices"])
        self.assertFalse(payload["rows"][-1]["tradable"]["USD"])

    def test_missing_required_asset_blocks_monitor_health(self):
        (self.raw / "SQQQ.csv").unlink()
        payload = self.build()
        self.assertFalse(payload["health"]["ok"])
        self.assertNotIn("SQQQ", payload["rows"][-1]["prices"])

    def test_stale_asset_preserves_its_real_last_date(self):
        self.write("SOXX", [(day, 10, 11, 10_000_000) for day in self.days[:-1]])
        payload = self.build()
        self.assertFalse(payload["health"]["ok"])
        self.assertEqual(payload["assets"]["SOXX"]["lastDate"], self.days[-2])
        self.assertNotIn("SOXX", payload["rows"][-1]["prices"])

    def test_adv_does_not_look_at_current_day_volume(self):
        # Previous 20 rows are only $1m/day, today's row is $100bn.
        rows = [(day, 10, 10, 100_000 if index < 20 else 10_000_000_000)
                for index, day in enumerate(self.days)]
        self.write("TQQQ", rows)
        output = self.build()["rows"]
        self.assertFalse(output[20]["tradable"]["TQQQ"])
        self.assertTrue(output[21]["tradable"]["TQQQ"])

    def test_twenty_day_maturity_matches_qqq_calendar(self):
        output = self.build()["rows"]
        self.assertFalse(output[19]["tradable"]["QQQ"])
        self.assertTrue(output[20]["tradable"]["QQQ"])

    def test_maturity_requires_a_price_exactly_twenty_calendar_rows_ago(self):
        self.write("TQQQ", [(day, 90.5, 100, 1_000_000)
                            for index, day in enumerate(self.days) if index != 280])
        output = self.build()["rows"]
        self.assertFalse(output[300]["tradable"]["TQQQ"])

    def test_nonfinite_or_nonpositive_prices_are_rejected(self):
        for invalid in ("nan", "inf", "-inf", 0, -1):
            with self.subTest(value=invalid):
                self.write("TQQQ", [(day, invalid if day == self.days[-1] else 10, 10, 1_000_000) for day in self.days])
                payload = self.build()
                self.assertFalse(payload["health"]["ok"])
                self.assertNotIn("TQQQ", payload["rows"][-1]["prices"])

    def test_invalid_date_volume_and_conflicting_duplicate_are_visible(self):
        with (self.raw / "TQQQ.csv").open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["2026-02-30", 100, 100, 100])
            writer.writerow([self.days[-1], 90.5, 100, -1])
            writer.writerow([self.days[0], 999, 999, 100])
        issues = self.build()["assets"]["TQQQ"]["health"]["issues"]
        self.assertGreaterEqual(sum(issue["code"] == "invalid_row" for issue in issues), 2)
        self.assertTrue(any(issue["code"] == "conflicting_duplicate" for issue in issues))

    def test_finite_inputs_cannot_overflow_dollar_volume(self):
        self.write("SOXL", [(day, 100, 1e308, 1e308) for day in self.days])
        payload = self.build()
        self.assertFalse(payload["health"]["ok"])
        self.assertNotIn("SOXL", payload["rows"][-1]["prices"])

    def test_insufficient_history_and_bad_source_sha_are_explicit(self):
        payload = self.build(self.days[25])
        self.assertFalse(payload["health"]["ok"])
        self.assertTrue(any(issue["code"] == "insufficient_history" for issue in payload["health"]["issues"]))
        with self.assertRaises(ValueError):
            M.build_monitor(self.raw, self.days[-1], "short-sha")
        with self.assertRaises(ValueError):
            self.build("2026-02-30")

    def test_atomic_writer_retains_complete_monitor_contract(self):
        output = self.root / "subdir" / "monitor-inputs.json"
        payload = self.build()
        size = M.write_monitor(payload, output)
        self.assertEqual(size, len(output.read_bytes()))
        loaded = json.loads(output.read_text())
        self.assertTrue(loaded["rawRowsIncluded"])
        self.assertEqual(loaded["rows"], payload["rows"])
        self.assertEqual(list(output.parent.glob(".monitor-inputs-*")), [])

    def test_size_budget_discards_only_duplicate_raw_rows(self):
        payload = self.build()
        raw_size = len(json.dumps(payload, separators=(",", ":")).encode())
        compact = json.loads(json.dumps(payload))
        for asset in compact["assets"].values():
            asset.pop("rows", None)
        compact_size = len(json.dumps(compact, separators=(",", ":")).encode())
        output = self.root / "monitor.json"
        with patch.object(M, "MAX_BYTES", (raw_size + compact_size) // 2):
            M.write_monitor(payload, output)
        loaded = json.loads(output.read_text())
        self.assertFalse(loaded["rawRowsIncluded"])
        self.assertEqual(loaded["rows"], compact["rows"])

    def test_cli_writes_unhealthy_artifact_but_invalid_model_input_fails(self):
        data = self.root / "data.json"
        output = self.root / "monitor.json"
        data.write_text(json.dumps({"as_of": self.days[-1]}))
        self.assertEqual(M.main(["--raw", str(self.raw), "--data", str(data), "--output", str(output), "--source-commit", COMMIT]), 0)
        data.write_text('{"as_of":"bad"}')
        self.assertEqual(M.main(["--raw", str(self.raw), "--data", str(data), "--output", str(output)]), 1)


if __name__ == "__main__":
    unittest.main()

