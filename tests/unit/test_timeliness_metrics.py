"""Unit tests for the timeliness metrics (data-freshness, data-latency).

Mirrors the eager-Celery + _write_csv pattern used by the other metric suites,
and exercises the multi-format datetime handling (string columns and epoch
columns) plus the required-parameter and error paths.
"""

import base64
import binascii
import os
import sys
import tempfile
import types
import unittest

import pandas as pd

if "pkg_resources" not in sys.modules:
    _pkg = types.ModuleType("pkg_resources")

    class _FakeDist:
        version = "0.0.0"

    _pkg.get_distribution = lambda _name: _FakeDist()
    sys.modules["pkg_resources"] = _pkg

from celery import Celery  # noqa: E402

_celery_app = Celery("tests")
_celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)
_celery_app.set_default()

from aidrin.structured_data_metrics.data_freshness import data_freshness  # noqa: E402
from aidrin.structured_data_metrics.data_latency import data_latency  # noqa: E402


def _write_csv(df: pd.DataFrame) -> tuple:
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return (tmp.name, os.path.basename(tmp.name), ".csv")


def _clean(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _is_base64(s):
    try:
        return isinstance(s, str) and len(base64.b64decode(s, validate=True)) > 0
    except (binascii.Error, ValueError):
        return False


class TestDataFreshness(unittest.TestCase):
    def test_age_of_newest_record(self):
        df = pd.DataFrame({"ts": ["2024-01-10", "2024-01-15", "2024-01-05"]})
        fi = _write_csv(df)
        try:
            r = data_freshness.apply(args=("ts", "2024-01-20", fi)).get()
        finally:
            _clean(fi[0])
        # newest = 2024-01-15, reference 2024-01-20 -> 5 days
        self.assertAlmostEqual(r["Data Freshness (days)"], 5.0, places=3)
        self.assertEqual(r["Unparsable Timestamps"], 0)
        self.assertTrue(_is_base64(r["Data Freshness Visualization"]))
        self.assertEqual(r["Reference Time"], "2024-01-20 00:00:00")

    def test_epoch_seconds_column(self):
        # 2024-01-15 as epoch seconds — exercises the multi-format path.
        epoch = int(pd.Timestamp("2024-01-15").timestamp())
        df = pd.DataFrame({"ts": [epoch]})
        fi = _write_csv(df)
        try:
            r = data_freshness.apply(args=("ts", "2024-01-20", fi)).get()
        finally:
            _clean(fi[0])
        self.assertAlmostEqual(r["Data Freshness (days)"], 5.0, places=3)

    def test_bad_reference_errors(self):
        df = pd.DataFrame({"ts": ["2024-01-10"]})
        fi = _write_csv(df)
        try:
            r = data_freshness.apply(args=("ts", "not-a-date", fi)).get()
        finally:
            _clean(fi[0])
        self.assertIn("Error", r)

    def test_missing_column_errors(self):
        df = pd.DataFrame({"ts": ["2024-01-10"]})
        fi = _write_csv(df)
        try:
            r = data_freshness.apply(args=("nope", "2024-01-20", fi)).get()
        finally:
            _clean(fi[0])
        self.assertIn("Error", r)


class TestDataLatency(unittest.TestCase):
    def test_delay_stats(self):
        df = pd.DataFrame({
            "event": ["2024-01-01 00:00:00", "2024-01-01 00:00:00"],
            "avail": ["2024-01-01 00:01:00", "2024-01-01 00:02:00"],  # 60s, 120s
        })
        fi = _write_csv(df)
        try:
            r = data_latency.apply(args=("event", "avail", fi)).get()
        finally:
            _clean(fi[0])
        self.assertAlmostEqual(r["Median Latency (s)"], 90.0, places=1)
        self.assertAlmostEqual(r["Max Latency (s)"], 120.0, places=1)
        self.assertEqual(r["Negative Latency Count"], 0)
        self.assertEqual(r["Records Considered"], 2)
        self.assertTrue(_is_base64(r["Data Latency Visualization"]))

    def test_negative_latency_flagged(self):
        df = pd.DataFrame({
            "event": ["2024-01-01 00:05:00"],
            "avail": ["2024-01-01 00:00:00"],  # availability before event
        })
        fi = _write_csv(df)
        try:
            r = data_latency.apply(args=("event", "avail", fi)).get()
        finally:
            _clean(fi[0])
        self.assertEqual(r["Negative Latency Count"], 1)

    def test_same_column_errors(self):
        df = pd.DataFrame({"t": ["2024-01-01"]})
        fi = _write_csv(df)
        try:
            r = data_latency.apply(args=("t", "t", fi)).get()
        finally:
            _clean(fi[0])
        self.assertIn("Error", r)

    def test_unparsable_pairs_error(self):
        df = pd.DataFrame({"event": ["x", "y"], "avail": ["p", "q"]})
        fi = _write_csv(df)
        try:
            r = data_latency.apply(args=("event", "avail", fi)).get()
        finally:
            _clean(fi[0])
        self.assertIn("Error", r)


if __name__ == "__main__":
    unittest.main()
