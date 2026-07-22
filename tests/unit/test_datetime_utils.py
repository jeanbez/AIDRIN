"""Tests for the multi-format datetime coercion helper.

The timeliness metrics rely on coerce_datetime accepting datetime64 (tz-aware
and naive), Unix epochs (s/ms/us), and strings (ISO + mixed formats), and on it
reporting how many values failed to parse.
"""

import unittest

import pandas as pd

from aidrin.file_handling.datetime_utils import (
    coerce_datetime,
    coerce_datetime_scalar,
)

REF = pd.Timestamp("2024-01-15 00:00:00")


class TestCoerceDatetime(unittest.TestCase):
    def test_native_datetime64_naive_unchanged(self):
        s = pd.to_datetime(pd.Series(["2024-01-15", "2024-01-16"]))
        out, n = coerce_datetime(s)
        self.assertEqual(n, 0)
        self.assertEqual(out.iloc[0], REF)

    def test_tz_aware_normalized_to_utc_naive(self):
        # 2024-01-15 05:00 +05:00  ==  2024-01-15 00:00 UTC
        s = pd.Series(pd.to_datetime(["2024-01-15T05:00:00+05:00"]))
        out, n = coerce_datetime(s)
        self.assertEqual(n, 0)
        self.assertIsNone(out.dt.tz)  # tz-naive after normalization
        self.assertEqual(out.iloc[0], REF)

    def test_iso_strings(self):
        s = pd.Series(["2024-01-15", "2024-01-16"])
        out, n = coerce_datetime(s)
        self.assertEqual(n, 0)
        self.assertEqual(out.iloc[0], REF)

    def test_mixed_string_formats(self):
        # Same instant expressed three different ways in one column.
        s = pd.Series(["2024-01-15", "01/15/2024", "2024-01-15 00:00:00"])
        out, n = coerce_datetime(s)
        self.assertEqual(n, 0)
        self.assertTrue((out == REF).all())

    def test_epoch_seconds(self):
        s = pd.Series([int(REF.timestamp())])  # 1705276800
        out, n = coerce_datetime(s)
        self.assertEqual(n, 0)
        self.assertEqual(out.iloc[0], REF)

    def test_epoch_milliseconds(self):
        s = pd.Series([int(REF.timestamp() * 1000)])
        out, n = coerce_datetime(s)
        self.assertEqual(n, 0)
        self.assertEqual(out.iloc[0], REF)

    def test_epoch_microseconds(self):
        s = pd.Series([int(REF.timestamp() * 1_000_000)])
        out, n = coerce_datetime(s)
        self.assertEqual(n, 0)
        self.assertEqual(out.iloc[0], REF)

    def test_unparsable_values_counted(self):
        s = pd.Series(["2024-01-15", "not a date", "also bad", "2024-01-16"])
        out, n = coerce_datetime(s)
        self.assertEqual(n, 2)
        self.assertEqual(out.notna().sum(), 2)

    def test_preexisting_nulls_are_not_parse_failures(self):
        s = pd.Series(["2024-01-15", None, "2024-01-16"])
        out, n = coerce_datetime(s)
        self.assertEqual(n, 0)  # the None was null on input, not a failed parse

    def test_scalar_reference_parsing(self):
        self.assertEqual(coerce_datetime_scalar("2024-01-15"), REF)
        self.assertEqual(coerce_datetime_scalar("01/15/2024"), REF)
        self.assertIsNone(coerce_datetime_scalar("garbage"))


if __name__ == "__main__":
    unittest.main()
