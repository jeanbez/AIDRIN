"""Robust multi-format datetime coercion shared by the timeliness metrics.

Timestamp columns arrive in many shapes — native ``datetime64`` (tz-aware or
naive), ISO/other date strings (possibly mixed formats in one column), and
numeric Unix epochs in seconds/milliseconds/microseconds/nanoseconds. The
timeliness metrics (data-freshness, data-latency) must compare timestamps and a
reference date consistently regardless of how the column was stored, so all of
them funnel through :func:`coerce_datetime`.

Timezone strategy: everything is normalized to **tz-naive UTC**. tz-aware values
are converted to UTC then made naive; epochs are UTC by definition; naive values
are left as-is. Parsing the reference date the same way keeps both sides of every
comparison on the same clock.
"""

import numpy as np
import pandas as pd


def _detect_epoch_unit(abs_values: np.ndarray) -> str:
    """Guess the Unix-epoch unit of a numeric column from its magnitude.

    Anchored to plausible modern timestamps: ~1.7e9 s, ~1.7e12 ms, ~1.7e15 us,
    ~1.7e18 ns. Uses the median so a few outliers don't skew the choice.
    """
    if len(abs_values) == 0:
        return "s"
    median = float(np.nanmedian(abs_values))
    if median < 1e11:
        return "s"
    if median < 1e14:
        return "ms"
    if median < 1e17:
        return "us"
    return "ns"


def coerce_datetime(series: pd.Series):
    """Coerce *series* to tz-naive UTC datetimes, tolerating many input formats.

    Handles native ``datetime64`` (tz-aware normalized to UTC), numeric Unix
    epochs (unit auto-detected), and strings (ISO + common + mixed formats).
    Unparsable values become ``NaT``.

    Returns
    -------
    (pandas.Series, int)
        The coerced ``datetime64[ns]`` series and the number of values that were
        non-null in the input but failed to parse (became ``NaT``).
    """
    non_null_before = int(series.notna().sum())

    if isinstance(series.dtype, pd.DatetimeTZDtype):
        parsed = series.dt.tz_convert("UTC").dt.tz_localize(None)
    elif pd.api.types.is_datetime64_any_dtype(series):
        parsed = series  # already tz-naive datetime64
    elif pd.api.types.is_numeric_dtype(series):
        unit = _detect_epoch_unit(series.dropna().abs().to_numpy())
        parsed = pd.to_datetime(series, unit=unit, errors="coerce")
    else:
        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            parsed = parsed.dt.tz_localize(None)

    n_unparsed = non_null_before - int(parsed.notna().sum())
    return parsed, n_unparsed


def coerce_datetime_scalar(value):
    """Parse a single date value (e.g. a user-supplied reference date) to a
    tz-naive UTC ``pd.Timestamp``, or ``None`` if it cannot be parsed."""
    parsed, _ = coerce_datetime(pd.Series([value]))
    ts = parsed.iloc[0]
    return None if pd.isna(ts) else ts
