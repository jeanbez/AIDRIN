import base64
import io
import logging

import matplotlib.pyplot as plt
from celery import Task, shared_task
from celery.exceptions import SoftTimeLimitExceeded

from aidrin.file_handling.datetime_utils import coerce_datetime, coerce_datetime_scalar
from aidrin.file_handling.file_parser import read_file

logger = logging.getLogger(__name__)

PRIMARY = "#4485F4"
NEGATIVE = "#D86470"
TEXT = "#6b7280"


@shared_task(bind=True, ignore_result=False)
def data_freshness(self: Task, timestamp_column, reference_time, file_info):
    """Age of the newest record relative to an explicit as-of date.

    Freshness = ``reference_time - max(timestamp_column)``. The reference date is
    required (a static file has no meaningful "now"), and it is echoed back so
    results are reproducible. The timestamp column may be datetime, epoch, or
    string form — see :func:`coerce_datetime`.

    Parameters
    ----------
    timestamp_column : str
        Column defining each record's time.
    reference_time : str
        The as-of date to measure staleness against (e.g. ``"2024-09-20"``).
    file_info : tuple
        ``(file_path, file_name, file_type)``.

    Returns
    -------
    dict
        Newest-record age (days), an age distribution, the reference used, the
        unparsable-timestamp count, and a histogram; or ``{"Error": str}``.
    """
    try:
        logger.info("Data Freshness task started")
        df = read_file(file_info)
        if len(df) == 0:
            return {"Error": "Dataset is empty"}
        if timestamp_column not in df.columns:
            return {"Error": f"Column not found: {timestamp_column!r}"}

        reference = coerce_datetime_scalar(reference_time)
        if reference is None:
            return {"Error": f"Could not parse reference_time {reference_time!r} as a date."}

        timestamps, n_unparsed = coerce_datetime(df[timestamp_column])
        timestamps = timestamps.dropna()
        if timestamps.empty:
            return {"Error": f"No parseable timestamps in {timestamp_column!r}."}

        newest = timestamps.max()
        age_days = (reference - newest).total_seconds() / 86400.0
        ages = (reference - timestamps).dt.total_seconds() / 86400.0

        pct = ages.quantile([0.5, 0.9, 0.99]).round(3)
        age_distribution = {
            "p50": float(pct.loc[0.5]),
            "p90": float(pct.loc[0.9]),
            "p99": float(pct.loc[0.99]),
            "max": float(ages.max()),
        }

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.hist(ages.values, bins=30, color=PRIMARY, edgecolor="white")
        ax.axvline(age_days, color=NEGATIVE, linestyle="--", linewidth=2,
                   label=f"newest = {age_days:.1f} d")
        ax.set_xlabel("Record age at reference date (days)", fontsize=10, color=TEXT)
        ax.set_ylabel("Number of records", fontsize=10, color=TEXT)
        ax.set_title("Record age distribution", fontsize=11, color=TEXT)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(TEXT)
        ax.legend(facecolor="none", edgecolor=TEXT, labelcolor=TEXT, fontsize=8)
        fig.tight_layout(pad=0.5)

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=150, transparent=True)
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        plt.close(fig)

        logger.info("Data Freshness task completed: %.2f days", age_days)
        return {
            "Data Freshness (days)": round(age_days, 3),
            "Newest Record": str(newest),
            "Reference Time": str(reference),
            "Age Distribution (days)": age_distribution,
            "Unparsable Timestamps": int(n_unparsed),
            "Total Records": int(len(df)),
            "Data Freshness Visualization": img_base64,
            "Description": (
                f"The newest record is {age_days:.1f} days old as of {reference.date()}. "
                "Negative values mean records dated after the reference. Lower = fresher."
            ),
        }

    except SoftTimeLimitExceeded:
        logger.error("Data Freshness task timed out")
        raise Exception("Data Freshness task timed out.")
