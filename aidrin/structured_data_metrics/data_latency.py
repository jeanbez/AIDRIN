import base64
import io
import logging

import matplotlib.pyplot as plt
from celery import Task, shared_task
from celery.exceptions import SoftTimeLimitExceeded

from aidrin.file_handling.datetime_utils import coerce_datetime
from aidrin.file_handling.file_parser import read_file

logger = logging.getLogger(__name__)

PRIMARY = "#4485F4"
NEGATIVE = "#D86470"
TEXT = "#6b7280"


@shared_task(bind=True, ignore_result=False)
def data_latency(self: Task, event_column, availability_column, file_info):
    """Delay between an event and when it became available, per record.

    Latency = ``availability_column - event_column`` (seconds). Both columns may
    be datetime, epoch, or string form — see :func:`coerce_datetime`. Negative
    latencies (availability before the event) usually mean clock skew or column
    mislabeling and are counted separately rather than hidden.

    Parameters
    ----------
    event_column : str
        Column with the earlier (event) timestamp.
    availability_column : str
        Column with the later (availability/ingestion) timestamp.
    file_info : tuple
        ``(file_path, file_name, file_type)``.

    Returns
    -------
    dict
        Mean/median/p95/max latency (seconds), the negative-latency count, the
        unparsable-timestamp count, and a histogram; or ``{"Error": str}``.
    """
    try:
        logger.info("Data Latency task started")
        df = read_file(file_info)
        if len(df) == 0:
            return {"Error": "Dataset is empty"}
        for col in (event_column, availability_column):
            if col not in df.columns:
                return {"Error": f"Column not found: {col!r}"}
        if event_column == availability_column:
            return {"Error": "event_column and availability_column must differ."}

        event_ts, n_event_bad = coerce_datetime(df[event_column])
        avail_ts, n_avail_bad = coerce_datetime(df[availability_column])

        delay = (avail_ts - event_ts).dt.total_seconds()
        delay = delay.dropna()
        if delay.empty:
            return {"Error": "No records have both timestamps parseable."}

        negative = int((delay < 0).sum())
        pct = delay.quantile([0.5, 0.95])

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.hist(delay.values, bins=30, color=PRIMARY, edgecolor="white")
        ax.axvline(0, color=NEGATIVE, linestyle="--", linewidth=1)
        ax.set_xlabel("Latency (seconds)", fontsize=10, color=TEXT)
        ax.set_ylabel("Number of records", fontsize=10, color=TEXT)
        ax.set_title("Latency distribution", fontsize=11, color=TEXT)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(TEXT)
        fig.tight_layout(pad=0.5)

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=150, transparent=True)
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        plt.close(fig)

        logger.info("Data Latency task completed: median %.1fs", float(pct.loc[0.5]))
        return {
            "Mean Latency (s)": round(float(delay.mean()), 3),
            "Median Latency (s)": round(float(pct.loc[0.5]), 3),
            "P95 Latency (s)": round(float(pct.loc[0.95]), 3),
            "Max Latency (s)": round(float(delay.max()), 3),
            "Negative Latency Count": negative,
            "Unparsable Timestamps": int(n_event_bad + n_avail_bad),
            "Records Considered": int(len(delay)),
            "Data Latency Visualization": img_base64,
            "Description": (
                f"Median delay from {event_column!r} to {availability_column!r} is "
                f"{float(pct.loc[0.5]):.1f}s. {negative} record(s) have negative latency "
                "(availability before event — check for clock skew or swapped columns)."
            ),
        }

    except SoftTimeLimitExceeded:
        logger.error("Data Latency task timed out")
        raise Exception("Data Latency task timed out.")
