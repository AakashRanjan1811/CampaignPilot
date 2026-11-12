"""Segmentation Agent: RFM feature engineering + K-Means.

This step is deliberately deterministic (no LLM): customer segments are a
data-science problem, and the LLM agents downstream reason over the profiles
this module produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import CHANNELS, Dataset


@dataclass
class SegmentationResult:
    rfm: pd.DataFrame  # customer_id, recency, frequency, monetary, segment_id
    profiles: list[dict]
    k: int
    silhouette_by_k: dict[int, float]


def compute_rfm(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.assign(line_value=transactions["quantity"] * transactions["unit_price"])
    snapshot = tx["invoice_date"].max() + pd.Timedelta(days=1)
    rfm = tx.groupby("customer_id").agg(
        last_purchase=("invoice_date", "max"),
        frequency=("invoice_no", "nunique"),
        monetary=("line_value", "sum"),
    )
    rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
    return rfm.drop(columns="last_purchase").reset_index()


def _features(rfm: pd.DataFrame) -> np.ndarray:
    # RFM values are heavily right-skewed; log1p then standardise.
    X = np.log1p(rfm[["recency", "frequency", "monetary"]].to_numpy(dtype=float))
    return StandardScaler().fit_transform(X)


def choose_k(X: np.ndarray, k_min: int, k_max: int, random_state: int) -> tuple[int, dict[int, float]]:
    scores = {}
    sample = min(len(X), 3000)
    for k in range(k_min, min(k_max, len(X) - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels, sample_size=sample, random_state=random_state))
    best = max(scores, key=scores.get)
    return best, scores


def percentile_position(rfm: pd.DataFrame, group: pd.DataFrame) -> dict[str, float]:
    """Where a cluster's typical customer sits in the whole customer base (0 = lowest)."""
    pos = {col: float((rfm[col] < group[col].median()).mean()) for col in ("recency", "frequency", "monetary")}
    pos["avg_order_value"] = float((rfm["avg_order_value"] < group["avg_order_value"].mean()).mean())
    return pos


def name_segment(pos: dict[str, float]) -> str:
    """Standard RFM segment semantics, expressed on percentile positions.

    Lower recency percentile = bought more recently.
    """
    r, f, aov = pos["recency"], pos["frequency"], pos["avg_order_value"]
    if aov < 0.15 and f >= 0.35:
        return "Bargain Hunters"
    if r < 0.25 and f >= 0.85:
        return "Champions"
    if r < 0.5 and f >= 0.55:
        return "Loyal Customers"
    if r < 0.5 and f < 0.35:
        return "New & Promising"
    if r >= 0.55 and f >= 0.55:
        return "At Risk"
    if r >= 0.7:
        return "Hibernating"
    return "Needs Attention"


def profile_segments(rfm: pd.DataFrame, customers: pd.DataFrame) -> list[dict]:
    merged = rfm.merge(customers, on="customer_id", how="left")
    profiles, used = [], {}
    for seg_id, group in merged.groupby("segment_id"):
        medians = group[["recency", "frequency", "monetary"]].median()
        name = name_segment(percentile_position(rfm, group))
        used[name] = used.get(name, 0) + 1
        if used[name] > 1:
            name = f"{name} {used[name]}"  # two clusters share a semantic name
        channel_stats = {}
        for ch in CHANNELS:
            opted = group[group[f"{ch}_opt_in"]]
            channel_stats[ch] = {
                "opt_in_rate": round(float(group[f"{ch}_opt_in"].mean()), 3),
                "avg_engagement_opted_in": round(float(opted[f"{ch}_engagement"].mean()) if len(opted) else 0.0, 3),
            }
        profiles.append({
            "segment_id": int(seg_id),
            "name": name,
            "size": int(len(group)),
            "median_recency_days": int(medians["recency"]),
            "median_orders": float(medians["frequency"]),
            "median_spend_inr": round(float(medians["monetary"]), 0),
            "avg_order_value_inr": round(float(group["avg_order_value"].mean()), 0),
            "channels": channel_stats,
        })
    return sorted(profiles, key=lambda p: p["segment_id"])


def segment_customers(data: Dataset, k_min: int = 3, k_max: int = 7, random_state: int = 42,
                      k: int | None = None) -> SegmentationResult:
    rfm = compute_rfm(data.transactions)
    X = _features(rfm)
    if k is None:
        k, scores = choose_k(X, k_min, k_max, random_state)
    else:
        scores = {}
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    rfm["segment_id"] = model.fit_predict(X)
    return SegmentationResult(rfm, profile_segments(rfm, data.customers), k, scores)
