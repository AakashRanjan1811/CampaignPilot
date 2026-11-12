import pandas as pd
import pytest

from campaignpilot.data import generate_synthetic
from campaignpilot.segmentation import compute_rfm, name_segment, segment_customers


def test_rfm_values_are_computed_correctly():
    tx = pd.DataFrame({
        "invoice_no": ["A", "A", "B", "C"],
        "customer_id": ["x", "x", "x", "y"],
        "invoice_date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-11", "2026-01-06"]),
        "quantity": [1, 2, 1, 4],
        "unit_price": [100.0, 50.0, 300.0, 25.0],
    })
    rfm = compute_rfm(tx).set_index("customer_id")
    # snapshot = last date + 1 day = 2026-01-12
    assert rfm.loc["x", "recency"] == 1 and rfm.loc["y", "recency"] == 6
    assert rfm.loc["x", "frequency"] == 2  # two distinct invoices, not three lines
    assert rfm.loc["x", "monetary"] == pytest.approx(500.0)
    assert rfm.loc["x", "avg_order_value"] == pytest.approx(250.0)


def test_every_customer_gets_exactly_one_segment(data, seg):
    assert len(seg.rfm) == data.customers.shape[0]
    assert seg.rfm["segment_id"].notna().all()
    assert sum(p["size"] for p in seg.profiles) == len(seg.rfm)


def test_k_is_the_silhouette_argmax(seg):
    assert seg.k == max(seg.silhouette_by_k, key=seg.silhouette_by_k.get)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_segments_recover_hidden_personas(seed):
    data = generate_synthetic(n_customers=1500, seed=seed)
    result = segment_customers(data)
    merged = result.rfm.merge(data.ground_truth, on="customer_id")
    purity = merged.groupby("segment_id")["persona"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(merged)
    assert purity > 0.9
    names = [p["name"] for p in result.profiles]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("pos,expected", [
    (dict(recency=0.1, frequency=0.95, monetary=0.95, avg_order_value=0.9), "Champions"),
    (dict(recency=0.4, frequency=0.65, monetary=0.7, avg_order_value=0.7), "Loyal Customers"),
    (dict(recency=0.2, frequency=0.2, monetary=0.3, avg_order_value=0.5), "New & Promising"),
    (dict(recency=0.7, frequency=0.75, monetary=0.8, avg_order_value=0.8), "At Risk"),
    (dict(recency=0.9, frequency=0.2, monetary=0.2, avg_order_value=0.3), "Hibernating"),
    (dict(recency=0.5, frequency=0.5, monetary=0.5, avg_order_value=0.05), "Bargain Hunters"),
])
def test_segment_naming_rules(pos, expected):
    assert name_segment(pos) == expected
