"""Customer and transaction data.

Two sources are supported:

* ``generate_synthetic`` – a seeded generator that produces realistic retail
  transactions from six hidden customer personas. The persona is kept in a
  separate ``ground_truth`` table that the agents never see; only the outcome
  simulator uses it (the same way a real market "knows" how customers behave).
* ``load_online_retail`` – loader for the public UCI *Online Retail* CSV
  (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice,
  CustomerID, Country). Consent and channel-engagement attributes do not exist
  in that dataset, so they are simulated with a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CHANNELS = ["email", "sms", "push", "whatsapp"]

# Hidden personas used only by the synthetic generator / outcome simulator.
# share, mean orders/year, recency window (days since last order), AOV range (INR)
PERSONAS = {
    "champion": dict(share=0.10, orders=(14, 24), last=(1, 20), aov=(1800, 3200)),
    "loyal": dict(share=0.20, orders=(6, 12), last=(5, 60), aov=(1200, 2200)),
    "new": dict(share=0.18, orders=(1, 2), last=(1, 40), aov=(700, 1600)),
    "at_risk": dict(share=0.15, orders=(6, 14), last=(100, 220), aov=(1200, 2400)),
    "hibernating": dict(share=0.27, orders=(1, 3), last=(180, 360), aov=(400, 1100)),
    "bargain": dict(share=0.10, orders=(4, 9), last=(15, 90), aov=(250, 650)),
}


@dataclass
class Dataset:
    customers: pd.DataFrame  # one row per customer: consent + engagement + contact history
    transactions: pd.DataFrame  # one row per invoice line
    ground_truth: pd.DataFrame | None = None  # customer_id -> persona (hidden from agents)
    source: str = "synthetic"


def _customer_attributes(ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame({
        "customer_id": ids,
        "email_opt_in": rng.random(n) < 0.85,
        "sms_opt_in": rng.random(n) < 0.55,
        "push_opt_in": rng.random(n) < 0.40,  # has the mobile app with notifications on
        "whatsapp_opt_in": rng.random(n) < 0.35,
        "email_engagement": rng.beta(3, 7, n).round(3),
        "sms_engagement": rng.beta(2, 8, n).round(3),
        "push_engagement": rng.beta(2, 6, n).round(3),
        "whatsapp_engagement": rng.beta(4, 6, n).round(3),
        "days_since_last_contact": rng.integers(0, 31, n),
    })


def generate_synthetic(n_customers: int = 2000, seed: int = 7, end_date: str = "2026-10-31") -> Dataset:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date)
    names = list(PERSONAS)
    shares = np.array([PERSONAS[p]["share"] for p in names])
    personas = rng.choice(names, size=n_customers, p=shares / shares.sum())
    ids = np.array([f"C{100000 + i}" for i in range(n_customers)])

    rows = []
    invoice = 500000
    for cid, persona in zip(ids, personas, strict=True):
        spec = PERSONAS[persona]
        n_orders = int(rng.integers(spec["orders"][0], spec["orders"][1] + 1))
        last = int(rng.integers(spec["last"][0], spec["last"][1] + 1))
        # The remaining orders are spread over the year before the most recent one.
        offsets = np.sort(np.concatenate([[last], rng.integers(last, 366, n_orders - 1)]))
        for offset in offsets:
            invoice += 1
            order_value = rng.uniform(*spec["aov"])
            lines = int(rng.integers(1, 4))
            for _ in range(lines):
                qty = int(rng.integers(1, 4))
                rows.append({
                    "invoice_no": f"INV{invoice}",
                    "customer_id": cid,
                    "invoice_date": end - pd.Timedelta(days=int(offset)),
                    "quantity": qty,
                    "unit_price": round(order_value / lines / qty, 2),
                })

    transactions = pd.DataFrame(rows)
    customers = _customer_attributes(ids, rng)
    # Customers who use the app engage more with push; heavy buyers open more email.
    heavy = np.isin(personas, ["champion", "loyal"])
    customers.loc[heavy, "email_engagement"] = (customers.loc[heavy, "email_engagement"] + 0.12).clip(upper=0.95)
    ground_truth = pd.DataFrame({"customer_id": ids, "persona": personas})
    return Dataset(customers, transactions, ground_truth, source=f"synthetic(n={n_customers}, seed={seed})")


def load_online_retail(path: str, seed: int = 7) -> Dataset:
    """Load the UCI Online Retail CSV/XLSX and attach simulated consent/engagement."""
    raw = pd.read_excel(path) if str(path).endswith((".xlsx", ".xls")) else pd.read_csv(path, encoding="latin-1")
    raw = raw.dropna(subset=["CustomerID"])
    raw = raw[(raw["Quantity"] > 0) & (raw["UnitPrice"] > 0)]
    raw = raw[~raw["InvoiceNo"].astype(str).str.startswith("C")]  # drop cancellations
    transactions = pd.DataFrame({
        "invoice_no": raw["InvoiceNo"].astype(str),
        "customer_id": raw["CustomerID"].astype(int).astype(str),
        "invoice_date": pd.to_datetime(raw["InvoiceDate"]),
        "quantity": raw["Quantity"].astype(int),
        "unit_price": raw["UnitPrice"].astype(float),
    })
    ids = np.sort(transactions["customer_id"].unique())
    customers = _customer_attributes(ids, np.random.default_rng(seed))
    return Dataset(customers, transactions, None, source=f"UCI Online Retail ({path})")
