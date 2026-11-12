from dataclasses import replace

import pytest

from campaignpilot.config import Settings
from campaignpilot.data import generate_synthetic
from campaignpilot.llm import MockLLM
from campaignpilot.segmentation import segment_customers


@pytest.fixture(scope="session")
def data():
    return generate_synthetic(n_customers=1200, seed=11)


@pytest.fixture(scope="session")
def seg(data):
    return segment_customers(data)


@pytest.fixture
def settings():
    return replace(Settings(), llm_provider="mock", groq_api_key="", budget=250.0, max_discount_pct=30.0,
                   vip_max_discount_pct=10.0, frequency_cap_days=7, max_revisions=2)


@pytest.fixture
def llm():
    return MockLLM()
