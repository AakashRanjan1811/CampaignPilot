"""Runtime configuration, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


@dataclass
class Settings:
    # LLM provider: "mock" (offline, deterministic), "groq" or "ollama"
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock").lower())
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))

    # Campaign and governance policy
    budget: float = field(default_factory=lambda: _env_float("CAMPAIGN_BUDGET", 250.0))
    max_discount_pct: float = field(default_factory=lambda: _env_float("MAX_DISCOUNT_PCT", 30.0))
    vip_max_discount_pct: float = field(default_factory=lambda: _env_float("VIP_MAX_DISCOUNT_PCT", 10.0))
    frequency_cap_days: int = field(default_factory=lambda: _env_int("FREQUENCY_CAP_DAYS", 7))
    quiet_hours_start: int = 21  # no sends at or after 21:00
    quiet_hours_end: int = 9  # no sends before 09:00
    min_channel_coverage: float = 0.15  # chosen channel must reach >= 15% of a segment
    gross_margin: float = field(default_factory=lambda: _env_float("GROSS_MARGIN", 0.45))
    max_revisions: int = field(default_factory=lambda: _env_int("MAX_REVISIONS", 2))

    # Segmentation
    k_min: int = 3
    k_max: int = 7
    random_state: int = 42

    def describe_llm(self) -> str:
        if self.llm_provider == "groq":
            return f"Groq · {self.groq_model}"
        if self.llm_provider == "ollama":
            return f"Ollama · {self.ollama_model}"
        return "Mock planner (offline)"
