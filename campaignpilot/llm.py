"""LLM providers behind one tiny interface: ``structured(system, user, schema)``.

* ``GroqLLM``   – hosted Llama models on Groq (free tier works).
* ``OllamaLLM`` – fully local models through Ollama.
* ``MockLLM``   – deterministic, offline planner used for tests, CI and demos
  without an API key. It reads the same JSON payload the real models receive.

Every provider returns a validated Pydantic object, so downstream agents never
deal with free-form text.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
PAYLOAD_MARKER = "INPUT_JSON:"


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str

    def structured(self, system: str, user: str, schema: type[T]) -> T: ...


class _LangChainLLM:
    """Shared retry logic for LangChain chat models with structured output."""

    name = "langchain"
    max_attempts = 3

    def _model(self):  # pragma: no cover - provided by subclasses
        raise NotImplementedError

    def _bind(self, schema):
        return self._model().with_structured_output(schema)

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        runnable = self._bind(schema)
        prompt, last_error = user, None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = runnable.invoke([("system", system), ("human", prompt)])
                if isinstance(result, dict):
                    result = schema.model_validate(result)
                if result is None:
                    raise LLMError("model returned no structured output")
                return result
            except (ValidationError, LLMError, ValueError) as exc:
                last_error = exc
                log.warning("%s attempt %d produced invalid output: %s", self.name, attempt, exc)
                prompt = (f"{user}\n\nYour previous answer was invalid: {str(exc)[:600]}\n"
                          "Return output that matches the schema exactly.")
            except Exception as exc:
                # Groq returns HTTP 400 "tool_use_failed" when the model emits a
                # malformed tool call; that is worth a retry. Anything else
                # (network, auth, rate limit) is fatal for this request.
                if "tool_use_failed" in str(exc) and attempt < self.max_attempts:
                    last_error = exc
                    log.warning("%s attempt %d: malformed tool call, retrying", self.name, attempt)
                    continue
                raise LLMError(f"{self.name} request failed: {exc}") from exc
        raise LLMError(f"{self.name} could not produce valid output after {self.max_attempts} attempts: {last_error}")


class GroqLLM(_LangChainLLM):
    def __init__(self, settings: Settings):
        if not settings.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set. Add it to .env or choose another provider.")
        self.settings = settings
        self.name = f"groq:{settings.groq_model}"

    def _model(self):
        from langchain_groq import ChatGroq

        return ChatGroq(api_key=self.settings.groq_api_key, model=self.settings.groq_model,
                        temperature=self.settings.llm_temperature, max_retries=2)


class OllamaLLM(_LangChainLLM):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.name = f"ollama:{settings.ollama_model}"

    def _model(self):
        from langchain_ollama import ChatOllama

        return ChatOllama(model=self.settings.ollama_model, base_url=self.settings.ollama_base_url,
                          temperature=self.settings.llm_temperature)

    def _bind(self, schema):
        return self._model().with_structured_output(schema, method="json_schema")


class MockLLM:
    """Deterministic rule-based stand-in for an LLM strategist.

    Its first draft deliberately proposes one over-limit win-back discount (a
    common real-LLM failure) so the governance -> revision loop is exercised
    in tests and demos. On revision it fixes every violation it is told about.
    """

    name = "mock"

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        if PAYLOAD_MARKER not in user:
            raise LLMError("MockLLM needs an INPUT_JSON payload")
        payload = json.loads(user.split(PAYLOAD_MARKER, 1)[1])
        if schema.__name__ == "StrategyPlan":
            return schema.model_validate(_mock_strategy_plan(payload))
        raise LLMError(f"MockLLM has no handler for {schema.__name__}")


def _best_channel(profile: dict) -> str:
    stats = profile["channels"]
    return max(stats, key=lambda c: stats[c]["opt_in_rate"] * stats[c]["avg_engagement_opted_in"])


_TEMPLATES = {
    "Champions": ("retain", "early_access", 0, "Hi {{first_name}}, as one of our top customers you get early access "
                  "to the new collection 48 hours before everyone else."),
    "Loyal Customers": ("upsell", "loyalty_points", 0, "Thanks for shopping with us, {{first_name}}! Earn double "
                        "loyalty points on your next order this week."),
    "New & Promising": ("nurture", "free_shipping", 0, "Welcome back, {{first_name}}! Your next order ships free - "
                        "here are picks based on your first purchase."),
    "At Risk": ("win_back", "percent_discount", 35, "We miss you, {{first_name}}! Take 35% off your next order - "
                "your favourites are waiting."),
    "Hibernating": ("reactivate", "percent_discount", 20, "It's been a while, {{first_name}}. Enjoy 20% off "
                    "anything in store this month."),
    "Bargain Hunters": ("upsell", "bundle", 0, "Hi {{first_name}}, bundle and save: grab our best-value combos "
                        "picked for you."),
}
_DEFAULT = ("nurture", "free_shipping", 0, "Hi {{first_name}}, your next order ships free. See what's new for you.")


def _mock_strategy_plan(payload: dict) -> dict:
    feedback = payload.get("governance_feedback") or []
    policy = payload.get("policy", {})
    by_segment: dict[int | None, list[dict]] = {}
    for v in feedback:
        by_segment.setdefault(v.get("segment_id"), []).append(v)

    strategies = []
    for profile in payload["segments"]:
        base = re.sub(r"\s+\d+$", "", profile["name"])  # "At Risk 2" -> "At Risk"
        objective, offer, discount, message = _TEMPLATES.get(base, _DEFAULT)
        channel = _best_channel(profile)
        send_hour = 19 if base in ("Loyal Customers", "Bargain Hunters") else 11
        rationale = (f"{profile['name']}: median recency {profile['median_recency_days']}d, "
                     f"{profile['median_orders']:g} orders; {channel} has the best reach x engagement.")

        for v in by_segment.get(profile["segment_id"], []):
            rule = v["rule"]
            if rule in ("max_discount", "vip_max_discount"):
                limit = policy.get("vip_max_discount_pct" if rule == "vip_max_discount" else "max_discount_pct", 30)
                discount = min(discount, limit)
                message = re.sub(r"\d+(\.\d+)?%", f"{discount:g}%", message)
            elif rule == "banned_phrase":
                for phrase in policy.get("banned_phrases", []):
                    message = re.sub(re.escape(phrase), "", message, flags=re.IGNORECASE)
            elif rule == "quiet_hours":
                send_hour = 11
            elif rule == "message_length":
                message = message[:110].rsplit(" ", 1)[0] + "."
            elif rule == "low_channel_coverage":
                channel = "email"
        if offer != "percent_discount":
            discount = 0
        strategies.append(dict(segment_id=profile["segment_id"], objective=objective, offer=offer,
                               discount_pct=discount, preferred_channel=channel, send_hour=send_hour,
                               message=message, rationale=rationale))
    return {"strategies": strategies}


def make_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider
    if provider == "groq":
        return GroqLLM(settings)
    if provider == "ollama":
        return OllamaLLM(settings)
    if provider == "mock":
        return MockLLM()
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use mock, groq or ollama.")
