"""Phase 3 — the opt-in LLM ``agent`` semantic inferrer.

The agent inferrer asks Claude to label columns; the anthropic SDK is lazy and
the client is injectable, so these tests exercise the prompt-build / parse /
parent-link path with a fake client — no network, no SDK required.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from datagenic import Configs
from datagenic.semantics import resolve_inferrer
from datagenic.semantics_agent import agent_inferrer

REPO_ROOT = Path(__file__).resolve().parents[1]

# the live smoke test needs both the SDK and a key; otherwise it's skipped
_AGENT_AVAILABLE = importlib.util.find_spec("anthropic") is not None and bool(
    os.environ.get("ANTHROPIC_API_KEY")
)


# ── fake anthropic client ─────────────────────────────────────────────────────


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason


class _Messages:
    def __init__(self, payload, captured, stop_reason):
        self._payload = payload
        self._captured = captured
        self._stop_reason = stop_reason

    def create(self, **kwargs):
        self._captured.update(kwargs)
        return _Response(json.dumps(self._payload), self._stop_reason)


class FakeClient:
    def __init__(self, payload, stop_reason="end_turn"):
        self.captured: dict = {}
        self.messages = _Messages(payload, self.captured, stop_reason)


def _columns():
    return [
        {"name": "trade_id", "type": "int"},
        {"name": "ac", "type": "string"},
        {"name": "it", "type": "string"},
    ]


# ── parsing + wiring ──────────────────────────────────────────────────────────


def test_agent_parses_and_filters_none():
    payload = {
        "columns": [
            {"name": "trade_id", "semantic": "none", "confidence": 0.0, "parent": ""},
            {"name": "ac", "semantic": "asset_class", "confidence": 0.96, "parent": ""},
            {"name": "it", "semantic": "instrument_type", "confidence": 0.9, "parent": "ac"},
        ]
    }
    fake = FakeClient(payload)
    out = agent_inferrer("trades", _columns(), client=fake)

    assert "trade_id" not in out  # 'none' is dropped
    assert out["ac"]["semantic"] == "asset_class"
    assert out["ac"]["by"] == "agent"
    assert out["it"]["parent"] == "ac"


def test_agent_links_parent_when_model_omits_it():
    payload = {
        "columns": [
            {"name": "ac", "semantic": "asset_class", "confidence": 0.9, "parent": ""},
            {"name": "it", "semantic": "instrument_type", "confidence": 0.9, "parent": ""},
        ]
    }
    out = agent_inferrer(
        "t",
        [{"name": "ac", "type": "string"}, {"name": "it", "type": "string"}],
        client=FakeClient(payload),
    )
    assert out["it"]["parent"] == "ac"  # linked to the asset_class column


def test_agent_refusal_returns_empty():
    fake = FakeClient({"columns": []}, stop_reason="refusal")
    assert agent_inferrer("t", _columns(), client=fake) == {}


def test_agent_uses_opus_model_and_structured_output():
    fake = FakeClient({"columns": []})
    agent_inferrer("t", _columns(), client=fake)
    assert fake.captured["model"] == "claude-opus-4-8"
    # structured output + a reasoning effort are requested
    assert fake.captured["output_config"]["format"]["type"] == "json_schema"
    assert "effort" in fake.captured["output_config"]


def test_agent_passes_sample_values_into_prompt():
    fake = FakeClient({"columns": []})
    agent_inferrer(
        "t", [{"name": "ac", "type": "string"}], samples={"ac": ["Rates", "Credit"]}, client=fake
    )
    prompt = fake.captured["messages"][0]["content"]
    assert "Rates" in prompt and "sample_values" in prompt


# ── Configs integration (inject the fake-backed inferrer) ─────────────────────


def test_configs_with_agent_inferrer():
    payload = {
        "columns": [
            {"name": "trade_id", "semantic": "none", "confidence": 0.0, "parent": ""},
            {"name": "dim1", "semantic": "asset_class", "confidence": 0.95, "parent": ""},
            {"name": "dim2", "semantic": "instrument_type", "confidence": 0.9, "parent": "dim1"},
        ]
    }
    fake = FakeClient(payload)
    inferrer = lambda t, c, samples=None: agent_inferrer(t, c, samples=samples, client=fake)  # noqa: E731

    ir = {
        "name": "m",
        "tables": [
            {
                "type": "fact",
                "name": "trades",
                "keys": ["trade_id"],
                "columns": [
                    {"name": "trade_id", "type": "int"},
                    {"name": "dim1", "type": "string"},
                    {"name": "dim2", "type": "string"},
                ],
            }
        ],
    }
    cfg = Configs(
        input=ir, output_format="parquet", number_of_batches=1, semantic_inferrer=inferrer
    ).config_template
    cols = {c["col_name"]: c for c in cfg["tables"][0]["columns"]}
    assert cols["dim1"]["column_generator"]["udf_name"] == "FinancialAssetClass"
    assert cols["dim2"]["column_generator"]["args"]["asset_class_column"] == "dim1"
    assert cols["dim1"]["_inferred"]["by"] == "agent"


# ── boundary: resolving 'agent' must not import the SDK ───────────────────────


def test_resolving_agent_does_not_import_anthropic():
    code = (
        "import datagenic.scaffolding, sys; "
        "from datagenic.semantics import resolve_inferrer; "
        "fn = resolve_inferrer('agent'); "
        "print(callable(fn) and 'anthropic' not in sys.modules "
        "and 'datagenic.semantics_agent' not in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_resolve_agent_returns_callable():
    assert callable(resolve_inferrer("agent"))


# ── live smoke test (real Claude call; skipped without SDK + key) ─────────────


@pytest.mark.integration
@pytest.mark.skipif(not _AGENT_AVAILABLE, reason="needs the anthropic SDK and ANTHROPIC_API_KEY")
def test_agent_live_smoke():
    """End-to-end against the real API: unambiguous finance columns are labelled,
    the hierarchy child links to the asset_class column, and a plain id is left
    alone. Lenient on the LLM's exact choices — asserts only the obvious mappings."""
    columns = [
        {"name": "trade_id", "type": "int"},
        {"name": "asset_class", "type": "string"},
        {"name": "instrument_type", "type": "string"},
        {"name": "ccy", "type": "string"},
        {"name": "isin", "type": "string"},
    ]
    out = agent_inferrer("trades", columns)  # builds a real anthropic.Anthropic()

    assert out.get("ccy", {}).get("semantic") == "currency"
    assert out.get("isin", {}).get("semantic") == "isin"
    assert out.get("asset_class", {}).get("semantic") == "asset_class"
    if "instrument_type" in out:  # if detected, it must be coherent-wired
        assert out["instrument_type"]["semantic"] == "instrument_type"
        assert out["instrument_type"]["parent"] == "asset_class"
    assert "trade_id" not in out  # not a finance semantic
    # confidences are usable floats in [0, 1]
    assert all(0.0 <= rec["confidence"] <= 1.0 for rec in out.values())
