"""Phase 5 — the agent inferrer proposes numeric distributions.

Beyond labelling a numeric column with a quantity semantic, the LLM proposes its
distribution family + scale (and, for a conditional quantity, the column to
condition on). These flow through as ``build_generator`` overrides. Exercised
with a fake client — no network, no SDK — plus a Configs integration check.
Invalid proposals are sanitised and fall back to the curated prior.
"""

import json

from datagenic import Configs
from datagenic.semantics_agent import _output_schema, agent_inferrer

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
    def __init__(self, payload, captured):
        self._payload = payload
        self._captured = captured

    def create(self, **kwargs):
        self._captured.update(kwargs)
        return _Response(json.dumps(self._payload))


class FakeClient:
    def __init__(self, payload):
        self.captured: dict = {}
        self.messages = _Messages(payload, self.captured)


def _col(name, semantic, parent="", distribution="none", mu=0, sigma=0, confidence=0.8):
    return {
        "name": name,
        "semantic": semantic,
        "confidence": confidence,
        "parent": parent,
        "distribution": distribution,
        "mu": mu,
        "sigma": sigma,
    }


# ── parsing: numeric proposals become args ────────────────────────────────────


def test_flat_numeric_proposal_becomes_args():
    payload = {"columns": [_col("yld", "rate", distribution="normal", mu=0.04, sigma=0.012)]}
    out = agent_inferrer("t", [{"name": "yld", "type": "float"}], client=FakeClient(payload))
    assert out["yld"]["args"] == {"distribution": "normal", "mu": 0.04, "sigma": 0.012}


def test_conditional_proposal_overrides_default_and_keeps_parent():
    payload = {
        "columns": [
            _col("ac", "asset_class"),
            _col("exposure", "notional", parent="ac", distribution="lognormal", mu=5e6, sigma=1.1),
        ]
    }
    out = agent_inferrer(
        "t",
        [{"name": "ac", "type": "string"}, {"name": "exposure", "type": "float"}],
        client=FakeClient(payload),
    )
    assert out["exposure"]["parent"] == "ac"  # the model's conditioning column is respected
    # only the pooled default is overridden; per-category cases stay curated
    assert out["exposure"]["args"] == {
        "default": {"distribution": "lognormal", "mu": 5e6, "sigma": 1.1}
    }


def test_invalid_proposal_is_sanitised():
    payload = {
        "columns": [
            _col("bad", "price", distribution="lognormal", mu=-5, sigma=0.3),  # mu<=0
            _col("neg_sigma", "quantity", distribution="normal", mu=10, sigma=-1),
        ]
    }
    out = agent_inferrer(
        "t",
        [{"name": "bad", "type": "float"}, {"name": "neg_sigma", "type": "float"}],
        client=FakeClient(payload),
    )
    assert "args" not in out["bad"]  # falls back to the curated prior
    assert "args" not in out["neg_sigma"]


def test_distribution_ignored_on_non_numeric_semantic():
    payload = {"columns": [_col("ccy", "currency", distribution="lognormal", mu=100, sigma=1)]}
    out = agent_inferrer("t", [{"name": "ccy", "type": "string"}], client=FakeClient(payload))
    assert "args" not in out["ccy"]


def test_schema_exposes_numeric_fields():
    props = _output_schema()["properties"]["columns"]["items"]["properties"]
    assert set(props) >= {"distribution", "mu", "sigma"}
    assert props["distribution"]["enum"] == ["normal", "lognormal", "none"]


# ── Configs integration: agent numeric proposal wires the generator ───────────


def test_configs_wires_agent_numeric_proposals():
    payload = {
        "columns": [
            _col("trade_id", "none"),
            _col("ac", "asset_class"),
            _col("exposure", "notional", parent="ac", distribution="lognormal", mu=5e6, sigma=1.1),
            _col("yld", "rate", distribution="normal", mu=0.04, sigma=0.012),
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
                    {"name": "ac", "type": "string"},
                    {"name": "exposure", "type": "float"},
                    {"name": "yld", "type": "float"},
                ],
            }
        ],
    }
    cfg = Configs(
        input=ir, output_format="parquet", number_of_batches=1, semantic_inferrer=inferrer
    ).config_template
    cols = {c["col_name"]: c for c in cfg["tables"][0]["columns"]}

    exposure = cols["exposure"]["column_generator"]
    assert exposure["udf_name"] == "ConditionalFloats"
    assert exposure["args"]["by_column"] == "ac"  # conditioned on the model's column
    assert exposure["args"]["default"] == {"distribution": "lognormal", "mu": 5e6, "sigma": 1.1}
    assert exposure["args"]["cases"]  # curated per-category cases retained

    yld = cols["yld"]["column_generator"]
    assert yld["udf_name"] == "RandomFloats"
    assert yld["args"]["distribution"] == "normal"
    assert yld["args"]["mu"] == 0.04

    # parent (ac) ordered before the conditional child (exposure)
    order = [c["col_name"] for c in cfg["tables"][0]["columns"]]
    assert order.index("ac") < order.index("exposure")
