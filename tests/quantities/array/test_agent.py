"""Phase 5 — the agent inferrer proposes array-quantity distributions.

For an array(float) column the LLM picks an array semantic and proposes the element
distribution/mu/sigma plus `size_of_arrays` (and, for a conditional series, the
conditioning column). These flow through as `build_generator` overrides; invalid
proposals are sanitised and fall back to the curated prior. Exercised with a fake
client — no network, no SDK.
"""

import json

from datagenic import Configs
from datagenic.semantics_agent import _output_schema, agent_inferrer


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text):
        self.content = [_Block(text)]
        self.stop_reason = "end_turn"


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


def _col(name, semantic, parent="", distribution="none", mu=0, sigma=0, size=0):
    return {
        "name": name,
        "semantic": semantic,
        "confidence": 0.8,
        "parent": parent,
        "distribution": distribution,
        "mu": mu,
        "sigma": sigma,
        "size_of_arrays": size,
    }


# ── parsing ───────────────────────────────────────────────────────────────────


def test_flat_array_proposal_carries_size():
    payload = {
        "columns": [_col("rets", "returns", distribution="normal", mu=0, sigma=0.015, size=252)]
    }
    out = agent_inferrer(
        "t", [{"name": "rets", "type": "array(float)"}], client=FakeClient(payload)
    )
    assert out["rets"]["args"] == {
        "distribution": "normal",
        "mu": 0.0,
        "sigma": 0.015,
        "size_of_arrays": 252,
    }


def test_conditional_array_proposal_overrides_default_and_keeps_parent():
    payload = {
        "columns": [
            _col("ac", "asset_class"),
            _col(
                "pnl", "pnl_series", parent="ac", distribution="normal", mu=0, sigma=1.5e5, size=252
            ),
        ]
    }
    out = agent_inferrer(
        "t",
        [{"name": "ac", "type": "string"}, {"name": "pnl", "type": "array(float)"}],
        client=FakeClient(payload),
    )
    assert out["pnl"]["parent"] == "ac"
    assert out["pnl"]["args"] == {
        "default": {"distribution": "normal", "mu": 0.0, "sigma": 1.5e5},
        "size_of_arrays": 252,
    }


def test_invalid_array_proposal_is_sanitised():
    payload = {"columns": [_col("c", "curve", distribution="lognormal", mu=-1, sigma=0.1, size=10)]}
    out = agent_inferrer("t", [{"name": "c", "type": "array(float)"}], client=FakeClient(payload))
    assert "args" not in out["c"]  # mu<=0 lognormal -> fall back to the curated prior


def test_missing_size_omits_size_key():
    payload = {
        "columns": [_col("rets", "returns", distribution="normal", mu=0, sigma=0.02, size=0)]
    }
    out = agent_inferrer(
        "t", [{"name": "rets", "type": "array(float)"}], client=FakeClient(payload)
    )
    assert "size_of_arrays" not in out["rets"]["args"]  # 0 -> keep the prior's default length
    assert out["rets"]["args"]["sigma"] == 0.02


def test_schema_exposes_size_of_arrays():
    props = _output_schema()["properties"]["columns"]["items"]["properties"]
    assert "size_of_arrays" in props


# ── Configs integration ───────────────────────────────────────────────────────


def test_configs_wires_agent_array_proposals():
    payload = {
        "columns": [
            _col("ac", "asset_class"),
            _col(
                "pnl", "pnl_series", parent="ac", distribution="normal", mu=0, sigma=1.5e5, size=252
            ),
            _col("rets", "returns", distribution="normal", mu=0, sigma=0.02, size=252),
        ]
    }
    fake = FakeClient(payload)
    inferrer = lambda t, c, samples=None: agent_inferrer(t, c, samples=samples, client=fake)  # noqa: E731

    ir = {
        "name": "m",
        "tables": [
            {
                "type": "fact",
                "name": "positions",
                "keys": ["pos_id"],
                "columns": [
                    {"name": "pos_id", "type": "int"},
                    {"name": "ac", "type": "string"},
                    {"name": "pnl", "type": "array(float)"},
                    {"name": "rets", "type": "array(float)"},
                ],
            }
        ],
    }
    cfg = Configs(
        input=ir, output_format="parquet", number_of_batches=1, semantic_inferrer=inferrer
    ).config_template
    cols = {c["col_name"]: c for c in cfg["tables"][0]["columns"]}

    pnl = cols["pnl"]["column_generator"]
    assert pnl["udf_name"] == "ConditionalArrayFloats"
    assert pnl["args"]["by_column"] == "ac"
    assert pnl["args"]["default"] == {"distribution": "normal", "mu": 0.0, "sigma": 1.5e5}
    assert pnl["args"]["size_of_arrays"] == 252
    assert pnl["args"]["cases"]  # curated per-category cases retained

    rets = cols["rets"]["column_generator"]
    assert rets["udf_name"] == "RandomArrayFloats"
    assert rets["args"]["sigma"] == 0.02
    assert rets["args"]["size_of_arrays"] == 252

    order = [c["col_name"] for c in cfg["tables"][0]["columns"]]
    assert order.index("ac") < order.index("pnl")  # parent before the conditional child
