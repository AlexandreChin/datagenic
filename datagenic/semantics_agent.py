"""LLM-backed semantic inferrer (Phase 3) — the opt-in ``agent`` strategy.

Resolves ambiguous columns by asking Claude to label each one with a finance
semantic, given the column names, types, and (when scaffolding from a sample)
real values. Like the other inferrers this runs at scaffold time only, so its
non-determinism never touches the byte-identical generation invariant — the
output is a config the user reviews and may hand-edit.

The ``anthropic`` SDK is imported lazily inside ``agent_inferrer`` so a bare
``import datagenic`` (and even ``import datagenic.scaffolding``) pulls in no SDK.
Install it with ``pip install 'datagenic[agent]'``. The client is injectable for
testing, so the parsing/wiring path runs with no network and no SDK.

Model default: ``claude-opus-4-8`` (override via ``DATAGENIC_AGENT_MODEL``).
Reasoning effort default: ``low`` (override via ``DATAGENIC_AGENT_EFFORT``) — this
is a short classification, not a long-horizon task.
"""

import json
import os

from datagenic.semantics import (
    SEMANTIC_GENERATORS,
    _link_hierarchy_parents,
)

# Short catalog descriptions shown to the model (one line per semantic).
_SEMANTIC_DESCRIPTIONS = {
    "asset_class": "top-level financial asset class (Rates, Credit, Equity, FX, Commodity, Inflation)",
    "instrument_type": "instrument/product type, valid for the row's asset class (IR Swap, CDS, FX Forward, ...)",
    "risk_factor_group": "risk factor group, valid for the row's asset class (IR Curve, Credit Spread, FX Spot, ...)",
    "currency": "ISO 4217 currency code (USD, EUR, GBP, ...)",
    "currency_pair": "FX currency pair (EUR/USD, USD/JPY, ...)",
    "tenor": "tenor / maturity bucket (ON, 1M, 5Y, 30Y, ...)",
    "credit_rating": "credit rating (S&P AAA..D or Moody's Aaa..C)",
    "gics_sector": "GICS sector (Energy, Financials, Information Technology, ...)",
    "isin": "ISIN security identifier (12 chars)",
    "cusip": "CUSIP security identifier (9 chars)",
    "sedol": "SEDOL security identifier (7 chars)",
    "instrument_name": "human-readable instrument name (e.g. 'EUR IR Swap 5Y')",
    "risk_factor_name": "human-readable risk factor name (e.g. 'EUR IR Curve 10Y')",
    "counterparty": "counterparty / legal entity name",
    "trading_book": "trading book label",
    "desk": "trading desk label",
}

_NONE = "none"  # the model's choice when no catalog entry fits

_SYSTEM = (
    "You label database columns with a finance-domain semantic so a synthetic-data "
    "scaffolder can pick a realistic generator. For each column choose exactly one "
    "semantic from the catalog, or 'none' if no entry fits. Use the column name, type, "
    "and any sample values. When you label a column 'instrument_type' or "
    "'risk_factor_group', set its 'parent' to the name of the column you labeled "
    "'asset_class' (else empty string). Set a calibrated confidence in [0,1]."
)

_MAX_SAMPLES = 15


def _output_schema() -> dict:
    semantics = sorted(SEMANTIC_GENERATORS) + [_NONE]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["columns"],
        "properties": {
            "columns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "semantic", "confidence", "parent"],
                    "properties": {
                        "name": {"type": "string"},
                        "semantic": {"type": "string", "enum": semantics},
                        "confidence": {"type": "number"},
                        "parent": {"type": "string"},
                    },
                },
            }
        },
    }


def _build_prompt(table_name, columns, samples) -> str:
    samples = samples or {}
    catalog = "\n".join(f"- {sem}: {desc}" for sem, desc in _SEMANTIC_DESCRIPTIONS.items())
    payload = []
    for col in columns:
        name = col.get("name", "")
        entry = {"name": name, "type": col.get("type")}
        values = samples.get(name)
        if values:
            seen, uniq = set(), []
            for v in values:
                s = str(v)
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
                if len(uniq) >= _MAX_SAMPLES:
                    break
            entry["sample_values"] = uniq
        payload.append(entry)
    return (
        f"Semantic catalog:\n{catalog}\n\n"
        f"Table: {table_name}\n"
        f"Columns (JSON):\n{json.dumps(payload, indent=2)}\n\n"
        "Return the label for every column."
    )


def _build_client():
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the 'agent' semantic inferrer needs the anthropic SDK; install it with "
            "`pip install 'datagenic[agent]'` (or pass a client=), or use "
            "--infer-semantics heuristic|values."
        ) from e
    return anthropic.Anthropic()


def _first_text(response) -> str | None:
    if getattr(response, "stop_reason", None) == "refusal":
        return None
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            return block.text
    return None


def agent_inferrer(
    table_name,
    columns,
    samples=None,
    *,
    model: str | None = None,
    effort: str | None = None,
    client=None,
):
    """Infer column semantics with Claude. ``client`` is injectable for tests;
    when omitted, a default ``anthropic.Anthropic()`` is built lazily."""
    client = client or _build_client()
    model = model or os.environ.get("DATAGENIC_AGENT_MODEL", "claude-opus-4-8")
    effort = effort or os.environ.get("DATAGENIC_AGENT_EFFORT", "low")

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM,
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": _output_schema()},
        },
        messages=[{"role": "user", "content": _build_prompt(table_name, columns, samples)}],
    )

    text = _first_text(response)
    if not text:
        return {}  # refusal or empty — fall back to no inference (type-based defaults)

    result: dict[str, dict] = {}
    for item in json.loads(text).get("columns", []):
        semantic = item.get("semantic")
        if semantic not in SEMANTIC_GENERATORS:  # 'none' or anything off-catalog
            continue
        result[item["name"]] = {
            "semantic": semantic,
            "confidence": float(item.get("confidence", 0.0)),
            "parent": item.get("parent") or None,
            "alternatives": [],
            "by": "agent",
        }
    return _link_hierarchy_parents(result)
