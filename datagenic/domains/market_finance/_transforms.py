"""Scaffold transforms contributed by the ``market_finance`` domain pack.

A scaffold transform runs *after* semantic inference and rewrites the inferred
semantics (and may contribute config-level default-parameter overrides). It is opted
into per module via ``Configs(domains={"market_finance": {"transforms": {...}}})`` and
registered through ``registry.add_scaffold_transform`` in ``_semantics.register``.

Pure dict manipulation over the ``ScaffoldContext`` — no numpy — so importing this stays
light. See ``generators.RollingTradeDate`` & co. for the runtime side.
"""


def apply_rolling_book(ctx, scenarios: int = 1) -> None:
    """Turn every trade table into a rolling book: remap the inferred trade_date /
    tenor / maturity_date semantics to their Rolling* equivalents and map the single
    unicity key to a rolling trade id. Requires an inferrer (heuristic) to have tagged
    those columns. On each date the book is a constant-size set of live trades; matured
    trades roll to freshly-booked ones with new ids. See RollingTradeDate & co.

    Rolling-book generators key off row index == slot; the post-generation key sort would
    scramble that mapping across dates (recall regenerates row by row), so the transform
    sets ``scale_skip_global_key_sort`` — row *order* changes only; the row set and key
    uniqueness do not. ``scenarios`` (>=1) repeats each position across that many scenario
    rows (see below)."""
    ir, semantics = ctx.ir, ctx.semantics
    remap = {
        "trade_date": "rolling_trade_date",
        "tenor": "rolling_tenor",
        "maturity_date": "rolling_maturity_date",
    }
    for table in ir["tables"]:
        tname = table["name"]
        # the column carrying the (rolling) trade date, used as the coherence parent
        trade_date_col = next(
            (
                c["name"]
                for c in table["columns"]
                if semantics.get((tname, c["name"]), {}).get("semantic") == "trade_date"
            ),
            None,
        )
        if trade_date_col is None:  # not a trade table (or inference is off) -> leave it
            continue
        for col in table["columns"]:
            rec = semantics.get((tname, col["name"]))
            if rec and rec["semantic"] in remap:
                rec["semantic"] = remap[rec["semantic"]]
                rec["by"] = "rolling_book"
                # all three derive from the shared per-slot schedule (seed-based), so none
                # reads another column: trade_date & maturity are fixed roots, tenor is the
                # residual maturity (maturity − as-of).
                rec["parent"] = None
        # The book's identity is a per-slot trade id that rolls (new id on each
        # roll). Support composite keys, not just a single key: map an id-like key
        # column (or the first key that isn't a rolling-schedule column) to
        # rolling_trade_id. It is unique per slot, so the composite key tuple is
        # unique BY CONSTRUCTION -> the engine's per-unit dedup is a no-op and the
        # book keeps its constant size (a shrunk frame would desync recall()).
        keys = table.get("keys", [])
        schedule_cols = {trade_date_col} | {
            c["name"]
            for c in table["columns"]
            if semantics.get((tname, c["name"]), {}).get("semantic")
            in ("rolling_tenor", "rolling_maturity_date")
        }
        free_keys = [k for k in keys if k not in schedule_cols]
        id_key = next(
            (k for k in free_keys if k.lower().endswith("id") or "ref" in k.lower()),
            free_keys[0] if free_keys else None,
        )
        scenario_keys = [k for k in free_keys if k != id_key]

        # Scenario dimension (scenarios=m): repeat each position across m scenario rows
        # so the trade id recurs while (rolling_trade_id, scenario) stays unique. Needs
        # exactly one non-id key to carry the scenario labels; otherwise fall back to one
        # row per position (m == 1).
        m = max(1, int(scenarios or 1))
        if m > 1 and len(scenario_keys) != 1:
            m = 1
        if m > 1:
            # tile the schedule across the scenario rows: trade_date, tenor and maturity
            # all derive from the per-slot schedule, so each needs the same repeat.
            for col in table["columns"]:
                rec = semantics.get((tname, col["name"]))
                if rec and rec.get("semantic") in (
                    "rolling_trade_date",
                    "rolling_tenor",
                    "rolling_maturity_date",
                ):
                    rec["args"] = {**rec.get("args", {}), "repeat": m}
            semantics[(tname, scenario_keys[0])] = {
                "semantic": "rolling_scenario",
                "confidence": 1.0,
                "parent": None,
                "alternatives": [],
                "by": "rolling_book",
                "args": {"count": m},
            }
        if id_key is not None:
            semantics[(tname, id_key)] = {
                "semantic": "rolling_trade_id",
                "confidence": 1.0,
                "parent": trade_date_col,
                "alternatives": [],
                "by": "rolling_book",
                "args": {"repeat": m} if m > 1 else {},
            }

    ctx.default_parameters["scale_skip_global_key_sort"] = True
