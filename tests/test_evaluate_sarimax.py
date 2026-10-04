"""Tests for SARIMAX experiment exogenous feature preparation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd

EVALUATOR_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_sarimax.py"
EVALUATOR_SPEC = spec_from_file_location("evaluate_sarimax", EVALUATOR_PATH)
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
evaluator = module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(evaluator)


def test_holiday_indicator_respects_transfers_bridge_and_work_day() -> None:
    """Moved source holidays are omitted and special event types are distinct."""
    dates = pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04"])
    holidays = pd.DataFrame(
        {
            "date": dates,
            "type": ["Holiday", "Transfer", "Bridge", "Work Day"],
            "locale": ["National"] * 4,
            "locale_name": ["Ecuador"] * 4,
            "transferred": [True, False, False, False],
        }
    )
    store = pd.Series({"city": "Quito", "state": "Pichincha"})

    values = [
        evaluator.holiday_value_for_store(date, store, holidays) for date in dates
    ]

    assert values == [0, 1, 1, -1]


def test_holiday_indicator_ignores_events_outside_store_locale() -> None:
    """A local event for another city does not mark this store's date."""
    date = pd.Timestamp("2017-01-05")
    holidays = pd.DataFrame(
        {
            "date": [date],
            "type": ["Holiday"],
            "locale": ["Local"],
            "locale_name": ["Guayaquil"],
            "transferred": [False],
        }
    )
    store = pd.Series({"city": "Quito", "state": "Pichincha"})

    assert evaluator.holiday_value_for_store(date, store, holidays) == 0
