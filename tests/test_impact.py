"""Testes das métricas de impacto de negócio."""

import numpy as np
import pandas as pd
import pytest

from src.impact import (
    break_even_effort_cost, compare_focus, decile_table, focus_mask, gain_curve,
)


@pytest.fixture
def scored():
    """Score sintético em que leads de score alto convertem mais."""
    rng = np.random.default_rng(0)
    scores = rng.random(5_000)
    y = pd.Series((rng.random(5_000) < scores * 0.1).astype(int))
    cost = pd.Series(rng.uniform(5, 20, 5_000))
    return y, scores, cost


def test_decile_table_is_consistent(scored):
    y, scores, _ = scored
    table = decile_table(y, scores)
    assert len(table) == 10
    assert table["leads"].sum() == len(y)
    assert table["conversoes"].sum() == y.sum()
    assert table["pct_conversoes_acum"].iloc[-1] == pytest.approx(1.0)
    assert table["taxa_conversao"].iloc[0] > table["taxa_conversao"].iloc[-1]


def test_gain_curve_goes_from_zero_to_one(scored):
    y, scores, _ = scored
    gain = gain_curve(y, scores)
    assert gain["pct_conversoes"].iloc[-1] == pytest.approx(1.0)
    assert gain["pct_conversoes"].is_monotonic_increasing


def test_focus_mask_selects_top_fraction():
    mask = focus_mask(np.array([0.1, 0.9, 0.5, 0.7, 0.3]), 0.4)
    assert mask.tolist() == [False, True, False, True, False]


def test_cpa_is_equal_at_break_even(scored):
    y, scores, cost = scored
    c = break_even_effort_cost(y, scores, cost, fraction=0.3)
    cpa = compare_focus(y, scores, cost, fraction=0.3, effort_cost=c)["cpa"]
    assert cpa.iloc[0] == pytest.approx(cpa.iloc[1])
