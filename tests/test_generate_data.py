"""Testes do gerador de dados simulados."""

import pandas as pd
import pytest

from src.generate_data import COLUMNS_WITH_MISSING, generate_leads

EXPECTED_COLUMNS = [
    "lead_id", "data_cadastro", "horario_cadastro", "canal", "campanha", "criativo",
    "dispositivo", "uf", "idade", "paginas_vistas", "tempo_no_site_seg",
    "visitou_pagina_preco", "abriu_email", "custo_aquisicao_lead", "converteu",
]


@pytest.fixture(scope="module")
def leads() -> pd.DataFrame:
    """Base completa, gerada uma única vez para todos os testes do módulo."""
    return generate_leads()


def test_generator_creates_expected_columns(leads):
    assert sorted(leads.columns) == sorted(EXPECTED_COLUMNS)


def test_generator_creates_50k_unique_leads(leads):
    assert len(leads) == 50_000
    assert leads["lead_id"].is_unique


def test_conversion_rate_within_expected_range(leads):
    assert 0.03 <= leads["converteu"].mean() <= 0.05


def test_categorical_domains(leads):
    assert leads["canal"].nunique() == 5
    assert leads["campanha"].nunique() == 8
    assert set(leads["criativo"]) == {"video", "carrossel", "imagem_estatica"}
    assert set(leads["dispositivo"].dropna()) == {"mobile", "desktop"}


def test_numeric_ranges(leads):
    assert leads["idade"].dropna().between(18, 60).all()
    dates = pd.to_datetime(leads["data_cadastro"])
    assert dates.min() >= pd.Timestamp("2025-01-01")
    assert dates.max() <= pd.Timestamp("2025-12-31")
    assert (leads["custo_aquisicao_lead"] > 0).all()


def test_missing_values_around_two_percent(leads):
    missing = leads[COLUMNS_WITH_MISSING].isna().mean()
    assert missing.between(0.015, 0.025).all()


def test_generator_is_reproducible():
    pd.testing.assert_frame_equal(generate_leads(n=2_000, seed=7), generate_leads(n=2_000, seed=7))
