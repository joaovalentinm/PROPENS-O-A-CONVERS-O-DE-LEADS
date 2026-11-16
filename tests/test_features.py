"""Testes da preparação de dados e do pipeline de features."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    MAX_SESSION_SECONDS, add_features, build_preprocessor, split_data,
)
from src.generate_data import generate_leads
from src.train import build_logistic_pipeline


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    """Amostra pequena da base simulada, com a mesma estrutura da base real."""
    return generate_leads(n=5_000, seed=1)


def test_add_features_creates_columns_and_caps_outliers(sample):
    out = add_features(sample)
    for col in ["faixa_etaria", "periodo_do_dia", "dia_da_semana", "engajamento", "mobile_noite"]:
        assert col in out.columns
    assert out["tempo_no_site_seg"].max() <= MAX_SESSION_SECONDS
    assert set(out["periodo_do_dia"]) <= {"madrugada", "manha", "tarde", "noite"}


def test_pipeline_handles_missing_values(sample):
    dirty = sample.copy()
    for col in ["idade", "tempo_no_site_seg", "dispositivo", "uf"]:
        dirty.loc[dirty.sample(frac=0.3, random_state=0).index, col] = np.nan

    transformed = build_preprocessor().fit_transform(dirty.drop(columns="converteu"))
    assert not np.isnan(transformed).any()


def test_pipeline_handles_row_with_all_optional_fields_missing(sample):
    pipe = build_logistic_pipeline().fit(sample.drop(columns="converteu"), sample["converteu"])
    lead = sample.drop(columns="converteu").head(1).copy()
    lead[["idade", "tempo_no_site_seg", "dispositivo", "uf"]] = np.nan
    proba = pipe.predict_proba(lead)[:, 1]
    assert 0 <= proba[0] <= 1


def test_pipeline_handles_unseen_category(sample):
    preprocessor = build_preprocessor().fit(sample.drop(columns="converteu"))
    lead = sample.drop(columns="converteu").head(1).copy()
    lead["canal"] = "linkedin_ads"
    assert preprocessor.transform(lead).shape[0] == 1


def test_split_is_70_15_15_and_stratified(sample):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample)
    n = len(sample)
    assert len(X_train) == pytest.approx(0.70 * n, abs=2)
    assert len(X_val) == pytest.approx(0.15 * n, abs=2)
    assert len(X_test) == pytest.approx(0.15 * n, abs=2)
    for y in (y_train, y_val, y_test):
        assert y.mean() == pytest.approx(sample["converteu"].mean(), abs=0.005)
    assert not set(X_train.index) & set(X_test.index)
