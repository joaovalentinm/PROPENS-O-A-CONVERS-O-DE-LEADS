"""Preparação dos dados e engenharia de features.

Todo o pré-processamento fica dentro de um ``Pipeline`` do scikit-learn. Assim,
tudo que é "aprendido" com os dados (medianas para imputação, médias e desvios
para padronização, categorias do one-hot) é calculado **só no treino** e
apenas aplicado na validação e no teste, sem vazamento de dados.

Decisões de tratamento (justificadas no notebook 01_eda):

- **Outliers de tempo no site:** sessões acima de 1 hora são abas esquecidas
  ou bots. O valor é limitado em 3.600 s (capping). O limite é uma regra de
  negócio fixa, não um quantil, então não depende dos dados e não vaza informação.
- **Ausentes numéricos (idade, tempo no site):** mediana do treino, que é
  robusta à assimetria do tempo no site.
- **Ausentes categóricos (dispositivo, UF):** categoria própria
  ``"desconhecido"``, pois a falha de rastreamento pode carregar informação.
- **custo_aquisicao_lead:** fica FORA do modelo (ver ``EXCLUDED_COLUMNS``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "leads.csv"

TARGET = "converteu"
MAX_SESSION_SECONDS = 3_600

# Colunas que não entram no modelo:
# - lead_id: identificador, não tem poder preditivo legítimo;
# - data_cadastro / horario_cadastro: substituídas por dia_da_semana e periodo_do_dia
#   (a data absoluta não se repete no futuro e faria o modelo decorar o calendário);
# - custo_aquisicao_lead: é quase todo explicado pelo canal (redundante), e o custo
#   por lead costuma ser rateado depois, por campanha, então não estaria disponível
#   no momento de pontuar o lead. Ele volta na análise de impacto (CPA).
EXCLUDED_COLUMNS = ["lead_id", "data_cadastro", "horario_cadastro", "custo_aquisicao_lead"]

NUMERIC_FEATURES = ["idade", "paginas_vistas", "tempo_no_site_seg", "engajamento"]
BINARY_FEATURES = ["visitou_pagina_preco", "abriu_email", "mobile_noite"]
CATEGORICAL_FEATURES = [
    "canal", "campanha", "criativo", "dispositivo", "uf",
    "faixa_etaria", "periodo_do_dia", "dia_da_semana",
]

AGE_BINS = [17, 24, 34, 44, 60]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-60"]
WEEKDAYS = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]


def load_leads(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lê a base bruta de leads (dados simulados)."""
    return pd.read_csv(path, parse_dates=["data_cadastro"])


def cap_session_time(time_seconds: pd.Series, limit: int = MAX_SESSION_SECONDS) -> pd.Series:
    """Limita o tempo de sessão ao teto de negócio, preservando os ausentes."""
    return time_seconds.clip(upper=limit)


def day_period(hour: pd.Series) -> pd.Series:
    """Classifica a hora do cadastro em madrugada, manha, tarde ou noite."""
    bins = [-1, 5, 11, 17, 23]
    labels = ["madrugada", "manha", "tarde", "noite"]
    return pd.cut(hour, bins=bins, labels=labels).astype(str)


def age_group(age: pd.Series) -> pd.Series:
    """Agrupa a idade em faixas; idade ausente vira ``NaN`` e é imputada depois."""
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS).astype(object)


def engagement_score(pages: pd.Series, time_seconds: pd.Series) -> pd.Series:
    """Combina páginas vistas e tempo no site em um único índice de engajamento.

    Usa log nas duas variáveis para que nenhuma domine a outra pela escala:
    ``log(1 + páginas) + log(1 + minutos)``.
    """
    return np.log1p(pages) + np.log1p(time_seconds / 60)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica o capping e cria as features derivadas.

    É uma transformação linha a linha, sem nada aprendido dos dados, por isso
    pode ser usada antes ou dentro do pipeline sem risco de vazamento.
    """
    out = df.copy()
    hour = pd.to_numeric(out["horario_cadastro"].str[:2], errors="coerce")
    dates = pd.to_datetime(out["data_cadastro"])

    out["tempo_no_site_seg"] = cap_session_time(out["tempo_no_site_seg"])
    out["faixa_etaria"] = age_group(out["idade"])
    out["periodo_do_dia"] = day_period(hour)
    out["dia_da_semana"] = dates.dt.dayofweek.map(dict(enumerate(WEEKDAYS)))
    out["engajamento"] = engagement_score(out["paginas_vistas"], out["tempo_no_site_seg"])
    is_night = (hour >= 22) | (hour < 6)
    out["mobile_noite"] = ((out["dispositivo"] == "mobile") & is_night).astype(int)
    return out


def build_column_transformer() -> ColumnTransformer:
    """Monta imputação, padronização e one-hot encoding por tipo de coluna."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="desconhecido")),
        # UFs com menos de 1% dos leads são agrupadas para não criar colunas ruidosas.
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                 sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessor() -> Pipeline:
    """Pipeline completo de pré-processamento: features derivadas e encoding."""
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("columns", build_column_transformer()),
    ])


def split_data(
    df: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Divide em treino, validação e teste (70/15/15), estratificado pelo alvo.

    Retorna ``X_train, X_val, X_test, y_train, y_val, y_test``. As features
    mantêm todas as colunas originais (inclusive o custo, usado na análise de
    impacto); o pipeline escolhe as colunas que de fato entram no modelo.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
