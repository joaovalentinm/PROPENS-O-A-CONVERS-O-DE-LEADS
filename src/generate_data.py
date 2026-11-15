"""Gera a base simulada de leads de uma empresa de cursos online.

ATENÇÃO: todos os dados produzidos por este módulo são FICTÍCIOS.
Eles foram criados apenas para fins de portfólio e não representam
nenhuma empresa, campanha ou pessoa real.

A probabilidade de conversão de cada lead vem de uma função logística
cujos coeficientes imitam relações plausíveis de marketing digital:

- visitar a página de preço, abrir e-mail e passar mais tempo no site
  aumentam a chance de compra;
- leads vindos de orgânico e e-mail convertem mais que mídia paga;
- cadastros pelo celular à noite convertem um pouco menos.

Também são inseridos ruído, ~2% de valores ausentes em algumas colunas e
outliers em ``tempo_no_site_seg``, para que a etapa de limpeza tenha o que
tratar.

Uso:
    python -m src.generate_data
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_LEADS = 50_000
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "leads.csv"

# Participação de cada canal no volume de leads.
CHANNELS = {
    "google_ads": 0.30,
    "meta_ads": 0.30,
    "tiktok_ads": 0.15,
    "email": 0.10,
    "organico": 0.15,
}

# Custo médio por lead (R$) em cada canal. Orgânico tem custo residual
# (conteúdo/SEO), e-mail tem custo baixo (ferramenta de disparo).
CHANNEL_CPL = {
    "google_ads": 18.0,
    "meta_ads": 12.0,
    "tiktok_ads": 9.0,
    "email": 2.5,
    "organico": 1.0,
}

# Efeito do canal na log-odds de conversão (referência: meta_ads).
CHANNEL_EFFECT = {
    "google_ads": 0.20,
    "meta_ads": 0.00,
    "tiktok_ads": -0.30,
    "email": 0.55,
    "organico": 0.70,
}

# Campanhas disponíveis por canal (8 no total).
CHANNEL_CAMPAIGNS = {
    "google_ads": ["search_marca", "search_generico"],
    "meta_ads": ["meta_prospeccao", "meta_remarketing"],
    "tiktok_ads": ["tiktok_awareness"],
    "email": ["email_newsletter", "email_black_friday"],
    "organico": ["organico_blog"],
}

CAMPAIGN_EFFECT = {
    "search_marca": 0.30,
    "search_generico": -0.10,
    "meta_prospeccao": -0.15,
    "meta_remarketing": 0.35,
    "tiktok_awareness": 0.00,
    "email_newsletter": 0.00,
    "email_black_friday": 0.25,
    "organico_blog": 0.00,
}

CREATIVES = {"video": 0.40, "carrossel": 0.30, "imagem_estatica": 0.30}
CREATIVE_EFFECT = {"video": 0.15, "carrossel": 0.05, "imagem_estatica": 0.00}

DEVICES = {"mobile": 0.68, "desktop": 0.32}

# Pesos aproximados por UF, inspirados na distribuição de população e
# acesso à internet: SP, RJ e MG concentram a maior parte dos leads.
UF_WEIGHTS = {
    "SP": 0.28, "RJ": 0.11, "MG": 0.10, "PR": 0.06, "RS": 0.06,
    "BA": 0.05, "SC": 0.04, "PE": 0.04, "GO": 0.035, "DF": 0.03,
    "CE": 0.03, "ES": 0.02, "PA": 0.02, "MT": 0.015, "MS": 0.015,
    "AM": 0.015, "PB": 0.01, "RN": 0.01, "MA": 0.01, "AL": 0.008,
    "PI": 0.007, "SE": 0.007, "RO": 0.006, "TO": 0.005, "AC": 0.003,
    "AP": 0.003, "RR": 0.003,
}

# Intercepto da função logística, calibrado para a conversão geral
# ficar entre 3% e 5%.
INTERCEPT = -4.7

# Colunas que recebem ~2% de valores ausentes (falhas de rastreamento).
COLUMNS_WITH_MISSING = ["idade", "tempo_no_site_seg", "dispositivo", "uf"]
MISSING_RATE = 0.02
OUTLIER_RATE = 0.005


def _normalized(weights: dict) -> tuple[list, np.ndarray]:
    """Devolve as chaves e as probabilidades normalizadas de um dicionário."""
    keys = list(weights)
    probs = np.array([weights[k] for k in keys], dtype=float)
    return keys, probs / probs.sum()


def _choice(rng: np.random.Generator, weights: dict, size: int) -> np.ndarray:
    """Sorteia categorias de acordo com os pesos informados."""
    keys, probs = _normalized(weights)
    return rng.choice(keys, size=size, p=probs)


def generate_acquisition(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Gera as colunas de origem do lead: data, canal, campanha e criativo."""
    start = pd.Timestamp("2025-01-01")
    days = rng.integers(0, 365, size=n)
    # Horário com pico à noite (20h) e um pico menor no almoço (12h).
    hour = np.where(
        rng.random(n) < 0.6,
        rng.normal(20, 2.5, n),
        rng.normal(12, 3.0, n),
    )
    hour = np.mod(np.round(hour), 24).astype(int)
    minute = rng.integers(0, 60, size=n)

    channel = _choice(rng, CHANNELS, n)
    campaign = np.array([rng.choice(CHANNEL_CAMPAIGNS[c]) for c in channel])
    creative = _choice(rng, CREATIVES, n)

    return pd.DataFrame({
        "data_cadastro": (start + pd.to_timedelta(days, unit="D")).date,
        "horario_cadastro": [f"{h:02d}:{m:02d}" for h, m in zip(hour, minute)],
        "canal": channel,
        "campanha": campaign,
        "criativo": creative,
    })


def generate_profile(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Gera o perfil do lead: dispositivo, UF e idade."""
    age = np.clip(rng.normal(32, 9, n), 18, 60).round().astype(int)
    return pd.DataFrame({
        "dispositivo": _choice(rng, DEVICES, n),
        "uf": _choice(rng, UF_WEIGHTS, n),
        "idade": age,
    })


def generate_behavior(rng: np.random.Generator, channel: np.ndarray) -> pd.DataFrame:
    """Gera o comportamento no site e no e-mail.

    Leads de e-mail e orgânico tendem a navegar mais, pois já conhecem a marca.
    """
    n = len(channel)
    warm = np.isin(channel, ["email", "organico"])

    # Um "interesse latente" correlaciona páginas, tempo e visita ao preço.
    interest = rng.normal(0, 1, n) + 0.4 * warm

    pages = rng.poisson(np.exp(1.1 + 0.35 * interest)) + 1
    time_on_site = rng.lognormal(mean=4.6 + 0.35 * interest, sigma=0.6)
    price_prob = 1 / (1 + np.exp(-(-1.3 + 0.9 * interest)))
    email_prob = np.where(channel == "email", 0.55, 0.25)

    return pd.DataFrame({
        "paginas_vistas": pages,
        "tempo_no_site_seg": time_on_site.round(0),
        "visitou_pagina_preco": (rng.random(n) < price_prob).astype(int),
        "abriu_email": (rng.random(n) < email_prob).astype(int),
    })


def generate_lead_cost(rng: np.random.Generator, channel: np.ndarray) -> np.ndarray:
    """Gera o custo de aquisição de cada lead (R$) com variação em torno do CPL do canal."""
    base = np.array([CHANNEL_CPL[c] for c in channel])
    return (base * rng.lognormal(0, 0.35, len(channel))).round(2)


def conversion_probability(df: pd.DataFrame) -> np.ndarray:
    """Calcula a probabilidade de conversão a partir de uma função logística."""
    hour = df["horario_cadastro"].str[:2].astype(int)
    mobile_night = ((df["dispositivo"] == "mobile") & ((hour >= 22) | (hour < 6))).astype(int)
    log_time = np.log1p(df["tempo_no_site_seg"])

    logit = (
        INTERCEPT
        + 1.60 * df["visitou_pagina_preco"]
        + 0.80 * df["abriu_email"]
        + 0.45 * (log_time - log_time.mean())
        + 0.06 * (df["paginas_vistas"] - df["paginas_vistas"].mean())
        + df["canal"].map(CHANNEL_EFFECT)
        + df["campanha"].map(CAMPAIGN_EFFECT)
        + df["criativo"].map(CREATIVE_EFFECT)
        + 0.15 * (df["dispositivo"] == "desktop")
        - 0.35 * mobile_night
        + 0.012 * (df["idade"] - 32)
    )
    return 1 / (1 + np.exp(-logit))


def add_outliers(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    """Insere tempos de sessão absurdos (abas esquecidas abertas, bots)."""
    df = df.copy()
    idx = rng.choice(df.index, size=int(len(df) * OUTLIER_RATE), replace=False)
    df.loc[idx, "tempo_no_site_seg"] = rng.uniform(10_000, 86_400, len(idx)).round(0)
    return df


def add_missing_values(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    """Apaga ~2% dos valores das colunas sujeitas a falhas de rastreamento."""
    df = df.copy()
    for col in COLUMNS_WITH_MISSING:
        mask = rng.random(len(df)) < MISSING_RATE
        df.loc[mask, col] = np.nan
    return df


def generate_leads(n: int = N_LEADS, seed: int = SEED) -> pd.DataFrame:
    """Gera a base completa de leads simulados (dados fictícios)."""
    rng = np.random.default_rng(seed)

    df = pd.concat([generate_acquisition(rng, n), generate_profile(rng, n)], axis=1)
    df = pd.concat([df, generate_behavior(rng, df["canal"].to_numpy())], axis=1)
    df["custo_aquisicao_lead"] = generate_lead_cost(rng, df["canal"].to_numpy())

    # A conversão é sorteada antes da sujeira entrar: os outliers e ausentes
    # representam problemas de medição, não do comportamento real do lead.
    df["converteu"] = (rng.random(n) < conversion_probability(df)).astype(int)

    df = add_outliers(rng, df)
    df = add_missing_values(rng, df)

    df.insert(0, "lead_id", [f"L{i:06d}" for i in range(1, n + 1)])
    return df.sort_values(["data_cadastro", "horario_cadastro"]).reset_index(drop=True)


def main() -> None:
    """Gera os dados e salva em data/raw/leads.csv."""
    df = generate_leads()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"{len(df):,} leads fictícios salvos em {OUTPUT_PATH}")
    print(f"Taxa de conversão: {df['converteu'].mean():.2%}")


if __name__ == "__main__":
    main()
