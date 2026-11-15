"""Métricas de impacto de negócio do score de propensão.

Transforma probabilidades em decisões: tabela por decil, curva de ganho,
lift e simulação de custo por conversão (CPA) quando a empresa concentra
o esforço comercial e o remarketing nos leads de maior score.

Todos os valores em reais vêm de dados SIMULADOS.
"""

import numpy as np
import pandas as pd

DEFAULT_EFFORT_COST = 25.0  # R$ por lead trabalhado (SDR + remarketing), premissa


def decile_table(y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    """Ordena os leads pelo score e resume a conversão em 10 decis (1 = maior score)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "score": scores})
    # Ranking antes do qcut evita decis desiguais quando há scores empatados.
    ranks = df["score"].rank(method="first", ascending=False)
    df["decil"] = pd.qcut(ranks, 10, labels=range(1, 11)).astype(int)

    table = df.groupby("decil").agg(
        leads=("y", "size"),
        conversoes=("y", "sum"),
        score_medio=("score", "mean"),
    )
    base_rate = df["y"].mean()
    table["taxa_conversao"] = table["conversoes"] / table["leads"]
    table["lift"] = table["taxa_conversao"] / base_rate
    table["pct_conversoes"] = table["conversoes"] / table["conversoes"].sum()
    table["pct_conversoes_acum"] = table["pct_conversoes"].cumsum()
    table["pct_leads_acum"] = table["leads"].cumsum() / table["leads"].sum()
    table["lift_acum"] = table["pct_conversoes_acum"] / table["pct_leads_acum"]
    return table


def gain_curve(y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    """Fração acumulada de conversões capturada ao trabalhar os leads em ordem de score."""
    y_sorted = np.asarray(y_true)[np.argsort(-scores, kind="stable")]
    n = len(y_sorted)
    return pd.DataFrame({
        "pct_leads": np.arange(1, n + 1) / n,
        "pct_conversoes": np.cumsum(y_sorted) / y_sorted.sum(),
    })


def focus_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Marca os ``fraction`` leads de maior score (o grupo que será trabalhado)."""
    n_selected = int(round(len(scores) * fraction))
    mask = np.zeros(len(scores), dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:n_selected]] = True
    return mask


def scenario_cost(
    y_true: pd.Series,
    acquisition_cost: pd.Series,
    worked: np.ndarray,
    effort_cost: float = DEFAULT_EFFORT_COST,
) -> dict:
    """Custo total, conversões e CPA de um cenário.

    Premissas (conservadoras):
    - o custo de aquisição de TODOS os leads já foi gasto, nos dois cenários;
    - cada lead trabalhado custa ``effort_cost`` (tempo de SDR + remarketing);
    - só os leads trabalhados convertem; as conversões fora do grupo são
      tratadas como perdidas (na prática, parte delas aconteceria mesmo assim).
    """
    y = np.asarray(y_true)
    acquisition = float(np.sum(acquisition_cost))
    effort = effort_cost * worked.sum()
    conversions = int(y[worked].sum())
    total = acquisition + effort
    return {
        "leads_trabalhados": int(worked.sum()),
        "conversoes": conversions,
        "custo_aquisicao": acquisition,
        "custo_esforco": effort,
        "custo_total": total,
        "cpa": total / conversions,
        "cpa_esforco": effort / conversions,
    }


def compare_focus(
    y_true: pd.Series,
    scores: np.ndarray,
    acquisition_cost: pd.Series,
    fraction: float = 0.30,
    effort_cost: float = DEFAULT_EFFORT_COST,
    seed: int = 42,
) -> pd.DataFrame:
    """Compara trabalhar todos os leads, os top ``fraction`` pelo modelo e uma amostra aleatória do mesmo tamanho."""
    rng = np.random.default_rng(seed)
    everyone = np.ones(len(scores), dtype=bool)
    random_pick = focus_mask(rng.random(len(scores)), fraction)
    scenarios = {
        "Todos os leads (atual)": everyone,
        f"Top {fraction:.0%} pelo modelo": focus_mask(scores, fraction),
        f"{fraction:.0%} aleatórios": random_pick,
    }
    table = pd.DataFrame({
        name: scenario_cost(y_true, acquisition_cost, mask, effort_cost)
        for name, mask in scenarios.items()
    }).T
    base = table.iloc[0]
    table["pct_conversoes_capturadas"] = table["conversoes"] / base["conversoes"]
    table["variacao_cpa"] = table["cpa"] / base["cpa"] - 1
    return table


def break_even_effort_cost(
    y_true: pd.Series, scores: np.ndarray, acquisition_cost: pd.Series, fraction: float = 0.30
) -> float:
    """Custo por lead trabalhado a partir do qual focar no top ``fraction`` reduz o CPA.

    Resolve (A + c·N) / C = (A + c·n) / Cf para c, em que A é o custo de aquisição,
    N e C são leads e conversões totais, e n e Cf são os do grupo focado.
    """
    y = np.asarray(y_true)
    mask = focus_mask(scores, fraction)
    A, N, C = float(np.sum(acquisition_cost)), len(y), y.sum()
    n, Cf = mask.sum(), y[mask].sum()
    return A * (C - Cf) / (N * Cf - n * C)
