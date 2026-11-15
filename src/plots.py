"""Estilo visual e funções de gráfico reutilizadas pelos notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# Paleta: uma cor principal para a série, cinza para referência e
# tons de texto neutros (o texto nunca usa a cor da série).
COLORS = {
    "primary": "#2a78d6",
    "secondary": "#eb6834",
    "reference": "#8a8984",
    "muted_bar": "#c9c8c2",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "grid": "#e6e5e0",
    "surface": "#fcfcfb",
}


def set_style() -> None:
    """Aplica o estilo padrão dos gráficos do projeto."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["surface"],
        "axes.facecolor": COLORS["surface"],
        "savefig.facecolor": COLORS["surface"],
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text_secondary"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": COLORS["text_secondary"],
        "ytick.color": COLORS["text_secondary"],
        "text.color": COLORS["text"],
        "legend.frameon": False,
    })


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Salva a figura em reports/figures com o nome informado."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def conversion_by(df: pd.DataFrame, column: str, target: str = "converteu") -> pd.DataFrame:
    """Calcula volume, conversões, taxa e intervalo de confiança de 95% por categoria."""
    out = df.groupby(column, observed=True)[target].agg(leads="size", conversoes="sum")
    out["taxa"] = out["conversoes"] / out["leads"]
    out["ic95"] = 1.96 * np.sqrt(out["taxa"] * (1 - out["taxa"]) / out["leads"])
    return out


def plot_rate_bars(
    rates: pd.DataFrame,
    title: str,
    overall_rate: float,
    sort: bool = True,
    show_ci: bool = False,
    highlight: list | None = None,
) -> plt.Figure:
    """Barras horizontais com a taxa de conversão por categoria.

    A linha tracejada marca a taxa geral, para facilitar a leitura de quem
    está acima ou abaixo da média. Categorias em ``highlight`` ganham a cor
    principal; as demais ficam em cinza. Sem ``highlight``, todas são coloridas.
    """
    data = rates.sort_values("taxa") if sort else rates.iloc[::-1]
    labels = data.index.astype(str)
    colors = [
        COLORS["primary"] if highlight is None or lab in highlight else COLORS["muted_bar"]
        for lab in labels
    ]

    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(data) + 1.2))
    ax.barh(labels, data["taxa"], color=colors, height=0.62,
            xerr=data["ic95"] if show_ci else None,
            error_kw={"ecolor": COLORS["text_secondary"], "elinewidth": 1, "capsize": 3})
    ax.axvline(overall_rate, color=COLORS["reference"], linestyle="--", linewidth=1.2)
    ax.set_ylim(-0.5, len(data) + 0.1)
    ax.text(overall_rate, len(data) - 0.55, f" média geral {overall_rate:.1%}",
            color=COLORS["text_secondary"], fontsize=9, va="bottom")

    errors = data["ic95"] if show_ci else pd.Series(0.0, index=data.index)
    for i, (value, err) in enumerate(zip(data["taxa"], errors)):
        ax.text(value + err + 0.001, i, f"{value:.1%}", va="center",
                fontsize=9, color=COLORS["text"])

    ax.set_title(title)
    ax.set_xlabel("Taxa de conversão")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, (data["taxa"] + errors).max() * 1.2)
    fig.tight_layout()
    return fig
