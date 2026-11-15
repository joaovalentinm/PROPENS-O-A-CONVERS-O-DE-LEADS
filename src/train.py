"""Treina, compara e salva os modelos de propensão à conversão.

1. Baseline: regressão logística com ``class_weight="balanced"``.
2. Modelo principal: XGBoost com hiperparâmetros escolhidos por
   ``RandomizedSearchCV`` e validação cruzada estratificada.

A métrica principal é a PR-AUC (``average_precision``): com 4% de compradores,
ela mede o que importa para o negócio, a qualidade do topo do ranking.

Os dados usados são simulados (ver ``src/generate_data.py``).

Uso:
    python -m src.train
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import SEED, build_preprocessor, load_leads, split_data

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
BASELINE_PATH = MODELS_DIR / "baseline_logistica.joblib"
MODEL_PATH = MODELS_DIR / "modelo_xgboost.joblib"
METRICS_PATH = REPORTS_DIR / "metricas.json"
CV_RESULTS_PATH = REPORTS_DIR / "busca_hiperparametros.csv"

# Fatia de leads que o time comercial consegue trabalhar (cenário da Etapa 6).
# O limiar de decisão é o score que separa os 30% melhores leads.
TOP_FRACTION = 0.30
N_ITER_SEARCH = 15
CV_FOLDS = 3

XGB_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [200, 300, 500],
    "model__learning_rate": [0.02, 0.05, 0.1],
    "model__max_depth": [2, 3, 4, 5],
    "model__min_child_weight": [1, 5, 10, 20],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__reg_lambda": [1, 5, 10],
}


def build_logistic_pipeline() -> Pipeline:
    """Baseline: pré-processamento + regressão logística balanceada."""
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=SEED)),
    ])


def build_xgb_pipeline() -> Pipeline:
    """Modelo principal: pré-processamento + XGBoost.

    Não usamos ``scale_pos_weight``: o objetivo é ordenar leads, e o
    rebalanceamento distorceria as probabilidades usadas na análise de impacto.
    """
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def tune_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    """Busca aleatória de hiperparâmetros com validação cruzada estratificada."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        build_xgb_pipeline(),
        param_distributions=XGB_PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        scoring="average_precision",
        cv=cv,
        random_state=SEED,
        n_jobs=1,
        refit=True,
    )
    return search.fit(X_train, y_train)


def top_fraction_threshold(scores: np.ndarray, fraction: float = TOP_FRACTION) -> float:
    """Score mínimo para um lead estar entre os ``fraction`` melhores."""
    return float(np.quantile(scores, 1 - fraction))


def evaluate(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    """Calcula as métricas de ranking, de classificação e de calibração."""
    predicted = (scores >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted),
        "brier": brier_score_loss(y_true, scores),
        "taxa_base": float(np.mean(y_true)),
    }


def save_artifacts(baseline: Pipeline, model: Pipeline, metrics: dict, cv_results: pd.DataFrame) -> None:
    """Salva os modelos (joblib), as métricas e o resultado da busca."""
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(baseline, BASELINE_PATH)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    cv_results.to_csv(CV_RESULTS_PATH, index=False)


def summarize_search(search: RandomizedSearchCV) -> pd.DataFrame:
    """Organiza o resultado da busca, do melhor para o pior conjunto de hiperparâmetros."""
    results = pd.DataFrame(search.cv_results_)
    params = results["params"].apply(lambda p: {k.replace("model__", ""): v for k, v in p.items()})
    return (
        pd.concat([pd.json_normalize(params), results[["mean_test_score", "std_test_score"]]], axis=1)
        .rename(columns={"mean_test_score": "pr_auc_cv", "std_test_score": "pr_auc_cv_std"})
        .sort_values("pr_auc_cv", ascending=False)
        .reset_index(drop=True)
    )


def main() -> dict:
    """Executa o treino completo e devolve as métricas."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(load_leads())

    print("Treinando baseline (regressão logística)...")
    baseline = build_logistic_pipeline().fit(X_train, y_train)

    print(f"Buscando hiperparâmetros do XGBoost ({N_ITER_SEARCH} combinações x {CV_FOLDS} folds)...")
    search = tune_xgb(X_train, y_train)
    model = search.best_estimator_

    metrics = {"melhores_hiperparametros": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
               "pr_auc_validacao_cruzada": search.best_score_}
    for name, pipe in [("logistica", baseline), ("xgboost", model)]:
        val_scores = pipe.predict_proba(X_val)[:, 1]
        threshold = top_fraction_threshold(val_scores)
        metrics[name] = {
            "limiar_top30": threshold,
            "validacao": evaluate(y_val, val_scores, threshold),
            "teste": evaluate(y_test, pipe.predict_proba(X_test)[:, 1], threshold),
        }

    save_artifacts(baseline, model, metrics, summarize_search(search))
    for name in ["logistica", "xgboost"]:
        m = metrics[name]["teste"]
        print(f"{name:>10} | teste: PR-AUC {m['pr_auc']:.3f} | ROC-AUC {m['roc_auc']:.3f} "
              f"| precision {m['precision']:.3f} | recall {m['recall']:.3f}")
    print(f"Modelos salvos em {MODELS_DIR}")
    return metrics


if __name__ == "__main__":
    main()
