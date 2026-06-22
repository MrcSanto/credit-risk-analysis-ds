"""Treino e avaliação dos modelos de classificação de inadimplência.

Responsável por dividir os dados em treino/teste, montar os pipelines
(pré-processamento + classificador), treiná-los, avaliá-los com métricas
adequadas a dados desbalanceados (ROC AUC, recall, F1, matriz de confusão)
e persistir/carregar o modelo final.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.features import build_features, build_preprocessor

RANDOM_STATE = 42


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide os dados em treino e teste de forma estratificada pelo alvo."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def get_models() -> dict[str, Pipeline]:
    """Retorna os modelos candidatos, cada um como um Pipeline completo.

    Todos usam ``class_weight='balanced'`` para tratar o desbalanceamento
    (~22% de inadimplentes). Cada Pipeline é **autocontido**: parte das colunas
    cruas (dataset limpo), cria as features (ratios), aplica o pré-processador
    (encoding/scaling ajustado só no treino) e então o classificador. Assim o
    modelo salvo recebe dados crus e devolve a previsão direto — ideal para o
    dashboard.
    """
    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    return {
        name: Pipeline(
            steps=[
                ("features", FunctionTransformer(build_features)),
                ("preprocessor", build_preprocessor()),
                ("clf", clf),
            ]
        )
        for name, clf in classifiers.items()
    }


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model: Pipeline | None = None) -> Pipeline:
    """Treina um pipeline de classificação. Sem ``model``, usa o gradient boosting."""
    if model is None:
        model = get_models()["hist_gradient_boosting"]
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Avalia o modelo e retorna as métricas adequadas a base desbalanceada."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def ks_gini(y_true: pd.Series, y_proba) -> dict:
    """Métricas clássicas de risco de crédito: KS e Gini.

    - **KS** (Kolmogorov-Smirnov): máxima distância entre as taxas acumuladas de
      verdadeiros positivos e falsos positivos ao longo do score; mede o poder de
      separação entre bons e maus pagadores.
    - **Gini** = 2 · ROC-AUC − 1; padrão de mercado em credit scoring.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    return {
        "ks": float(np.max(tpr - fpr)),
        "gini": float(2 * auc - 1),
        "roc_auc": float(auc),
    }


def cross_validate_auc(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> dict:
    """ROC-AUC por validação cruzada estratificada (StratifiedKFold).

    Mostra a estabilidade/robustez do modelo para além do split único de teste.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


def save_model(model: Any, path: str = "models/model.pkl") -> None:
    """Persiste o modelo (pipeline completo) treinado em disco."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = "models/model.pkl") -> Any:
    """Carrega um modelo previamente treinado (usado também pelo dashboard)."""
    return joblib.load(path)
