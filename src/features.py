"""Engenharia de features para o modelo de risco de crédito.

Responsável por transformar as variáveis limpas em features prontas para a
modelagem: criação de novas variáveis (linha a linha, sem vazamento) e a
construção do pré-processador (encoding + scaling) usado dentro de um
``Pipeline`` ajustado apenas no conjunto de treino.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "loan_status"

# Features numéricas após `build_features` (inclui os ratios criados).
NUM_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "loan_to_income_ratio",
    "loan_to_emp_length_ratio",
    "int_rate_to_loan_amt_ratio",
]

# Features categóricas.
CAT_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria novas features a partir do dataset limpo.

    São razões calculadas linha a linha (não usam estatísticas do conjunto,
    portanto são seguras contra vazamento de dados):

    - ``loan_to_income_ratio``: valor do empréstimo / renda
    - ``loan_to_emp_length_ratio``: tempo de emprego / valor do empréstimo
    - ``int_rate_to_loan_amt_ratio``: taxa de juros / valor do empréstimo
    """
    df = df.copy()

    df["loan_to_income_ratio"] = df["loan_amnt"] / df["person_income"]
    df["loan_to_emp_length_ratio"] = df["person_emp_length"] / df["loan_amnt"]
    df["int_rate_to_loan_amt_ratio"] = df["loan_int_rate"] / df["loan_amnt"]

    # Protege contra divisões por zero / infinitos
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def split_features_target(
    df: pd.DataFrame, target: str = TARGET
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa o DataFrame em features (X) e variável-alvo (y)."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Cria o pré-processador (encoding + scaling) para usar no Pipeline.

    - One-hot encoding nas variáveis categóricas.
    - Padronização (StandardScaler) nas variáveis numéricas.

    Deve ser ajustado apenas no treino (via ``Pipeline.fit``) para evitar
    vazamento de informação do conjunto de teste.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUM_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CAT_FEATURES,
            ),
        ],
        remainder="drop",
    )
