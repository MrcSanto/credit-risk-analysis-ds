"""Carregamento e limpeza dos dados de risco de crédito.

Responsável por ler o dataset bruto em ``data/raw/``, aplicar a limpeza
inicial (tratamento de valores ausentes, tipos, duplicatas, outliers) e
persistir o resultado em ``data/processed/`` para uso nas etapas seguintes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Limiares de outliers definidos na EDA (01_eda):
# idade > 100 e tempo de emprego > 60 anos são erros de digitação (ex.: 144, 123).
AGE_MAX = 100
EMP_LENGTH_MAX = 60

# Colunas com valores ausentes que serão imputadas pela mediana.
COLS_IMPUTAR_MEDIANA = ["loan_int_rate", "person_emp_length"]


def load_raw_data(path: str = "data/raw/credit_risk_dataset.csv") -> pd.DataFrame:
    """Carrega o dataset original (Kaggle) a partir de ``data/raw/``."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a limpeza inicial e devolve um novo DataFrame.

    Etapas (decisões tomadas na EDA):
    1. Remove linhas duplicadas.
    2. Remove outliers impossíveis de ``person_age`` (> 100) e
       ``person_emp_length`` (> 60).
    3. Imputa valores ausentes de ``loan_int_rate`` e ``person_emp_length``
       pela mediana da própria coluna.
    """
    df = df.copy()

    # 1. Duplicatas
    df = df.drop_duplicates()

    # 2. Outliers impossíveis (valores de digitação, ex.: idade 144, emprego 123)
    df = df[df["person_age"] <= AGE_MAX]
    df = df[(df["person_emp_length"] <= EMP_LENGTH_MAX) | (df["person_emp_length"].isna())]

    # 3. Imputação de ausentes pela mediana
    for col in COLS_IMPUTAR_MEDIANA:
        df[col] = df[col].fillna(df[col].median())

    return df.reset_index(drop=True)


def save_processed_data(
    df: pd.DataFrame, path: str = "data/processed/credit_clean.csv"
) -> None:
    """Salva o dataset limpo/transformado em ``data/processed/``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
