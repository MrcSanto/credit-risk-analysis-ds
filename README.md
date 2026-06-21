# Análise de Risco de Crédito 💳

Trabalho da disciplina de **Data Science**: previsão de **inadimplência em
operações de crédito** com um modelo de classificação binária treinado sobre um
dataset do Kaggle.

O objetivo é, a partir de características do cliente e do empréstimo, estimar a
probabilidade de **inadimplência** (`loan_status`) e apoiar a decisão de
concessão de crédito.

## 📊 Dataset

- **Fonte:** [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Tamanho:** ~32,5 mil registros, 12 colunas
- **Alvo:** `loan_status` (`1` = inadimplente, `0` = adimplente)

| Coluna | Descrição |
|---|---|
| `person_age` | Idade do cliente |
| `person_income` | Renda anual |
| `person_home_ownership` | Tipo de moradia (RENT, OWN, MORTGAGE, OTHER) |
| `person_emp_length` | Tempo de emprego (anos) |
| `loan_intent` | Finalidade do empréstimo |
| `loan_grade` | Nota de risco do empréstimo (A a G) |
| `loan_amnt` | Valor do empréstimo |
| `loan_int_rate` | Taxa de juros |
| `loan_status` | **Alvo** — 1 = inadimplente, 0 = adimplente |
| `loan_percent_income` | Razão empréstimo / renda |
| `cb_person_default_on_file` | Histórico de calote (Y/N) |
| `cb_person_cred_hist_length` | Tempo de histórico de crédito (anos) |

> O dataset **não é versionado** (a pasta `data/` está no `.gitignore`). Baixe o
> arquivo `credit_risk_dataset.csv` no [link acima](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
> e coloque-o em `data/raw/`.

## 🗂️ Estrutura do projeto

```
credit-risk-analysis-ds/
├── data/
│   ├── raw/             # dataset bruto (baixado do Kaggle)
│   └── processed/       # dataset limpo (gerado pelo notebook 02)
├── notebooks/
│   ├── 01_eda.ipynb            # análise exploratória
│   ├── 02_preprocessing.ipynb  # limpeza → credit_clean.csv
│   └── 03_modeling.ipynb       # treino, avaliação e escolha do modelo
├── src/
│   ├── data.py          # carga e limpeza dos dados
│   ├── features.py      # engenharia de features e pré-processador
│   └── models.py        # treino, avaliação e persistência do modelo
├── models/
│   └── model.pkl        # pipeline final treinado
├── dashboard/           # app interativo (Streamlit)
├── reports/             # relatório técnico e figuras
└── docs/                # documentação e enunciado
```

## 🔄 Fluxo de trabalho

1. **EDA** ([`01_eda.ipynb`](notebooks/01_eda.ipynb)) — distribuições, valores ausentes, outliers, balanceamento do alvo e correlações.
2. **Pré-processamento** ([`02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb)) — remove duplicatas e outliers, imputa ausentes e salva `data/processed/credit_clean.csv`.
3. **Modelagem** ([`03_modeling.ipynb`](notebooks/03_modeling.ipynb)) — engenharia de features, comparação de modelos e treino do pipeline final.

## 📈 Resultados

Comparamos três modelos (todos com `class_weight='balanced'` para tratar o
desbalanceamento de ~22% de inadimplentes). O **HistGradientBoosting** foi o
escolhido por melhor ROC-AUC e recall:

| Modelo | ROC-AUC | Recall | F1 |
|---|---|---|---|
| **HistGradientBoosting** | **0,95** | **0,81** | 0,81 |
| Random Forest | 0,94 | 0,72 | 0,83 |
| Regressão Logística | 0,88 | 0,80 | 0,66 |

O pipeline final (pré-processamento + modelo) fica salvo em `models/model.pkl`.

## 🚀 Como executar

O projeto usa [**uv**](https://docs.astral.sh/uv/) e **Python 3.12**.

```bash
# 1. reproduzir o ambiente
uv sync

# 2. rodar os notebooks (na ordem 01 → 02 → 03)
uv run jupyter lab

# 3. iniciar o dashboard
uv run streamlit run dashboard/app.py
```

