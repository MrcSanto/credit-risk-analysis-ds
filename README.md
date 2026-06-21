# Análise de Risco de Crédito

Trabalho final da disciplina de **Data Science**: previsão de **inadimplência em operações de crédito** com classificação binária treinada sobre um dataset do Kaggle.

**Equipe:**
- Igor Zanette — Matrícula 198862
- Marco Antônio Santolin — Matrícula 198769
- Marcos Paulo de Medeiros — Matrícula 151994

---

## Dataset

- **Fonte:** [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Tamanho:** ~32.500 registros, 12 colunas
- **Alvo:** `loan_status` (`1` = inadimplente, `0` = adimplente)

| Coluna | Descrição |
|---|---|
| `person_age` | Idade do cliente |
| `person_income` | Renda anual |
| `person_home_ownership` | Tipo de moradia (RENT, OWN, MORTGAGE, OTHER) |
| `person_emp_length` | Tempo de emprego (anos) |
| `loan_intent` | Finalidade do empréstimo |
| `loan_grade` | Nota de risco (A a G) |
| `loan_amnt` | Valor do empréstimo |
| `loan_int_rate` | Taxa de juros |
| `loan_status` | **Alvo** — 1 = inadimplente, 0 = adimplente |
| `loan_percent_income` | Razão empréstimo / renda |
| `cb_person_default_on_file` | Histórico de calote (Y/N) |
| `cb_person_cred_hist_length` | Tempo de histórico de crédito (anos) |

> O dataset **não é versionado**. Baixe `credit_risk_dataset.csv` no link acima e coloque em `data/raw/`.

---

## Estrutura do projeto

```
credit-risk-analysis-ds/
├── data/
│   ├── raw/              # dataset bruto (não versionado)
│   └── processed/        # dataset limpo (gerado via make data)
├── notebooks/
│   ├── 01_eda.ipynb            # análise exploratória
│   ├── 02_preprocessing.ipynb  # limpeza e geração de credit_clean.csv
│   └── 03_modeling.ipynb       # treino, comparação e escolha do modelo
├── src/
│   ├── data.py           # carga e limpeza dos dados
│   ├── features.py       # engenharia de features e pré-processador
│   └── models.py         # treino, avaliação e persistência do modelo
├── models/
│   └── model.pkl         # pipeline final treinado
├── dashboard/
│   └── app.py            # dashboard interativo (Streamlit)
├── reports/
│   └── relatorio.ipynb   # relatório técnico (executar via make report)
├── docs/
│   └── DIVISAO_TAREFAS.md
├── Makefile
└── requirements.txt
```

---

## Resultados do modelo

Comparamos três modelos com `class_weight='balanced'` para tratar o desbalanceamento (~22% inadimplentes). O **HistGradientBoosting** foi escolhido por melhor ROC-AUC e recall:

| Modelo | ROC-AUC | Recall | F1 | Acurácia |
|---|---|---|---|---|
| **HistGradientBoosting** | **0,95** | **0,81** | 0,81 | 0,92 |
| Random Forest | 0,94 | 0,72 | 0,83 | 0,92 |
| Regressão Logística | 0,88 | 0,80 | 0,66 | 0,84 |

---

## Como executar

Requer **Python 3.12** e `pip`. Clone o repositório e siga os passos:

```bash
# 1. Instalar dependências
make install

# 2. Gerar dataset limpo (coloque credit_risk_dataset.csv em data/raw/ antes)
make data

# 3. Executar os notebooks (opcional — outputs já salvos no repo)
make notebooks

# 4. Iniciar o dashboard interativo
make dashboard
# acesse http://localhost:8501

# 5. Gerar o relatório em PDF
make report
# gera reports/relatorio.pdf via Playwright/Chromium

# Ou sem PDF (só HTML):
make report-html
```

### Sem Makefile (manual)

```bash
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

---

## Dashboard

O app Streamlit (`dashboard/app.py`) tem três abas:

- **EDA** — KPIs, distribuição de inadimplência por grade/finalidade, boxplots e heatmap de correlação, com filtros interativos na sidebar
- **Simulador de Risco** — formulário com os 11 campos do dataset; retorna probabilidade de default com indicador de cor e comparativo com médias do dataset
- **Resultados do Modelo** — comparação de métricas entre os 3 modelos, matriz de confusão e importância das features por permutação

---

## Relatório técnico

O relatório está em `reports/relatorio.ipynb`. Para exportar:

```bash
make report       # gera reports/relatorio.pdf
make report-html  # gera reports/relatorio.html
```

O PDF é gerado automaticamente via Playwright (Chromium) — não precisa de pandoc nem LaTeX.
