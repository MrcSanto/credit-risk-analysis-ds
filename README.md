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

Requer **Python 3.12**. Siga os passos na ordem abaixo para garantir reprodutibilidade.

### 1. Clonar o repositório

```bash
git clone https://github.com/IgorZanette/credit-risk-analysis-ds.git
cd credit-risk-analysis-ds
```

### 2. Criar e ativar um ambiente virtual

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

> Use sempre o ambiente virtual para evitar conflitos de versão entre projetos.

### 3. Instalar dependências

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

> O `scikit-learn` está fixado na versão **1.8.0** — a mesma com que o modelo foi treinado. Não atualize essa versão ou o `model.pkl` não carregará.

### 4. Baixar o dataset

Baixe o arquivo `credit_risk_dataset.csv` em [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) e coloque em `data/raw/`.

### 5. Gerar o dataset limpo

```bash
python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from src.data import load_raw_data, clean_data, save_processed_data
df = load_raw_data('data/raw/credit_risk_dataset.csv')
df = clean_data(df)
Path('data/processed').mkdir(parents=True, exist_ok=True)
save_processed_data(df, 'data/processed/credit_clean.csv')
print('Gerado: data/processed/credit_clean.csv')
"
```

### 6. Iniciar o dashboard

```bash
python -m streamlit run dashboard/app.py
```

Acesse **http://localhost:8501** no navegador.

### 7. Gerar o relatório em PDF (opcional)

```bash
python -m jupyter nbconvert --to html --no-input reports/relatorio.ipynb --output relatorio --output-dir reports/
python -c "
from playwright.sync_api import sync_playwright
from pathlib import Path
html = Path('reports/relatorio.html').resolve()
pdf  = Path('reports/relatorio.pdf').resolve()
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(html.as_uri(), wait_until='networkidle')
    pg.pdf(path=str(pdf), format='A4', margin={'top':'15mm','bottom':'15mm','left':'15mm','right':'15mm'}, print_background=True)
    b.close()
print(f'PDF gerado: {pdf}')
"
```

### Com Makefile (se tiver `make` instalado)

```bash
make install                   # instala dependências
make data                      # gera credit_clean.csv
make dashboard                 # inicia o dashboard
make report                    # gera reports/relatorio.pdf
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
