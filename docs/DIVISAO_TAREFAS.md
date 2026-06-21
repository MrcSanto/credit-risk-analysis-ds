# Divisão de Tarefas — Projeto Análise de Risco de Crédito

## Equipe

| Pessoa | Matrícula | Responsabilidade principal |
|---|---|---|
| **Marco Antônio Santolin** | 198769 | EDA + Pré-processamento + Modelagem + Avaliação |
| **Igor Zanette** | 198862 | Dashboard interativo (Streamlit) + Relatório técnico |
| **Marcos Paulo de Medeiros** | 151994 | Relatório PDF + Revisão geral + Organização do repositório |

---

## O que é o projeto

Trabalho final de Data Science. Previsão de **inadimplência de crédito** (coluna
`loan_status`: `1` = cliente deu calote, `0` = pagou) a partir de dados de empréstimos.

**Dataset:** `data/raw/credit_risk_dataset.csv` — ~32.500 linhas, 12 colunas.
Fonte: [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

---

## Entregáveis

1. **Repositório no GitHub** ✅ — organizado, com README, código comentado e reprodutível.
   - Acesso do professor: `holbig@upf.br`
2. **Relatório técnico em PDF** ✅ — `reports/relatorio.ipynb` → exportar com `make report-html`
3. **Dashboard interativo** ✅ — Streamlit em `dashboard/app.py`, rodar com `make dashboard`

---

## Checklist de tarefas

### Marco Antônio — Técnico
- [x] EDA completa (`notebooks/01_eda.ipynb`)
- [x] Limpeza: outliers + valores ausentes (`notebooks/02_preprocessing.ipynb`)
- [x] Engenharia de features + encoding (dentro do pipeline em `src/features.py`)
- [x] Geração de `data/processed/credit_clean.csv`
- [x] Split treino/teste estratificado + pipeline scikit-learn
- [x] Treino de 3 modelos (Regressão Logística, Random Forest, HistGradientBoosting)
- [x] Tratamento do desbalanceamento (`class_weight='balanced'`)
- [x] Avaliação com ROC-AUC, precision, recall, F1 e matriz de confusão
- [x] Modelo salvo em `models/model.pkl`
- [x] Notebooks comentados com conclusões (`notebooks/03_modeling.ipynb`)

### Igor Zanette — Dashboard + Relatório
- [x] App Streamlit em `dashboard/app.py`
  - [x] Sidebar com filtros interativos (grade, finalidade, moradia, renda)
  - [x] Aba EDA: 4 KPIs + 6 gráficos interativos
  - [x] Aba Simulador de Risco: formulário + predição + fatores de alerta
  - [x] Aba Resultados do Modelo: comparação de modelos + matriz de confusão + importância das features
- [x] Instruções de execução no `README.md`
- [x] Seções Dataset e Análise Exploratória do relatório (`reports/relatorio.ipynb`)
- [x] `Makefile` com targets: `install`, `data`, `notebooks`, `dashboard`, `report-html`
- [x] `requirements.txt`

### Marcos Paulo — Relatório + Revisão
- [x] Seções Modelagem e Resultados do relatório
- [x] Introdução, Conclusão, Limitações e Trabalhos futuros
- [ ] Revisão final do texto e coesão
- [ ] Exportar PDF e submeter no Moodle
- [ ] Dar acesso ao repositório para o professor (`holbig@upf.br`)

---

## Como rodar o projeto

```bash
# 1. Instalar dependências
make install

# 2. Gerar dataset limpo (necessário ter credit_risk_dataset.csv em data/raw/)
make data

# 3. Executar os notebooks em ordem (opcional — outputs já estão salvos)
make notebooks

# 4. Iniciar o dashboard
make dashboard

# 5. Gerar o relatório em HTML → imprimir como PDF no browser
make report-html
```

---

## Resultados do modelo

| Modelo | ROC-AUC | Recall | F1 | Acurácia |
|---|---|---|---|---|
| **HistGradientBoosting** ⭐ | **0,95** | **0,81** | 0,81 | 0,92 |
| Random Forest | 0,94 | 0,72 | **0,83** | 0,92 |
| Regressão Logística | 0,88 | 0,80 | 0,66 | 0,84 |

**Modelo escolhido:** HistGradientBoosting — melhor ROC-AUC e Recall (identificar inadimplentes é mais crítico que recusar bons pagadores).

---

## Prazo

Entrega: **23/06/2026** — submissão do PDF no Moodle + apresentação oral (5 min).
