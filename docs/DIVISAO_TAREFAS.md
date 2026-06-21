# 📋 Divisão de Tarefas — Projeto Análise de Risco de Crédito

> Documento de alinhamento da equipe. Lê com calma antes de começar a tua parte. 🙂

## 👥 Equipe

| Pessoa | Bloco principal | Peso na nota |
|---|---|---|
| **Marco** | Bloco técnico: EDA + preparação dos dados + modelagem + avaliação | 50% (25% + 25%) |
| **Igor** | Dashboard interativo (Streamlit) + escrita do relatório + repo limpo | 20% |
| **Marcos** | Montagem do relatório PDF + slides + escrita do relatório + repo limpo | 15% + 15% (repo) |

---

## 🎯 O que é o projeto

Trabalho final de Data Science. Vamos prever **inadimplência de crédito** (a coluna
`loan_status`: `1` = cliente deu calote, `0` = pagou) a partir de dados de empréstimos.

**Dataset:** `data/raw/credit_risk_dataset.csv` — ~32.500 linhas, 12 colunas.
Vem do Kaggle (Credit Risk Dataset).

**Dicionário das variáveis:**

| Coluna | Significado |
|---|---|
| `person_age` | Idade do cliente |
| `person_income` | Renda anual |
| `person_home_ownership` | Moradia (RENT, OWN, MORTGAGE...) |
| `person_emp_length` | Tempo de emprego (anos) |
| `loan_intent` | Finalidade do empréstimo (PERSONAL, EDUCATION, MEDICAL...) |
| `loan_grade` | Nota/risco do empréstimo (A a G) |
| `loan_amnt` | Valor do empréstimo |
| `loan_int_rate` | Taxa de juros |
| `loan_status` | **ALVO** — 1 = inadimplente, 0 = adimplente |
| `loan_percent_income` | Empréstimo / renda |
| `cb_person_default_on_file` | Já teve calote no histórico? (Y/N) |
| `cb_person_cred_hist_length` | Tempo de histórico de crédito (anos) |

---

## 📦 Entregáveis (são 3, todos obrigatórios)

1. **Repositório no GitHub** — organizado, com README, código comentado e reprodutível.
   👉 Dar acesso ao professor: `holbig@upf.br`
2. **Relatório técnico em PDF** — feito em Jupyter/Quarto, exportado pra PDF.
3. **Dashboard interativo** — Streamlit, com filtros e exploração dos resultados.

---

## 🗂️ Estrutura do repositório (onde cada um trabalha)

```
credit-risk-analysis-ds/
├── data/
│   ├── raw/           # dados brutos (CSV já está aqui) — NÃO mexer
│   └── processed/     # Marco gera o CSV limpo aqui
├── notebooks/         # Marco: EDA e modelagem
├── src/               # Marco: scripts reutilizáveis (limpeza, treino)
├── models/            # modelo treinado salvo (.pkl)
├── reports/
│   └── figures/       # gráficos exportados
├── dashboard/         # Igor: app Streamlit
└── docs/              # este arquivo + enunciado
```

> ⚠️ Os dados (`data/raw/` e `data/processed/`) estão no `.gitignore` — não entram no Git.
> Cada um precisa ter o CSV localmente.

---

## 🔗 Contrato de dados (IMPORTANTE — desbloqueia o time)

O Marco vai gerar um dataset **já limpo e tratado** em:

```
data/processed/credit_clean.csv
```

É a partir desse arquivo que Igor (dashboard) e o relatório vão trabalhar — assim ninguém
fica refazendo limpeza. Marco avisa no grupo quando esse arquivo estiver pronto. ✅

---

## 🙋 Responsabilidades detalhadas

### 👤 Marco — EDA + Preparação + Modelagem + Avaliação
- Análise exploratória: estatísticas, distribuição do alvo (~22% são inadimplentes → **base desbalanceada**), correlações, gráficos por `loan_grade`, `loan_intent`, taxa de juros.
- Limpeza: tratar **outliers** (tem idade tipo 144 anos e tempo de emprego de 123 anos 👀) e **valores ausentes** (`loan_int_rate` e `person_emp_length` têm faltantes).
- Engenharia de features: encoding das categóricas + features novas se fizer sentido.
- Gerar o `data/processed/credit_clean.csv` (o contrato acima).
- Modelagem: split treino/teste estratificado + pipeline.
- Treinar **≥3 modelos** (ex.: Regressão Logística, Random Forest, Gradient Boosting) e tratar o desbalanceamento (class_weight ou SMOTE).
- Avaliar com métricas certas pra crédito: **ROC-AUC, precision, recall, F1, matriz de confusão** (acurácia sozinha engana com base desbalanceada).
- Salvar o modelo em `models/` pro Igor usar no dashboard.
- **Deixar os notebooks bem comentados** (resultados, gráficos e conclusões) — é a partir deles que Igor e Marcos vão escrever as seções técnicas do relatório.

### 👤 Igor — Dashboard + escrita do relatório
- App em `dashboard/` usando o `credit_clean.csv` e o modelo salvo em `models/`.
- Visualizações da EDA com **filtros** (por grade, finalidade, faixa de renda...).
- **Simulador de risco**: usuário insere os dados de um cliente e o app prevê a chance de inadimplência usando o modelo do Marco.
- Deixar instruções de como rodar o dashboard no `README.md`.
- Escrever as seções **Dataset e Análise Exploratória** do relatório (a partir dos notebooks do Marco).
- Ajudar a manter o repositório limpo e organizado.

### 👤 Marcos — Relatório PDF + escrita + Apresentação
- Escrever as seções **Modelagem e Resultados** do relatório (a partir dos notebooks do Marco).
- Montar o **relatório final em PDF** (Jupyter/Quarto): juntar todas as seções, mais **Introdução**, **Conclusão** (limitações + trabalhos futuros), título, autores e URL do repo.
- Cuidar da formatação, coesão e revisão geral do texto.
- Preparar os **slides da apresentação** (5 min): visão do problema → demo do dashboard → principais resultados e dificuldades.
- Submeter o PDF no Moodle.
- Ajudar a manter o repositório limpo e organizado.

> 💡 A escrita do relatório e a organização do repo ficam com **Igor e Marcos**. O Marco entra
> só dando suporte técnico/tirando dúvidas sobre o que foi feito nos notebooks.

---

## ✅ Checklist de tarefas

### Setup (todos)
- [ ] Clonar o repo e colocar o CSV em `data/raw/`
- [ ] Criar/atualizar o `requirements.txt` com as bibliotecas usadas (cada um adiciona as suas)
- [ ] Dar acesso do repo ao professor (`holbig@upf.br`)

### Marco — Técnico
- [ ] EDA completa (notebook `01_eda`)
- [ ] Limpeza: outliers + valores ausentes
- [ ] Engenharia de features + encoding
- [ ] Gerar `data/processed/credit_clean.csv` ✅ (avisar o time)
- [ ] Split treino/teste + pipeline
- [ ] Treinar ≥3 modelos + tratar desbalanceamento
- [ ] Avaliação (ROC-AUC, precision, recall, F1, matriz de confusão)
- [ ] Salvar modelo em `models/`
- [ ] Deixar os notebooks comentados (base pro relatório do Igor e Marcos)

### Igor — Dashboard + relatório
- [ ] Estrutura inicial do app Streamlit em `dashboard/`
- [ ] Visualizações da EDA com filtros
- [ ] Simulador de risco (usando o modelo salvo)
- [ ] Instruções de execução no README
- [ ] Escrever seções Dataset e Análise Exploratória do relatório
- [ ] Ajudar a manter o repo limpo

### Marcos — Relatório + Apresentação
- [ ] Escrever seções Modelagem e Resultados do relatório
- [ ] Estrutura do relatório (Jupyter/Quarto) + Introdução
- [ ] Juntar todas as seções (com as do Igor)
- [ ] Conclusão (limitações + trabalhos futuros)
- [ ] Preencher o `README.md` (descrição + como rodar)
- [ ] Exportar pra PDF e revisar
- [ ] Slides da apresentação (5 min)
- [ ] Submeter PDF no Moodle
- [ ] Ajudar a manter o repo limpo

---

## ⏰ Prazo

O enunciado diz **16/06/2026** para entrega do relatório (PDF) e apresentação.
**Atenção:** essa data já passou — confirmar com o professor se houve prorrogação antes de seguir.

---

*Dúvidas? Manda no grupo. Bom trabalho, equipe! 🚀*
