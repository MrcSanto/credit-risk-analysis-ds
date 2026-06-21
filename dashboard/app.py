"""Dashboard interativo (Streamlit) — Análise de Risco de Crédito.

Duas seções principais:
  - Análise Exploratória: KPIs, gráficos com filtros interativos na sidebar.
  - Simulador de Risco: formulário que usa o modelo treinado para estimar
    a probabilidade de inadimplência de um cliente.

Executar com:
    uv run streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import load_model

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Risco de Crédito",
    page_icon="💳",
    layout="wide",
)

sns.set_theme(style="whitegrid")


# ─── Carregamento de dados e modelo (cacheado) ───────────────────────────────
@st.cache_data
def _load_data() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data/processed/credit_clean.csv")


@st.cache_resource
def _load_pipeline():
    return load_model(PROJECT_ROOT / "models/model.pkl")


df_full = _load_data()
model = _load_pipeline()

# ─── Sidebar: filtros ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💳 Risco de Crédito")
    st.markdown("**Filtros (aba EDA)**")

    grades = st.multiselect(
        "Nota de risco (loan_grade)",
        options=sorted(df_full["loan_grade"].unique()),
        default=sorted(df_full["loan_grade"].unique()),
    )
    intents = st.multiselect(
        "Finalidade do empréstimo",
        options=sorted(df_full["loan_intent"].unique()),
        default=sorted(df_full["loan_intent"].unique()),
    )
    ownership_opts = sorted(df_full["person_home_ownership"].unique())
    ownership = st.multiselect(
        "Tipo de moradia",
        options=ownership_opts,
        default=ownership_opts,
    )
    income_p99 = int(df_full["person_income"].quantile(0.99))
    income_range = st.slider(
        "Renda anual (US$)",
        min_value=int(df_full["person_income"].min()),
        max_value=income_p99,
        value=(int(df_full["person_income"].min()), income_p99),
        step=1_000,
    )

# Aplica filtros
df = df_full[
    df_full["loan_grade"].isin(grades)
    & df_full["loan_intent"].isin(intents)
    & df_full["person_home_ownership"].isin(ownership)
    & df_full["person_income"].between(*income_range)
].copy()

df["status_label"] = df["loan_status"].map({0: "Adimplente", 1: "Inadimplente"})

PALETTE = {"Adimplente": "#55a868", "Inadimplente": "#c44e52"}

# Estatísticas do dataset completo (usadas no simulador para comparação)
STATS = df_full.agg({
    "loan_int_rate": "mean",
    "loan_percent_income": "mean",
    "person_income": "mean",
    "loan_amnt": "mean",
    "person_emp_length": "mean",
}).to_dict()

# Métricas dos modelos (obtidas no notebook 03_modeling)
MODEL_METRICS = pd.DataFrame({
    "Modelo": ["HistGradientBoosting ⭐", "Random Forest", "Regressão Logística"],
    "ROC-AUC": [0.95, 0.94, 0.88],
    "Recall": [0.81, 0.72, 0.80],
    "F1": [0.81, 0.83, 0.66],
    "Acurácia": [0.92, 0.92, 0.84],
}).set_index("Modelo")

# Matriz de confusão do melhor modelo no conjunto de teste (6 482 amostras)
CONF_MATRIX = pd.DataFrame(
    [[4811, 253], [269, 1149]],
    index=["Real: Adimplente", "Real: Inadimplente"],
    columns=["Prev: Adimplente", "Prev: Inadimplente"],
)

# ─── Tabs principais ─────────────────────────────────────────────────────────
tab_eda, tab_sim, tab_model = st.tabs([
    "📊 Análise Exploratória", "🔮 Simulador de Risco", "🏆 Resultados do Modelo"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Análise Exploratória
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("Análise Exploratória — Risco de Crédito")

    if df.empty:
        st.warning("Nenhum registro com os filtros selecionados. Ajuste os filtros na sidebar.")
        st.stop()

    # ── KPIs ─────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clientes analisados", f"{len(df):,}")
    k2.metric("Taxa de inadimplência", f"{df['loan_status'].mean():.1%}")
    k3.metric("Valor médio do empréstimo", f"US$ {df['loan_amnt'].mean():,.0f}")
    k4.metric("Taxa de juros média", f"{df['loan_int_rate'].mean():.2f}%")

    st.divider()

    # ── Linha 1: inadimplência por grade e por finalidade ────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Inadimplência por nota de risco")
        taxa_grade = (
            df.groupby("loan_grade")["loan_status"].mean().sort_index() * 100
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(taxa_grade.index, taxa_grade.values, color="#c44e52", edgecolor="white")
        media = df["loan_status"].mean() * 100
        ax.axhline(media, color="gray", linestyle="--", linewidth=1.2, label=f"Média ({media:.1f}%)")
        ax.set_ylabel("Taxa de inadimplência (%)")
        ax.set_xlabel("loan_grade")
        ax.set_title("Inadimplência por nota de risco (A → maior risco → G)")
        ax.legend()
        for bar, val in zip(bars, taxa_grade.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9,
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Inadimplência por finalidade")
        taxa_intent = (
            df.groupby("loan_intent")["loan_status"].mean().sort_values() * 100
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(taxa_intent.index, taxa_intent.values, color="#4c72b0", edgecolor="white")
        media = df["loan_status"].mean() * 100
        ax.axvline(media, color="gray", linestyle="--", linewidth=1.2, label=f"Média ({media:.1f}%)")
        ax.set_xlabel("Taxa de inadimplência (%)")
        ax.set_title("Inadimplência por finalidade do empréstimo")
        ax.legend()
        for bar, val in zip(bars, taxa_intent.values):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Linha 2: alvo e taxa de juros ────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Proporção adimplente / inadimplente")
        contagem = df["loan_status"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            contagem.values,
            labels=["Adimplente (0)", "Inadimplente (1)"],
            colors=["#55a868", "#c44e52"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Distribuição da variável-alvo")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_d:
        st.subheader("Taxa de juros por status")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            x="status_label", y="loan_int_rate", data=df, ax=ax,
            palette=PALETTE, order=["Adimplente", "Inadimplente"],
        )
        ax.set_title("Distribuição da taxa de juros por status")
        ax.set_xlabel("")
        ax.set_ylabel("Taxa de juros (%)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Linha 3: renda e comprometimento de renda ────────────────────────────
    col_e, col_f = st.columns(2)

    with col_e:
        st.subheader("Renda anual por status")
        p99 = df["person_income"].quantile(0.99)
        df_trunc = df[df["person_income"] <= p99]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            x="status_label", y="person_income", data=df_trunc, ax=ax,
            palette=PALETTE, order=["Adimplente", "Inadimplente"],
        )
        ax.set_title("Renda anual por status (até p99)")
        ax.set_xlabel("")
        ax.set_ylabel("Renda anual (US$)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_f:
        st.subheader("Comprometimento de renda (empréstimo / renda)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            data=df, x="loan_percent_income", hue="status_label",
            bins=30, ax=ax, palette=PALETTE,
            stat="density", common_norm=False, alpha=0.6,
        )
        ax.set_title("Distribuição do comprometimento de renda")
        ax.set_xlabel("loan_percent_income")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tabela de dados ──────────────────────────────────────────────────────
    with st.expander("Ver dados filtrados"):
        st.dataframe(df.drop(columns=["status_label"]), use_container_width=True)
        st.caption(f"{len(df):,} registros exibidos após aplicação dos filtros.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Simulador de Risco
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.header("Simulador de Risco de Crédito")
    st.markdown(
        "Preencha as informações do cliente e do empréstimo para estimar a "
        "**probabilidade de inadimplência** usando o modelo **HistGradientBoosting** "
        "(ROC-AUC 0,95 | Recall 0,81)."
    )

    with st.form("simulador"):
        st.subheader("Dados do cliente")
        c1, c2, c3 = st.columns(3)
        person_age = c1.number_input("Idade", min_value=18, max_value=100, value=30)
        person_income = c2.number_input(
            "Renda anual (US$)", min_value=1_000, max_value=2_000_000, value=50_000, step=1_000
        )
        person_emp_length = c3.number_input(
            "Tempo de emprego (anos)", min_value=0.0, max_value=41.0, value=3.0, step=0.5
        )
        person_home_ownership = c1.selectbox("Tipo de moradia", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        cb_person_default_on_file = c2.selectbox("Histórico de calote?", ["N", "Y"])
        cb_person_cred_hist_length = c3.number_input(
            "Histórico de crédito (anos)", min_value=2, max_value=30, value=4
        )

        st.subheader("Dados do empréstimo")
        d1, d2, d3 = st.columns(3)
        loan_amnt = d1.number_input(
            "Valor do empréstimo (US$)", min_value=500, max_value=35_000, value=10_000, step=500
        )
        loan_int_rate = d2.number_input(
            "Taxa de juros (%)", min_value=5.42, max_value=23.22, value=11.0, step=0.1, format="%.2f"
        )
        loan_intent = d3.selectbox(
            "Finalidade",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )
        loan_grade = d1.selectbox("Nota de risco", ["A", "B", "C", "D", "E", "F", "G"])
        loan_percent_income = loan_amnt / person_income if person_income > 0 else 0.0
        d2.metric("Comprometimento de renda", f"{loan_percent_income:.1%}")
        d3.markdown("")

        submitted = st.form_submit_button("Calcular Risco", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }])

        prob = float(model.predict_proba(input_df)[0, 1])

        if prob < 0.30:
            color, label, icon = "#55a868", "BAIXO RISCO", "✅"
        elif prob < 0.60:
            color, label, icon = "#f0ad4e", "RISCO MODERADO", "⚠️"
        else:
            color, label, icon = "#c44e52", "ALTO RISCO", "🚨"

        st.divider()
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown(
                f"""
                <div style="background:{color}22; border-left:6px solid {color};
                            padding:24px; border-radius:8px; text-align:center;">
                    <p style="font-size:3rem; margin:0;">{icon}</p>
                    <p style="font-size:1.8rem; font-weight:bold; color:{color}; margin:6px 0;">
                        {label}
                    </p>
                    <p style="font-size:1.2rem; margin:0;">
                        Probabilidade de inadimplência: <strong>{prob:.1%}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("")

            # Barra de progresso visual
            fig, ax = plt.subplots(figsize=(7, 1.4))
            ax.barh(0, 1.0, color="#e8e8e8", height=0.5)
            ax.barh(0, prob, color=color, height=0.5)
            ax.axvline(0.30, color="#f0ad4e", linestyle="--", linewidth=1.5, label="30% (risco moderado)")
            ax.axvline(0.60, color="#c44e52", linestyle="--", linewidth=1.5, label="60% (alto risco)")
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.set_xticklabels([f"{x:.0%}" for x in ax.get_xticks()], fontsize=8)
            ax.set_yticks([])
            ax.set_xlabel("Probabilidade de inadimplência", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"Score: {prob:.1%}", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("---")
            st.markdown("**Limiares de decisão:**")
            st.markdown("- 🟢 &lt; 30% → Baixo risco — perfil favorável para concessão")
            st.markdown("- 🟡 30–60% → Risco moderado — avaliar com cautela")
            st.markdown("- 🔴 &gt; 60% → Alto risco — perfil desfavorável")

        # ── Fatores de alerta ──────────────────────────────────────────────
        st.divider()
        st.subheader("Fatores de risco identificados")
        st.caption("Comparação com a média dos 32.409 clientes do dataset.")

        GRADE_RISK = {"A": 0.055, "B": 0.119, "C": 0.173, "D": 0.318,
                      "E": 0.462, "F": 0.529, "G": 0.579}
        grade_default_rate = GRADE_RISK.get(loan_grade, 0)

        alertas = []
        positivos = []

        if loan_grade in ("E", "F", "G"):
            alertas.append(f"**Nota de risco {loan_grade}** — taxa histórica de inadimplência dessa nota: {grade_default_rate:.0%}")
        else:
            positivos.append(f"**Nota de risco {loan_grade}** — nota baixa, bom sinal ({grade_default_rate:.0%} de inadimplência histórica)")

        if loan_int_rate > STATS["loan_int_rate"] * 1.3:
            alertas.append(f"**Taxa de juros alta** — {loan_int_rate:.1f}% vs média do dataset ({STATS['loan_int_rate']:.1f}%)")
        elif loan_int_rate < STATS["loan_int_rate"] * 0.8:
            positivos.append(f"**Taxa de juros baixa** — {loan_int_rate:.1f}% abaixo da média ({STATS['loan_int_rate']:.1f}%)")

        if loan_percent_income > STATS["loan_percent_income"] * 1.5:
            alertas.append(f"**Comprometimento de renda alto** — {loan_percent_income:.0%} vs média ({STATS['loan_percent_income']:.0%})")
        else:
            positivos.append(f"**Comprometimento de renda ok** — {loan_percent_income:.0%} (média: {STATS['loan_percent_income']:.0%})")

        if cb_person_default_on_file == "Y":
            alertas.append("**Histórico de calote** — cliente já teve inadimplência registrada")
        else:
            positivos.append("**Sem histórico de calote** — bom sinal")

        if person_income < STATS["person_income"] * 0.6:
            alertas.append(f"**Renda baixa** — US$ {person_income:,.0f} vs média (US$ {STATS['person_income']:,.0f})")
        elif person_income > STATS["person_income"] * 1.4:
            positivos.append(f"**Renda alta** — US$ {person_income:,.0f} acima da média (US$ {STATS['person_income']:,.0f})")

        fa, fb = st.columns(2)
        with fa:
            st.markdown("**Pontos de atenção**")
            if alertas:
                for a in alertas:
                    st.error(f"🔴 {a}")
            else:
                st.success("Nenhum fator de alerta identificado.")
        with fb:
            st.markdown("**Pontos favoráveis**")
            if positivos:
                for p in positivos:
                    st.success(f"🟢 {p}")
            else:
                st.warning("Nenhum fator positivo de destaque.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Resultados do Modelo
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.header("Resultados do Modelo — Comparação e Avaliação")
    st.markdown(
        "Treinamos e comparamos **3 modelos** de classificação binária para prever "
        "inadimplência. Todos usam `class_weight='balanced'` para compensar o "
        "desbalanceamento (~22% de inadimplentes). Avaliados em 20% dos dados (6.482 amostras)."
    )

    st.divider()

    # ── Tabela de métricas ───────────────────────────────────────────────────
    st.subheader("Comparação de métricas")
    display_df = MODEL_METRICS.copy()
    for col in display_df.columns:
        best_idx = display_df[col].idxmax()
        display_df[col] = display_df[col].map(lambda v: f"{v:.2f}")
        display_df.loc[best_idx, col] = display_df.loc[best_idx, col] + " 🏆"
    st.dataframe(display_df, use_container_width=True)
    st.caption(
        "ROC-AUC: capacidade de separar as classes | "
        "Recall: % de inadimplentes corretamente identificados | "
        "F1: equilíbrio precision/recall | Acurácia: % geral de acertos"
    )

    st.divider()

    # ── Gráfico de métricas + Matriz de confusão ─────────────────────────────
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("ROC-AUC por modelo")
        fig, ax = plt.subplots(figsize=(6, 4))
        cores = ["#c44e52" if "⭐" in m else "#aec6cf" for m in MODEL_METRICS.index]
        bars = ax.barh(MODEL_METRICS.index, MODEL_METRICS["ROC-AUC"], color=cores, edgecolor="white")
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("ROC-AUC")
        ax.set_title("ROC-AUC por modelo (maior = melhor)")
        for bar, val in zip(bars, MODEL_METRICS["ROC-AUC"]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_m2:
        st.subheader("Matriz de confusão — HistGradientBoosting")
        fig, ax = plt.subplots(figsize=(6, 4))
        import numpy as np
        data = CONF_MATRIX.values
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Prev: Adimplente", "Prev: Inadimplente"])
        ax.set_yticklabels(["Real: Adimplente", "Real: Inadimplente"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{data[i, j]:,}", ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if data[i, j] > data.max() / 2 else "black")
        ax.set_title("Matriz de confusão (conjunto de teste)")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ── Gráfico multi-métrica ────────────────────────────────────────────────
    st.subheader("Todas as métricas por modelo")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(MODEL_METRICS))
    width = 0.2
    metricas = ["ROC-AUC", "Recall", "F1", "Acurácia"]
    cores_bar = ["#4c72b0", "#c44e52", "#55a868", "#dd8452"]
    for i, (metrica, cor) in enumerate(zip(metricas, cores_bar)):
        offset = (i - 1.5) * width
        bars = ax.bar([xi + offset for xi in x], MODEL_METRICS[metrica], width, label=metrica, color=cor, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODEL_METRICS.index, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparação completa das métricas por modelo")
    ax.legend(loc="lower right")
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Importância das features ─────────────────────────────────────────────
    st.divider()
    st.subheader("Importância das features (Permutation Importance — ROC-AUC)")
    st.caption("Mede quanto o ROC-AUC cai ao embaralhar cada variável. Obtido no notebook 03_modeling.")

    feature_imp = pd.Series({
        "loan_grade": 0.185,
        "loan_int_rate": 0.142,
        "loan_percent_income": 0.098,
        "person_income": 0.071,
        "cb_person_default_on_file": 0.058,
        "loan_amnt": 0.034,
        "loan_to_income_ratio": 0.029,
        "person_emp_length": 0.018,
        "person_age": 0.011,
        "cb_person_cred_hist_length": 0.007,
        "int_rate_to_loan_amt_ratio": 0.005,
        "loan_to_emp_length_ratio": 0.003,
        "loan_intent": 0.002,
        "person_home_ownership": 0.001,
    }).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c44e52" if v >= 0.05 else "#4c72b0" for v in feature_imp.values]
    feature_imp.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Queda média no ROC-AUC ao embaralhar a feature")
    ax.set_title("Importância das features — HistGradientBoosting")
    ax.axvline(0, color="black", linewidth=0.5)
    for bar, val in zip(ax.patches, feature_imp.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(
        "As 3 features mais importantes são **loan_grade**, **loan_int_rate** e "
        "**loan_percent_income** — coerente com as hipóteses levantadas na EDA: "
        "quanto pior a nota, maior a taxa e maior o comprometimento de renda, "
        "maior o risco de inadimplência."
    )
