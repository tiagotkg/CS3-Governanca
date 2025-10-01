import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(
    page_title="Combate a Produtos Falsificados – Dashboard IA",
    layout="wide",
)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_path: str) -> pd.DataFrame:
    """Read a CSV from the app directory or an uploaded file."""
    return pd.read_csv(file_path, parse_dates=["Data"])

def display_insight(text: str, icon: str = "💡"):
    st.markdown(f"{icon} **{text}**")

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
DATA_DIR = Path(__file__).parent

chamados_df  = load_csv("Dados/chamados_suporte.csv")
nps_df       = load_csv("Dados/nps_cartuchos.csv")
devolucao_df = load_csv("Dados/taxa_devolucao.csv")
variacao_df  = load_csv("Dados/variacoes_regionais.csv")

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
with st.sidebar:
    st.header("Filtros")
    regions = sorted(chamados_df['Região'].unique())
    selected_regions = st.multiselect("Região", regions, default=regions)

    date_min = chamados_df['Data'].min()
    date_max = chamados_df['Data'].max()
    date_range = st.date_input(
        "Período",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

# Apply filters
def filter_df(df: pd.DataFrame, require_region: bool = True) -> pd.DataFrame:
    mask_date = (df['Data'] >= pd.to_datetime(date_range[0])) & (df['Data'] <= pd.to_datetime(date_range[1]))
    if require_region and 'Região' in df.columns:
        mask_region = df['Região'].isin(selected_regions)
        return df.loc[mask_region & mask_date].copy()
    return df.loc[mask_date].copy()

chamados_df  = filter_df(chamados_df)
nps_df       = filter_df(nps_df)
devolucao_df = filter_df(devolucao_df, require_region=False)
variacao_df  = filter_df(variacao_df)

# ------------------------------------------------------------
# Layout – Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Chamados por Cartucho & Região",
        "Comparativo NPS",
        "Mapa de Calor – Falsificação",
        "Tendências Devoluções vs Suporte",
    ]
)

# ------------------------------------------------------------
# Tab 1 – Chamados por Cartucho & Região
# ------------------------------------------------------------
with tab1:
    st.subheader("Volume de Chamados por Região")

    calls_chart = (
        alt.Chart(chamados_df)
        .mark_bar(size=15)
        .encode(
            x=alt.X("yearmonth(Data):T", title="Mês"),
            y=alt.Y("sum(Total_Chamados):Q", title="Total de Chamados"),
            color=alt.Color("Região", legend=None),
            tooltip=[
                alt.Tooltip("Região"),
                alt.Tooltip("sum(Total_Chamados)", title="Chamados Totais", format=",.0f"),
            ],
            column="Região:N",
        )
        .properties(height=200)
    )
    st.altair_chart(calls_chart, use_container_width=True)

    fals_chart = (
        alt.Chart(chamados_df)
        .mark_bar(size=15, color="#d62728")
        .encode(
            x=alt.X("yearmonth(Data):T", title="Mês"),
            y=alt.Y("sum(Chamados_Falsificados):Q", title="Chamados por Falsificação"),
            tooltip=[
                alt.Tooltip("sum(Chamados_Falsificados)", title="Chamados Falsificados", format=",.0f")
            ],
            column="Região:N",
        )
        .properties(height=200)
    )
    st.altair_chart(fals_chart, use_container_width=True)

    # Simple insight example
    region_ratio = (
        chamados_df.groupby("Região")["Chamados_Falsificados"]
        .sum()
        .sort_values(ascending=False)
    )
    if len(region_ratio) >= 2:
        top_region = region_ratio.index[0]
        second_region = region_ratio.index[1]
        ratio = region_ratio.iloc[0] / region_ratio.iloc[1]
        display_insight(
            f"A região **{top_region}** apresenta **{ratio:.1f}×** mais chamados por cartuchos falsos do que **{second_region}**."
        )

# ------------------------------------------------------------
# Tab 2 – Comparativo NPS
# ------------------------------------------------------------
with tab2:
    st.subheader("NPS – Original vs Genérico")

    nps_melt = nps_df.melt(
        id_vars=["Data", "Região"],
        value_vars=["NPS_Original", "NPS_Falsificado"],
        var_name="Tipo",
        value_name="NPS",
    )
    nps_chart = (
        alt.Chart(nps_melt)
        .mark_line(point=True)
        .encode(
            x=alt.X("yearmonth(Data):T", title="Mês"),
            y=alt.Y("NPS:Q"),
            color=alt.Color("Tipo:N", title="Tipo de Cartucho"),
            strokeDash="Tipo:N"
        )
        .properties(height=400)
    )
    st.altair_chart(nps_chart, use_container_width=True)

    # Insight
    nps_avg = nps_df[["NPS_Original", "NPS_Falsificado"]].mean()
    display_insight(
        f"NPS médio de **cartucho original** = **{nps_avg['NPS_Original']:.0f}** | "
        f"**Genérico** = **{nps_avg['NPS_Falsificado']:.0f}**"
    )

# ------------------------------------------------------------
# Tab 3 – Mapa de Calor
# ------------------------------------------------------------
with tab3:
    st.subheader("Incidência de Falsificação – Heatmap")

    heat_df = (
        variacao_df.groupby(["Região", pd.Grouper(key="Data", freq="M")])["Penetração_Falsificados"]
        .mean()
        .reset_index()
    )
    heat_df["Mês"] = heat_df["Data"].dt.strftime("%Y-%m")

    heat_chart = (
        alt.Chart(heat_df)
        .mark_rect()
        .encode(
            x=alt.X("Mês:N", title="Mês"),
            y=alt.Y("Região:N", title="Região"),
            color=alt.Color(
                "Penetração_Falsificados:Q",
                scale=alt.Scale(scheme="reds"),
                title="Penetração (%)",
            ),
            tooltip=[
                alt.Tooltip("Região"),
                alt.Tooltip("Mês"),
                alt.Tooltip("Penetração_Falsificados", format=".2%"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(heat_chart, use_container_width=True)

# ------------------------------------------------------------
# Tab 4 – Tendências Devoluções vs Suporte
# ------------------------------------------------------------
with tab4:
    st.subheader("Tendência de Devoluções & Chamados de Suporte")

    # Prepare Devoluções
    dev_melt = devolucao_df.melt(
        id_vars=["Data", "Categoria"],
        value_vars=["Taxa_Devolucao_Original", "Taxa_Devolucao_Falsificado"],
        var_name="Tipo",
        value_name="Taxa_Devolução",
    )
    dev_chart = (
        alt.Chart(dev_melt)
        .mark_line(point=True)
        .encode(
            x=alt.X("yearmonth(Data):T", title="Mês"),
            y=alt.Y("Taxa_Devolução:Q", title="Taxa de Devolução"),
            color=alt.Color("Tipo:N", title="Tipo de Cartucho"),
            strokeDash="Tipo:N"
        )
        .properties(height=300)
    )

    # Prepare Support Calls
    support_chart = (
        alt.Chart(chamados_df)
        .mark_line(point=True, color="#1f77b4")
        .encode(
            x=alt.X("yearmonth(Data):T", title="Mês"),
            y=alt.Y("sum(Total_Chamados):Q", title="Total de Chamados de Suporte"),
            tooltip=[
                alt.Tooltip("sum(Total_Chamados)", title="Chamados Totais", format=",.0f")
            ]
        )
        .properties(height=300)
    )

    st.altair_chart(dev_chart, use_container_width=True)
    st.altair_chart(support_chart, use_container_width=True)

    # Combined insight example
    if "Diferenca_Percentual" in devolucao_df.columns:
        avg_dev_diff = devolucao_df["Diferenca_Percentual"].mean()
        display_insight(
            f"A taxa média de devolução para cartuchos falsificados supera a dos originais em **{avg_dev_diff:.1%}**."
        )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.caption(
    "Protótipo de dashboard interativo para análise de suporte, NPS e falsificação de cartuchos. "
    "Sprint 2 – Business Analytics & Governança de IA."
)
