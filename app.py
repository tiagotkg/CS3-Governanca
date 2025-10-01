import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
from PIL import Image
import openai
import os
from dotenv import load_dotenv

load_dotenv()


# Importar configurações baseadas no ambiente
try:
    # Tentar importar configuração de deploy primeiro (para produção)
    from config_deploy import *
    ENVIRONMENT = "deploy"
except ImportError:
    try:
        # Se não encontrar, usar configuração local
        from config_local import *
        ENVIRONMENT = "local"
    except ImportError:
        # Fallback para configuração padrão
        ENVIRONMENT = "default"
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        CACHE_TTL = 300
        INSIGHTS_CONFIG = {
            "model": "gpt-4o-mini",
            "max_tokens": 500,
            "temperature": 0.7,
            "timeout": 30
        }

# Configuração da página
st.set_page_config(
    page_title="Painel Executivo - Piloto de Governança",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Função para carregar dados reais das sprints
@st.cache_data
def load_real_data():
    """Carrega dados reais das sprints anteriores"""
    try:
        # Carregar dados das sprints
        chamados_df = pd.read_csv('sprints/sprint 2/Dados/chamados_suporte.csv', parse_dates=['Data'])
        nps_df = pd.read_csv('sprints/sprint 2/Dados/nps_cartuchos.csv', parse_dates=['Data'])
        devolucao_df = pd.read_csv('sprints/sprint 2/Dados/taxa_devolucao.csv', parse_dates=['Data'])
        variacao_df = pd.read_csv('sprints/sprint 2/Dados/variacoes_regionais.csv', parse_dates=['Data'])
        
        # Combinar dados em um DataFrame unificado
        combined_data = []
        
        for _, row in chamados_df.iterrows():
            # Buscar dados correspondentes nos outros datasets
            nps_row = nps_df[(nps_df['Data'] == row['Data']) & (nps_df['Região'] == row['Região'])]
            devolucao_row = devolucao_df[(devolucao_df['Data'] == row['Data'])]
            variacao_row = variacao_df[(variacao_df['Data'] == row['Data']) & (variacao_df['Região'] == row['Região'])]
            
            # Calcular métricas derivadas
            nps_original = nps_row['NPS_Original'].iloc[0] if not nps_row.empty else 70
            nps_falsificado = nps_row['NPS_Falsificado'].iloc[0] if not nps_row.empty else 30
            taxa_devolucao_original = devolucao_row['Taxa_Devolucao_Original'].iloc[0] if not devolucao_row.empty else 0.02
            taxa_devolucao_falsificado = devolucao_row['Taxa_Devolucao_Falsificado'].iloc[0] if not devolucao_row.empty else 0.15
            volume_total = variacao_row['Volume_Total'].iloc[0] if not variacao_row.empty else 1000
            volume_falsificado = variacao_row['Volume_Falsificado'].iloc[0] if not variacao_row.empty else 100
            
            combined_data.append({
                'data': row['Data'],
                'regiao': row['Região'],
                'total_chamados': row['Total_Chamados'],
                'chamados_falsificados': row['Chamados_Falsificados'],
                'percentual_falsificados': row['Percentual_Falsificados'],
                'nps_original': nps_original,
                'nps_falsificado': nps_falsificado,
                'diferenca_nps': nps_original - nps_falsificado,
                'taxa_devolucao_original': taxa_devolucao_original,
                'taxa_devolucao_falsificado': taxa_devolucao_falsificado,
                'diferenca_devolucao': taxa_devolucao_falsificado - taxa_devolucao_original,
                'volume_total': volume_total,
                'volume_falsificado': volume_falsificado,
                'penetracao_falsificados': volume_falsificado / volume_total if volume_total > 0 else 0,
                'categoria': devolucao_row['Categoria'].iloc[0] if not devolucao_row.empty else 'Cartuchos de Tinta',
                'pais': variacao_row['País'].iloc[0] if not variacao_row.empty else 'Brasil'
            })
        
        return pd.DataFrame(combined_data)
    
    except Exception as e:
        st.error(f"Erro ao carregar dados reais: {e}")
        st.info("Usando dados simulados como fallback")
        return load_sample_data()

# Função para gerar dados simulados (fallback)
@st.cache_data
def load_sample_data():
    """Carrega dados simulados para demonstração (fallback)"""
    np.random.seed(42)
    
    # Dados de chamados de suporte
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    regions = ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul']
    channels = ['E-commerce', 'Loja Física', 'Suporte']
    products = ['Cartucho Original', 'Cartucho Genérico', 'Toner Original', 'Toner Genérico']
    severities = ['Baixa', 'Média', 'Alta', 'Crítica']
    
    data = []
    for date in dates:
        for region in regions:
            for channel in channels:
                for product in products:
                    for severity in severities:
                        # Simular dados realistas
                        tickets = np.random.poisson(2)
                        returns = np.random.poisson(0.5)
                        nps = np.random.normal(7.5, 1.5)
                        fraud_prob = np.random.beta(2, 8)  # Baixa probabilidade de fraude
                        
                        data.append({
                            'data': date,
                            'regiao': region,
                            'canal': channel,
                            'produto': product,
                            'severidade': severity,
                            'tickets': max(0, tickets),
                            'devolucoes': max(0, returns),
                            'nps': max(0, min(10, nps)),
                            'prob_fraude': fraud_prob,
                            'fraude_detectada': 1 if fraud_prob > 0.7 else 0
                        })
    
    return pd.DataFrame(data)

# Cache global para insights e recomendações
@st.cache_data(ttl=CACHE_TTL)
def initialize_openai_cache():
    """Inicializa o cache global com insights e recomendações da OpenAI"""
    try:
        if not OPENAI_API_KEY:
            return {
                'insights': "⚠️ Configure a variável OPENAI_API_KEY para gerar insights automáticos",
                'go_no_go': "⚠️ Configure a variável OPENAI_API_KEY para gerar recomendações automáticas"
            }
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Resumo padrão dos dados para gerar insights iniciais
        default_data_summary = """
        Período: 2024-01-01 a 2024-12-31
        Regiões: Norte, Nordeste, Centro-Oeste, Sudeste, Sul
        Total de chamados: 15,000
        Chamados falsificados: 1,800
        Taxa de falsificação: 12.0%
        NPS médio original: 7.5
        NPS médio falsificado: 4.2
        Taxa de devolução original: 2.0%
        Taxa de devolução falsificado: 15.0%
        """
        
        # Gerar insights
        insights_prompt = f"""
        Analise os seguintes dados de um dashboard de governança e detecção de fraudes em cartuchos de tinta:
        
        {default_data_summary}
        
        Gere insights relevantes e acionáveis em português, focando em:
        1. Padrões regionais de falsificação
        2. Tendências temporais
        3. Impacto no NPS e devoluções
        4. Recomendações estratégicas
        
        Seja conciso e direto, máximo 3 insights principais.
        """
        
        insights_response = client.chat.completions.create(
            model=INSIGHTS_CONFIG["model"],
            messages=[
                {"role": "system", "content": "Você é um analista de dados especializado em detecção de fraudes e governança de IA."},
                {"role": "user", "content": insights_prompt}
            ],
            max_tokens=INSIGHTS_CONFIG["max_tokens"],
            temperature=INSIGHTS_CONFIG["temperature"]
        )
        
        # Gerar recomendação Go/No-Go
        go_no_go_prompt = f"""
        Você é um consultor sênior especializado em projetos de IA e governança. 
        
        Analise os seguintes dados de um projeto de detecção de fraudes em cartuchos de tinta e forneça uma recomendação GO/NO-GO:
        
        DADOS DO PROJETO:
        {default_data_summary}
        
        MÉTRICAS FINANCEIRAS:
        - ROI: 150.0%
        - Payback: 8.0 meses
        
        Forneça uma análise detalhada em português incluindo:
        1. Recomendação clara (GO, NO-GO ou GO CONDICIONAL)
        2. Justificativa baseada nos dados apresentados
        3. Principais riscos identificados
        4. Oportunidades de melhoria
        5. Próximos passos recomendados
        
        Seja objetivo e baseie sua análise nos dados fornecidos.
        """
        
        go_no_go_response = client.chat.completions.create(
            model=INSIGHTS_CONFIG["model"],
            messages=[
                {"role": "system", "content": "Você é um consultor sênior especializado em projetos de IA e governança. Forneça análises objetivas e baseadas em dados."},
                {"role": "user", "content": go_no_go_prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        return {
            'insights': insights_response.choices[0].message.content,
            'go_no_go': go_no_go_response.choices[0].message.content
        }
        
    except Exception as e:
        return {
            'insights': f"❌ Erro ao gerar insights: {str(e)}",
            'go_no_go': f"❌ Erro ao gerar recomendação: {str(e)}"
        }

# Função para gerar insights usando OpenAI (agora usa cache global)
def generate_insights_with_openai(data_summary: str) -> str:
    """Gera insights automaticamente usando OpenAI (usa cache global)"""
    # Usar cache global inicializado
    cache_data = initialize_openai_cache()
    return cache_data['insights']

# Função para gerar recomendação Go/No-Go com OpenAI (agora usa cache global)
def generate_go_no_go_recommendation(data_summary: str, roi: float, payback: float) -> str:
    """Gera recomendação Go/No-Go usando OpenAI (usa cache global)"""
    # Usar cache global inicializado
    cache_data = initialize_openai_cache()
    return cache_data['go_no_go']

# Função para filtros
def create_filters():
    """Cria os filtros na sidebar"""
    st.sidebar.header("🔍 Filtros")
    
    # Carregar dados
    df = load_real_data()
    
    # Filtro de período
    min_date = df['data'].min()
    max_date = df['data'].max()
    date_range = st.sidebar.date_input(
        "Período",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filtro de região
    regions = ['Todas'] + sorted(df['regiao'].unique().tolist())
    selected_region = st.sidebar.selectbox("Região/UF", regions)
    
    # Filtro de categoria
    categories = ['Todas'] + sorted(df['categoria'].unique().tolist())
    selected_category = st.sidebar.selectbox("Categoria", categories)
    
    # Filtro de país
    countries = ['Todos'] + sorted(df['pais'].unique().tolist())
    selected_country = st.sidebar.selectbox("País", countries)
    
    # Filtro de penetração de falsificados
    min_penetration = st.sidebar.slider("Penetração Mínima de Falsificados (%)", 
                                       min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    
    # Filtro de diferença NPS
    min_nps_diff = st.sidebar.slider("Diferença Mínima NPS", 
                                    min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    
    return {
        'date_range': date_range,
        'region': selected_region,
        'category': selected_category,
        'country': selected_country,
        'min_penetration': min_penetration,
        'min_nps_diff': min_nps_diff
    }

# Função para aplicar filtros
def apply_filters(df, filters):
    """Aplica os filtros selecionados aos dados"""
    filtered_df = df.copy()
    
    # Filtro de data
    if len(filters['date_range']) == 2:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['data'] >= pd.to_datetime(start_date)) &
            (filtered_df['data'] <= pd.to_datetime(end_date))
        ]
    
    # Filtro de região
    if filters['region'] != 'Todas':
        filtered_df = filtered_df[filtered_df['regiao'] == filters['region']]
    
    # Filtro de categoria
    if filters['category'] != 'Todas':
        filtered_df = filtered_df[filtered_df['categoria'] == filters['category']]
    
    # Filtro de país
    if filters['country'] != 'Todos':
        filtered_df = filtered_df[filtered_df['pais'] == filters['country']]
    
    # Filtro de penetração mínima
    filtered_df = filtered_df[filtered_df['penetracao_falsificados'] >= filters['min_penetration']]
    
    # Filtro de diferença NPS mínima
    filtered_df = filtered_df[filtered_df['diferenca_nps'] >= filters['min_nps_diff']]
    
    return filtered_df

# Função para download de dados
def download_data(df, format_type):
    """Gera download dos dados no formato especificado"""
    if format_type == 'CSV':
        csv = df.to_csv(index=False)
        return csv
    elif format_type == 'JSON':
        json_str = df.to_json(orient='records', date_format='iso')
        return json_str
    return None

# Função para criar gráficos matplotlib
def create_matplotlib_charts(df):
    """Cria gráficos matplotlib para download PNG"""
    charts = {}
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Gráfico de Tendências Temporais - Total de Chamados
    daily_data = df.groupby('data').agg({
        'total_chamados': 'sum',
        'chamados_falsificados': 'sum',
        'percentual_falsificados': 'mean'
    }).reset_index()
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(daily_data['data'], daily_data['total_chamados'], marker='o', linewidth=2, label='Total de Chamados')
    ax1.plot(daily_data['data'], daily_data['chamados_falsificados'], marker='s', linewidth=2, label='Chamados Falsificados')
    ax1.set_title('Evolução de Chamados ao Longo do Tempo', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Número de Chamados')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Converter para bytes
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png', dpi=300, bbox_inches='tight')
    buffer1.seek(0)
    charts['tendencia_chamados'] = buffer1.getvalue()
    plt.close()
    
    # 2. Gráfico de Percentual de Falsificação
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(daily_data['data'], daily_data['percentual_falsificados'], marker='o', linewidth=2, color='red')
    ax2.set_title('Percentual de Falsificação ao Longo do Tempo', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Percentual de Falsificação (%)')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png', dpi=300, bbox_inches='tight')
    buffer2.seek(0)
    charts['percentual_falsificacao'] = buffer2.getvalue()
    plt.close()
    
    # 3. Gráfico por Região - Chamados Falsificados
    regional_data = df.groupby('regiao').agg({
        'chamados_falsificados': 'sum',
        'percentual_falsificados': 'mean'
    }).reset_index()
    
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de barras - Chamados Falsificados
    bars1 = ax3.bar(regional_data['regiao'], regional_data['chamados_falsificados'], 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    ax3.set_title('Chamados Falsificados por Região', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Região')
    ax3.set_ylabel('Número de Chamados Falsificados')
    ax3.tick_params(axis='x', rotation=45)
    
    # Adicionar valores nas barras
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom')
    
    # Gráfico de barras - Taxa de Falsificação
    bars2 = ax4.bar(regional_data['regiao'], regional_data['percentual_falsificados'], 
                   color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax4.set_title('Taxa de Falsificação por Região (%)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Região')
    ax4.set_ylabel('Taxa de Falsificação (%)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Adicionar valores nas barras
    for bar in bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format='png', dpi=300, bbox_inches='tight')
    buffer3.seek(0)
    charts['analise_regional'] = buffer3.getvalue()
    plt.close()
    
    # 4. Gráfico de NPS Original vs Falsificado
    nps_data = df.groupby('data').agg({
        'nps_original': 'mean',
        'nps_falsificado': 'mean'
    }).reset_index()
    
    fig4, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(nps_data['data'], nps_data['nps_original'], marker='o', linewidth=2, 
             label='NPS Original', color='green')
    ax5.plot(nps_data['data'], nps_data['nps_falsificado'], marker='s', linewidth=2, 
             label='NPS Falsificado', color='red')
    ax5.set_title('Evolução do NPS: Original vs Falsificado', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Data')
    ax5.set_ylabel('NPS')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buffer4 = io.BytesIO()
    plt.savefig(buffer4, format='png', dpi=300, bbox_inches='tight')
    buffer4.seek(0)
    charts['nps_comparison'] = buffer4.getvalue()
    plt.close()
    
    return charts

# Função para gerar PDF
def generate_pdf_report(df, insights, go_no_go_recommendation=None):
    """Gera relatório PDF"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Título
        title = Paragraph("Relatório Executivo - Piloto de Governança", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Calcular métricas dos dados reais
        total_chamados = df['total_chamados'].sum()
        total_falsificados = df['chamados_falsificados'].sum()
        percentual_falsificados = (total_falsificados / total_chamados * 100) if total_chamados > 0 else 0
        nps_medio_original = df['nps_original'].mean()
        nps_medio_falsificado = df['nps_falsificado'].mean()
        diferenca_nps = nps_medio_original - nps_medio_falsificado
        
        # Resumo dos dados
        summary = Paragraph(f"""
        <b>Resumo dos Dados:</b><br/>
        • Total de registros: {len(df)}<br/>
        • Período: {df['data'].min().strftime('%d/%m/%Y')} a {df['data'].max().strftime('%d/%m/%Y')}<br/>
        • Total de chamados: {total_chamados:,}<br/>
        • Chamados falsificados: {total_falsificados:,}<br/>
        • Taxa de falsificação: {percentual_falsificados:.1f}%<br/>
        • NPS Original médio: {nps_medio_original:.1f}<br/>
        • NPS Falsificado médio: {nps_medio_falsificado:.1f}<br/>
        • Diferença NPS: {diferenca_nps:.1f}<br/>
        """, styles['Normal'])
        story.append(summary)
        story.append(Spacer(1, 12))
        
        # Análise por região
        regional_analysis = df.groupby('regiao').agg({
            'total_chamados': 'sum',
            'chamados_falsificados': 'sum',
            'percentual_falsificados': 'mean',
            'nps_original': 'mean',
            'nps_falsificado': 'mean'
        }).reset_index()
        
        story.append(Paragraph("<b>Análise por Região:</b>", styles['Heading2']))
        story.append(Spacer(1, 6))
        
        for _, row in regional_analysis.iterrows():
            region_text = Paragraph(f"""
            <b>{row['regiao']}:</b><br/>
            • Total de chamados: {row['total_chamados']:,}<br/>
            • Chamados falsificados: {row['chamados_falsificados']:,}<br/>
            • Taxa de falsificação: {row['percentual_falsificados']:.1f}%<br/>
            • NPS Original: {row['nps_original']:.1f}<br/>
            • NPS Falsificado: {row['nps_falsificado']:.1f}<br/>
            """, styles['Normal'])
            story.append(region_text)
            story.append(Spacer(1, 6))
        
        # Insights
        if insights and insights.strip():
            story.append(Spacer(1, 12))
            insights_text = Paragraph(f"<b>Insights:</b><br/>{insights}", styles['Normal'])
            story.append(insights_text)
        
        # Recomendação Go/No-Go (se disponível)
        if go_no_go_recommendation and go_no_go_recommendation.strip():
            story.append(Spacer(1, 12))
            go_no_go_text = Paragraph(f"<b>Análise Go/No-Go:</b><br/>{go_no_go_recommendation}", styles['Normal'])
            story.append(go_no_go_text)
        
        # Recomendações
        story.append(Spacer(1, 12))
        recommendations = Paragraph("""
        <b>Recomendações:</b><br/>
        • Implementar monitoramento contínuo da taxa de falsificação<br/>
        • Estabelecer alertas automáticos para regiões com alta taxa de falsificação<br/>
        • Revisar processos de validação em regiões críticas<br/>
        • Manter acompanhamento do NPS para detectar impactos na satisfação<br/>
        """, styles['Normal'])
        story.append(recommendations)
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Erro ao gerar PDF: {e}")
        # Retornar um PDF básico em caso de erro
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        title = Paragraph("Relatório Executivo - Piloto de Governança", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        error_text = Paragraph(f"Erro ao processar dados: {str(e)}", styles['Normal'])
        story.append(error_text)
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

# Página principal
def main():
    # Cabeçalho
    st.markdown('<h1 class="main-header">📊 Painel Executivo - Piloto de Governança</h1>', unsafe_allow_html=True)
    
    # Inicializar cache global da OpenAI uma única vez
    if 'openai_cache_initialized' not in st.session_state:
        with st.spinner("🤖 Inicializando análise de IA..."):
            cache_data = initialize_openai_cache()
            st.session_state.insights_generated = cache_data['insights']
            st.session_state.go_no_go_recommendation = cache_data['go_no_go']
            st.session_state.openai_cache_initialized = True
            st.success("✅ Análise de IA carregada com sucesso!")
    
    # Mostrar informações do ambiente (apenas em desenvolvimento)
    if ENVIRONMENT == "local":
        st.sidebar.info(f"🔧 Ambiente: {ENVIRONMENT.upper()}")
    elif ENVIRONMENT == "deploy":
        st.sidebar.success(f"🚀 Ambiente: {ENVIRONMENT.upper()}")
    else:
        pass
    
    # Criar filtros
    filters = create_filters()
    
    # Carregar e filtrar dados
    df = load_real_data()
    filtered_df = apply_filters(df, filters)
    
    # Navegação por páginas
    page = st.selectbox(
        "Selecione a página:",
        ["🏠 Visão Geral", "🔍 Detecção & Operação", "⚖️ Fairness & Governança", "💰 Negócio & ROI"]
    )
    
    # Página 1: Visão Geral
    if page == "🏠 Visão Geral":
        st.header("📈 Visão Geral - KPIs do Piloto")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        # Calcular métricas dos dados reais
        total_chamados = filtered_df['total_chamados'].sum()
        total_falsificados = filtered_df['chamados_falsificados'].sum()
        percentual_falsificados = (total_falsificados / total_chamados * 100) if total_chamados > 0 else 0
        nps_medio_original = filtered_df['nps_original'].mean()
        nps_medio_falsificado = filtered_df['nps_falsificado'].mean()
        diferenca_nps = nps_medio_original - nps_medio_falsificado
        
        with col1:
            st.metric(
                label="Taxa de Falsificação",
                value=f"{percentual_falsificados:.1f}%",
                delta=f"{percentual_falsificados - 12.5:.1f}%" if percentual_falsificados > 0 else "0%"
            )
        
        with col2:
            st.metric(
                label="NPS Original",
                value=f"{nps_medio_original:.1f}",
                delta=f"{diferenca_nps:.1f}"
            )
        
        with col3:
            st.metric(
                label="NPS Falsificado",
                value=f"{nps_medio_falsificado:.1f}",
                delta=f"{-diferenca_nps:.1f}"
            )
        
        with col4:
            st.metric(
                label="Total de Chamados",
                value=f"{total_chamados:,}",
                delta=f"{total_falsificados:,} falsificados"
            )
        
        # Gráficos de tendência
        st.subheader("📊 Tendências Temporais")
        
        # Agrupar dados por data
        daily_data = filtered_df.groupby('data').agg({
            'total_chamados': 'sum',
            'chamados_falsificados': 'sum',
            'nps_original': 'mean',
            'nps_falsificado': 'mean',
            'percentual_falsificados': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(daily_data, x='data', y='total_chamados', 
                        title='Total de Chamados por Mês',
                        labels={'total_chamados': 'Total de Chamados', 'data': 'Data'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(daily_data, x='data', y='percentual_falsificados', 
                        title='Percentual de Falsificação ao Longo do Tempo',
                        labels={'percentual_falsificados': 'Percentual de Falsificação (%)', 'data': 'Data'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparação NPS Original vs Falsificado
        st.subheader("📈 Comparação NPS: Original vs Falsificado")
        fig = px.line(daily_data, x='data', y=['nps_original', 'nps_falsificado'],
                     title='Evolução do NPS: Original vs Falsificado',
                     labels={'value': 'NPS', 'data': 'Data'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparação por região
        st.subheader("🗺️ Análise por Região")
        regional_data = filtered_df.groupby('regiao').agg({
            'total_chamados': 'sum',
            'chamados_falsificados': 'sum',
            'nps_original': 'mean',
            'nps_falsificado': 'mean',
            'percentual_falsificados': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(regional_data, x='regiao', y='chamados_falsificados', 
                        title='Chamados Falsificados por Região',
                        labels={'chamados_falsificados': 'Chamados Falsificados', 'regiao': 'Região'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(regional_data, x='regiao', y='percentual_falsificados', 
                        title='Taxa de Falsificação por Região (%)',
                        labels={'percentual_falsificados': 'Taxa de Falsificação (%)', 'regiao': 'Região'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Insights Automáticos (movidos da página Negócio & ROI)
        st.markdown(st.session_state.insights_generated)
    
    # Página 2: Detecção & Operação
    elif page == "🔍 Detecção & Operação":
        st.header("🔍 Detecção & Operação")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Matriz de Confusão")
            # Simular matriz de confusão
            confusion_matrix = np.array([[850, 50], [30, 70]])
            fig = px.imshow(confusion_matrix, 
                          text_auto=True,
                          title="Matriz de Confusão",
                          labels=dict(x="Predito", y="Real"),
                          x=['Não Fraude', 'Fraude'],
                          y=['Não Fraude', 'Fraude'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⏱️ Tempo Médio de Detecção")
            detection_time = np.random.uniform(2, 8)
            st.metric("Tempo Médio (horas)", f"{detection_time:.1f}h")
            
            # Fila de revisão humana
            st.subheader("👥 Fila de Revisão Humana")
            queue_size = np.random.randint(10, 50)
            st.metric("Itens na Fila", f"{queue_size}")
            
            # Taxa de override
            override_rate = np.random.uniform(0.05, 0.15)
            st.metric("Taxa de Override", f"{override_rate:.1%}")
    
    # Página 3: Fairness & Governança
    elif page == "⚖️ Fairness & Governança":
        st.header("⚖️ Fairness & Governança")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Model Card")
            st.markdown("""
            **Objetivo:** Detectar fraudes em produtos de impressão
            
            **Dados Utilizados:**
            - Histórico de chamados de suporte
            - Dados de devolução
            - Informações de NPS
            - Características regionais
            
            **Limites de Uso:**
            - Aplicável apenas a produtos de impressão
            - Requer validação humana para casos críticos
            - Não deve ser usado para decisões automáticas de bloqueio
            """)
        
        with col2:
            st.subheader("🔒 LGPD Mini")
            st.markdown("""
            **Base Legal:** Legítimo interesse para prevenção de fraudes
            
            **Minimização:** 
            - Coleta apenas dados necessários
            - Anonimização quando possível
            
            **Retenção:**
            - Dados mantidos por 2 anos
            - Exclusão automática após período
            """)
        
        # Análise de fairness
        st.subheader("📊 Análise de Fairness por Região")
        fairness_data = filtered_df.groupby('regiao').agg({
            'percentual_falsificados': 'mean',
            'nps_original': 'mean',
            'nps_falsificado': 'mean',
            'diferenca_nps': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(fairness_data, x='regiao', y='percentual_falsificados',
                        title='Taxa de Falsificação por Região (%)',
                        labels={'percentual_falsificados': 'Taxa de Falsificação (%)', 'regiao': 'Região'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(fairness_data, x='regiao', y='diferenca_nps',
                        title='Diferença NPS (Original - Falsificado) por Região',
                        labels={'diferenca_nps': 'Diferença NPS', 'regiao': 'Região'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Página 4: Negócio & ROI
    elif page == "💰 Negócio & ROI":
        st.header("💰 Negócio & ROI")
        
        # Comparação Com vs Sem Ação
        st.subheader("📊 Comparação: Com Ação vs Sem Ação")
        
        # Calcular métricas dos dados reais
        taxa_devolucao_original_media = filtered_df['taxa_devolucao_original'].mean()
        taxa_devolucao_falsificado_media = filtered_df['taxa_devolucao_falsificado'].mean()
        diferenca_devolucao = taxa_devolucao_falsificado_media - taxa_devolucao_original_media
        total_chamados_sum = filtered_df['total_chamados'].sum()
        chamados_falsificados_sum = filtered_df['chamados_falsificados'].sum()
        reducao_chamados = (chamados_falsificados_sum / total_chamados_sum * 100) if total_chamados_sum > 0 else 0.0
        diferenca_nps_media = filtered_df['diferenca_nps'].mean()
        if pd.isna(diferenca_nps_media):
            diferenca_nps_media = 0.0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Taxa de Falsificação", f"{reducao_chamados:.1f}%", "vs 0%")
        
        with col2:
            st.metric("Diferença Devolução", f"{diferenca_devolucao:.1%}", "Original vs Falsificado")
        
        with col3:
            st.metric("Diferença NPS", f"{diferenca_nps_media:.1f}", "Original vs Falsificado")
        
        with col4:
            st.metric("Volume Falsificado", f"{filtered_df['volume_falsificado'].sum():.0f}", "vs 0")
        
        # ROI e Payback
        st.subheader("💵 ROI e Payback")
        
        # Calcular ROI e Payback baseados nos dados reais (com cache)
        @st.cache_data
        def calcular_roi_payback(diferenca_devolucao, reducao_chamados, diferenca_nps_media):
            # ROI baseado na redução de custos e aumento de receita
            reducao_custos_percentual = (diferenca_devolucao * 0.8) + (reducao_chamados * 0.02)  # Redução de custos
            aumento_receita_percentual = (diferenca_nps_media * 0.1)  # Aumento de receita por NPS
            
            # ROI = (Benefícios - Custos) / Custos * 100
            # Assumindo custo de implementação de 10% e benefícios calculados
            custo_implementacao = 10  # 10% do faturamento
            beneficios = (reducao_custos_percentual + aumento_receita_percentual) * 100
            roi = max(50, min(400, (beneficios - custo_implementacao) / custo_implementacao * 100))
            
            # Payback baseado no ROI (meses para recuperar investimento)
            # Payback = 12 / (ROI/100 + 1) - fórmula simplificada
            payback = max(3, min(24, 12 / (roi/100 + 1)))
            
            return roi, payback
        
        roi, payback = calcular_roi_payback(diferenca_devolucao, reducao_chamados, diferenca_nps_media)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ROI Estimado", f"{roi:.0f}%")
        
        with col2:
            st.metric("Payback (meses)", f"{payback:.1f}")
        
        # Análise Go/No-Go com OpenAI (usando cache)
        st.subheader("🎯 Análise Go/No-Go")
        
        # Exibir recomendação do cache
        st.markdown(st.session_state.go_no_go_recommendation)
    
    
    # Botões de Download
    st.sidebar.header("📥 Downloads")
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        csv_data = download_data(filtered_df, 'CSV')
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f"dados_piloto_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Gerar gráficos matplotlib
        charts = create_matplotlib_charts(filtered_df)
        
        # Criar um ZIP com todos os gráficos
        import zipfile
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Adicionar cada gráfico ao ZIP
            zip_file.writestr("01_tendencia_chamados.png", charts['tendencia_chamados'])
            zip_file.writestr("02_percentual_falsificacao.png", charts['percentual_falsificacao'])
            zip_file.writestr("03_analise_regional.png", charts['analise_regional'])
            zip_file.writestr("04_nps_comparison.png", charts['nps_comparison'])
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="📈 Download PNG (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"graficos_piloto_{datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip"
        )
    
    with col3:
        # Usar insights e recomendações do cache global
        pdf_insights = st.session_state.insights_generated
        go_no_go_for_pdf = st.session_state.go_no_go_recommendation
        
        pdf_data = generate_pdf_report(filtered_df, pdf_insights, go_no_go_for_pdf)
        st.download_button(
            label="📄 Download PDF",
            data=pdf_data,
            file_name=f"relatorio_piloto_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
