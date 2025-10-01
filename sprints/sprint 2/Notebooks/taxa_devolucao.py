import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Configurar o estilo dos gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Criar diretório para salvar as visualizações
os.makedirs('../visualizacoes', exist_ok=True)

# Definir semente para reprodutibilidade
np.random.seed(42)

# Simulação de dados: Taxa de devolução de produtos
# Período de 12 meses (2023)
meses = pd.date_range(start='2023-01-01', end='2023-12-31', freq='MS')
categorias = ['Cartuchos de Tinta', 'Toners', 'Impressoras', 'Notebooks', 'Desktops', 'Acessórios']

# Criar DataFrame para taxa de devolução
devolucao_df = pd.DataFrame()

# Gerar dados para cada categoria
for categoria in categorias:
    # Definir taxa base de devolução para cada categoria
    if categoria == 'Cartuchos de Tinta':
        taxa_base_original = 0.015  # 1.5% para produtos originais
        taxa_base_falsificado = 0.12  # 12% para produtos falsificados
    elif categoria == 'Toners':
        taxa_base_original = 0.018  # 1.8% para produtos originais
        taxa_base_falsificado = 0.15  # 15% para produtos falsificados
    elif categoria == 'Impressoras':
        taxa_base_original = 0.025  # 2.5% para produtos originais
        taxa_base_falsificado = 0.08  # 8% para produtos falsificados (menos comum)
    elif categoria == 'Notebooks':
        taxa_base_original = 0.022  # 2.2% para produtos originais
        taxa_base_falsificado = 0.09  # 9% para produtos falsificados (menos comum)
    elif categoria == 'Desktops':
        taxa_base_original = 0.020  # 2% para produtos originais
        taxa_base_falsificado = 0.07  # 7% para produtos falsificados (menos comum)
    else:  # Acessórios
        taxa_base_original = 0.012  # 1.2% para produtos originais
        taxa_base_falsificado = 0.11  # 11% para produtos falsificados
    
    # Gerar dados para cada mês
    for i, mes in enumerate(meses):
        # Adicionar variação sazonal e tendência
        fator_sazonal = 1 + 0.2 * np.sin(np.pi * i / 6)  # Variação sazonal
        fator_tendencia = 1 + 0.01 * i  # Leve tendência de aumento ao longo do ano
        
        # Calcular taxas com variação
        taxa_original = taxa_base_original * fator_sazonal * fator_tendencia * np.random.uniform(0.9, 1.1)
        taxa_falsificado = taxa_base_falsificado * fator_sazonal * fator_tendencia * np.random.uniform(0.9, 1.1)
        
        # Adicionar ao DataFrame
        devolucao_df = pd.concat([devolucao_df, pd.DataFrame({
            'Data': [mes],
            'Categoria': [categoria],
            'Taxa_Devolucao_Original': [taxa_original],
            'Taxa_Devolucao_Falsificado': [taxa_falsificado],
            'Diferenca_Percentual': [(taxa_falsificado - taxa_original) / taxa_original * 100]
        })])

# Resetar índice
devolucao_df = devolucao_df.reset_index(drop=True)

# Salvar dados
devolucao_df.to_csv('taxa_devolucao.csv', index=False)

# Visualização 1: Comparação de taxas de devolução (Original vs Falsificado) por categoria
plt.figure(figsize=(14, 8))
categorias_ordenadas = devolucao_df.groupby('Categoria')['Taxa_Devolucao_Falsificado'].mean().sort_values(ascending=False).index

# Calcular médias por categoria
media_por_categoria = devolucao_df.groupby('Categoria')[['Taxa_Devolucao_Original', 'Taxa_Devolucao_Falsificado']].mean().reset_index()
media_por_categoria = media_por_categoria.set_index('Categoria').loc[categorias_ordenadas].reset_index()

# Criar gráfico de barras agrupadas
x = np.arange(len(categorias_ordenadas))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, media_por_categoria['Taxa_Devolucao_Original'] * 100, width, label='Produtos Originais', color='#0096D6')
rects2 = ax.bar(x + width/2, media_por_categoria['Taxa_Devolucao_Falsificado'] * 100, width, label='Produtos Falsificados', color='#E61E50')

ax.set_title('Taxa Média de Devolução: Produtos Originais vs. Falsificados', fontsize=16)
ax.set_xlabel('Categoria de Produto', fontsize=14)
ax.set_ylabel('Taxa de Devolução (%)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categorias_ordenadas, rotation=45, ha='right')
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionar rótulos nas barras
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 pontos de deslocamento vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('../visualizacoes/comparacao_taxa_devolucao.png', dpi=300, bbox_inches='tight')

# Visualização 2: Evolução da diferença percentual ao longo do tempo para cartuchos e toners
plt.figure(figsize=(14, 8))

for categoria in ['Cartuchos de Tinta', 'Toners']:
    dados_categoria = devolucao_df[devolucao_df['Categoria'] == categoria]
    plt.plot(dados_categoria['Data'], dados_categoria['Diferenca_Percentual'], 
             marker='o', linewidth=2, label=categoria)

plt.title('Diferença Percentual na Taxa de Devolução (Falsificado vs Original)', fontsize=16)
plt.xlabel('Mês', fontsize=14)
plt.ylabel('Diferença Percentual (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Categoria', fontsize=12)
plt.tight_layout()
plt.savefig('../visualizacoes/diferenca_percentual_devolucao.png', dpi=300, bbox_inches='tight')

print("Dados de taxa de devolução gerados e visualizações criadas com sucesso!")
