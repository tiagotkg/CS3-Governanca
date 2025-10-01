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

# Simulação de dados: Variações regionais no consumo de itens não oficiais
# Período de 12 meses (2023)
meses = pd.date_range(start='2023-01-01', end='2023-12-31', freq='MS')
regioes = ['América do Norte', 'Europa', 'Ásia-Pacífico', 'América Latina', 'Oriente Médio e África']
paises = {
    'América do Norte': ['EUA', 'Canadá', 'México'],
    'Europa': ['Alemanha', 'Reino Unido', 'França', 'Itália', 'Espanha'],
    'Ásia-Pacífico': ['China', 'Índia', 'Japão', 'Austrália', 'Coreia do Sul'],
    'América Latina': ['Brasil', 'Argentina', 'Colômbia', 'Chile', 'Peru'],
    'Oriente Médio e África': ['Emirados Árabes', 'África do Sul', 'Arábia Saudita', 'Egito', 'Nigéria']
}

# Criar DataFrame para variações regionais
regional_df = pd.DataFrame()

# Gerar dados para cada país
for regiao, lista_paises in paises.items():
    # Definir penetração base de produtos falsificados para cada região
    if regiao == 'América do Norte':
        penetracao_base = 0.12  # 12% de penetração de produtos falsificados
    elif regiao == 'Europa':
        penetracao_base = 0.18
    elif regiao == 'Ásia-Pacífico':
        penetracao_base = 0.35
    elif regiao == 'América Latina':
        penetracao_base = 0.28
    else:  # Oriente Médio e África
        penetracao_base = 0.32
    
    # Gerar dados para cada país na região
    for pais in lista_paises:
        # Variação específica do país (alguns países têm mais problemas com falsificações)
        if pais in ['China', 'Índia', 'México', 'Brasil', 'Nigéria']:
            fator_pais = np.random.uniform(1.1, 1.3)  # Países com maior incidência
        elif pais in ['EUA', 'Canadá', 'Japão', 'Alemanha', 'Reino Unido']:
            fator_pais = np.random.uniform(0.7, 0.9)  # Países com menor incidência
        else:
            fator_pais = np.random.uniform(0.9, 1.1)  # Países com incidência média
        
        # Penetração ajustada para o país
        penetracao_pais = penetracao_base * fator_pais
        
        # Gerar dados para cada mês
        for i, mes in enumerate(meses):
            # Adicionar variação sazonal e tendência
            fator_sazonal = 1 + 0.1 * np.sin(np.pi * i / 6)  # Variação sazonal
            fator_tendencia = 1 + 0.005 * i  # Leve tendência de aumento ao longo do ano
            
            # Calcular penetração com variação
            penetracao = penetracao_pais * fator_sazonal * fator_tendencia * np.random.uniform(0.95, 1.05)
            
            # Estimar volume de vendas total (em milhares de unidades)
            if regiao == 'América do Norte':
                volume_base = np.random.uniform(800, 1200)
            elif regiao == 'Europa':
                volume_base = np.random.uniform(600, 900)
            elif regiao == 'Ásia-Pacífico':
                volume_base = np.random.uniform(1000, 1500)
            elif regiao == 'América Latina':
                volume_base = np.random.uniform(300, 500)
            else:  # Oriente Médio e África
                volume_base = np.random.uniform(200, 400)
            
            # Ajustar volume por país
            if pais in ['EUA', 'China', 'Índia', 'Brasil', 'Alemanha']:
                volume_base *= np.random.uniform(1.2, 1.5)  # Países maiores
            elif pais in ['Canadá', 'Austrália', 'Arábia Saudita', 'África do Sul']:
                volume_base *= np.random.uniform(0.8, 1.0)  # Países médios
            else:
                volume_base *= np.random.uniform(0.5, 0.8)  # Países menores
            
            # Calcular volume de produtos falsificados
            volume_falsificado = volume_base * penetracao
            
            # Adicionar ao DataFrame
            regional_df = pd.concat([regional_df, pd.DataFrame({
                'Data': [mes],
                'Região': [regiao],
                'País': [pais],
                'Penetração_Falsificados': [penetracao],
                'Volume_Total': [volume_base],
                'Volume_Falsificado': [volume_falsificado]
            })])

# Resetar índice
regional_df = regional_df.reset_index(drop=True)

# Salvar dados
regional_df.to_csv('variacoes_regionais.csv', index=False)

# Visualização 1: Mapa de calor da penetração de produtos falsificados por país
plt.figure(figsize=(16, 10))

# Calcular média de penetração por país
penetracao_media = regional_df.groupby('País')['Penetração_Falsificados'].mean().reset_index()
penetracao_media = penetracao_media.sort_values('Penetração_Falsificados', ascending=False)

# Criar mapa de calor
plt.figure(figsize=(16, 10))
heatmap_data = regional_df.pivot_table(
    values='Penetração_Falsificados', 
    index='País', 
    columns='Data',
    aggfunc='mean'
)

# Ordenar países por penetração média
heatmap_data = heatmap_data.reindex(penetracao_media['País'])

# Criar mapa de calor
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='.1%', linewidths=.5, cbar_kws={'label': 'Penetração de Produtos Falsificados'})
plt.title('Evolução da Penetração de Produtos Falsificados por País (2023)', fontsize=16)
plt.xlabel('Mês', fontsize=14)
plt.ylabel('País', fontsize=14)
plt.tight_layout()
plt.savefig('../visualizacoes/heatmap_penetracao_paises.png', dpi=300, bbox_inches='tight')

# Visualização 2: Top 10 países com maior volume de produtos falsificados
plt.figure(figsize=(14, 8))

# Calcular volume total de produtos falsificados por país
volume_por_pais = regional_df.groupby('País')['Volume_Falsificado'].sum().reset_index()
volume_por_pais = volume_por_pais.sort_values('Volume_Falsificado', ascending=False).head(10)

# Criar gráfico de barras
sns.barplot(x='Volume_Falsificado', y='País', data=volume_por_pais, palette='viridis')
plt.title('Top 10 Países com Maior Volume de Produtos Falsificados (2023)', fontsize=16)
plt.xlabel('Volume Total de Produtos Falsificados (Milhares de Unidades)', fontsize=14)
plt.ylabel('País', fontsize=14)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('../visualizacoes/top10_paises_volume.png', dpi=300, bbox_inches='tight')

# Visualização 3: Comparação de penetração média por região
plt.figure(figsize=(14, 8))

# Calcular penetração média por região
penetracao_por_regiao = regional_df.groupby('Região')['Penetração_Falsificados'].mean().reset_index()
penetracao_por_regiao = penetracao_por_regiao.sort_values('Penetração_Falsificados', ascending=False)

# Criar gráfico de barras
ax = sns.barplot(x='Região', y='Penetração_Falsificados', data=penetracao_por_regiao, palette='viridis')
plt.title('Penetração Média de Produtos Falsificados por Região (2023)', fontsize=16)
plt.xlabel('Região', fontsize=14)
plt.ylabel('Penetração Média (%)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adicionar rótulos nas barras
for i, p in enumerate(ax.patches):
    ax.annotate(f'{p.get_height():.1%}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'bottom', fontsize=12)

plt.tight_layout()
plt.savefig('../visualizacoes/penetracao_por_regiao.png', dpi=300, bbox_inches='tight')

print("Dados de variações regionais gerados e visualizações criadas com sucesso!")
