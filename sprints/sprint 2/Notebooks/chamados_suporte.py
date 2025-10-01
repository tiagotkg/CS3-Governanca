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

# Simulação de dados: Volume de chamados com produtos falsificados
# Período de 12 meses (2023)
meses = pd.date_range(start='2023-01-01', end='2023-12-31', freq='MS')
regioes = ['América do Norte', 'Europa', 'Ásia-Pacífico', 'América Latina', 'Oriente Médio e África']

# Criar DataFrame para chamados de suporte
chamados_df = pd.DataFrame()

# Gerar dados para cada região
for regiao in regioes:
    # Definir tendência base para cada região (algumas regiões têm mais problemas com falsificações)
    if regiao == 'Ásia-Pacífico':
        base = 1200  # Maior incidência
        sazonalidade = 200
    elif regiao == 'América do Norte':
        base = 800
        sazonalidade = 150
    elif regiao == 'Europa':
        base = 900
        sazonalidade = 180
    elif regiao == 'América Latina':
        base = 600
        sazonalidade = 120
    else:  # Oriente Médio e África
        base = 700
        sazonalidade = 140
    
    # Gerar dados com tendência crescente e componente sazonal
    for i, mes in enumerate(meses):
        # Tendência crescente (5% ao mês) + sazonalidade + ruído aleatório
        tendencia = base * (1 + 0.05 * i)
        # Sazonalidade (mais chamados no final do ano fiscal - Q4)
        componente_sazonal = sazonalidade * np.sin(np.pi * i / 6)
        # Ruído aleatório
        ruido = np.random.normal(0, base * 0.1)
        
        # Total de chamados
        total_chamados = int(tendencia + componente_sazonal + ruido)
        
        # Proporção de chamados relacionados a produtos falsificados (varia por região)
        if regiao == 'Ásia-Pacífico':
            prop_falsificados = np.random.uniform(0.25, 0.35)  # 25-35%
        elif regiao == 'América do Norte':
            prop_falsificados = np.random.uniform(0.10, 0.18)  # 10-18%
        elif regiao == 'Europa':
            prop_falsificados = np.random.uniform(0.15, 0.22)  # 15-22%
        elif regiao == 'América Latina':
            prop_falsificados = np.random.uniform(0.20, 0.30)  # 20-30%
        else:  # Oriente Médio e África
            prop_falsificados = np.random.uniform(0.22, 0.32)  # 22-32%
        
        # Calcular chamados relacionados a produtos falsificados
        chamados_falsificados = int(total_chamados * prop_falsificados)
        
        # Adicionar ao DataFrame
        chamados_df = pd.concat([chamados_df, pd.DataFrame({
            'Data': [mes],
            'Região': [regiao],
            'Total_Chamados': [total_chamados],
            'Chamados_Falsificados': [chamados_falsificados],
            'Percentual_Falsificados': [prop_falsificados * 100]
        })])

# Resetar índice
chamados_df = chamados_df.reset_index(drop=True)

# Salvar dados
chamados_df.to_csv('chamados_suporte.csv', index=False)

# Visualização 1: Evolução mensal de chamados relacionados a produtos falsificados por região
plt.figure(figsize=(12, 8))
for regiao in regioes:
    dados_regiao = chamados_df[chamados_df['Região'] == regiao]
    plt.plot(dados_regiao['Data'], dados_regiao['Chamados_Falsificados'], marker='o', linewidth=2, label=regiao)

plt.title('Evolução Mensal de Chamados Relacionados a Produtos Falsificados por Região', fontsize=16)
plt.xlabel('Mês', fontsize=12)
plt.ylabel('Número de Chamados', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Região', fontsize=10)
plt.tight_layout()
plt.savefig('../visualizacoes/evolucao_chamados_falsificados.png', dpi=300, bbox_inches='tight')

# Visualização 2: Percentual de chamados relacionados a produtos falsificados por região
plt.figure(figsize=(12, 8))
media_percentual = chamados_df.groupby('Região')['Percentual_Falsificados'].mean().sort_values(ascending=False)

sns.barplot(x=media_percentual.index, y=media_percentual.values, palette='viridis')
plt.title('Percentual Médio de Chamados Relacionados a Produtos Falsificados por Região', fontsize=16)
plt.xlabel('Região', fontsize=12)
plt.ylabel('Percentual Médio (%)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('../visualizacoes/percentual_chamados_falsificados.png', dpi=300, bbox_inches='tight')

print("Dados de chamados de suporte gerados e visualizações criadas com sucesso!")
