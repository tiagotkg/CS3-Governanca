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

# Simulação de dados: NPS por tipo de cartucho (original x genérico)
# Período de 12 meses (2023)
meses = pd.date_range(start='2023-01-01', end='2023-12-31', freq='MS')
regioes = ['América do Norte', 'Europa', 'Ásia-Pacífico', 'América Latina', 'Oriente Médio e África']

# Criar DataFrame para NPS
nps_df = pd.DataFrame()

# Gerar dados para cada região
for regiao in regioes:
    # Definir NPS base para cada região (produtos originais)
    if regiao == 'América do Norte':
        nps_base_original = 72  # NPS base para produtos originais
    elif regiao == 'Europa':
        nps_base_original = 68
    elif regiao == 'Ásia-Pacífico':
        nps_base_original = 65
    elif regiao == 'América Latina':
        nps_base_original = 70
    else:  # Oriente Médio e África
        nps_base_original = 67
    
    # NPS para produtos falsificados (significativamente menor)
    nps_base_falsificado = nps_base_original * 0.4  # 40% do NPS de produtos originais
    
    # Gerar dados para cada mês
    for i, mes in enumerate(meses):
        # Adicionar variação sazonal e tendência
        fator_sazonal = 1 + 0.05 * np.sin(np.pi * i / 6)  # Pequena variação sazonal
        
        # Calcular NPS com variação
        nps_original = nps_base_original * fator_sazonal + np.random.normal(0, 3)  # Adicionar ruído
        nps_falsificado = nps_base_falsificado * fator_sazonal + np.random.normal(0, 5)  # Mais variabilidade em falsificados
        
        # Garantir que NPS esteja no intervalo -100 a 100
        nps_original = max(-100, min(100, nps_original))
        nps_falsificado = max(-100, min(100, nps_falsificado))
        
        # Adicionar ao DataFrame
        nps_df = pd.concat([nps_df, pd.DataFrame({
            'Data': [mes],
            'Região': [regiao],
            'NPS_Original': [nps_original],
            'NPS_Falsificado': [nps_falsificado],
            'Diferença_NPS': [nps_original - nps_falsificado]
        })])

# Resetar índice
nps_df = nps_df.reset_index(drop=True)

# Salvar dados
nps_df.to_csv('nps_cartuchos.csv', index=False)

# Visualização 1: Comparação de NPS médio por região
plt.figure(figsize=(14, 8))

# Calcular médias por região
media_por_regiao = nps_df.groupby('Região')[['NPS_Original', 'NPS_Falsificado']].mean().reset_index()
media_por_regiao = media_por_regiao.sort_values('NPS_Original', ascending=False)

# Criar gráfico de barras agrupadas
x = np.arange(len(regioes))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, media_por_regiao['NPS_Original'], width, label='Cartuchos Originais', color='#0096D6')
rects2 = ax.bar(x + width/2, media_por_regiao['NPS_Falsificado'], width, label='Cartuchos Falsificados', color='#E61E50')

ax.set_title('Net Promoter Score (NPS) Médio: Cartuchos Originais vs. Falsificados', fontsize=16)
ax.set_xlabel('Região', fontsize=14)
ax.set_ylabel('NPS', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(media_por_regiao['Região'], rotation=45, ha='right')
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionar linha horizontal em NPS = 0
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Adicionar rótulos nas barras
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -15),  # Ajuste para valores negativos
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('../visualizacoes/comparacao_nps_regiao.png', dpi=300, bbox_inches='tight')

# Visualização 2: Evolução do NPS ao longo do tempo (média global)
plt.figure(figsize=(14, 8))

# Calcular média global por mês
nps_mensal = nps_df.groupby('Data')[['NPS_Original', 'NPS_Falsificado']].mean().reset_index()

plt.plot(nps_mensal['Data'], nps_mensal['NPS_Original'], marker='o', linewidth=2, label='Cartuchos Originais', color='#0096D6')
plt.plot(nps_mensal['Data'], nps_mensal['NPS_Falsificado'], marker='s', linewidth=2, label='Cartuchos Falsificados', color='#E61E50')

plt.title('Evolução do NPS ao Longo do Tempo: Cartuchos Originais vs. Falsificados', fontsize=16)
plt.xlabel('Mês', fontsize=14)
plt.ylabel('NPS Médio Global', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizacoes/evolucao_nps_tempo.png', dpi=300, bbox_inches='tight')

print("Dados de NPS gerados e visualizações criadas com sucesso!")
