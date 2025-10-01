# Integrantes:
- **555183** - Danilo Ramalho Silva
- **554668** - Israel Dalcin Alves Diniz
- **556213** - João Vitor Pires da Silva
- **555677** - Matheus Hungaro Fidelis
- **556389** - Pablo Menezes Barreto
- **556984** - Tiago Toshio Kumagai Gibo


# Painel Executivo - Piloto de Governança

Dashboard interativo desenvolvido em Streamlit para monitoramento do piloto de governança e analytics.

## 📁 Link do Github
https://github.com/tiagotkg/CS3-Governanca.git

## 🤖 Link do Streamlit
https://cs-governanca.streamlit.app

## 🚀 Como Executar

### 1. Ativar o ambiente virtual
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Instalar dependências (se necessário)
```bash
pip install -r requirements.txt
```

### 3. Configurar API da OpenAI

#### Para Desenvolvimento Local:
1. Copie o arquivo `env_example.txt` para `.env`:
```bash
cp env_example.txt .env
```

2. Edite o arquivo `.env` e adicione sua chave da API:
```
OPENAI_API_KEY=sua_chave_da_api_aqui
```


### 4. Executar a aplicação

#### Desenvolvimento Local:
```bash
streamlit run app.py
```


A aplicação será aberta automaticamente no navegador em `http://localhost:8501`

## 📊 Funcionalidades

### Páginas Disponíveis:
- **🏠 Visão Geral**: KPIs do piloto, precisão, recall, tendências temporais
- **🔍 Detecção & Operação**: Matriz de confusão, tempo de detecção, fila de revisão
- **⚖️ Fairness & Governança**: Model Card, LGPD, análise de fairness
- **💰 Negócio & ROI**: Comparação com/sem ação, ROI, recomendação go/no-go

### Filtros Dinâmicos:
- Período (data inicial e final)
- Região/UF
- Canal (E-commerce, Loja Física, Suporte)
- Linha de produto (Cartucho/Toner, Original/Genérico)
- Severidade do chamado

### Downloads:
- **CSV**: Exportar dados filtrados
- **PNG**: Exportar gráficos (em desenvolvimento)
- **PDF**: Relatório executivo completo

### Insights Automáticos:
- 🤖 Geração automática de insights usando OpenAI
- Análise inteligente de padrões regionais
- Recomendações estratégicas baseadas nos dados
- **NOVO**: Insights gerados automaticamente na seção "Recomendação"

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Framework para aplicações web
- **Pandas**: Manipulação de dados
- **Plotly**: Visualizações interativas
- **NumPy**: Computação numérica
- **Scikit-learn**: Machine learning
- **ReportLab**: Geração de PDFs

## 📁 Estrutura do Projeto

```
CS3-Governanca/
├── app.py                 # Aplicação principal
├── env_example.txt        # Exemplo de variáveis de ambiente
├── requirements.txt       # Dependências
├── README.md              # Este arquivo
├── venv/                 # Ambiente virtual
└── sprints/              # Dados das sprints anteriores
```

## 🎯 Critérios de Aceite Atendidos

✅ Dashboard único com filtros funcionais  
✅ KPIs do modelo e de negócio visíveis  
✅ Seção de governança com Model Card + LGPD  
✅ Comparativo Com ação vs Sem ação  
✅ Botões de download ativos  
✅ Seção de insights no painel  
