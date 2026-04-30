import csv
import random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Carregamento dos Dados de Treinamento
dados_treino = []
with open('treinamento.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # Pular o cabeçalho
    for row in reader:
        dados_treino.append([float(x) for x in row])

# 2. Carregamento das Amostras de Teste
dados_teste = []
with open('teste.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # Pular o cabeçalho
    for row in reader:
        dados_teste.append([float(x) for x in row[1:]]) # Ignorar a coluna 'Amostra'

# 3. Hiperparâmetros
taxa_aprendizagem = 0.01
epocas_max = 1000

print("Iniciando treinamento do Perceptron (5 Processos)...")
print(f"Taxa de aprendizagem: {taxa_aprendizagem}")
print("-" * 40)

modelos_pesos = [] # Lista para salvar os pesos de cada um dos 5 treinamentos
historico_erros = [] # Lista para salvar a contagem de erros por época para os gráficos

# 4. Loop para os 5 Processos de Treinamento (T1 a T5)
for t in range(1, 6):
    # Inicialização dos Pesos Aleatórios (w0 é o bias, w1, w2, w3 os pesos)
    # Inicializando com valores entre 0 e 1 conforme a especificação do trabalho
    w_inicial = [random.uniform(0, 1) for _ in range(4)]
    w = w_inicial.copy()
    print(f"\n[T{t}] Pesos Iniciais: w0(bias)={w_inicial[0]:.4f}, w1={w_inicial[1]:.4f}, w2={w_inicial[2]:.4f}, w3={w_inicial[3]:.4f}")
    
    epoca_atual = 0
    erros_por_epoca = []
    
    while epoca_atual < epocas_max:
        erros_nesta_epoca = 0
        
        for padrao in dados_treino:
            # x0 = -1 (bias fixo)
            # Entrada: [x0, x1, x2, x3]
            x = [-1.0] + padrao[:3]
            d = padrao[3]  # saída desejada
            
            # Campo local induzido v = w0*x0 + w1*x1 + w2*x2 + w3*x3
            v = sum(w[i] * x[i] for i in range(4))
            
            # Função de Ativação (Degrau Bipolar)
            y = 1.0 if v >= 0 else -1.0
            
            # Regra de Hebb Supervisionada
            if y != d:
                erros_nesta_epoca += 1
                erro = d - y
                for i in range(4):
                    w[i] = w[i] + taxa_aprendizagem * erro * x[i]
                    
        erros_por_epoca.append(erros_nesta_epoca)
        epoca_atual += 1
        
        if erros_nesta_epoca == 0:
            break
            
    print(f"[T{t}] Convergiu na época {epoca_atual} | Pesos finais: w0(bias)={w[0]:.4f}, w1={w[1]:.4f}, w2={w[2]:.4f}, w3={w[3]:.4f}")
    modelos_pesos.append(w.copy())
    historico_erros.append(erros_por_epoca)

# 5. Classificação das Novas Amostras
print("-" * 40)
print("Resultados da Classificação das Novas Amostras:")
print("| Amostra | x1 | x2 | x3 | y(T1) | y(T2) | y(T3) | y(T4) | y(T5) |")
print("|---------|----|----|----|-------|-------|-------|-------|-------|")

for i, amostra in enumerate(dados_teste):
    # x0 = -1 (bias)
    x = [-1.0] + amostra
    saidas_t = []
    
    for w in modelos_pesos:
        v = sum(w[j] * x[j] for j in range(4))
        y = 1.0 if v >= 0 else -1.0
        saidas_t.append(y)
        
    print(f"| {i+1} | {amostra[0]:.4f} | {amostra[1]:.4f} | {amostra[2]:.4f} | {saidas_t[0]:.0f} | {saidas_t[1]:.0f} | {saidas_t[2]:.0f} | {saidas_t[3]:.0f} | {saidas_t[4]:.0f} |")

# 6. Geração do Dashboard Interativo com Plotly
print("-" * 40)
print("Gerando gráficos interativos aprimorados...")

# Configurar subplots: 1 linha, 2 colunas
# Coluna 1: Gráfico 3D, Coluna 2: Erros por época
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'xy'}]],
    subplot_titles=('Fronteira de Decisão (T1) e Dados', 'Histórico de Erros por Época')
)

# --- Gráfico de Erros por Época (Coluna 2) ---
cores = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f']
for t in range(5):
    fig.add_trace(go.Scatter(
        x=list(range(1, len(historico_erros[t])+1)), 
        y=historico_erros[t], 
        mode='lines',
        name=f'Treinamento T{t+1}',
        line=dict(color=cores[t], width=2),
        legendgroup='erros'
    ), row=1, col=2)

# --- Gráfico 3D (Coluna 1) ---
# Separar dados de treino por classe para plotar cores diferentes
treino_c1 = [p for p in dados_treino if p[3] == -1]
treino_c2 = [p for p in dados_treino if p[3] == 1]

# Plotar Classe C1 (Treino)
fig.add_trace(go.Scatter3d(
    x=[p[0] for p in treino_c1], y=[p[1] for p in treino_c1], z=[p[2] for p in treino_c1],
    mode='markers', marker=dict(color='#ff4757', size=6, symbol='circle', line=dict(color='white', width=1)),
    name='Classe C1 (-1)', legendgroup='dados'
), row=1, col=1)

# Plotar Classe C2 (Treino)
fig.add_trace(go.Scatter3d(
    x=[p[0] for p in treino_c2], y=[p[1] for p in treino_c2], z=[p[2] for p in treino_c2],
    mode='markers', marker=dict(color='#1e90ff', size=6, symbol='diamond', line=dict(color='white', width=1)),
    name='Classe C2 (+1)', legendgroup='dados'
), row=1, col=1)

# Gerar malha para o hiperplano de separação usando os pesos do T1
# v = w0*(-1) + w1*x1 + w2*x2 + w3*x3 = 0 => x3 = (w0 - w1*x1 - w2*x2) / w3
w_t1 = modelos_pesos[0]
w0, w1, w2, w3 = w_t1[0], w_t1[1], w_t1[2], w_t1[3]

x1_range = np.linspace(min(p[0] for p in dados_treino)-0.5, max(p[0] for p in dados_treino)+0.5, 10)
x2_range = np.linspace(min(p[1] for p in dados_treino)-0.5, max(p[1] for p in dados_treino)+0.5, 10)
X1, X2 = np.meshgrid(x1_range, x2_range)
X3 = (w0 - w1 * X1 - w2 * X2) / w3

# Plotar o hiperplano
fig.add_trace(go.Surface(
    x=X1, y=X2, z=X3, 
    colorscale='Viridis', opacity=0.7, showscale=False,
    name='Fronteira (T1)'
), row=1, col=1)

# Estilizar layout
fig.update_layout(
    title='Dashboard do Perceptron (Espaço de Decisão e Convergência)',
    template='plotly_dark', # Tema escuro premium
    scene=dict(
        xaxis_title='Físico-química x1',
        yaxis_title='Físico-química x2',
        zaxis_title='Físico-química x3'
    ),
    xaxis_title='Época',
    yaxis_title='Número de Erros',
    height=750,
    margin=dict(l=20, r=20, t=60, b=20)
)

# Salvar e concluir
fig.write_html("dashboard_perceptron.html")
print("Dashboard gerado e salvo em 'dashboard_perceptron.html' com sucesso!")
