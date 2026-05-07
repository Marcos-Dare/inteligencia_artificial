import numpy as np
import matplotlib.pyplot as plt

train_data_str = """
01 0.4329 -1.3719 0.7022 -0.8535 1.0000
02 0.3024 0.2286 0.8630 2.7909 -1.0000
03 0.1349 -0.6445 1.0530 0.5687 -1.0000
04 0.3374 -1.7163 0.3670 -0.6283 -1.0000
05 1.1434 -0.0485 0.6637 1.2606 1.0000
06 1.3749 -0.5071 0.4464 1.3009 1.0000
07 0.7221 -0.7587 0.7681 -0.5592 1.0000
08 0.4403 -0.8072 0.5154 -0.3129 1.0000
09 -0.5231 0.3548 0.2538 1.5776 -1.0000
10 0.3255 -2.0000 0.7112 -1.1209 1.0000
11 0.5824 1.3915 -0.2291 4.1735 -1.0000
12 0.1340 0.6081 0.4450 3.2230 -1.0000
13 0.1480 -0.2988 0.4778 0.8649 1.0000
14 0.7359 0.1869 -0.0872 2.3584 1.0000
15 0.7115 -1.1469 0.3394 0.9573 -1.0000
16 0.8251 -1.2840 0.8452 1.2382 -1.0000
17 0.1569 0.3712 0.8825 1.7633 1.0000
18 0.0033 0.6835 0.5389 2.8249 -1.0000
19 0.4243 0.8313 0.2634 3.5855 -1.0000
20 1.0490 0.1326 0.9138 1.9792 1.0000
21 1.4276 0.5331 -0.0145 3.7286 1.0000
22 0.5971 1.4865 0.2904 4.6069 -1.0000
23 0.8475 2.1479 0.3179 5.8235 -1.0000
24 1.3967 -0.4171 0.6443 1.3927 1.0000
25 0.0044 1.5378 0.6099 4.7755 -1.0000
26 0.2201 -0.5668 0.0515 0.7829 1.0000
27 0.6300 -1.2480 0.8591 0.8093 -1.0000
28 -0.2479 0.8960 0.0547 1.7381 1.0000
29 -0.3088 -0.0929 0.8659 1.5483 -1.0000
30 -0.5180 1.4974 0.5453 2.3993 1.0000
31 0.6833 0.8266 0.0829 2.8864 1.0000
32 0.4353 -1.4066 0.4207 -0.4879 1.0000
33 -0.1069 -3.2329 0.1856 -2.4572 -1.0000
34 0.4662 0.6261 0.7304 3.4370 -1.0000
35 0.8298 -1.4089 0.3119 1.3235 -1.0000
"""

test_data_str = """
1 0.9694 0.6909 0.4334 3.4965
2 0.5427 1.3832 0.6390 4.0352
3 0.6081 -0.9196 0.5925 0.1016
4 -0.1618 0.4694 0.2030 3.0117
5 0.1870 -0.2578 0.6124 1.7749
6 0.4891 -0.5276 0.4378 0.6439
7 0.3777 2.0149 0.7423 3.3932
8 1.1498 -0.4067 0.2469 1.5866
9 0.9325 1.0950 1.0359 3.3591
10 0.5060 1.3317 0.9222 3.7174
11 0.0497 -2.0656 0.6124 -0.6585
12 0.4004 3.5369 0.9766 5.3532
13 -0.1874 1.3343 0.5374 3.2189
14 0.5060 1.3317 0.9222 3.7174
15 1.6375 -0.7911 0.7537 0.5515
"""

def parse_data(data_str, has_d=True):
    data = []
    for line in data_str.strip().split('\n'):
        parts = line.strip().split()
        if not parts:
            continue
        if has_d:
            row = [float(p) for p in parts[1:]] # skip id
        else:
            row = [float(p) for p in parts[1:]]
        data.append(row)
    return np.array(data)

train_data = parse_data(train_data_str, has_d=True)
test_data = parse_data(test_data_str, has_d=False)

X_train = train_data[:, 0:4]
d_train = train_data[:, 4]
X_test = test_data

# Insert x0 = -1
X_train = np.insert(X_train, 0, -1, axis=1)
X_test = np.insert(X_test, 0, -1, axis=1)

alpha = 0.0025
precision = 1e-6

results = []
test_predictions = []
eqms_for_plot = []

for i in range(5):
    np.random.seed(i + 42) # Different seed for each run
    w = np.random.rand(5)
    initial_w = w.copy()
    
    epochs = 0
    previous_eqm = float('inf')
    eqms = []
    
    while True:
        eqm_sum = 0
        for j in range(X_train.shape[0]):
            x = X_train[j]
            d = d_train[j]
            u = np.dot(w, x)
            eqm_sum += (d - u)**2
            w = w + alpha * (d - u) * x
            
        eqm = eqm_sum / X_train.shape[0]
        eqms.append(eqm)
        epochs += 1
        
        if abs(eqm - previous_eqm) <= precision:
            break
        previous_eqm = eqm
        
    results.append({
        'training': f'T{i+1}',
        'initial_w': initial_w,
        'final_w': w.copy(),
        'epochs': epochs
    })
    
    if i < 2:
        eqms_for_plot.append(eqms)
        
    # Test
    y_test = []
    for j in range(X_test.shape[0]):
        u = np.dot(w, X_test[j])
        y = 1 if u >= 0 else -1
        y_test.append(y)
    test_predictions.append(y_test)

# Plotting EQMs
plt.figure(figsize=(10, 6))
plt.plot(eqms_for_plot[0], label='Treinamento T1')
plt.plot(eqms_for_plot[1], label='Treinamento T2')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio (EQM)')
plt.title('Gráfico do EQM por Épocas (T1 e T2)')
plt.legend()
plt.grid(True)
plt.savefig('/home/marcos/Área de Trabalho/Projetos/inteligencia_artificial/adaline/grafico_eqm.png')

# Create README.md content
readme_content = "# Resultados da Rede Adaline\n\n"
readme_content += "## Tabela de Treinamentos\n\n"
readme_content += "| Treinamento | Número de Épocas | Vetor de Pesos Inicial (w0, w1, w2, w3, w4) | Vetor de Pesos Final (w0, w1, w2, w3, w4) |\n"
readme_content += "|---|---|---|---|\n"
for r in results:
    w_ini = ", ".join([f"{x:.4f}" for x in r['initial_w']])
    w_fin = ", ".join([f"{x:.4f}" for x in r['final_w']])
    readme_content += f"| {r['training']} | {r['epochs']} | [{w_ini}] | [{w_fin}] |\n"
    
readme_content += "\n## Gráfico de EQM\n\n"
readme_content += "![Gráfico do EQM](grafico_eqm.png)\n\n"

readme_content += "## Tabela de Classificação das Amostras de Teste\n\n"
readme_content += "| Amostra | x1 | x2 | x3 | x4 | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) | Classificação Final (Válvula) |\n"
readme_content += "|---|---|---|---|---|---|---|---|---|---|---|\n"
for i in range(len(test_data)):
    row_str = f"| {i+1} | " + " | ".join([f"{x:.4f}" for x in test_data[i]]) + " | "
    preds = [test_predictions[t][i] for t in range(5)]
    valvula = "A" if preds[0] == -1 else "B" # since they are identical
    row_str += " | ".join([str(p) for p in preds]) + f" | {valvula} |\n"
    readme_content += row_str

readme_content += "\n## Explicação sobre os Pesos Inalterados\n\n"
readme_content += "**Embora o número de épocas de cada treinamento seja diferente, explique por que então os valores dos pesos continuam praticamente inalterados:**\n\n"
readme_content += "A rede Adaline encontra um único ponto de mínimo global para o Erro Quadrático Médio (EQM) no espaço de pesos, uma vez que a superfície de erro para redes com funções de ativação lineares no cálculo do erro é um paraboloide (função quadrática convexa). Devido a esta característica, não existem mínimos locais. Independentemente do vetor de pesos inicial (aleatório), o algoritmo da Regra Delta guiado pela descida do gradiente convergirá sempre para o mesmo conjunto de pesos ótimo (ou muito próximo a ele, dada a tolerância/precisão). A diferença no número de épocas ocorre porque cada treinamento parte de um ponto inicial diferente no espaço de pesos, necessitando de mais ou menos iterações (épocas) para descer o gradiente e atingir o mínimo global com a tolerância de precisão exigida ($10^{-6}$)."

with open('/home/marcos/Área de Trabalho/Projetos/inteligencia_artificial/adaline/README.md', 'w') as f:
    f.write(readme_content)

print("Processamento concluído. README.md e grafico_eqm.png gerados na pasta adaline.")
