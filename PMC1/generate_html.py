import json
import pandas as pd
import numpy as np

def generate_html():
    with open('results.json', 'r') as f:
        results = json.load(f)
        
    test_df = pd.read_csv('test.csv')
    
    # Extract values for Table 1
    t1_table_rows = ""
    for idx, r in enumerate(results):
        t1_table_rows += f"<tr><td>{idx+1}º ({r['T']})</td><td>{r['MSE']:.8f}</td><td>{r['Epochs']}</td></tr>\n"
        
    # Extract values for Table 2 (Test Set)
    t2_table_rows = ""
    y_preds = [r['y_pred'] for r in results]
    
    # For relative error calculations, the exercise asks for "Erro Relativo Médio (%)" and "Variância (%)".
    # The variance is of the relative errors for each training.
    for i in range(len(test_df)):
        sample = test_df.iloc[i]
        d = sample['d']
        preds = [p[i] for p in y_preds]
        
        row = f"<tr><td>{i+1}</td><td>{sample['x1']:.4f}</td><td>{sample['x2']:.4f}</td><td>{sample['x3']:.4f}</td><td>{d:.4f}</td>"
        for p in preds:
            row += f"<td>{p:.4f}</td>"
        row += "</tr>\n"
        t2_table_rows += row
        
    # Add Erro Relativo Médio (%) and Variância (%)
    mean_errors = []
    vars_errors = []
    for r in results:
        mean_errors.append(r['MeanRelError'])
        vars_errors.append(r['VarRelError'])
        
    mean_err_row = "<tr><td colspan='5'><b>Erro Relativo Médio (%)</b></td>"
    for e in mean_errors:
        mean_err_row += f"<td>{e:.4f}</td>"
    mean_err_row += "</tr>\n"
    
    var_err_row = "<tr><td colspan='5'><b>Variância (%)</b></td>"
    for v in vars_errors:
        var_err_row += f"<td>{v:.4f}</td>"
    var_err_row += "</tr>\n"
    
    t2_table_rows += mean_err_row + var_err_row
    
    # Determine the best model
    best_idx = np.argmin(mean_errors)
    best_model = results[best_idx]['T']
    
    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Resultados - Treinamento da Rede Perceptron Multicamadas</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #0056b3;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .question-box {{
            background-color: #eef7ff;
            border-left: 5px solid #0056b3;
            padding: 10px;
            margin-bottom: 20px;
        }}
        .img-container {{
            text-align: center;
            margin: 20px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ccc;
        }}
    </style>
</head>
<body>

    <h1>Trabalho - Lab. Inteligência Artificial</h1>
    <h2>Resultados do Algoritmo Backpropagation</h2>

    <h3>1. Resultados Finais dos Treinamentos</h3>
    <table>
        <thead>
            <tr>
                <th>Treinamento</th>
                <th>Erro Quadrático Médio (EQM)</th>
                <th>Número de Épocas</th>
            </tr>
        </thead>
        <tbody>
            {t1_table_rows}
        </tbody>
    </table>

    <h3>2. Gráficos de EQM x Épocas (Dois maiores números de épocas)</h3>
    <div class="img-container">
        <img src="graficos_eqm.png" alt="Gráficos EQM x Épocas">
    </div>

    <h3>3. Variação do Erro Quadrático Médio e Número de Épocas</h3>
    <div class="question-box">
        <p><b>Pergunta:</b> Explique de forma detalhada por que tanto o erro quadrático médio quanto o número de épocas variam de treinamento para treinamento.</p>
        <p><b>Resposta:</b> A variação ocorre devido à inicialização aleatória dos pesos sinápticos e dos vieses (biases) antes de cada treinamento. Como o algoritmo Backpropagation utiliza o método de descida de gradiente para minimizar o erro, a posição inicial da rede na superfície de erro (landscape) é diferente a cada execução. Consequentemente, o caminho percorrido pelo gradiente até o mínimo local também muda. Se os pesos iniciais estiverem mais "próximos" de uma boa solução, a rede convergirá em menos épocas; caso contrário, precisará de mais épocas. Da mesma forma, diferentes inicializações podem levar a rede a convergir para diferentes mínimos locais na superfície de erro, resultando em valores finais distintos para o Erro Quadrático Médio (EQM).</p>
    </div>

    <h3>4. Validação da Rede (Conjunto de Teste)</h3>
    <table>
        <thead>
            <tr>
                <th>Amostra</th>
                <th>x1</th>
                <th>x2</th>
                <th>x3</th>
                <th>d</th>
                <th>y_rede (T1)</th>
                <th>y_rede (T2)</th>
                <th>y_rede (T3)</th>
                <th>y_rede (T4)</th>
                <th>y_rede (T5)</th>
            </tr>
        </thead>
        <tbody>
            {t2_table_rows}
        </tbody>
    </table>

    <h3>5. Análise da Melhor Configuração</h3>
    <div class="question-box">
        <p><b>Pergunta:</b> Indique qual das configurações finais de treinamento {{T1, T2, T3, T4 ou T5}} seria a mais adequada para o sistema de ressonância magnética, ou seja, qual delas está oferecendo a melhor generalização.</p>
        <p><b>Resposta:</b> Com base na tabela de validação, a configuração mais adequada é a <b>{best_model}</b>. Esta configuração apresentou o menor Erro Relativo Médio (%) no conjunto de teste, o que indica que as suas predições (y_rede) foram as mais próximas dos valores desejados (d) para amostras que não foram vistas durante o treinamento. Consequentemente, a rede {best_model} obteve a melhor capacidade de generalização.</p>
    </div>

</body>
</html>
"""
    with open('respostas.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print("respostas.html gerado com sucesso!")

if __name__ == '__main__':
    generate_html()
