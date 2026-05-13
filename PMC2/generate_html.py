import json
import pandas as pd

def generate_html():
    with open('results.json', 'r') as f:
        results = json.load(f)
        
    test_df = pd.read_csv('test.csv')
    
    # Generate HTML content
    html_content = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Resultados - Classificação de Conservantes</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; color: #333; }
        h1, h2, h3 { color: #0056b3; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        .img-container { text-align: center; margin: 20px 0; }
        img { max-width: 100%; height: auto; border: 1px solid #ccc; }
    </style>
</head>
<body>

    <h1>Trabalho - Lab. Inteligência Artificial</h1>
    <h2>Resultados - Classificação de Conservantes (PMC2)</h2>

    <h3>1. Desempenho dos Treinamentos</h3>
    <table>
        <thead>
            <tr>
                <th>Algoritmo</th>
                <th>Tempo de Processamento (s)</th>
                <th>Número de Épocas</th>
                <th>Erro Quadrático Médio Final (EQM)</th>
            </tr>
        </thead>
        <tbody>
"""
    for r in results:
        html_content += f"            <tr><td>{r['T']}</td><td>{r['Time']:.2f}</td><td>{r['Epochs']}</td><td>{r['MSE']:.8f}</td></tr>\n"
        
    html_content += """        </tbody>
    </table>

    <h3>2. Gráficos de EQM x Épocas</h3>
    <div class="img-container">
        <img src="graficos_eqm.png" alt="Gráficos EQM x Épocas">
    </div>

    <h3>3. Validação no Conjunto de Teste</h3>
"""

    for r in results:
        html_content += f"    <h4>Resultados para o Treinamento: {r['T']}</h4>\n"
        html_content += """    <table>
        <thead>
            <tr>
                <th>Amostra</th>
                <th>x1</th>
                <th>x2</th>
                <th>x3</th>
                <th>x4</th>
                <th>d1</th>
                <th>d2</th>
                <th>d3</th>
                <th>y1</th>
                <th>y2</th>
                <th>y3</th>
            </tr>
        </thead>
        <tbody>
"""
        y_preds = r['y_pred']
        for i in range(len(test_df)):
            sample = test_df.iloc[i]
            html_content += f"            <tr><td>{i+1}</td><td>{sample['x1']:.4f}</td><td>{sample['x2']:.4f}</td><td>{sample['x3']:.4f}</td><td>{sample['x4']:.4f}</td><td>{int(sample['d1'])}</td><td>{int(sample['d2'])}</td><td>{int(sample['d3'])}</td><td>{y_preds[i][0]}</td><td>{y_preds[i][1]}</td><td>{y_preds[i][2]}</td></tr>\n"
            
        html_content += f"""        </tbody>
    </table>
    <p><b>Taxa de Acerto:</b> {r['Accuracy']:.2f}%</p>
"""

    html_content += """
</body>
</html>
"""
    with open('respostas.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print("respostas.html gerado com sucesso!")

if __name__ == '__main__':
    generate_html()
