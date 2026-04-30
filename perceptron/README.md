# Trabalho: Laboratório de Inteligência Artificial - Perceptron

## Atividade 1: Treinamentos com pesos aleatórios entre 0 e 1

A rede Perceptron foi treinada 5 vezes (T1 a T5), inicializando os pesos de forma randômica no intervalo de 0 a 1. A taxa de aprendizagem utilizada foi de $\eta = 0.01$, e o viés de ativação foi mantido fixo em $x_0 = -1$.

| Treinamento | Vetor de Pesos Inicial (w0, w1, w2, w3) | Vetor de Pesos Final (w0, w1, w2, w3) | Número de Épocas |
| :---: | :--- | :--- | :---: |
| **1º (T1)** | `w0=0.5762`, `w1=0.4819`, `w2=0.0797`, `w3=0.3578` | `w0=-3.1438`, `w1=1.6030`, `w2=2.5174`, `w3=-0.7472` | 438 |
| **2º (T2)** | `w0=0.4965`, `w1=0.2178`, `w2=0.2719`, `w3=0.5538` | `w0=-3.1235`, `w1=1.5623`, `w2=2.4980`, `w3=-0.7425` | 424 |
| **3º (T3)** | `w0=0.8909`, `w1=0.7494`, `w2=0.5675`, `w3=0.9677` | `w0=-3.1091`, `w1=1.5755`, `w2=2.5121`, `w3=-0.7428` | 448 |
| **4º (T4)** | `w0=0.9290`, `w1=0.3534`, `w2=0.9118`, `w3=0.6027` | `w0=-3.0310`, `w1=1.4725`, `w2=2.4645`, `w3=-0.7239` | 405 |
| **5º (T5)** | `w0=0.5304`, `w1=0.7407`, `w2=0.3437`, `w3=0.9656` | `w0=-3.1296`, `w1=1.6096`, `w2=2.5203`, `w3=-0.7460` | 434 |

*(Obs: $w_0$ representa o limiar de ativação/bias)*

---

## Atividade 2: Classificação Automática de Novas Amostras

Após o treinamento, o perceptron foi aplicado na classificação das 10 novas amostras. Os resultados das saídas $y$ (onde $-1$ representa a classe **C1** e $+1$ a classe **C2**) para os 5 processos (T1 a T5) estão registrados abaixo:

| Amostra | x1 | x2 | x3 | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | -0.3565 | 0.0620 | 5.9891 | -1 | -1 | -1 | -1 | -1 |
| **2** | -0.7842 | 1.1267 | 5.5912 | 1 | 1 | 1 | 1 | 1 |
| **3** | 0.3012 | 0.5611 | 5.8234 | 1 | 1 | 1 | 1 | 1 |
| **4** | 0.7757 | 1.0648 | 8.0677 | 1 | 1 | 1 | 1 | 1 |
| **5** | 0.1570 | 0.8028 | 6.3040 | 1 | 1 | 1 | 1 | 1 |
| **6** | -0.7014 | 1.0316 | 3.6005 | 1 | 1 | 1 | 1 | 1 |
| **7** | 0.3748 | 0.1536 | 6.1537 | -1 | -1 | -1 | -1 | -1 |
| **8** | -0.6920 | 0.9404 | 4.4058 | 1 | 1 | 1 | 1 | 1 |
| **9** | -1.3970 | 0.7141 | 4.9263 | -1 | -1 | -1 | -1 | -1 |
| **10**| -1.8842 | -0.2805 | 1.2548 | -1 | -1 | -1 | -1 | -1 |

---

## Questões Teóricas

**Explique por que o número de épocas de treinamento varia a cada vez que executamos o treinamento do perceptron:**

O número de épocas varia porque os pesos iniciais da rede são gerados aleatoriamente a cada execução. Cada conjunto de pesos iniciais representa um ponto de partida (um hiperplano inicial) diferente no espaço de busca. Dependendo de quão favorável for esse ponto de partida (mais próximo ou mais distante da solução ideal que separa perfeitamente as duas classes), o algoritmo precisará de um número diferente de iterações (ajustes de peso via regra de Hebb) até convergir para uma fronteira de decisão válida.

**Qual a principal limitação do perceptron quando aplicado em problemas de classificação de padrões:**

A principal limitação do Perceptron de camada única é que ele **só consegue classificar corretamente padrões que são linearmente separáveis**. Se as características das classes formarem uma distribuição complexa no espaço e não puderem ser separadas perfeitamente por um único hiperplano reto (como ocorre no clássico problema lógico do OU-Exclusivo, o XOR), o algoritmo não irá convergir, ficando eternamente oscilando e ajustando os pesos em um loop infinito, caso não exista um critério de parada forçado, como um número máximo de épocas pré-estabelecido.
