import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1.0 - sx)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, seed):
        np.random.seed(seed)
        # Weights between 0 and 1 as requested
        self.W1 = np.random.uniform(0, 1, (input_size, hidden_size))
        self.b1 = np.random.uniform(0, 1, (1, hidden_size))
        self.W2 = np.random.uniform(0, 1, (hidden_size, output_size))
        self.b2 = np.random.uniform(0, 1, (1, output_size))
        
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2
        
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        # Output layer error
        dZ2 = (self.A2 - y) * sigmoid_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer error
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

def train_mlp(X, y, seed, max_epochs=200000, lr=0.1, precision=1e-6):
    mlp = MLP(3, 10, 1, seed)
    mse_history = []
    
    for epoch in range(1, max_epochs + 1):
        output = mlp.forward(X)
        mse = np.mean((output - y) ** 2)
        
        # Guardar MSE history a cada 10 epocas para o plot não ficar pesado
        if epoch % 10 == 0 or epoch == 1:
            mse_history.append(mse)
        
        if mse <= precision:
            print(f"Converged at epoch {epoch} with MSE {mse}")
            break
            
        mlp.backward(X, y, lr)
        
        if epoch % 50000 == 0:
            print(f"Epoch {epoch}, MSE: {mse}")
            
    return mlp, epoch, mse_history

def main():
    train_df = pd.read_csv('train.csv')
    X_train = train_df[['x1', 'x2', 'x3']].values
    y_train = train_df[['d']].values
    
    test_df = pd.read_csv('test.csv')
    X_test = test_df[['x1', 'x2', 'x3']].values
    y_test = test_df[['d']].values
    
    results = []
    history_dict = {}
    
    for i in range(1, 6):
        print(f"--- Training T{i} ---")
        start_time = time.time()
        # Different seeds for initializations between 0 and 1
        mlp, epochs, mse_hist = train_mlp(X_train, y_train, seed=i*42)
        end_time = time.time()
        print(f"T{i} finished in {end_time - start_time:.2f} seconds. Epochs: {epochs}")
        
        final_mse = mse_hist[-1]
        
        # Predict on test
        y_pred = mlp.forward(X_test)
        
        rel_errors = np.abs(y_test - y_pred) / np.abs(y_test) * 100
        mean_rel_error = np.mean(rel_errors)
        var_rel_error = np.var(rel_errors)
        
        results.append({
            'T': f'T{i}',
            'Epochs': epochs,
            'MSE': final_mse,
            'y_pred': y_pred.flatten().tolist(),
            'MeanRelError': mean_rel_error,
            'VarRelError': var_rel_error
        })
        history_dict[f'T{i}'] = mse_hist
        
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    results_sorted = sorted(results, key=lambda x: x['Epochs'], reverse=True)
    longest1 = results_sorted[0]['T']
    longest2 = results_sorted[1]['T']
    
    plt.figure(figsize=(12, 5))
    
    # The history points are sampled every 10 epochs. We create x-axis values accordingly.
    # But actually, since it might stop at an exact epoch, we just plot.
    
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(history_dict[longest1]) * 10 + 1, 10)[:len(history_dict[longest1])], history_dict[longest1])
    plt.title(f'EQM x Época ({longest1})')
    plt.xlabel('Época')
    plt.ylabel('Erro Quadrático Médio (EQM)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(history_dict[longest2]) * 10 + 1, 10)[:len(history_dict[longest2])], history_dict[longest2])
    plt.title(f'EQM x Época ({longest2})')
    plt.xlabel('Época')
    plt.ylabel('Erro Quadrático Médio (EQM)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('graficos_eqm.png')
    print("Saved graficos_eqm.png")

if __name__ == '__main__':
    main()
