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
        self.W1 = np.random.uniform(0, 1, (input_size, hidden_size))
        self.b1 = np.random.uniform(0, 1, (1, hidden_size))
        self.W2 = np.random.uniform(0, 1, (hidden_size, output_size))
        self.b2 = np.random.uniform(0, 1, (1, output_size))
        
        # Momentum variables
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2
        
    def backward(self, X, y, learning_rate, momentum=0.0):
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
        
        # Calculate velocities
        self.vW2 = momentum * self.vW2 - learning_rate * dW2
        self.vb2 = momentum * self.vb2 - learning_rate * db2
        self.vW1 = momentum * self.vW1 - learning_rate * dW1
        self.vb1 = momentum * self.vb1 - learning_rate * db1
        
        # Update weights
        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1

def train_mlp(X, y, seed, momentum, max_epochs=300000, lr=0.1, precision=1e-6):
    mlp = MLP(4, 15, 3, seed)
    mse_history = []
    
    for epoch in range(1, max_epochs + 1):
        output = mlp.forward(X)
        mse = np.mean((output - y) ** 2)
        
        if epoch % 10 == 0 or epoch == 1:
            mse_history.append(mse)
            
        if mse <= precision:
            print(f"Converged at epoch {epoch} with MSE {mse}")
            break
            
        mlp.backward(X, y, lr, momentum)
        
        if epoch % 50000 == 0:
            print(f"Epoch {epoch}, MSE: {mse}")
            
    return mlp, epoch, mse_history

def main():
    train_df = pd.read_csv('train.csv')
    X_train = train_df[['x1', 'x2', 'x3', 'x4']].values
    y_train = train_df[['d1', 'd2', 'd3']].values
    
    test_df = pd.read_csv('test.csv')
    X_test = test_df[['x1', 'x2', 'x3', 'x4']].values
    y_test = test_df[['d1', 'd2', 'd3']].values
    
    results = []
    history_dict = {}
    
    seed = 100
    configs = [
        {'name': 'Padrão', 'momentum': 0.0},
        {'name': 'Momentum', 'momentum': 0.9}
    ]
    
    for config in configs:
        name = config['name']
        momentum = config['momentum']
        
        print(f"--- Training {name} ---")
        start_time = time.time()
        # Same seed for both
        mlp, epochs, mse_hist = train_mlp(X_train, y_train, seed=seed, momentum=momentum, max_epochs=5000)
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"{name} finished in {processing_time:.2f} seconds. Epochs: {epochs}")
        
        final_mse = mse_hist[-1]
        
        # Predict on test
        y_pred_raw = mlp.forward(X_test)
        
        # Post-processing
        y_pred = np.where(y_pred_raw >= 0.5, 1, 0)
        
        # Accuracy: a pattern is correct only if all 3 outputs match exactly
        correct = 0
        for i in range(len(y_test)):
            if np.array_equal(y_pred[i], y_test[i]):
                correct += 1
        accuracy = (correct / len(y_test)) * 100
        
        results.append({
            'T': name,
            'Epochs': epochs,
            'MSE': final_mse,
            'Time': processing_time,
            'y_pred_raw': y_pred_raw.tolist(),
            'y_pred': y_pred.tolist(),
            'Accuracy': accuracy
        })
        history_dict[name] = mse_hist
        
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # Plotting
    plt.figure(figsize=(12, 5))
    
    t1_name = configs[0]['name']
    t2_name = configs[1]['name']
    
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(history_dict[t1_name]) * 10 + 1, 10)[:len(history_dict[t1_name])], history_dict[t1_name])
    plt.title(f'EQM x Época ({t1_name})')
    plt.xlabel('Época')
    plt.ylabel('Erro Quadrático Médio (EQM)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(history_dict[t2_name]) * 10 + 1, 10)[:len(history_dict[t2_name])], history_dict[t2_name])
    plt.title(f'EQM x Época ({t2_name})')
    plt.xlabel('Época')
    plt.ylabel('Erro Quadrático Médio (EQM)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('graficos_eqm.png')
    print("Saved graficos_eqm.png")

if __name__ == '__main__':
    main()
