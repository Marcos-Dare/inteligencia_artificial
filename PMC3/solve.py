import numpy as np
import os
import json
import time

train_data = [
    0.1701, 0.1023, 0.4405, 0.3609, 0.7192, 0.2258, 0.3175, 0.0127, 0.4290, 0.0544,
    0.8000, 0.0450, 0.4268, 0.0112, 0.3218, 0.2185, 0.7240, 0.3516, 0.4420, 0.0984,
    0.1747, 0.3964, 0.5114, 0.6183, 0.3330, 0.2398, 0.0508, 0.4497, 0.2178, 0.7762,
    0.1078, 0.3773, 0.0001, 0.3877, 0.0821, 0.7836, 0.1887, 0.4483, 0.0424, 0.2539,
    0.3164, 0.6386, 0.4862, 0.4068, 0.1611, 0.1101, 0.4372, 0.3795, 0.7092, 0.2400,
    0.3087, 0.0159, 0.4330, 0.0733, 0.7995, 0.0262, 0.4223, 0.0085, 0.3303, 0.2037,
    0.7332, 0.3328, 0.4445, 0.0909, 0.1838, 0.3888, 0.5277, 0.6042, 0.3435, 0.2304,
    0.0568, 0.4500, 0.2371, 0.7705, 0.1246, 0.3701, 0.0006, 0.3943, 0.0646, 0.7878,
    0.1694, 0.4468, 0.0372, 0.2632, 0.3048, 0.6516, 0.4690, 0.4132, 0.1523, 0.1182,
    0.4334, 0.3978, 0.6987, 0.2538, 0.2998, 0.0195, 0.4366, 0.0924, 0.7984, 0.0077
]

test_data = [
    0.4173, 0.0062, 0.3387, 0.1886, 0.7418, 0.3138, 0.4466, 0.0835, 0.1930, 0.3807,
    0.5438, 0.5897, 0.3536, 0.2210, 0.0631, 0.4499, 0.2564, 0.7642, 0.1411, 0.3626
]

all_data = train_data + test_data

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def train_mlp(p, N1, seed, lr=0.1, momentum=0.8, precision=0.5e-6, max_epochs=50000):
    np.random.seed(seed)
    X_train = []
    y_train = []
    for t in range(p, 100):
        x_in = [train_data[t - i] for i in range(1, p+1)]
        X_train.append(x_in)
        y_train.append([train_data[t]])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    num_samples = X_train.shape[0]
    
    W1 = np.random.rand(p, N1)
    b1 = np.random.rand(1, N1)
    W2 = np.random.rand(N1, 1)
    b2 = np.random.rand(1, 1)
    
    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)
    
    mse_history = []
    
    for epoch in range(max_epochs):
        # Online training (pattern by pattern) often works better for these small datasets and specific hyperparams
        mse = 0
        
        # shuffle for online training
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for i in indices:
            x = X_train[i:i+1]
            y = y_train[i:i+1]
            
            Z1 = np.dot(x, W1) + b1
            A1 = sigmoid(Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = sigmoid(Z2)
            
            error = y - A2
            mse += error[0,0]**2
            
            dZ2 = error * sigmoid_derivative(Z2)
            dA1 = np.dot(dZ2, W2.T)
            dZ1 = dA1 * sigmoid_derivative(Z1)
            
            dW2 = np.dot(A1.T, dZ2)
            db2 = dZ2
            dW1 = np.dot(x.T, dZ1)
            db1 = dZ1
            
            vW2 = lr * dW2 + momentum * vW2
            vb2 = lr * db2 + momentum * vb2
            vW1 = lr * dW1 + momentum * vW1
            vb1 = lr * db1 + momentum * vb1
            
            W2 += vW2
            b2 += vb2
            W1 += vW1
            b1 += vb1
            
        mse = mse / num_samples
        mse_history.append(mse)
        
        if mse < precision:
            break
            
    return W1, b1, W2, b2, mse_history, epoch + 1

networks = [
    {"name": "Rede 1", "p": 5, "N1": 10},
    {"name": "Rede 2", "p": 10, "N1": 15},
    {"name": "Rede 3", "p": 15, "N1": 25}
]

for net in networks:
    print(f"Training {net['name']}...")
    for t_idx in range(3):
        start = time.time()
        W1, b1, W2, b2, history, epochs = train_mlp(net['p'], net['N1'], seed=t_idx*100)
        print(f"  T{t_idx+1}: Epochs={epochs}, MSE={history[-1]:.8f}, Time={time.time()-start:.2f}s")
