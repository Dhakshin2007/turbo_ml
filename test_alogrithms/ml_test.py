import time
import random
import turbo_ml

# 1. Generate Fake Training Data
# 10,000 samples, 10 features each
print("Generating 100,000 data points...")
X = [[random.random() for _ in range(10)] for _ in range(10000)]
y = [sum(row) + random.random() for row in X] # Target is sum of inputs + noise

# --- COMPETITOR: Pure Python Implementation ---
class PythonLinearRegression:
    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters
        self.weights = []
        self.bias = 0.0

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        
        for _ in range(self.iters):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n_samples):
                # Predict
                pred = self.bias + sum(w*x for w, x in zip(self.weights, X[i]))
                error = pred - y[i]
                # Gradient
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            # Update
            self.weights = [w - (self.lr * d / n_samples) for w, d in zip(self.weights, dw)]
            self.bias -= self.lr * db / n_samples

print("\nTraining Pure Python Model...")
py_model = PythonLinearRegression(iters=100) # Only 100 iterations or it takes forever
start = time.time()
py_model.fit(X, y)
py_time = time.time() - start
print(f"Python Time: {py_time:.4f}s")

# --- HERO: Turbo ML Model ---
print("\nTraining Turbo ML Model...")
turbo_model = turbo_ml.TurboLinearRegression(0.01, 100)
start = time.time()
turbo_model.fit(X, y)
turbo_time = time.time() - start
print(f"Turbo ML Time: {turbo_time:.4f}s")

# --- VERDICT ---
print(f"\n---------------------------------------------")
print(f"SPEEDUP: {py_time / turbo_time:.2f}x FASTER")
print(f"---------------------------------------------")