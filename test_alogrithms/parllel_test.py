import time
import random
import turbo_ml

# 1. Create Data (Increase size to see parallel benefits)
# 500,000 samples!
print("Generating 500,000 samples...")
X = [[random.random() for _ in range(10)] for _ in range(500_000)]
y = [sum(x) for x in X] # Dummy target

# 2. Train the model (We just need it initialized)
model = turbo_ml.TurboLinearRegression(0.01, 10)
model.fit(X[:1000], y[:1000]) # Quick train just to set weights

print("\n--- Running Parallel Prediction ---")

# Python list comprehension (Single Core)
start = time.time()
# Simulating what Python would do:
py_preds = []
for sample in X:
    pred = 0.0
    for val in sample:
        pred += val * 0.5 # Dummy weight
    py_preds.append(pred)
print(f"Python Prediction Time: {time.time() - start:.4f}s")

# Turbo ML (Multi-Core Rust)
start = time.time()
rust_preds = model.predict(X)
rust_time = time.time() - start
print(f"Turbo ML Prediction Time: {rust_time:.4f}s")

print(f"\nSPEEDUP: {(time.time() - start - rust_time) / rust_time:.2f}x FASTER")