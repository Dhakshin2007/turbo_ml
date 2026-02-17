import time
import random
import turbo_ml

# CONFIGURATION
ROWS = 500_000
COLS = 10

print(f"Generating {ROWS} samples with {COLS} features...")

# 1. Generate Data
# We create a FLAT list directly to simulate reading a binary file or buffer
# (This simulates how real 'Big Data' is loaded)
flat_X = [random.random() for _ in range(ROWS * COLS)]
y = [random.random() for _ in range(ROWS)]

# Create a slow nested version just for the Python comparison
nested_X = [flat_X[i*COLS : (i+1)*COLS] for i in range(ROWS)]

print("Data Generation Complete.\n")

# --- COMPETITOR: Standard Python ---
print("Running Standard Python (Nested Lists)...")
start = time.time()
py_preds = []
# Standard slow python logic
for row in nested_X:
    pred = 0.0
    for val in row:
        pred += val * 0.5 
    py_preds.append(pred)
py_time = time.time() - start
print(f"Python Time: {py_time:.4f}s")

# --- HERO: Turbo ML (Flat Memory) ---
print("\nRunning Turbo ML (Flat Memory + Parallel)...")
model = turbo_ml.TurboLinearRegression(0.01, 10)
# We treat training as instant for this test to focus on prediction throughput
model.fit(flat_X[:1000], 100, COLS, y[:100]) 

start = time.time()
# Notice we pass ROWS and COLS so Rust knows how to slice the flat array
rust_preds = model.predict(flat_X, ROWS, COLS)
rust_time = time.time() - start
print(f"Turbo ML Time: {rust_time:.4f}s")

# --- VERDICT ---
speedup = py_time / rust_time
print(f"\n---------------------------------------------")
print(f"SPEEDUP: {speedup:.2f}x FASTER")
print(f"---------------------------------------------")