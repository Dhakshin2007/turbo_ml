import time
import random

def get_time():
    return time.perf_counter()

print("--- STARTING PURE PYTHON BASELINE ---")

# 1. SETUP DATA (Same as your Rust test)
SIZE = 500
print(f"Generating {SIZE}x{SIZE} data...")
mat_a = [[random.random() for _ in range(SIZE)] for _ in range(SIZE)]
mat_b = [[random.random() for _ in range(SIZE)] for _ in range(SIZE)]

# 2. PURE PYTHON MATRIX MULTIPLICATION
print(f"Running Pure Python matmul ({SIZE}x{SIZE})...")
start = get_time()

# Standard O(n^3) nested loop implementation
result = [[0.0] * SIZE for _ in range(SIZE)]
for i in range(SIZE):
    for j in range(SIZE):
        for k in range(SIZE):
            result[i][j] += mat_a[i][k] * mat_b[k][j]

py_mat_time = get_time() - start
print(f"Python Matrix Time: {py_mat_time:.6f}s")


# 3. PURE PYTHON LINEAR REGRESSION
print(f"\nRunning Pure Python fit (100k samples)...")
ROWS, COLS = 100_000, 10
X = [[random.random() for _ in range(COLS)] for _ in range(ROWS)]
y = [random.random() for _ in range(ROWS)]
weights = [0.0] * COLS
lr = 0.01

start = get_time()
# Run 100 iterations (Same as your TurboSolver setup)
for _ in range(100):
    for i in range(ROWS):
        # Dot product
        pred = sum(X[i][j] * weights[j] for j in range(COLS))
        error = pred - y[i]
        # Gradient update
        for j in range(COLS):
            weights[j] -= lr * error * X[i][j]

py_ml_time = get_time() - start
print(f"Python ML Fit Time: {py_ml_time:.6f}s")

print("\n--- BASELINE COMPLETE ---")