import turbo_ml as tm
import time
import random

# Use high-precision timer
def get_time():
    return time.perf_counter()

print("--- STARTING RAW ENGINE TEST ---\n")

# 1. SETUP DATA
SIZE = 500
print(f"Generating {SIZE}x{SIZE} data...")
flat_data = [random.random() for _ in range(SIZE * SIZE)]

# 2. TEST MATRIX MULTIPLICATION (The 436x Beast)
# We call TurboMatrix directly to avoid wrapper shape errors
print("Running TurboMatrix matmul...")
A = tm.TurboMatrix(flat_data, SIZE, SIZE)
B = tm.TurboMatrix(flat_data, SIZE, SIZE)

start = get_time()
C = A.matmul(B)
turbo_mat_time = get_time() - start

# Estimate Python Time (Pure Python is too slow to run full 500x500)
py_est_time = 21.98  # Based on your previous benchmark result
print(f"Matrix Time: {turbo_mat_time:.6f}s")
print(f"🚀 Speedup:  {py_est_time / turbo_mat_time:.2f}x Faster\n")

# 3. TEST ML SOLVER (The Parallel Scikit-Learn)
print("Running TurboSolver fit (100k samples)...")
ROWS, COLS = 100_000, 10
X_data = [random.random() for _ in range(ROWS * COLS)]
y_data = [random.random() for _ in range(ROWS)]

X = tm.TurboMatrix(X_data, ROWS, COLS)
solver = tm.TurboSolver(0.01, 100) # lr=0.01, iterations=100

start = get_time()
solver.fit(X, y_data)
turbo_ml_time = get_time() - start

py_ml_est = 25.63 # Based on your previous benchmark result
print(f"ML Fit Time: {turbo_ml_time:.6f}s")
print(f"🚀 Speedup:   {py_ml_est / turbo_ml_time:.2f}x Faster\n")

print("--- TEST COMPLETE ---")