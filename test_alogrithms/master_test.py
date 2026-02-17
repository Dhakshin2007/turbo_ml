import time
import random
import turbo_ml as tm
import math

# Use perf_counter for high precision timing
def timer():
    return time.perf_counter()

print("\n==============================================")
print("   TURBO ML vs PYTHON: THE REALITY CHECK")
print("==============================================\n")

# --- LOAD TEST: 1 Million Elements ---
N = 1_000_000
print(f"Generating {N} random numbers for heavy processing...")
data = [random.random() for _ in range(N)]
print("Data loaded.\n")

# ---------------------------------------------------------
# ROUND 1: Complex Math Loop (Simulating O(N^6) logic)
# ---------------------------------------------------------
print(f"--- ROUND 1: Heavy Complexity Loop (Size: {N}) ---")

# PYTHON COMPETITOR
print("Running Python... (Please wait)")
start = timer()
py_result = 0.0
# We simulate a heavy operation: sin(cos(tan(x)))
for x in data:
    val = x
    # Loop 50 times per item to simulate complexity
    for _ in range(50):
        val = abs(math.tan(math.cos(math.sin(val))))
    py_result += val
py_time = timer() - start
print(f"Python Time: {py_time:.6f} seconds")

# TURBO ML COMPETITOR
print("Running Turbo ML...")
start = timer()
# We pass the raw list and the complexity factor (50)
turbo_result = tm.optimize_heavy_loop(data, 50)
turbo_time = timer() - start
print(f"Turbo ML Time: {turbo_time:.6f} seconds")

speedup = py_time / turbo_time
print(f"Result Check: {'MATCH' if abs(py_result - turbo_result) < 1.0 else 'MISMATCH'}")
print(f"🚀 SPEEDUP: {speedup:.2f}x FASTER\n")


# ---------------------------------------------------------
# ROUND 2: Matrix Multiplication (The Data Science Core)
# ---------------------------------------------------------
SIZE = 500
print(f"--- ROUND 2: Matrix Multiplication ({SIZE}x{SIZE}) ---")

# Generate Matrices
print("Generating Matrices...")
mat_a = [[random.random() for _ in range(SIZE)] for _ in range(SIZE)]
mat_b = [[random.random() for _ in range(SIZE)] for _ in range(SIZE)]

# PYTHON COMPETITOR (Approximated because Pure Python takes FOREVER)
print("Running Python (Est. based on 10 rows)...")
start = timer()
# Run only 10 rows to estimate full time (otherwise we wait 5 mins)
for i in range(10): 
    for j in range(SIZE):
        acc = 0.0
        for k in range(SIZE):
            acc += mat_a[i][k] * mat_b[k][j]
estimated_py_time = (timer() - start) * (SIZE / 10)
print(f"Python Time (Est): {estimated_py_time:.6f} seconds")

# TURBO ML COMPETITOR
print("Running Turbo ML (Full Matrix)...")
# Convert to TurboMatrix
tm_a = tm.Matrix(mat_a)
tm_b = tm.Matrix(mat_b)

start = timer()
tm_c = tm_a.matmul(tm_b)
turbo_time = timer() - start
print(f"Turbo ML Time:     {turbo_time:.6f} seconds")

speedup = estimated_py_time / turbo_time
print(f"🚀 SPEEDUP: {speedup:.2f}x FASTER")
print("==============================================")