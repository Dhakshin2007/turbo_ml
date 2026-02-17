import time
import random
import turbo_ml

# Matrix Size: 500 x 500
# This requires roughly 250,000,000 computations (500^3 * 2)
N = 500

print(f"Generating {N}x{N} Matrices...")
A = [random.random() for _ in range(N * N)]
B = [random.random() for _ in range(N * N)]
print("Data Ready.\n")

# --- COMPETITOR: Pure Python ---
print("Running Python Matrix Multiplication (This might take a while)...")
start = time.time()

# Standard Naive Matrix Mult in Python
C_py = [0.0] * (N * N)
for i in range(N):
    for k in range(N):
        val_a = A[i * N + k]
        for j in range(N):
            C_py[i * N + j] += val_a * B[k * N + j]

py_time = time.time() - start
print(f"Python Time: {py_time:.4f}s")

# --- HERO: Turbo ML (AVX + Parallel) ---
print("\nRunning Turbo ML...")
start = time.time()
C_rust = turbo_ml.matrix_multiply(A, B, N)
rust_time = time.time() - start
print(f"Turbo ML Time: {rust_time:.4f}s")

# --- VERDICT ---
print(f"\n---------------------------------------------")
print(f"SPEEDUP: {py_time / rust_time:.2f}x FASTER")
print(f"---------------------------------------------")