import time
import random
import turbo_ml  # Importing YOUR library!

# 1. Generate a massive dataset (10 Million numbers)
# This simulates a heavy ML dataset loading phase.
print("Generating data...")
data = [random.random() for _ in range(10_000_000)]
print("Data ready.\n")

# --- COMPETITOR: Standard Python ---
print("Running Standard Python...")
start_time = time.time()

# Simulating the same algorithm in Python
python_result = 0.0
for x in data:
    val = (x * x + x) / 0.5
    python_result += val

python_time = time.time() - start_time
print(f"Python Time: {python_time:.4f} seconds")

# --- HERO: Turbo ML (Your Library) ---
print("\nRunning Turbo ML...")
start_time = time.time()

# Calling your Rust function
rust_result = turbo_ml.compute_heavy(data)

rust_time = time.time() - start_time
print(f"Turbo ML Time: {rust_time:.4f} seconds")

# --- THE VERDICT ---
speedup = python_time / rust_time
print(f"\n---------------------------------------------")
print(f"SPEEDUP FACTOR: {speedup:.2f}x FASTER")
print(f"---------------------------------------------")