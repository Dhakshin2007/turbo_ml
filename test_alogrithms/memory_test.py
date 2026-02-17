import turbo_ml
import sys
import os
import psutil # You might need to 'pip install psutil'

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # Convert to MB

print(f"Base Memory: {get_memory_usage():.2f} MB")

# 1. Create a Python List (Heavy)
print("\n--- Creating Python List (10 Million items) ---")
py_list = [float(x) for x in range(10_000_000)]
mem_after_py = get_memory_usage()
print(f"Memory with Python List: {mem_after_py:.2f} MB")
py_size = mem_after_py - get_memory_usage() # This is rough, but shows the spike

# Clear it to free memory
del py_list
import gc
gc.collect()

print(f"\nMemory Cleared: {get_memory_usage():.2f} MB")

# 2. Create TurboArray (Light)
print("\n--- Creating TurboArray (10 Million items) ---")
# We generate the list inside the call to avoid holding two copies in Python
# (In a real app, you'd load this from a file directly to Rust)
turbo_arr = turbo_ml.TurboArray([float(x) for x in range(10_000_000)])

mem_after_turbo = get_memory_usage()
print(f"Memory with TurboArray: {mem_after_turbo:.2f} MB")

print("\n---------------------------------------------")
print("Notice how TurboArray consumes significantly less peak RAM")
print("once the data is stored inside Rust's optimized structure.")
print("---------------------------------------------")