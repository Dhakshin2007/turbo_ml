# turbo_ml 🚀

**turbo_ml** is an experimental high-performance Python library backed by Rust, designed to accelerate selected numerical and ML-related operations compared to pure Python implementations.

⚠️ **Important:** This project is currently **under active development**.  
It is **not guaranteed** to work for every Python use case or environment.

---

## 📌 Project Status (Read This First)

- 🧪 **Development stage:** Early / Experimental
- ⚡ Focus: Performance-critical operations
- 🔧 Backend: Rust (via PyO3)
- 🛑 Not a drop-in replacement for all ML libraries

If you are looking for a fully mature ML framework, this may **not** yet be suitable for production use.

---

## ✅ Supported Python Versions (Guaranteed)

**Fully supported and tested:**

- Python **3.8**
- Python **3.9**
- Python **3.10**
- Python **3.11**
- Python **3.12**

> ✅ If `py --list` shows Python **3.12**, installation is guaranteed to work.

❌ **Not supported yet:**
- Python **3.13**
- Python **3.14+**

This limitation exists due to upstream Rust–Python bindings (PyO3).  
Support will be added once upstream compatibility is stable.

---

## 📦 Installation

### Standard Installation (Recommended)

```bash
pip install turbo_ml
```

If you are using Windows, Linux, or macOS with Python 3.8–3.12, this will install a **prebuilt binary wheel**.  
You **do NOT** need Rust or a C/C++ compiler.

---

## 🔍 Verifying Installation

After installation, verify with:

```bash
python -c "import turbo_ml; print('turbo_ml installed successfully')"
```

If no error appears, installation is complete.

---

## ▶️ How to Run the Example Scripts

This repository includes **two test files** to compare performance.

### 1️⃣ `base_python_test.py`
A normal Python script without any acceleration.

Run:
```bash
python base_python_test.py
```

Purpose:
- Acts as a baseline
- Uses standard Python logic
- Slower execution for heavy computation

---

### 2️⃣ `turbo_test.py`
Uses the `turbo_ml` library.

Run:
```bash
python turbo_test.py
```

Purpose:
- Imports `turbo_ml`
- Executes the same logic using Rust-accelerated functions
- Expected to run faster for supported operations

---

## ⚠️ Common Errors and How to Fix Them

### ❌ Error: Python version not supported

**Error message example:**
```
ERROR: turbo_ml requires Python < 3.13
```

**Cause:**  
You are using Python 3.13 or newer.

**Fix:**
1. Install Python **3.12**
2. Create a virtual environment using it:
   ```bash
   py -3.12 -m venv venv
   ```
3. Activate the environment and reinstall:
   ```bash
   pip install turbo_ml
   ```

---

### ❌ Error: pip tries to build from source (Rust / MSVC errors)

**Cause:**
- Unsupported Python version
- pip cache using an old build
- Wheel not selected

**Fix:**
```bash
pip uninstall turbo_ml -y
pip cache purge
pip install turbo_ml --no-cache-dir
```

If it still tries to compile, check:
```bash
python --version
```
Ensure it is **≤ 3.12**.

---

### ❌ Error: ModuleNotFoundError: turbo_ml

**Cause:**
- Installed in a different Python environment
- Virtual environment not activated

**Fix:**
```bash
which python
pip show turbo_ml
```

Ensure both point to the same environment.

---

## 🧪 Experimental Nature (Caution)

- This library **does not accelerate arbitrary Python code**
- Only specific operations are optimized
- Performance gains depend on:
  - Data size
  - Operation type
  - System architecture

You may observe:
- No speedup for small inputs
- Different behavior compared to pure Python
- Missing features (for now)

This is expected during early development.

---

## 🛠 Development Disclaimer

- APIs may change without notice
- Performance claims may evolve
- Backward compatibility is not guaranteed yet

If something breaks, it is likely a **known limitation**, not a user mistake.

---

## 📬 Feedback & Contributions

This project is actively evolving.

- Bug reports are welcome
- Performance benchmarks are appreciated
- Contributions should focus on **measurable speedups**

Repository:
https://github.com/Dhakshin2007/turbo_ml

---

## 🧠 Final Note

**turbo_ml** is an exploration of what’s possible when Python and Rust work together.

Use it to:
- Learn
- Experiment
- Benchmark
- Push performance boundaries

Not (yet) to:
- Replace full ML frameworks
- Assume universal compatibility

Thank you for testing an early-stage project.
