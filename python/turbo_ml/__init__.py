from .turbo_ml import TurboMatrix, TurboSolver, run_heavy_logic
import time

# --- 1. The Matrix (NumPy Replacement) ---
class Matrix:
    def __init__(self, data):
        # Auto-detect dimensions (Adaptable to list of lists)
        if isinstance(data, list):
            if isinstance(data[0], list):
                self.rows = len(data)
                self.cols = len(data[0])
                # Flatten the data for Rust
                flat_data = [float(x) for row in data for x in row]
            else:
                # User passed a flat list, assume 1D vector (Rx1)
                self.rows = len(data)
                self.cols = 1
                flat_data = [float(x) for x in data]
            
            self._core = TurboMatrix(flat_data, self.rows, self.cols)
        
        elif isinstance(data, TurboMatrix):
            self._core = data
            self.rows, self.cols = data.shape()

    def matmul(self, other):
        """Matrix Multiplication (A @ B)"""
        if not isinstance(other, Matrix):
            raise TypeError("Can only multiply with another TurboML Matrix")
        
        result_core = self._core.matmul(other._core)
        return Matrix(result_core)

    def to_list(self):
        return self._core.to_list()

    def __repr__(self):
        return f"<TurboMatrix shape=({self.rows}, {self.cols})>"

# --- 2. The Model (Scikit-Learn Replacement) ---
class LinearRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.model = TurboSolver(lr, iterations)
    
    def fit(self, X, y):
        # X can be a Python list or a TurboMatrix
        if not isinstance(X, Matrix):
            X = Matrix(X)
        self.model.fit(X._core, y)
    
    def predict(self, X):
        if not isinstance(X, Matrix):
            X = Matrix(X)
        return self.model.predict(X._core)

# --- 3. The Complexity Optimizer ---
def optimize_heavy_loop(data, iterations=1000):
    """Runs generic heavy math logic in parallel on Rust."""
    return run_heavy_logic(data, iterations)