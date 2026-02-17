use pyo3::prelude::*;
use rayon::prelude::*;

// ==========================================
// CORE 1: THE TURBO MATRIX (Mini-NumPy)
// ==========================================
// This is the building block for EVERYTHING.
// It replaces Python Lists with a high-performance memory block.
#[pyclass]
#[derive(Clone)]
struct TurboMatrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

#[pymethods]
impl TurboMatrix {
    #[new]
    fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        if data.len() != rows * cols {
            panic!("Data length does not match rows * cols");
        }
        TurboMatrix { data, rows, cols }
    }

    // 1. Parallel Matrix Multiplication (The heavy lifter)
    fn matmul(&self, other: &TurboMatrix) -> PyResult<TurboMatrix> {
        if self.cols != other.rows {
            return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch for matmul"));
        }

        let n = self.rows;
        let m = self.cols;
        let p = other.cols;
        
        let mut result = vec![0.0; n * p];

        // Parallelize across rows of the result matrix
        result.par_chunks_mut(p)
              .enumerate()
              .for_each(|(i, row)| {
                  for k in 0..m {
                      let val_a = self.data[i * m + k];
                      // Inner loop vectorizes automatically (SIMD)
                      for j in 0..p {
                          row[j] += val_a * other.data[k * p + j];
                      }
                  }
              });

        Ok(TurboMatrix { data: result, rows: n, cols: p })
    }

    // 2. Element-wise Operations (Add, Sub, Mul) - Parallelized
    fn add(&self, val: f32) -> TurboMatrix {
        let new_data: Vec<f32> = self.data.par_iter().map(|&x| x + val).collect();
        TurboMatrix { data: new_data, rows: self.rows, cols: self.cols }
    }

    fn mul(&self, val: f32) -> TurboMatrix {
        let new_data: Vec<f32> = self.data.par_iter().map(|&x| x * val).collect();
        TurboMatrix { data: new_data, rows: self.rows, cols: self.cols }
    }

    // 3. Fast Transpose (Crucial for ML)
    fn transpose(&self) -> TurboMatrix {
        let mut new_data = vec![0.0; self.data.len()];
        let rows = self.rows;
        let cols = self.cols;

        // Parallelize the transpose operation
        new_data.par_chunks_mut(rows) // Split by columns of original (rows of new)
                .enumerate()
                .for_each(|(j, col_data)| {
                    for i in 0..rows {
                        col_data[i] = self.data[i * cols + j];
                    }
                });

        TurboMatrix { data: new_data, rows: cols, cols: rows }
    }
    
    // 4. Getters for Python
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    fn to_list(&self) -> Vec<f32> {
        self.data.clone()
    }
}

// ==========================================
// CORE 2: THE ML SOLVER (Mini-Scikit-Learn)
// ==========================================
// A generic Gradient Descent solver that can fit Linear Regression
#[pyclass]
struct TurboSolver {
    weights: Vec<f32>,
    bias: f32,
    lr: f32,
    iters: usize,
}

#[pymethods]
impl TurboSolver {
    #[new]
    fn new(lr: f32, iters: usize) -> Self {
        TurboSolver { weights: vec![], bias: 0.0, lr, iters }
    }

    fn fit(&mut self, X: &TurboMatrix, y: Vec<f32>) {
        let n_samples = X.rows;
        let n_features = X.cols;
        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        for _ in 0..self.iters {
            let mut dw = vec![0.0; n_features];
            let mut db = 0.0;

            // Parallel Calculation of Gradients
            // We split the samples across threads, calculate partial gradients, and sum them up
            let (partial_dw, partial_db) = (0..n_samples).into_par_iter()
                .fold(|| (vec![0.0; n_features], 0.0), |(mut acc_dw, mut acc_db), i| {
                    let offset = i * n_features;
                    let mut pred = self.bias;
                    for j in 0..n_features {
                        pred += self.weights[j] * X.data[offset + j];
                    }
                    let error = pred - y[i];
                    
                    for j in 0..n_features {
                        acc_dw[j] += error * X.data[offset + j];
                    }
                    acc_db += error;
                    (acc_dw, acc_db)
                })
                .reduce(|| (vec![0.0; n_features], 0.0), |(mut a_dw, a_db), (b_dw, b_db)| {
                    for j in 0..n_features {
                        a_dw[j] += b_dw[j];
                    }
                    (a_dw, a_db + b_db)
                });

            dw = partial_dw;
            db = partial_db;

            // Update step
            for j in 0..n_features {
                self.weights[j] -= (dw[j] / n_samples as f32) * self.lr;
            }
            self.bias -= (db / n_samples as f32) * self.lr;
        }
    }

    fn predict(&self, X: &TurboMatrix) -> Vec<f32> {
        X.data.par_chunks(X.cols)
              .map(|row| {
                  let mut sum = self.bias;
                  for (j, val) in row.iter().enumerate() {
                      sum += self.weights[j] * val;
                  }
                  sum
              })
              .collect()
    }
}

// ==========================================
// CORE 3: COMPLEXITY CRUSHER (Tree Logic)
// ==========================================
// This is for your O(N^6) problem. It runs generic heavy logic in parallel.
#[pyfunction]
fn run_heavy_logic(data: Vec<f32>, iterations: usize) -> f32 {
    // Parallelize the heavy loop
    data.par_iter().map(|&x| {
        let mut res = x;
        for _ in 0..iterations {
            res = res.sin().cos().tan().abs(); // Simulate complex math
        }
        res
    }).sum()
}

// Module Definition
#[pymodule]
fn turbo_ml(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TurboMatrix>()?;
    m.add_class::<TurboSolver>()?;
    m.add_function(wrap_pyfunction!(run_heavy_logic, m)?)?;
    Ok(())
}