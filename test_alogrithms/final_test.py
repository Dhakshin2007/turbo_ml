import turbo_ml as tm # Now it looks professional!

# Standard clean usage (like NumPy)
A = tm.Matrix([[1.0, 2.0], [3.0, 4.0]])
B = tm.Matrix([[1.0, 0.0], [0.0, 1.0]])

# Super fast multiply
C = A.matmul(B)

print(C) 
# Should output: <TurboMatrix (2x2)>