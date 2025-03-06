import sympy
import numpy as np

d1 = 4
d2 = 7
d3 = 13
d4 = 21
d5 = 33
L = np.array([d1, d2, d3, d4, d5])
A = np.array([[1, 0, 0, 1],
              [1, 1, 0, 0],
              [2, 1, 0, 1],
              [1, 0, 1, 0],
              [0, 0, 0, 1]])

# Calculando o posto e os índices das colunas principais
_, inds = sympy.Matrix(A).T.rref()

# Construindo a submatriz A_* usando as colunas principais de A
def A_(i):
    return A[inds[i]]

def L_(i):
    return L[inds[i]]

A_star = np.array([A_(0), A_(1), A_(2), A_(3)])
L_star = np.array([L_(0), L_(1), L_(2), L_(3)])
x0 = np.linalg.solve(A_star, L_star)

print (A_(0))
print (A_(1))
print (A_(2))
print (inds)
print(A_star)
print(A)
print(L_star)
print(x0)