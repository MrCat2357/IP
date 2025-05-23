import numpy as np
from sympy import symbols, Matrix, sqrt, lambdify

# Chute inicial
X0 = np.array([200, 65, 200, 200, 100, 200, 300, 150, 50, 150], dtype=float)

# Constantes
x1, y1 = 100, 100

# Observações (distâncias)
Lb = np.array([
    104.4226, 141.4264, 100.0186, 206.1519, 70.6993,
    130.0119, 164.0010, 128.0688, 169.9940,
    100.0009, 111.7834, 158.1090,
    206.1490, 70.6874,
    250.0094
])

# Definição simbólica
x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = symbols('x2 y2 x3 y3 x4 y4 x5 y5 x6 y6')
vars_sym = Matrix([x2, y2, x3, y3, x4, y4, x5, y5, x6, y6])

# Funções de distância ao quadrado
def dist2(xa, ya, xb, yb):
    return (xa - xb)**2 + (ya - yb)**2

F = Matrix([
    sqrt(dist2(x1, y1, x2, y2)),
    sqrt(dist2(x1, y1, x3, y3)),
    sqrt(dist2(x1, y1, x4, y4)),
    sqrt(dist2(x1, y1, x5, y5)),
    sqrt(dist2(x1, y1, x6, y6)),
    sqrt(dist2(x2, y2, x3, y3)),
    sqrt(dist2(x2, y2, x4, y4)),
    sqrt(dist2(x2, y2, x5, y5)),
    sqrt(dist2(x2, y2, x6, y6)),
    sqrt(dist2(x3, y3, x4, y4)),
    sqrt(dist2(x3, y3, x5, y5)),
    sqrt(dist2(x3, y3, x6, y6)),
    sqrt(dist2(x4, y4, x5, y5)),
    sqrt(dist2(x4, y4, x6, y6)),
    sqrt(dist2(x5, y5, x6, y6))
])

# Jacobiana
J = F.jacobian(vars_sym)

# Peso (matriz identidade vezes sigma^2)
P = (1 / 0.01**2) * np.eye(15)

# Funções lambdificadas para avaliação numérica
F_func = lambdify(vars_sym, F, 'numpy')
J_func = lambdify(vars_sym, J, 'numpy')

# Iteração
Xprox = X0.copy()
X = np.ones_like(Xprox)
j = 0

while np.sum(np.abs(X)) > 0.01:
    j += 1
    Xant = Xprox.copy()
    F0 = np.array(F_func(*Xant)).astype(float).flatten()
    J0 = np.array(J_func(*Xant)).astype(float)
    K = Lb - F0
    JT_P = J0.T @ P
    X = np.linalg.solve(JT_P @ J0, JT_P @ K)
    Xprox = Xant + X
    v = J0 @ X - K

# Pós-processamento
Xant = Xprox.copy()
J0 = np.array(J_func(*Xant)).astype(float)
Ex = np.linalg.inv(J0.T @ P @ J0)
dp = np.sqrt(np.diag(Ex))

sig_pos = (v.T @ P @ v) / (15 - 10)
Ex = sig_pos * Ex
dp = np.sqrt(np.diag(Ex))

# Resultados
print("Coordenadas ajustadas:")
print(Xprox)

print("\nDesvios padrão das coordenadas:")
print(dp)
