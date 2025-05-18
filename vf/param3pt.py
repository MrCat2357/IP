import numpy as np

# Coordenada fixa conhecida
P0 = np.array([0.0, 0.0])

# Coordenadas reais (não usadas na estimação, só para referência)
P1_real = np.array([2.0, 1.0])
P2_real = np.array([1.0, 3.0])

# Distâncias reais (calculadas entre os pontos reais)
def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

d01 = euclidean(P0, P1_real)
d02 = euclidean(P0, P2_real)
d12 = euclidean(P1_real, P2_real)

# Estimativas iniciais (x1, y1, x2, y2)
x = np.array([1.0, 0.0, 0.0, 2.0])  # x1, y1, x2, y2

# Função de resíduos
def residuals(x):
    x1, y1, x2, y2 = x
    P1 = np.array([x1, y1])
    P2 = np.array([x2, y2])
    r1 = euclidean(P0, P1) - d01
    r2 = euclidean(P0, P2) - d02
    r3 = euclidean(P1, P2) - d12
    return np.array([r1, r2, r3])

# Jacobiana analítica 3x4
def jacobian(x):
    x1, y1, x2, y2 = x
    P1 = np.array([x1, y1])
    P2 = np.array([x2, y2])

    # Evita divisão por zero
    d01_est = euclidean(P0, P1) + 1e-8
    d02_est = euclidean(P0, P2) + 1e-8
    d12_est = euclidean(P1, P2) + 1e-8

    J = np.zeros((3, 4))  # 3 resíduos, 4 variáveis

    # ∂r1/∂x1, ∂r1/∂y1
    J[0, 0] = (x1 - 0) / d01_est
    J[0, 1] = (y1 - 0) / d01_est

    # ∂r2/∂x2, ∂r2/∂y2
    J[1, 2] = (x2 - 0) / d02_est
    J[1, 3] = (y2 - 0) / d02_est

    # ∂r3/∂x1, ∂r3/∂y1, ∂r3/∂x2, ∂r3/∂y2
    J[2, 0] = (x1 - x2) / d12_est
    J[2, 1] = (y1 - y2) / d12_est
    J[2, 2] = (x2 - x1) / d12_est
    J[2, 3] = (y2 - y1) / d12_est

    return J

# Parâmetros de iteração
max_iter = 10
tolerance = 1e-6

print("Iterações:")
for i in range(max_iter):
    r = residuals(x)
    J = jacobian(x)
    
    # Gauss-Newton update
    JTJ = J.T @ J
    JTr = J.T @ r
    dx = -np.linalg.solve(JTJ, JTr)

    x = x + dx

    print(f"Iteração {i+1}")
    print("x =", x)
    print("Norma(dx) =", np.linalg.norm(dx))
    print("Erro total (norma dos resíduos):", np.linalg.norm(r))
    print("-" * 40)

    if np.linalg.norm(dx) < tolerance:
        break

# Resultado final
x1_opt = x[0:2]
x2_opt = x[2:4]
print("\nEstimativas finais:")
print("P1 estimado:", x1_opt)
print("P2 estimado:", x2_opt)

print("\nValores reais:")
print("P1 real:", P1_real)
print("P2 real:", P2_real)
