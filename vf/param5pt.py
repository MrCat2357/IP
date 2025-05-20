import numpy as np

# Ponto conhecido
x1 = 2
y1 = 1

# Chute inicial: 4 pontos estimados (x2, y2, x3, y3, x4, y4, x5, y5)
x = np.array([5.003, 0.98, 2.002, 4.994, 5.006, 4.989, 1.007, 2.017])

# Função do sistema com 4 distâncias ao ponto fixo e 6 distâncias entre pares dos 4 estimados
def f(x):
    x2, y2, x3, y3, x4, y4, x5, y5 = x

    return np.array([
        np.sqrt((x1 - x2)**2 + (y1 - y2)**2) - 3,
        np.sqrt((x1 - x3)**2 + (y1 - y3)**2) - 4,
        np.sqrt((x1 - x4)**2 + (y1 - y4)**2) - 5,
        np.sqrt((x1 - x5)**2 + (y1 - y5)**2) - 1.41,
        np.sqrt((x2 - x3)**2 + (y2 - y3)**2) - 5,
        np.sqrt((x2 - x4)**2 + (y2 - y4)**2) - 4,
        np.sqrt((x2 - x5)**2 + (y2 - y5)**2) - 4.12,
        np.sqrt((x3 - x4)**2 + (y3 - y4)**2) - 3,
        np.sqrt((x3 - x5)**2 + (y3 - y5)**2) - 3.16,
        np.sqrt((x4 - x5)**2 + (y4 - y5)**2) - 5,
    ])

# Jacobiana do sistema
def jacobian(x):
    x2, y2, x3, y3, x4, y4, x5, y5 = x

    def d_dx(xa, ya, xb, yb):
        d = np.sqrt((xa - xb)**2 + (ya - yb)**2)
        return [(xa - xb)/d, (ya - yb)/d]

    return np.array([
        d_dx(x2, y2, x1, y1) + [0]*6,
        [0]*2 + d_dx(x3, y3, x1, y1) + [0]*4,
        [0]*4 + d_dx(x4, y4, x1, y1) + [0]*2,
        [0]*6 + d_dx(x5, y5, x1, y1),
        d_dx(x2, y2, x3, y3)[:2] + d_dx(x3, y3, x2, y2)[:2] + [0]*4,
        d_dx(x2, y2, x4, y4)[:2] + [0]*2 + d_dx(x4, y4, x2, y2)[:2] + [0]*2,
        d_dx(x2, y2, x5, y5)[:2] + [0]*4 + d_dx(x5, y5, x2, y2)[:2],
        [0]*2 + d_dx(x3, y3, x4, y4)[:2] + d_dx(x4, y4, x3, y3)[:2] + [0]*2,
        [0]*2 + d_dx(x3, y3, x5, y5)[:2] + [0]*2 + d_dx(x5, y5, x3, y3)[:2],
        [0]*4 + d_dx(x4, y4, x5, y5)[:2] + d_dx(x5, y5, x4, y4)[:2],
    ])

# Parâmetros
max_iter = 2000000
tolerance = 1e-7

# Iterações de Newton
for i in range(max_iter):
    fx = f(x)
    J = jacobian(x)

    try:
        JT = J.T
        A = JT @ J
        b = JT @ (-fx)
        dx = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Jacobiana transposta * Jacobiana é singular na iteração", i)
        break

    x = x + dx
    print(f"Iteração {i+1}, x = {x}, ||dx|| = {np.linalg.norm(dx):.12f}")

    if np.linalg.norm(dx) < tolerance:
        break

# Resultado final
print("\nSolução aproximada encontrada:")
for i in range(4):
    print(f"x{i+2} = {x[2*i]:.6f}, y{i+2} = {x[2*i+1]:.6f}")
