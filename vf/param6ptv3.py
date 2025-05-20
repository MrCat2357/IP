import numpy as np

# Ponto fixo
x1, y1 = 100, 100

# Estimativas iniciais
x = np.array([200, 65, 200, 200, 100, 200, 300, 150, 50, 150])

# Distâncias observadas
d12, d13, d14, d15, d16 = 104.4226, 141.4264, 100.0186, 206.1519, 70.6993
d23, d24, d25, d26 = 130.0119, 164.0010, 128.0688, 169.9940
d34, d35, d36 = 100.0009, 111.7834, 158.1090
d45, d46 = 206.1490, 70.6874
d56 = 250.0094

# Função do sistema não-linear
def f(x):
    x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = x
    return np.array([
        np.sqrt((x1 - x2)**2 + (y1 - y2)**2) - d12,
        np.sqrt((x1 - x3)**2 + (y1 - y3)**2) - d13,
        np.sqrt((x1 - x4)**2 + (y1 - y4)**2) - d14,
        np.sqrt((x1 - x5)**2 + (y1 - y5)**2) - d15,
        np.sqrt((x1 - x6)**2 + (y1 - y6)**2) - d16,
        np.sqrt((x2 - x3)**2 + (y2 - y3)**2) - d23,
        np.sqrt((x2 - x4)**2 + (y2 - y4)**2) - d24,
        np.sqrt((x2 - x5)**2 + (y2 - y5)**2) - d25,
        np.sqrt((x2 - x6)**2 + (y2 - y6)**2) - d26,
        np.sqrt((x3 - x4)**2 + (y3 - y4)**2) - d34,
        np.sqrt((x3 - x5)**2 + (y3 - y5)**2) - d35,
        np.sqrt((x3 - x6)**2 + (y3 - y6)**2) - d36,
        np.sqrt((x4 - x5)**2 + (y4 - y5)**2) - d45,
        np.sqrt((x4 - x6)**2 + (y4 - y6)**2) - d46,
        np.sqrt((x5 - x6)**2 + (y5 - y6)**2) - d56,
    ])

def jacobian(x):
    x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = x

    def deriv(px, py, qx, qy):
        d = np.sqrt((px - qx)**2 + (py - qy)**2)
        if d == 0:
            return 0.0, 0.0
        return (px - qx) / d, (py - qy) / d

    J = np.zeros((15, 10))

    # f1 a f5
    J[0, 0], J[0, 1] = deriv(x2, y2, x1, y1)
    J[1, 2], J[1, 3] = deriv(x3, y3, x1, y1)
    J[2, 4], J[2, 5] = deriv(x4, y4, x1, y1)
    J[3, 6], J[3, 7] = deriv(x5, y5, x1, y1)
    J[4, 8], J[4, 9] = deriv(x6, y6, x1, y1)

    # Parciais das distâncias entre os pontos
    pares = [(2, 3, 5), (2, 4, 6), (2, 5, 7), (2, 6, 8),
             (3, 4, 9), (3, 5, 10), (3, 6, 11),
             (4, 5, 12), (4, 6, 13), (5, 6, 14)]

    idx = {2: 0, 3: 2, 4: 4, 5: 6, 6: 8}

    coords = {2: (x2, y2), 3: (x3, y3), 4: (x4, y4),
              5: (x5, y5), 6: (x6, y6)}

    for a, b, row in pares:
        ax, ay = coords[a]
        bx, by = coords[b]
        dax, day = deriv(ax, ay, bx, by)
        dbx, dby = deriv(bx, by, ax, ay)
        J[row, idx[a]], J[row, idx[a]+1] = dax, day
        J[row, idx[b]], J[row, idx[b]+1] = dbx, dby

    return J

# Parâmetros do ajuste
sigma = 0.01  # desvio padrão das observações (10 mm)
max_iter = 5000
tolerance = 1e-4

# Iterações de Newton com mínimos quadrados
for i in range(max_iter):
    fx = f(x) / sigma
    J = jacobian(x) / sigma

    try:
        JT = J.T
        A = np.dot(JT, J)
        b = np.dot(JT, -fx)
        dx = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Sistema singular na iteração", i)
        break

    x = x + dx
    print(f"Iteração {i+1}, ||dx|| = {np.linalg.norm(dx):.12f}")

    if np.linalg.norm(dx) < tolerance:
        break

# Solução final
print("\nSolução aproximada encontrada:")
for i in range(5):
    print(f"x{i+2} = {x[2*i]:.6f}, y{i+2} = {x[2*i+1]:.6f}")

# Cálculo da matriz de covariância a posteriori
fx_final = f(x) / sigma
J_final = jacobian(x) / sigma
A = np.dot(J_final.T, J_final)
n = 15  # observações
u = 10  # parâmetros
residuos = fx_final
s0_2 = np.sum(residuos**2) / (n - u)
cov_x = s0_2 * np.linalg.inv(A)

# Desvios padrão das coordenadas
desvios = np.sqrt(np.diag(cov_x))
print("\nDesvios padrão estimados:")
for i, name in enumerate(["x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5", "x6", "y6"]):
    print(f"{name}: {desvios[i]:.4f} m")

print(f"\nFator de variância a posteriori: {s0_2:.8f}")
print(f"Desvio padrão a posteriori: {np.sqrt(s0_2):.6f} m")
