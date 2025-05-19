import numpy as np

# --------------------------
# Parâmetros e normalização
# --------------------------

# Ponto fixo conhecido (normalizável também)
x1, y1 = 2.0, 1.0

# Distâncias reais entre os pontos (pode vir de sensores, por exemplo)
distancias = np.array([
    3, 4, 5, 3.2, 4.1,  # do ponto fixo para os demais
    5, 4, 3, 3,         # entre os estimados
    3, 4, 5,
    5, 3.3,
    2.8
])

max_dist = np.max(distancias)  # valor de normalização
D = distancias / max_dist      # distâncias normalizadas

# Chute inicial (em escala normalizada)
x = np.array([
    5.0, 1.0,
    6.0, 4.0,
    2.0, 6.0,
    -1.0, 4.0,
    -2.0, 1.0
]) / max_dist

# Normaliza também o ponto fixo
x1 /= max_dist
y1 /= max_dist

# ------------------------
# Funções do sistema
# ------------------------

def dist(px, py, qx, qy):
    return np.sqrt((px - qx)**2 + (py - qy)**2)

def f(x):
    x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = x
    return np.array([
        dist(x1, y1, x2, y2) - D[0],
        dist(x1, y1, x3, y3) - D[1],
        dist(x1, y1, x4, y4) - D[2],
        dist(x1, y1, x5, y5) - D[3],
        dist(x1, y1, x6, y6) - D[4],
        dist(x2, y2, x3, y3) - D[5],
        dist(x2, y2, x4, y4) - D[6],
        dist(x2, y2, x5, y5) - D[7],
        dist(x2, y2, x6, y6) - D[8],
        dist(x3, y3, x4, y4) - D[9],
        dist(x3, y3, x5, y5) - D[10],
        dist(x3, y3, x6, y6) - D[11],
        dist(x4, y4, x5, y5) - D[12],
        dist(x4, y4, x6, y6) - D[13],
        dist(x5, y5, x6, y6) - D[14],
    ])

def jacobian(x):
    def dfdx(px, py, qx, qy):
        d = dist(px, py, qx, qy)
        if d == 0:
            return 0.0, 0.0
        return (px - qx)/d, (py - qy)/d

    x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = x
    J = np.zeros((15, 10))

    # f1–f5: do ponto fixo para os estimados
    J[0, 0:2] = dfdx(x2, y2, x1, y1)
    J[1, 2:4] = dfdx(x3, y3, x1, y1)
    J[2, 4:6] = dfdx(x4, y4, x1, y1)
    J[3, 6:8] = dfdx(x5, y5, x1, y1)
    J[4, 8:10] = dfdx(x6, y6, x1, y1)

    # f6–f15: entre os pontos estimados
    pares = [
        (0, 2), (0, 4), (0, 6), (0, 8),
        (2, 4), (2, 6), (2, 8),
        (4, 6), (4, 8),
        (6, 8)
    ]
    for i, (i1, i2) in enumerate(pares, start=5):
        J[i, i1], J[i, i1+1] = dfdx(x[i1], x[i1+1], x[i2], x[i2+1])
        J[i, i2], J[i, i2+1] = dfdx(x[i2], x[i2+1], x[i1], x[i1+1])

    return J

# ------------------------
# Método de Newton-Gauss
# ------------------------

max_iter = 2000
tolerance = 1e-10

for i in range(max_iter):
    fx = f(x)
    J = jacobian(x)
    JT = J.T
    A = JT @ J
    b = JT @ (-fx)

    try:
        dx = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print(f"Matriz singular na iteração {i}")
        break

    x = x + dx
    erro = np.linalg.norm(dx)
    print(f"Iteração {i+1}, ||dx|| = {erro:.12e}")
    if erro < tolerance:
        break

# ------------------------
# Resultado final
# ------------------------

print("\nSolução aproximada encontrada:")
for i in range(5):
    x_real = x[2*i] * max_dist
    y_real = x[2*i+1] * max_dist
    print(f"x{i+2} = {x_real:.6f}, y{i+2} = {y_real:.6f}")
