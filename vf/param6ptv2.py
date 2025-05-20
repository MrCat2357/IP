import numpy as np

# Ponto fixo conhecido
x1 = 100
y1 = 100

# Chute inicial para os 5 pontos estimados: (x2, y2, ..., x6, y6)
x = np.array([200, 75, 195, 205, 95, 200, 300, 160, 50, 150])

# Distâncias conhecidas (você pode ajustar isso)
d12, d13, d14, d15, d16 = 104.4226, 141.4264, 100.0186, 200.1519, 70.6993
d23, d24, d25, d26 = 130.0119, 164.0010, 128.0688, 169.9940
d34, d35, d36 = 100.0009, 111.7834, 158.1090
d45, d46 = 206.1490, 70.6874
d56 = 250.0094

# Sistema não-linear: diferenças entre distâncias observadas e calculadas
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

    # f1 a f5 — derivadas de distâncias do ponto fixo (x1, y1) aos demais
    J[0, 0], J[0, 1] = deriv(x2, y2, x1, y1)
    J[1, 2], J[1, 3] = deriv(x3, y3, x1, y1)
    J[2, 4], J[2, 5] = deriv(x4, y4, x1, y1)
    J[3, 6], J[3, 7] = deriv(x5, y5, x1, y1)
    J[4, 8], J[4, 9] = deriv(x6, y6, x1, y1)

    # f6: dist(x2, x3)
    dx, dy = deriv(x2, y2, x3, y3)
    J[5, 0], J[5, 1] = dx, dy
    dx, dy = deriv(x3, y3, x2, y2)
    J[5, 2], J[5, 3] = dx, dy

    # f7: dist(x2, x4)
    dx, dy = deriv(x2, y2, x4, y4)
    J[6, 0], J[6, 1] = dx, dy
    dx, dy = deriv(x4, y4, x2, y2)
    J[6, 4], J[6, 5] = dx, dy

    # f8: dist(x2, x5)
    dx, dy = deriv(x2, y2, x5, y5)
    J[7, 0], J[7, 1] = dx, dy
    dx, dy = deriv(x5, y5, x2, y2)
    J[7, 6], J[7, 7] = dx, dy

    # f9: dist(x2, x6)
    dx, dy = deriv(x2, y2, x6, y6)
    J[8, 0], J[8, 1] = dx, dy
    dx, dy = deriv(x6, y6, x2, y2)
    J[8, 8], J[8, 9] = dx, dy

    # f10: dist(x3, x4)
    dx, dy = deriv(x3, y3, x4, y4)
    J[9, 2], J[9, 3] = dx, dy
    dx, dy = deriv(x4, y4, x3, y3)
    J[9, 4], J[9, 5] = dx, dy

    # f11: dist(x3, x5)
    dx, dy = deriv(x3, y3, x5, y5)
    J[10, 2], J[10, 3] = dx, dy
    dx, dy = deriv(x5, y5, x3, y3)
    J[10, 6], J[10, 7] = dx, dy

    # f12: dist(x3, x6)
    dx, dy = deriv(x3, y3, x6, y6)
    J[11, 2], J[11, 3] = dx, dy
    dx, dy = deriv(x6, y6, x3, y3)
    J[11, 8], J[11, 9] = dx, dy

    # f13: dist(x4, x5)
    dx, dy = deriv(x4, y4, x5, y5)
    J[12, 4], J[12, 5] = dx, dy
    dx, dy = deriv(x5, y5, x4, y4)
    J[12, 6], J[12, 7] = dx, dy

    # f14: dist(x4, x6)
    dx, dy = deriv(x4, y4, x6, y6)
    J[13, 4], J[13, 5] = dx, dy
    dx, dy = deriv(x6, y6, x4, y4)
    J[13, 8], J[13, 9] = dx, dy

    # f15: dist(x5, x6)
    dx, dy = deriv(x5, y5, x6, y6)
    J[14, 6], J[14, 7] = dx, dy
    dx, dy = deriv(x6, y6, x5, y5)
    J[14, 8], J[14, 9] = dx, dy

    return J


# Parâmetros
max_iter = 200
tolerance = 1e-10

# Iterações de Newton (versão de mínimos quadrados)
for i in range(max_iter):
    fx = f(x)
    J = jacobian(x)
    
    try:
        JT = J.T
        A = np.dot(JT, J)       # JT @ J → np.dot(JT, J)
        b = np.dot(JT, -fx)     # JT @ (-fx) → np.dot(JT, -fx)
        dx = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Jacobiana transposta * Jacobiana é singular na iteração", i)
        break

    x = x + dx
    print(f"Iteração {i+1}, x = {x}, ||dx|| = {np.linalg.norm(dx):.12f}")
    for i in range(5):
        print(f"x{i+2} = {x[2*i]:.6f}, y{i+2} = {x[2*i+1]:.6f}")

    if np.linalg.norm(dx) < tolerance:
        break


# Resultado final
print("\nSolução aproximada encontrada:")
for i in range(5):
    print(f"x{i+2} = {x[2*i]:.6f}, y{i+2} = {x[2*i+1]:.6f}")
