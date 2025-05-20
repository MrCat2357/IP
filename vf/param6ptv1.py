import numpy as np

# Ponto fixo conhecido
x1 = 2
y1 = 1

# Chute inicial para os 5 pontos estimados: (x2, y2, ..., x6, y6)
x = np.array([200, 65, 200, 200, 100, 200, 300, 150, 50, 150])

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

# Jacobiana numérica (para evitar escrever manualmente)
def jacobian(x, h=1e-6):
    n = len(x)
    m = len(f(x))
    J = np.zeros((m, n))
    fx = f(x)

    for i in range(n):
        x_perturbed = np.copy(x)
        x_perturbed[i] += h
        J[:, i] = (f(x_perturbed) - fx) / h

    return J

# Parâmetros
max_iter = 100
tolerance = 1e-10

# Iterações de Newton (versão de mínimos quadrados)
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
for i in range(5):
    print(f"x{i+2} = {x[2*i]:.6f}, y{i+2} = {x[2*i+1]:.6f}")
