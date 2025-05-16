import numpy as np

# Chute inicial
x = np.array([2.0, 2.0])  # Inicializa x com o chute inicial

# Função do sistema
def f(x):
    x1, x2 = x
    return np.array([
        x1 + x2 - 2 * x2**2 + 4,
        x1**2 + x2**2 - 8,
        3 * x1**2 - x2**2 - 7.7
    ])

# Jacobiana do sistema
def jacobian(x):
    x1, x2 = x
    return np.array([
        [1, 1 - 4 * x2],
        [2 * x1, 2 * x2],
        [6 * x1, -2 * x2]
    ])

# Parâmetros
max_iter = 3
tolerance = 1e-6

# Iterações de Newton (versão de mínimos quadrados)
for i in range(max_iter):
    fx = f(x)
    J = jacobian(x)
    
    # Resolve (J^T J) * dx = J^T * (-f)
    try:
        JT = J.T
        A = JT @ J              # J^T * J
        b = JT @ (-fx)          # J^T * (-fx)
        dx = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Jacobiana transposta * Jacobiana é singular na iteração", i)
        break

    x = x + dx
    print(f"Iteração {i+1}, x = {x}, ||dx|| = {np.linalg.norm(dx):.6f}")

    if np.linalg.norm(dx) < tolerance:
        break

# Resultado final
print("\nSolução aproximada encontrada:")
print(f"x1 = {x[0]:.6f}")
print(f"x2 = {x[1]:.6f}")
